module ModelCore

using LinearAlgebra
using SparseArrays
using Random
using Statistics
using PythonCall: pyconvert
using JuMP
using HiGHS

export neumann_apply, neumann_apply!, NeumannCache,
       compute_investment, solve_planner, fast_loop, solve_firm_lp,
       run_montecarlo

_v(x) = pyconvert(Vector{Float64}, x)
_m(x) = pyconvert(Matrix{Float64}, x)

struct NeumannCache
    tmp1::Vector{Float64}
    tmp2::Vector{Float64}
end
NeumannCache(n::Int) = NeumannCache(zeros(n), zeros(n))

# Allocating convenience version (unchanged semantics).
function neumann_apply(A::AbstractMatrix{Float64}, v::AbstractVecOrMat{Float64}, k::Int=20)
    res  = copy(v)
    term = copy(res)
    tmp  = similar(res)
    @inbounds for _ in 1:k
        mul!(tmp, A, term)
        res .+= tmp
        term, tmp = tmp, term
    end
    return res
end

function neumann_apply!(cache::NeumannCache, A::AbstractMatrix{Float64},
                        v::Vector{Float64}, k::Int=20)
    res = copy(v)                          # result accumulator (one alloc)
    t1  = cache.tmp1                       # bind to local variables
    t2  = cache.tmp2
    copyto!(t1, v)                         # initial term
    @inbounds for _ in 1:k
        mul!(t2, A, t1)
        res .+= t2
        t1, t2 = t2, t1                    
    end
    return res
end

function neumann_apply(A_py, v_py, k::Int=20)
    A = _m(A_py)
    try
        v = pyconvert(Vector{Float64}, v_py)
        return neumann_apply(A, v, k)
    catch
        V = _m(v_py)
        return neumann_apply(A, V, k)
    end
end

function compute_investment_native(G_hat_v::Vector{Float64},
                                   A_m::Matrix{Float64},
                                   B_m::Matrix{Float64},
                                   C_prev_v::Vector{Float64},
                                   G_vec_v::Vector{Float64},
                                   g::Float64, c::Float64;
                                   k::Int=20)
    n     = length(G_hat_v)
    cache = NeumannCache(n)

    B_apply = v -> B_m * v
    gC      = max.(G_hat_v, 0.0)

    # --- C_prev terms ---
    Gv      = gC .* C_prev_v
    term1_C = B_apply(neumann_apply!(cache, A_m, Gv, k))
    term2_C = B_apply(neumann_apply!(cache, A_m, gC .* term1_C, k))
    term3_C = B_apply(neumann_apply!(cache, A_m, gC .* term2_C, k))

    # --- G_vec terms ---
    G_v_g   = g .* G_vec_v
    term1_G = B_apply(neumann_apply!(cache, A_m, G_v_g, k))
    term2_G = B_apply(neumann_apply!(cache, A_m, g .* term1_G, k))
    term3_G = B_apply(neumann_apply!(cache, A_m, g .* term2_G, k))

    return term1_C .+ term2_C .+ term3_C .+ term1_G .+ term2_G .+ term3_G
end

function compute_investment(G_hat, A_bar, B, C_prev, G_vec, g_step, c_step; k::Int=20)
    compute_investment_native(
        _v(G_hat), _m(A_bar), _m(B), _v(C_prev), _v(G_vec),
        Float64(g_step), Float64(c_step); k=k)
end

"""
    solve_planner_native(...)

Solve the central planning optimization problem via a Preconditioned Barzilai-Borwein method on the dual.
Finds structural shadow prices and target consumption maximizing welfare.
"""
function solve_planner_native(alpha_v::Vector{Float64},
                              A_m::Matrix{Float64},
                              B_m::Matrix{Float64},
                              l_tilde_v::Vector{Float64},
                              dK_v::Vector{Float64},
                              K_v::Vector{Float64},
                              L_total_f::Float64,
                              G_v::Vector{Float64},
                              gamma_v::Vector{Float64},
                              C_prev_v::Vector{Float64};
                              k::Int=20,
                              tol::Float64=1e-4,
                              tol_p::Float64=tol, tol_d::Float64=tol,
                              eta_K::Float64=0.25, eta_L::Float64=0.35,
                              max_iter::Int=2000,
                              L0::Float64=1.0,
                              L_scale_up::Float64=2.0,
                              L_scale_dn::Float64=1.05)

    eps_val = 1e-15
    n       = length(alpha_v)
    cache   = NeumannCache(n)
    A_m_T   = collect(A_m')
    B_m_T   = collect(B_m')

    Bt  = v -> B_m * neumann_apply!(cache, A_m, v, k)
    BtT = w -> neumann_apply!(cache, A_m_T, B_m_T * w, k)

    K_eff = K_v .- Bt(dK_v)
    L_eff = L_total_f - dot(l_tilde_v, dK_v)
    # Cold-start duals: π₀ = α / max(C_prev − γ, ε)
    denom   = max.(C_prev_v .- gamma_v, 1e-4)
    pi_init = alpha_v ./ denom

    # Use the full vector space (no filtering of 'active' sectors)
    log_lam_K = log.(max.(BtT(pi_init), 1e-10))
    log_lam_L = log(sum(alpha_v) / max(n * L_eff, eps_val))

    log_y_K = copy(log_lam_K)
    log_y_L = log_lam_L

    log_y_K = copy(log_lam_K)
    log_y_L = log_lam_L

    # Pre-allocate iteration variables for the full N-sector space
    prev_dtheta_K    = zeros(n)
    prev_dtheta_L    = 0.0
    prev_grad_K_iter = zeros(n)
    prev_grad_L_iter = 0.0

    eta_K_cur = eta_K
    eta_L_cur = eta_L

    C_res     = zeros(n)
    converged = false
    opt_iter  = 0
    tk_curr   = 1.0
    log_lam_K_old = copy(log_lam_K)
    log_lam_L_old = log_lam_L

    for iter in 1:max_iter
        opt_iter = iter

        y_K    = exp.(log_y_K)
        y_L    = exp(log_y_L)
        pi_vec = BtT(y_K) .+ y_L .* l_tilde_v
        C_res  = gamma_v .+ alpha_v ./ max.(pi_vec, eps_val)

        # s_K / grad_K at Nesterov point — used for the gradient step only
        s_K = K_eff .- Bt(C_res)
        s_L = L_eff  - dot(l_tilde_v, C_res)

        # 1. Sector-Specific Preconditioning (Jacobi Diagonal)
        # We cap the curvature term to prevent numerical blowup in sectors with low prices.
        diag_H = (B_m .^ 2)' * (alpha_v ./ max.(pi_vec .^ 2, 1e-6))
        prec_K = max.(K_eff, 1e-3) .+ 0.1 .* diag_H
        
        grad_K_raw = .-s_K
        grad_L_raw = -s_L
        
        grad_K = grad_K_raw ./ prec_K
        grad_L = grad_L_raw / (abs(L_eff) + eps_val)

        # 2. KKT Convergence Check
        lam_K_cur  = exp.(log_lam_K)
        lam_L_cur  = exp(log_lam_L)
        pi_chk     = BtT(lam_K_cur) .+ lam_L_cur .* l_tilde_v
        C_chk      = gamma_v .+ alpha_v ./ max.(pi_chk, eps_val)
        s_K_chk    = K_eff .- Bt(C_chk)
        s_L_chk    = L_eff  - dot(l_tilde_v, C_chk)

        # User specified: Lam_k,i * s_k,i < 1e-5
        comp_ok_K   = all(abs.(lam_K_cur .* s_K_chk) .<= tol_d)
        comp_ok_L   = abs(lam_L_cur * s_L_chk) <= tol_d
        primal_ok_K = all(abs.(min.(0.0, s_K_chk)) ./ (max.(K_eff, eps_val)) .<= tol_p)
        primal_ok_L = (abs(min(0.0, s_L_chk)) / (abs(L_eff) + eps_val)) <= tol_p
 
        if comp_ok_K && comp_ok_L && primal_ok_K && primal_ok_L
            converged = true
            break
        end

        # 3. Preconditioned Spectral Step (PBB)
        # Using raw gradient differences (dy_raw) to estimate curvature accurately.
        bb_eta = 1.0
        if iter > 1
            dy_K_raw = grad_K_raw .- prev_grad_K_iter
            dy_L_raw = grad_L_raw  - prev_grad_L_iter
            
            # BB formula in preconditioned space: η = (s' P s) / (s' y)
            # where s is the change in λ and y is the change in raw gradient g.
            dot_ss = dot(prev_dtheta_K, prec_K .* prev_dtheta_K) + prev_dtheta_L^2 * abs(L_eff)
            dot_sy = dot(prev_dtheta_K, dy_K_raw) + prev_dtheta_L * dy_L_raw
            
            if abs(dot_sy) > 1e-25
                gamma_spec = abs(dot_sy / dot_ss)
                bb_eta     = 1.0 / (1.0 + gamma_spec) 
            end
        end

        copyto!(prev_grad_K_iter, grad_K_raw) # Track RAW gradient for next iter
        prev_grad_L_iter = grad_L_raw

        step_K = bb_eta .* grad_K
        step_L = bb_eta * grad_L

        copyto!(prev_dtheta_K, step_K)
        prev_dtheta_L = step_L

        log_lam_K_new = log_y_K .+ step_K
        log_lam_L_new = log_y_L  + step_L

        # tk update
        tk_next = (1.0 + sqrt(1.0 + 4.0 * tk_curr^2)) / 2.0
        beta_t  = (tk_curr - 1.0) / tk_next
        tk_curr = tk_next

        # Extrapolation
        log_y_K .= log_lam_K_new .+ beta_t .* (log_lam_K_new .- log_lam_K)
        log_y_L  = log_lam_L_new  + beta_t  * (log_lam_L_new  - log_lam_L)

        # Iterate update
        log_lam_K .= log_lam_K_new
        log_lam_L  = log_lam_L_new
    end

    lam_K = exp.(log_lam_K)
    lam_L = exp(log_lam_L)

    pi_vec_star = BtT(lam_K) .+ lam_L .* l_tilde_v
    X_star      = neumann_apply!(cache, A_m, C_res .+ dK_v .+ G_v, k)

    return (
        C_star     = C_res,
        X_star     = X_star,
        pi_star    = pi_vec_star,
        success    = converged,
        lambda_K   = lam_K,
        lambda_L   = lam_L,
        iterations = opt_iter
    )
end

# Python wrapper
function solve_planner(alpha, A_bar, B, l_tilde, dK, K, L_total::Real, G_vec, gamma, C_prev;
                       k::Int=20,
                       tol::Float64=1e-4,
                       tol_p::Float64=tol, tol_d::Float64=tol,
                       eta_K::Float64=0.25, eta_L::Float64=0.35,
                       max_iter::Int=2000)
    solve_planner_native(
        _v(alpha), _m(A_bar), _m(B), _v(l_tilde), _v(dK), _v(K),
        Float64(L_total), _v(G_vec), _v(gamma), _v(C_prev);
        k=k, tol=tol, tol_p=tol_p, tol_d=tol_d,
        eta_K=eta_K, eta_L=eta_L, max_iter=max_iter)
end

"""
    fast_loop_native(...)
"""
function fast_loop_native(P_base_v::Vector{Float64},
                          C_plan_v::Vector{Float64},
                          alpha_true_start_v::Vector{Float64},
                          alpha_s_v::Vector{Float64},
                          rng::AbstractRNG,
                          drift_rho::Float64,
                          drift_sigma::Float64,
                          noise_sigma::Float64,
                          Y_f::Float64,
                          gamma_v::Vector{Float64},
                          K_v_vec::Vector{Float64},
                          n_months::Int=3;
                          theta_drift::Float64=0.1,
                          max_price_iter::Int=50,
                          price_tol::Float64=0.005,
                          price_step_cap::Float64=0.5,
                          alpha_h::Union{Nothing, AbstractMatrix{Float64}}=nothing,
                          gamma_h::Union{Nothing, AbstractMatrix{Float64}}=nothing,
                          Y_h::Union{Nothing, AbstractVector{Float64}}=nothing,
                          alpha_slow_h::Union{Nothing, AbstractMatrix{Float64}}=nothing)

    multi_hh  = !isnothing(alpha_h) && !isnothing(gamma_h) && !isnothing(Y_h)
    n_h_count = multi_hh ? size(alpha_h, 1) : 0

    local alpha_h_ev
    if multi_hh; alpha_h_ev = copy(alpha_h); end

    n = length(P_base_v)
    C_m     = C_plan_v ./ n_months
    gamma_m = gamma_v  ./ n_months
    Y_m     = Y_f / n_months

    local gamma_h_m, Y_h_m
    if multi_hh
        gamma_h_m = gamma_h ./ n_months
        Y_h_m     = Y_h ./ n_months
    end

    C_monthly = zeros(n_months, n)
    P_monthly = zeros(n_months, n)

    C_hat_sum       = zeros(n)
    a_reveal_sum    = zeros(n)
    monthly_drifts  = zeros(n_months)
    monthly_resid_Y = zeros(n_months)

    a_f    = copy(alpha_true_start_v)
    active = a_f .> 0

    agg_shocks_drift = randn(rng, n_months, n) .* drift_sigma
    agg_shocks_noise = randn(rng, n_months, n) .* noise_sigma

    noise_persistent = zeros(n)

    local hh_shocks_drift, hh_shocks_noise
    if multi_hh
        hh_shocks_drift = randn(rng, n_months, n_h_count, n) .* drift_sigma
        hh_shocks_noise = randn(rng, n_months, n_h_count, n) .* noise_sigma
    end

    for tau in 1:n_months
        log_f = ifelse.(a_f .> 1e-25, log.(max.(a_f, 1e-30)), -25.0)
        log_s = ifelse.(alpha_s_v .> 1e-25, log.(max.(alpha_s_v, 1e-30)), -25.0)

        @inbounds for i in 1:n
            if active[i]
                drift     = theta_drift * (log_s[i] - log_f[i])
                noise_persistent[i] = drift_rho * noise_persistent[i] + agg_shocks_drift[tau, i]
                log_f[i] += drift + noise_persistent[i] + agg_shocks_noise[tau, i]
            end
        end
        exp_f = exp.(log_f)
        a_f   = exp_f ./ max(sum(exp_f), 1e-30)
        log_f_norm = ifelse.(a_f .> 1e-25, log.(max.(a_f, 1e-30)), -25.0)

        if multi_hh
            log_sh = log_f_norm
            @inbounds for i in 1:n
                if active[i]
                    for h in 1:n_h_count
                        log_ah = alpha_h_ev[h, i] > 1e-25 ? log(alpha_h_ev[h, i]) : -25.0
                        drift_h = theta_drift * (log_sh[i] - log_ah)
                        alpha_h_ev[h, i] = exp(log_ah + drift_h +
                                               hh_shocks_drift[tau, h, i] +
                                               hh_shocks_noise[tau, h, i])
                    end
                end
            end
            row_sums = max.(vec(sum(alpha_h_ev, dims=2)), 1e-30)
            alpha_h_ev ./= row_sums
        end

        P_iter = copy(P_base_v)
        local gamma_sum
        if multi_hh; gamma_sum = vec(sum(gamma_h_m, dims=1)); end

        for _ in 1:max_price_iter
            if multi_hh
                resid_Y_h_iter = max.(Y_h_m .- gamma_h_m * P_iter, 1e-30)
                C_d_iter = gamma_sum .+ (alpha_h_ev' * resid_Y_h_iter) ./ max.(P_iter, 1e-30)
            else
                resid_Y_iter = max(Y_m - dot(P_iter, gamma_m), 1e-30)
                C_d_iter     = gamma_m .+ a_f .* resid_Y_iter ./ max.(P_iter, 1e-30)
            end
            Z_iter = C_d_iter .- C_m
            denom_step = (C_d_iter .+ C_m) ./ 2.0
            P_iter = P_iter .* (1.0 .+ price_step_cap .* Z_iter ./ max.(denom_step, 1e-12))
            P_iter = max.(P_iter, 1e-12)

            if maximum(abs.(Z_iter) ./ max.((C_d_iter .+ C_m) ./ 2.0, 1e-12)) < 0.005
                break
            end
        end
        P_clear = P_iter

        if multi_hh
            resid_Y_h_star = max.(Y_h_m .- gamma_h_m * P_clear, 1e-30)
            resid_Y_star   = sum(resid_Y_h_star)
            C_d = gamma_sum .+ (alpha_h_ev' * resid_Y_h_star) ./ max.(P_clear, 1e-30)
        else
            resid_Y_star = max(Y_m - dot(P_clear, gamma_m), 1e-30)
            C_d          = gamma_m .+ a_f .* resid_Y_star ./ max.(P_clear, 1e-30)
        end

        a_reveal_sum .+= (P_clear .* (C_d .- gamma_m)) ./ max(resid_Y_star, 1e-30)

        eps_avg   = mean(-1.0 .+ (gamma_m ./ max.(C_d, 1e-30)) .* (1.0 .- a_f))
        eps_final = min(eps_avg, -1e-4)
        delta_p   = (P_clear .- P_base_v) ./ max.(P_base_v, 1e-30)
        signal    = clamp.(eps_final .* delta_p, -0.05, 0.0)
        denom_sig = 1.0 .+ signal
        C_hat_sum .+= C_m ./ denom_sig

        C_monthly[tau, :]       .= C_d
        P_monthly[tau, :]       .= P_clear
        monthly_resid_Y[tau]     = resid_Y_star
        monthly_drifts[tau]      = mean(abs.(P_clear ./ P_base_v .- 1.0))
    end

    G_hat_bare = (C_hat_sum .- C_plan_v) ./ max.(C_plan_v, 1e-30)
    P_final    = P_monthly[n_months, :]
    signed_drift = mean((P_final ./ P_base_v .- 1.0)[alpha_s_v .> 1e-12])
    
    abs_drifts_v = abs.(P_final ./ P_base_v .- 1.0)[alpha_s_v .> 1e-12]
    abs_drift = isempty(abs_drifts_v) ? 0.0 : length(abs_drifts_v) / sum(1.0 ./ (abs_drifts_v .+ 1e-12))

    return (
        C_monthly        = C_monthly,
        C_base_monthly   = repeat(C_m', n_months, 1),
        C_hat_monthly    = C_monthly,
        C_hat            = C_hat_sum,
        G_hat_bare       = G_hat_bare,
        P_monthly        = P_monthly,
        P_final          = P_final,
        price_drift      = abs_drift,
        signed_drift     = signed_drift,
        monthly_drifts   = monthly_drifts,
        monthly_resid_Y  = monthly_resid_Y,
        alpha_true_final = a_reveal_sum ./ n_months,
        alpha_mean       = a_reveal_sum ./ n_months,
        g_alpha_final    = zeros(n),
        alpha_noise_final = zeros(n),
        alpha_h_final    = multi_hh ? alpha_h_ev : zeros(0, 0),
        alpha_macro_final = a_f,
    )
end

function fast_loop(P_base, C_plan, alpha_true_start, alpha_slow,
                   rng::AbstractRNG,
                   drift_rho::Real, drift_sigma::Real, noise_sigma::Real,
                   Y::Real, gamma, K_v, n_months::Int=3;
                   theta_drift::Float64=0.1,
                   max_price_iter::Int=50,
                   price_tol::Float64=0.005,
                   price_step_cap::Float64=0.5,
                   alpha_h=nothing,
                   gamma_h=nothing,
                   Y_h=nothing,
                   alpha_slow_h=nothing)

    fast_loop_native(
        _v(P_base), _v(C_plan), _v(alpha_true_start), _v(alpha_slow),
        rng,
        Float64(drift_rho), Float64(drift_sigma), Float64(noise_sigma),
        Float64(Y), _v(gamma), _v(K_v), n_months;
        theta_drift=theta_drift,
        max_price_iter=max_price_iter,
        price_tol=price_tol,
        price_step_cap=price_step_cap,
        alpha_h = isnothing(alpha_h) ? nothing : _m(alpha_h),
        gamma_h = isnothing(gamma_h) ? nothing : _m(gamma_h),
        Y_h = isnothing(Y_h) ? nothing : _v(Y_h),
        alpha_slow_h = isnothing(alpha_slow_h) ? nothing : _m(alpha_slow_h))
end

const FIRM_SOLVER_CACHE = Dict{Int, Any}()

function solve_firm_lp(v_MIP_py, B_dense_py, K_firms_py, X_star_py, tol=0.001)
    v_plan  = max.(_v(v_MIP_py), 1e-8)
    B_dense = _m(B_dense_py)
    K_firms = _m(K_firms_py)
    X_star  = _v(X_star_py)

    n_firms = size(K_firms, 1)
    n       = length(X_star)
    B_nz = [findall(>(1e-12), B_dense[i, :]) for i in 1:n]

    if !haskey(FIRM_SOLVER_CACHE, n)
        models = Vector{Any}(undef, n_firms)
        for f in 1:n_firms
            m = Model(HiGHS.Optimizer); set_silent(m)
            set_attribute(m, "presolve", "off"); set_attribute(m, "threads", 1)
            x = @variable(m, x[1:n] >= 0); u = @variable(m, u[1:n] >= 0)
            u_cons1 = Vector{ConstraintRef}(undef, n)
            u_cons2 = Vector{ConstraintRef}(undef, n)
            cap_cons = Vector{ConstraintRef}(undef, n)
            for i in 1:n
                u_cons1[i] = @constraint(m, x[i] - u[i] <= 0.0)
                u_cons2[i] = @constraint(m, -x[i] - u[i] <= 0.0)
                if !isempty(B_nz[i])
                    cap_cons[i] = @constraint(m, sum(B_dense[i, j] * x[j] for j in B_nz[i]) <= 0.0)
                end
            end
            eps_reg = 0.001
            @objective(m, Max, sum(v_plan[i] * x[i] for i in 1:n) - eps_reg * sum(u[i] for i in 1:n))
            models[f] = (model=m, x_vars=x, u_vars=u, u_cons1=u_cons1, u_cons2=u_cons2, cap_cons=cap_cons)
        end
        FIRM_SOLVER_CACHE[n] = models
    end

    models_cache = FIRM_SOLVER_CACHE[n]
    K_total_sector = vec(sum(K_firms, dims=1))
    X_f_total = zeros(n, n_firms)

    Threads.@threads for f in 1:n_firms
        cache = models_cache[f]
        m, x_vars, u_cons1, u_cons2, cap_cons = cache.model, cache.x_vars, cache.u_cons1, cache.u_cons2, cache.cap_cons
        for i in 1:n
            share = K_total_sector[i] > 1e-12 ? K_firms[f, i] / K_total_sector[i] : 1.0/n_firms
            target = share * X_star[i]
            set_normalized_rhs(u_cons1[i], target)
            set_normalized_rhs(u_cons2[i], -target)
            if isassigned(cap_cons, i)
                set_normalized_rhs(cap_cons[i], max(K_firms[f, i], 0.0))
            end
        end
        optimize!(m)
        if has_values(m); X_f_total[:, f] .= value.(x_vars); end
    end
    return X_f_total
end

function run_montecarlo(n_runs::Int, run_one_fn::Function; kwargs...)
    results = Vector{Any}(undef, n_runs)
    Threads.@threads for i in 1:n_runs
        results[i] = run_one_fn(i, MersenneTwister(i); kwargs...)
    end
    return results
end


# --- Precompilation Hints ---
# These hints help Julia's JIT generator prepare machine code for the 
# specific argument types used in our simulation, reducing first-run latency.
if VERSION >= v"1.9"
    # Using precompile() is more efficient than a let block for loading speed.
    precompile(neumann_apply!, (NeumannCache, Matrix{Float64}, Vector{Float64}, Int))
    precompile(solve_planner_native, (Vector{Float64}, Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64, Vector{Float64}, Vector{Float64}, Vector{Float64}))
    precompile(fast_loop_native, (Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, MersenneTwister, Float64, Float64, Float64, Float64, Vector{Float64}, Vector{Float64}, Int))
end

end  # module ModelCore