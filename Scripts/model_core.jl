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
    # Check if v_py is a vector or matrix
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

Solve the central planning optimization problem via a projected Nesterov/Barzilai-Borwein method on the dual.
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
                              max_iter::Int=2000)

    eps_val = 1e-15
    n       = length(alpha_v)
    cache   = NeumannCache(n)
    A_m_T   = collect(A_m')             # pre-transpose once
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

    avg_start = max(3 * max_iter ÷ 4, 1)
    lam_K_sum = zeros(n)
    lam_L_sum = 0.0
    avg_count = 0

    log_y_K = copy(log_lam_K)
    log_y_L = log_lam_L

    # Pre-allocate iteration variables for the full N-sector space
    prev_dtheta_K    = zeros(n)
    prev_dtheta_L    = 0.0
    prev_grad_K_iter = zeros(n)
    prev_grad_L_iter = 0.0
    grad_K_iter_buf  = zeros(n)   # reusable buffer

    eta_K_cur = eta_K
    eta_L_cur = eta_L

    C_res     = zeros(n)
    converged = false
    opt_iter  = 0

    for iter in 1:max_iter
        opt_iter = iter

        y_K    = exp.(log_y_K)
        y_L    = exp(log_y_L)
        pi_vec = BtT(y_K) .+ y_L .* l_tilde_v
        C_res  = gamma_v .+ alpha_v ./ max.(pi_vec, eps_val)

        # s_K / grad_K at Nesterov point — used for the gradient step only
        s_K = K_eff .- Bt(C_res)
        s_L = L_eff  - dot(l_tilde_v, C_res)

        grad_K = (.-s_K) ./ (max.(K_eff, eps_val))
        grad_L = (-s_L) / (abs(L_eff) + eps_val)

        # BUG 2 FIX: convergence check must be evaluated at a single consistent
        # point.  Use the actual iterate (log_lam_K / log_lam_L), not the
        # Nesterov momentum point (log_y_K), so that lam and s come from the
        # same λ.  Previously s_K was from y_K while lam_K_cur was from
        # log_lam_K, making the complementarity product meaningless.
        lam_K_cur  = exp.(log_lam_K)
        lam_L_cur  = exp(log_lam_L)
        pi_chk     = BtT(lam_K_cur) .+ lam_L_cur .* l_tilde_v
        C_chk      = gamma_v .+ alpha_v ./ max.(pi_chk, eps_val)
        s_K_chk    = K_eff .- Bt(C_chk)
        s_L_chk    = L_eff  - dot(l_tilde_v, C_chk)

        primal_ok_K = all(abs.(min.(0.0, s_K_chk)) ./ (max.(K_eff, eps_val)) .<= tol_p)
        primal_ok_L = (abs(min(0.0, s_L_chk)) / (abs(L_eff) + eps_val)) <= tol_p

        # BUG 4 FIX: normalise complementarity tolerance by K_eff / L_eff so
        # the stopping criterion is scale-invariant (primal check already does
        # this; the raw absolute tol_d was far too loose for large K values).
        comp_ok_K = all(abs.(lam_K_cur .* s_K_chk) .<= tol_d)
        comp_ok_L = abs(lam_L_cur * s_L_chk) <= tol_d
 
        if primal_ok_K && primal_ok_L && comp_ok_K && comp_ok_L
            converged = true
            break
        end

        # BUG 3 FIX: BB must track gradient differences from the same sequence
        # as the steps.  Previously prev_dtheta_K held steps from the Nesterov
        # gradient (grad_K) while dy_K was the difference of gradients
        # re-evaluated at the actual iterate (log_lam_K) — mixing two different
        # sequences made the BB ratio unreliable.  Now both the step and the
        # gradient difference come from the Nesterov point, so the ratio is
        # consistent.  The redundant re-evaluation at log_lam_K is removed.
        if iter > 1
            dy_K   = grad_K         .- prev_grad_K_iter
            dy_L   = grad_L          - prev_grad_L_iter
            dot_ss = dot(prev_dtheta_K, prev_dtheta_K) + prev_dtheta_L^2
            dot_sg = dot(prev_dtheta_K, dy_K) + prev_dtheta_L * dy_L
            if abs(dot_sg) > 1e-30
                bb_eta    = clamp(abs(dot_ss / dot_sg), 0.01, 0.5)
                eta_K_cur = bb_eta
                eta_L_cur = bb_eta
            end
        end

        copyto!(prev_grad_K_iter, grad_K)
        prev_grad_L_iter = grad_L

        step_K = eta_K_cur .* grad_K
        step_L = eta_L_cur * grad_L

        copyto!(prev_dtheta_K, step_K)
        prev_dtheta_L = step_L

        log_lam_K_new = log_y_K .+ step_K
        log_lam_L_new = log_y_L  + step_L

        beta_t  = (iter - 1.0) / (iter + 2.0)
        log_y_K = log_lam_K_new .+ beta_t .* (log_lam_K_new .- log_lam_K)
        log_y_L = log_lam_L_new  + beta_t  * (log_lam_L_new  - log_lam_L)

        log_lam_K .= log_lam_K_new
        log_lam_L  = log_lam_L_new

        if iter >= avg_start
            lam_K_sum .+= exp.(log_lam_K)
            lam_L_sum  += exp(log_lam_L)
            avg_count  += 1
        end
    end

    if !converged && avg_count > 0
        lam_K  = lam_K_sum ./ avg_count
        lam_L  = lam_L_sum  / avg_count
        pi_vec = BtT(lam_K) .+ lam_L .* l_tilde_v
        C_res  = gamma_v .+ alpha_v ./ max.(pi_vec, eps_val)
    else
        lam_K = exp.(log_lam_K)
        lam_L = exp(log_lam_L)
    end

    pi_vec_star = BtT(lam_K) .+ lam_L .* l_tilde_v
    X_star = neumann_apply!(cache, A_m, C_res .+ dK_v .+ G_v, k)

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

Simulate the intra-quarter (monthly) market-clearing tatonnement.
Applies stochastic LN-AR drift to household preferences and finds 
market-clearing prices iteratively based on the Linear Expenditure System.
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

    # Determine if multi-household mode is active
    multi_hh  = !isnothing(alpha_h) && !isnothing(gamma_h) && !isnothing(Y_h)
    n_h_count = multi_hh ? size(alpha_h, 1) : 0

    # Mutable copy of per-household preferences for LN-AR evolution
    local alpha_h_ev
    if multi_hh
        alpha_h_ev = copy(alpha_h)
    end

    n = length(P_base_v)

    C_m     = C_plan_v ./ n_months
    gamma_m = gamma_v  ./ n_months
    Y_m     = Y_f / n_months

    # Per-household monthly quantities
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

    # Fix 6: Pre-generate all shock matrices before the month loop
    agg_shocks_drift = randn(rng, n_months, n) .* drift_sigma
    agg_shocks_noise = randn(rng, n_months, n) .* noise_sigma

    noise_persistent = zeros(n) # Shared macro-fad state

    # For multi-HH: pre-generate shocks for all households × months × sectors
    local hh_shocks_drift, hh_shocks_noise
    if multi_hh
        hh_shocks_drift = randn(rng, n_months, n_h_count, n) .* drift_sigma
        hh_shocks_noise = randn(rng, n_months, n_h_count, n) .* noise_sigma
    end

    for tau in 1:n_months

        # 1. Aggregate preference drift: log-OU mean-reversion toward alpha_slow
        log_f = ifelse.(a_f .> 1e-25, log.(max.(a_f, 1e-30)), -25.0)
        log_s = ifelse.(alpha_s_v .> 1e-25, log.(max.(alpha_s_v, 1e-30)), -25.0)

        @inbounds for i in 1:n
            if active[i]
                drift     = theta_drift * (log_s[i] - log_f[i])
                # Add persistent mean-reverting macro noise (fad model)
                noise_persistent[i] = drift_rho * noise_persistent[i] + agg_shocks_drift[tau, i]
                log_f[i] += drift + noise_persistent[i] + agg_shocks_noise[tau, i]
            end
        end
        exp_f = exp.(log_f)
        a_f   = exp_f ./ max(sum(exp_f), 1e-30)
        # Macro-aggregate target for households in log-space
        log_f_norm = ifelse.(a_f .> 1e-25, log.(max.(a_f, 1e-30)), -25.0)

        # 1b. Per-household LN-AR drift: Track the Macro Alpha
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
            # Vectorized fast re-normalization across rows:
            row_sums = max.(vec(sum(alpha_h_ev, dims=2)), 1e-30)
            alpha_h_ev ./= row_sums
        end

        # 2. Iterative price tatonnement toward market clearing
        P_iter = copy(P_base_v)
        
        local gamma_sum
        if multi_hh
            gamma_sum = vec(sum(gamma_h_m, dims=1))
        end

        for _ in 1:max_price_iter
            if multi_hh
                # Vectorized operations mapping cleanly to BLAS
                resid_Y_h_iter = max.(Y_h_m .- gamma_h_m * P_iter, 1e-30)
                C_d_iter = gamma_sum .+ (alpha_h_ev' * resid_Y_h_iter) ./ max.(P_iter, 1e-30)
            else
                resid_Y_iter = max(Y_m - dot(P_iter, gamma_m), 1e-30)
                C_d_iter     = gamma_m .+ a_f .* resid_Y_iter ./ max.(P_iter, 1e-30)
            end
            Z_iter = C_d_iter .- C_m
            # Midpoint Tatonnement: denom = (Demand + Supply) / 2
            # This is extremely stable even when Supply is near zero.
            denom_step = (C_d_iter .+ C_m) ./ 2.0
            P_iter = P_iter .* (1.0 .+ price_step_cap .* Z_iter ./ max.(denom_step, 1e-12))
            P_iter = max.(P_iter, 1e-12)

            # Symmetric Percent Error (0.5% tolerance)
            if maximum(abs.(Z_iter) ./ max.((C_d_iter .+ C_m) ./ 2.0, 1e-12)) < 0.005
                break
            end
        end
        P_clear = P_iter

        # Compute final cleared demand
        if multi_hh
            resid_Y_h_star = max.(Y_h_m .- gamma_h_m * P_clear, 1e-30)
            resid_Y_star   = sum(resid_Y_h_star)
            C_d = gamma_sum .+ (alpha_h_ev' * resid_Y_h_star) ./ max.(P_clear, 1e-30)
        else
            resid_Y_star = max(Y_m - dot(P_clear, gamma_m), 1e-30)
            C_d          = gamma_m .+ a_f .* resid_Y_star ./ max.(P_clear, 1e-30)
        end

        # 3. Revealed preference shares from LES expenditure
        a_reveal_sum .+= (P_clear .* (C_d .- gamma_m)) ./ max(resid_Y_star, 1e-30)

        # 4. LES price-correction signal for G_hat
        eps_final = min.(-1.0 .+ (gamma_m ./ max.(C_d, 1e-30)) .* (1.0 .- a_f), -1e-4)
        delta_p   = (P_clear .- P_base_v) ./ max.(P_base_v, 1e-30)
        signal    = clamp.(eps_final .* delta_p, -0.1, 0.0)
        denom_sig = 1.0 .+ signal
        C_hat_sum .+= C_m ./ denom_sig

        C_monthly[tau, :]       .= C_d
        P_monthly[tau, :]       .= P_clear
        monthly_resid_Y[tau]     = resid_Y_star
        monthly_drifts[tau]      = mean(abs.(P_clear ./ P_base_v .- 1.0))
    end

    G_hat_bare = (C_hat_sum .- C_plan_v) ./ max.(C_plan_v, 1e-30)

    active_consumer = alpha_s_v .> 1e-12
    P_final      = P_monthly[n_months, :]
    signed_drift = mean((P_final ./ P_base_v .- 1.0)[active_consumer])
    
    # Robust Diagnostic: Trimmed Harmonic Mean
    # We ignore the outliers (top/bottom 5% of sectors) so structural
    # shortages in tiny sectors don't blow up the reported mean.
    abs_drifts_v = abs.(P_final ./ P_base_v .- 1.0)[active_consumer]
    if !isempty(abs_drifts_v)
        n_act = length(abs_drifts_v)
        trim  = max(1, Int(floor(0.05 * n_act))) # Trim 5%
        sorted_drifts = sort(abs_drifts_v)
        trimmed = sorted_drifts[(trim + 1):(end - trim)]
        abs_drift = length(trimmed) / sum(1.0 ./ (trimmed .+ 1e-12))
    else
        abs_drift = 0.0
    end

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

# Python wrapper – converts once, delegates
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

    # Pre-extract sparse structure of B
    B_nz       = [findall(>(1e-12), B_dense[i, :]) for i in 1:n]
    B_trans_nz = [findall(>(1e-12), B_dense[:, j]) for j in 1:n]

    # Retrieve or initialize cached block-diagonal model
    if !haskey(FIRM_SOLVER_CACHE, n)
        m = Model(HiGHS.Optimizer)
        set_silent(m)
        set_attribute(m, "presolve", "off") # MUST be OFF so Dual Simplex can warm-start
        set_attribute(m, "threads", 4) # Allow HiGHS internal parallelization

        # Variable matrix: columns are firms, rows are sectors
        # Indexed as x[firm, sector]
        x = @variable(m, x[1:n_firms, 1:n] >= 0)
        
        # 1. Sectoral target quotas (to align with planner's X_star)
        q_cons = Matrix{ConstraintRef}(undef, n_firms, n)
        # 2. Capital constraints (to respect physical limits)
        cap_cons = Matrix{ConstraintRef}(undef, n_firms, n)

        for f in 1:n_firms
            for i in 1:n
                # Initialize quotas at 0, will be updated iteratively per quarter
                q_cons[f, i] = @constraint(m, x[f, i] <= 0.0)
                
                if !isempty(B_nz[i])
                    nz = B_nz[i]
                    # B[i, j] * x[f, j] summed over sectors j that require capital i
                    cap_cons[f, i] = @constraint(m, sum(B_dense[i, j] * x[f, j] for j in nz) <= 0.0)
                end
            end
        end
        FIRM_SOLVER_CACHE[n] = (model=m, x_vars=x, q_cons=q_cons, cap_cons=cap_cons)
    end

    cache = FIRM_SOLVER_CACHE[n]
    m, x_vars, q_cons, cap_cons = cache.model, cache.x_vars, cache.q_cons, cache.cap_cons

    # Objective: Maximize sum of value-added (v_MIP' * x_f) across the ensemble
    @objective(m, Max, sum(dot(v_plan, x_vars[f, :]) for f in 1:n_firms))

    # Calculate firm-specific quotas based on current capital shares
    # Share = firm_capital_in_sector_i / total_capital_in_sector_i
    K_total_sector = vec(sum(K_firms, dims=1))
    
    # Update Quota and Capital RHS values
    for f in 1:n_firms
        for i in 1:n
            # Update output quota
            share = K_total_sector[i] > 1e-12 ? K_firms[f, i] / K_total_sector[i] : 1.0/n_firms
            set_normalized_rhs(q_cons[f, i], max(share * X_star[i], 0.0))
            
            # Update capital constraint
            if !isempty(B_nz[i])
                set_normalized_rhs(cap_cons[f, i], max(K_firms[f, i], 0.0))
            end
        end
    end

    # Solve the ensemble in a single pass
    optimize!(m)
    
    X_f_total = zeros(n, n_firms)
    if has_values(m)
        x_vals = value.(x_vars)
        for f in 1:n_firms
            X_f_total[:, f] .= x_vals[f, :]
        end
    else
        @warn "Firm LP failed to produce values: status = $(termination_status(m))"
    end

    return X_f_total
end

"""
    run_montecarlo(n_runs, run_one_fn; kwargs...)

Run `n_runs` independent simulations in parallel using `Threads.@threads`.
Each thread gets its own `MersenneTwister` seeded with the run index.

`run_one_fn(run_idx::Int, rng::AbstractRNG; kwargs...)` must be a callable
that executes one full simulation and returns its result.  The function must
be fully thread-safe (no shared mutable state other than its own `rng`).

Returns a `Vector{Any}` of length `n_runs` containing the results.
"""
function run_montecarlo(n_runs::Int, run_one_fn::Function; kwargs...)
    results = Vector{Any}(undef, n_runs)
    Threads.@threads for i in 1:n_runs
        rng_i = MersenneTwister(i)          # deterministic, per-thread seed
        results[i] = run_one_fn(i, rng_i; kwargs...)
    end
    return results
end

let
    n_pre = 2
    A_pre = rand(n_pre, n_pre)
    B_pre = rand(n_pre, n_pre)
    v_pre = rand(n_pre)
    K_pre = rand(n_pre)
    C_pre = rand(n_pre)
    G_pre = rand(n_pre)
    a_pre = rand(n_pre); a_pre ./= sum(a_pre)
    l_pre = rand(n_pre)
    ga_pre = zeros(n_pre)
    
    # 1. Warm up Neumann series and Investment
    compute_investment_native(a_pre, A_pre, B_pre, C_pre, G_pre, 0.01, 0.01, k=2)
    
    # 2. Warm up Planner (JuMP + HiGHS + BB-loop)
    solve_planner_native(a_pre, A_pre, B_pre, l_pre, v_pre, K_pre, 10.0, G_pre, ga_pre, C_pre, max_iter=5)
    
    # 3. Warm up Tatonnement
    rng = MersenneTwister(42)
    fast_loop_native(v_pre, C_pre, a_pre, a_pre, rng, 0.9, 0.01, 0.01, 10.0, ga_pre, K_pre, 1)

    # 4. Warm up Firm LP (including cache initialization)
    K_f_pre = rand(5, n_pre)
    solve_firm_lp(v_pre, B_pre, K_f_pre, v_pre, 0.01)
end

end  # module ModelCore