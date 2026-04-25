module ModelCore

using LinearAlgebra
using SparseArrays
using Random
using Statistics
using PythonCall: pyconvert

export neumann_apply, evolve_structural_alpha, revealed_demand, infer_growth,
       compute_investment, solve_planner, compute_income, fast_loop

_v(x) = pyconvert(Vector{Float64}, x)
_m(x) = pyconvert(Matrix{Float64}, x)

# Approximate (I + A + A^2 + … + A^k) v via repeated multiplication.
function neumann_apply(A_bar, v, k::Int=20)
    A      = _m(A_bar)
    result = copy(_v(v))
    term   = copy(result)
    for _ in 1:k
        term    = A * term
        result .+= term
    end
    return result
end

# Log-space OU mean-reversion of preference weights toward a habit target.
function evolve_structural_alpha(alpha_slow_v,
                                 rng::AbstractRNG,
                                 alpha_habit_v,
                                 drift_slow::Real,
                                 kappa_slow::Real)
    n = length(alpha_slow_v)
    active = alpha_slow_v .> 0

    log_s     = zeros(n)
    log_habit = zeros(n)
    for i in 1:n
        if active[i]
            log_s[i]     = log(max(alpha_slow_v[i], 1e-30))
            log_habit[i] = log(max(alpha_habit_v[i], 1e-30))
        end
    end
    shocks_s = zeros(n)
    for i in 1:n
        active[i] && (shocks_s[i] = drift_slow * randn(rng))
    end
    log_s_new = log_s .+ kappa_slow .* (log_habit .- log_s) .+ shocks_s

    s_new = zeros(n)
    for i in 1:n
        active[i] && (s_new[i] = exp(log_s_new[i]))
    end
    s_new ./= sum(s_new)
    return s_new
end

# Apply LES price-elasticity correction to monthly consumption plans.
function revealed_demand(C_monthly, P_monthly, P_base, C_plan, alpha, gamma)
    Cm = pyconvert(Array{Float64, 2}, C_monthly)
    Pm = pyconvert(Array{Float64, 2}, P_monthly)
    Pb = pyconvert(Array{Float64, 1}, P_base)
    Cp = pyconvert(Array{Float64, 1}, C_plan)
    av = pyconvert(Array{Float64, 1}, alpha)
    gv = pyconvert(Array{Float64, 1}, gamma)

    n_months, n = size(Cm)
    Chat = zeros(n)
    eps_vec = -1.0 .+ (gv .* (1.0 .- av)) ./ max.(Cp, 1e-30)

    for tau in 1:n_months
        P_tau = Pm[tau, :]
        C_tau = Cm[tau, :]
        for i in 1:n
            denom = 1.0 + eps_vec[i] * (P_tau[i] - Pb[i]) / (Pb[i] + 1e-30)
            denom = max(denom, 0.1)
            Chat[i] += C_tau[i] / denom
        end
    end
    return Chat
end

# Compute sectoral growth signals from revealed vs planned consumption.
function infer_growth(C_hat, C_star)
    Chat_v  = _v(C_hat)
    Cstar_v = _v(C_star)
    res = zeros(length(Chat_v))
    for i in eachindex(Chat_v)
        denom = max(Cstar_v[i], 1e-30)
        res[i] = max(0.0, (Chat_v[i] - Cstar_v[i]) / denom)
    end
    return res
end

# Compute net investment requirements from growth signals and government needs.
function compute_investment(G_hat, A_bar, B, C_prev, G_vec, g_step, c_step; k::Int=20)
    G_hat_v  = _v(G_hat)
    C_prev_v = _v(C_prev)
    G_vec_v  = _v(G_vec)
    g        = Float64(g_step)

    is_B_1d = B isa AbstractArray ? ndims(B) == 1 : pyconvert(Int, B.ndim) == 1
    B_m = is_B_1d ? nothing : _m(B)
    B_v = is_B_1d ? _v(B)   : nothing
    A_m = _m(A_bar)

    B_apply = v -> is_B_1d ? B_v .* v : B_m * v

    gC      = max.(G_hat_v, 0)
    Gv      = gC .* C_prev_v
    term1_C = B_apply(neumann_apply(A_m, Gv, k))
    term2_C = B_apply(neumann_apply(A_m, gC .* term1_C, k))

    G_v_g   = g .* G_vec_v
    term1_G = B_apply(neumann_apply(A_m, G_v_g, k))
    term2_G = B_apply(neumann_apply(A_m, g .* term1_G, k))

    return term1_C .+ term2_C .+ term1_G .+ term2_G
end

# Nesterov dual ascent with Barzilai-Borwein step size and Polyak averaging.
# Solves the household LES utility maximisation subject to capital and labour
# constraints implied by the Leontief technology.
function solve_planner(alpha, A_bar, B, l_tilde, dK, K, L_total::Real, G_vec, gamma, C_prev;
                       k::Int=20,
                       tol::Float64=1e-4,
                       tol_p::Float64=tol, tol_d::Float64=tol,
                       eta_K::Float64=0.25, eta_L::Float64=0.35,
                       max_iter::Int=2000)

    alpha_v   = _v(alpha)
    A_m       = _m(A_bar)
    eps_val   = 1e-15
    n         = length(alpha_v)

    is_B_1d = B isa AbstractArray ? ndims(B) == 1 : pyconvert(Int, B.ndim) == 1
    B_v = is_B_1d ? _v(B)   : nothing
    B_m = is_B_1d ? nothing : _m(B)

    l_tilde_v = _v(l_tilde)
    dK_v      = _v(dK)
    G_v       = _v(G_vec)
    K_v       = _v(K)
    gamma_v   = _v(gamma)
    L_total_f = Float64(L_total)

    local Bt, BtT
    if is_B_1d
        Bt  = v -> B_v .* neumann_apply(A_m, v, k)
        BtT = w -> neumann_apply(A_m', B_v .* w, k)
    else
        Bt  = v -> B_m * neumann_apply(A_m, v, k)
        BtT = w -> neumann_apply(A_m', B_m' * w, k)
    end

    K_eff = K_v .- Bt(dK_v)
    L_eff = L_total_f - dot(l_tilde_v, dK_v)

    pos_Keff = K_eff[K_eff .> 0]
    K_mean   = isempty(pos_Keff) ? 1.0 : mean(pos_Keff)
    active   = K_eff .> 1e-6 * K_mean

    Kv          = K_eff[active]
    alpha_act   = alpha_v[active]
    gamma_act   = gamma_v[active]
    l_tilde_act = l_tilde_v[active]
    n_act       = length(Kv)

    Bt_active = v_act -> begin
        v_full = zeros(n); v_full[active] .= v_act
        Bt(v_full)[active]
    end
    BtT_active = w_act -> begin
        w_full = zeros(n); w_full[active] .= w_act
        BtT(w_full)[active]
    end

    # Cold-start duals: π₀ = α / max(C_prev − γ, ε)
    C0_act   = _v(C_prev)[active]
    denom    = max.(C0_act .- gamma_act, 1e-4)
    pi_init  = alpha_act ./ denom

    BtT_init = w_act -> begin
        w_full = zeros(n); w_full[active] .= w_act
        w_res  = w_full .- A_m' * w_full
        w_res[active]
    end
    lam_K_init = max.(BtT_init(pi_init), 1e-10)
    lam_L_init = sum(alpha_act) / max(n_act * L_eff, eps_val)

    log_lam_K = log.(lam_K_init)
    log_lam_L = log(lam_L_init)

    avg_start = max(3 * max_iter ÷ 4, 1)
    lam_K_sum = zeros(n_act)
    lam_L_sum = 0.0
    avg_count = 0

    log_y_K = copy(log_lam_K)
    log_y_L = log_lam_L

    prev_grad_K   = zeros(n_act)
    prev_grad_L   = 0.0
    prev_dtheta_K = zeros(n_act)
    prev_dtheta_L = 0.0

    eta_K_cur = eta_K
    eta_L_cur = eta_L
    LOG_CLAMP = 1.5

    C_act     = zeros(n_act)
    converged = false
    opt_iter  = 0

    for iter in 1:max_iter
        opt_iter = iter

        y_K    = exp.(log_y_K)
        y_L    = exp(log_y_L)
        pi_vec = BtT_active(y_K) .+ y_L .* l_tilde_act
        C_act  = gamma_act .+ alpha_act ./ max.(pi_vec, eps_val)

        s_K = Kv    .- Bt_active(C_act)
        s_L = L_eff  - dot(l_tilde_act, C_act)

        primal_ok_K = all(s_K ./ (Kv .+ eps_val) .>= -tol_p)
        primal_ok_L = (s_L / (abs(L_eff) + eps_val)) >= -tol_p
        lam_K_cur   = exp.(log_lam_K)
        lam_L_cur   = exp(log_lam_L)
        comp_ok_K   = all(abs.(lam_K_cur .* s_K) .<= tol_d)
        comp_ok_L   = abs(lam_L_cur * s_L) <= tol_d

        if primal_ok_K && primal_ok_L && comp_ok_K && comp_ok_L
            converged = true
            break
        end

        grad_K = (.-s_K) ./ (Kv .+ eps_val)
        grad_L = (-s_L) / (abs(L_eff) + eps_val)

        # Barzilai-Borwein adaptive step
        if iter > 1
            dy_K = grad_K .- prev_grad_K
            dot_ss = dot(prev_dtheta_K, prev_dtheta_K) + prev_dtheta_L^2
            dot_sg = dot(prev_dtheta_K, dy_K) + prev_dtheta_L * (grad_L - prev_grad_L)
            if abs(dot_sg) > 1e-30
                bb_eta = clamp(abs(dot_ss / dot_sg), 0.01, 0.5)
                eta_K_cur = bb_eta
                eta_L_cur = bb_eta
            end
        end

        prev_grad_K = copy(grad_K)
        prev_grad_L = grad_L

        step_K = clamp.(eta_K_cur .* grad_K, -LOG_CLAMP, LOG_CLAMP)
        step_L = clamp(eta_L_cur * grad_L, -LOG_CLAMP, LOG_CLAMP)

        prev_dtheta_K = copy(step_K)
        prev_dtheta_L = step_L

        log_lam_K_new = log_y_K .+ step_K
        log_lam_L_new = log_y_L  + step_L

        beta_t = (iter - 1.0) / (iter + 2.0)
        log_y_K    = log_lam_K_new .+ beta_t .* (log_lam_K_new .- log_lam_K)
        log_y_L    = log_lam_L_new  + beta_t  * (log_lam_L_new  - log_lam_L)

        log_lam_K .= log_lam_K_new
        log_lam_L  = log_lam_L_new

        if iter >= avg_start
            lam_K_sum .+= exp.(log_lam_K)
            lam_L_sum  += exp(log_lam_L)
            avg_count  += 1
        end
    end

    lam_K = exp.(log_lam_K)
    lam_L = exp(log_lam_L)

    if !converged && avg_count > 0
        lam_K  = lam_K_sum ./ avg_count
        lam_L  = lam_L_sum  / avg_count
        pi_vec = BtT_active(lam_K) .+ lam_L .* l_tilde_act
        C_act  = gamma_act .+ alpha_act ./ max.(pi_vec, eps_val)
    end

    lam_K_full         = zeros(n)
    lam_K_full[active] .= lam_K

    pi_vec_star = BtT(lam_K_full) .+ lam_L .* l_tilde_v
    C_star      = zeros(n)
    C_star[active] .= C_act
    X_star = neumann_apply(A_m, C_star .+ dK_v .+ G_v, k)

    return (
        C_star     = C_star,
        X_star     = X_star,
        pi_star    = pi_vec_star,
        success    = converged,
        lambda_K   = lam_K_full,
        lambda_L   = lam_L,
        iterations = opt_iter
    )
end

# Nominal income: scale total output by the ratio of value-added to output.
function compute_income(v, X_star, pi, A)
    X_v  = _v(X_star)
    pi_v = _v(pi)
    A_m  = _m(A)
    AX         = A_m * X_v
    net_output = X_v .- AX
    deflator   = dot(pi_v, net_output)
    total_X    = sum(X_v)
    return deflator > 0 ? v * total_X / deflator : v * total_X
end

# Intra-quarter tatonnement: evolves preferences via log-OU toward alpha_slow,
# clears monthly prices via iterative excess-demand correction, and accumulates
# revealed demand for the planner's G_hat signal.
function fast_loop(P_base, C_plan, alpha_true_start, alpha_slow,
                   rng::AbstractRNG,
                   drift_rho::Real, drift_sigma::Real, noise_sigma::Real,
                   Y::Real, gamma, K_v, n_months::Int=3;
                   theta_drift::Float64=0.1,
                   max_price_iter::Int=25,
                   price_tol::Float64=0.005,
                   price_step_cap::Float64=0.5)

    P_base_v  = _v(P_base)
    C_plan_v  = _v(C_plan)
    gamma_v   = _v(gamma)
    alpha_s_v = _v(alpha_slow)
    K_v_vec   = _v(K_v)
    Y_f       = Float64(Y)

    C_m     = C_plan_v ./ n_months
    gamma_m = gamma_v  ./ n_months
    Y_m     = Y_f / n_months

    n         = length(P_base_v)
    C_monthly = zeros(n_months, n)
    P_monthly = zeros(n_months, n)

    C_hat_sum       = zeros(n)
    a_reveal_sum    = zeros(n)
    monthly_drifts  = zeros(n_months)
    monthly_resid_Y = zeros(n_months)

    a_f    = copy(_v(alpha_true_start))
    active = a_f .> 0

    for tau in 1:n_months

        # 1. Preference drift: log-OU mean-reversion toward alpha_slow
        log_f = [a > 1e-25 ? log(a) : -25.0 for a in a_f]
        log_s = [a > 1e-25 ? log(a) : -25.0 for a in alpha_s_v]

        for i in 1:n
            if active[i]
                drift      = theta_drift * (log_s[i] - log_f[i])
                shock      = drift_sigma  * randn(rng)
                noise      = noise_sigma  * randn(rng)
                log_f[i]  += drift + shock + noise
            end
        end
        exp_f = exp.(log_f)
        a_f   = exp_f ./ max(sum(exp_f), 1e-30)

        # 2. Iterative price tatonnement toward market clearing
        P_iter = copy(P_base_v)
        for _ in 1:max_price_iter
            resid_Y_iter = max(Y_m - dot(P_iter, gamma_m), 1e-30)
            C_d_iter     = gamma_m .+ a_f .* resid_Y_iter ./ max.(P_iter, 1e-30)
            Z_iter       = C_d_iter .- C_m
            eps_iter = min.(-1.0 .+ (gamma_m ./ max.(C_d_iter, 1e-30)) .* (1.0 .- a_f), -1e-4)
            correction = clamp.(
                eps_iter .* Z_iter ./ max.(abs.(C_d_iter), 1e-30),
                -price_step_cap, price_step_cap
            )
            P_iter = max.(P_iter .* (1.0 .- correction), 1e-30)
            if maximum(abs.(Z_iter) ./ max.(C_m, 1e-30)) < price_tol
                break
            end
        end
        P_clear = P_iter

        resid_Y_star = max(Y_m - dot(P_clear, gamma_m), 1e-30)
        C_d          = gamma_m .+ a_f .* resid_Y_star ./ max.(P_clear, 1e-30)

        # 3. Revealed preference shares from LES expenditure
        a_reveal_sum .+= (P_clear .* (C_d .- gamma_m)) ./ resid_Y_star

        # 4. LES price-correction signal for G_hat
        eps_final = min.(-1.0 .+ (gamma_m ./ max.(C_d, 1e-30)) .* (1.0 .- a_f), -1e-4)
        delta_p  = (P_clear .- P_base_v) ./ max.(P_base_v, 1e-30)
        signal   = clamp.(eps_final .* delta_p, -0.1, 0.1)
        denom    = 1.0 .+ signal
        C_hat_sum .+= C_m ./ denom

        C_monthly[tau, :]       .= C_d
        P_monthly[tau, :]       .= P_clear
        monthly_resid_Y[tau]     = resid_Y_star
        monthly_drifts[tau]      = mean(abs.(P_clear ./ P_base_v .- 1.0))
    end

    G_hat_bare = max.(0.0, (C_hat_sum .- C_plan_v) ./ max.(C_plan_v, 1e-30))

    active_consumer = alpha_s_v .> 1e-12
    P_final      = P_monthly[n_months, :]
    signed_drift = mean((P_final ./ P_base_v .- 1.0)[active_consumer])
    abs_drift    = mean(abs.(P_final ./ P_base_v .- 1.0)[active_consumer])

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
    )
end

end  # module ModelCore