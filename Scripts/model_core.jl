module ModelCore

using LinearAlgebra
using SparseArrays
using Random
using Statistics
using PythonCall: pyconvert


export neumann_apply, evolve_true_alpha, revealed_demand, infer_growth,
       compute_investment, solve_planner, compute_income, fast_loop

_v(x) = pyconvert(Vector{Float64}, x)
_m(x) = pyconvert(Matrix{Float64}, x)

# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# --- Stochastic Preference Evolution (Structural Component) ------------------
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
        if active[i]
            s_new[i] = exp(log_s_new[i])
        end
    end

    s_new ./= sum(s_new)

    return s_new
end

# -----------------------------------------------------------------------------
function revealed_demand(C_monthly, P_monthly, P_base, C_plan, alpha, gamma)
    Cm = pyconvert(Array{Float64, 2}, C_monthly)
    Pm = pyconvert(Array{Float64, 2}, P_monthly)
    Pb = pyconvert(Array{Float64, 1}, P_base)
    Cp = pyconvert(Array{Float64, 1}, C_plan)
    av = pyconvert(Array{Float64, 1}, alpha)
    gv = pyconvert(Array{Float64, 1}, gamma)

    n_months, n = size(Cm)
    Chat = zeros(n)

    # LES price elasticity: eps_i = -1 + gamma_i * (1 - alpha_i) / C_i
    # We use C_plan as the reference consumption for the elasticity calculation
    eps_vec = -1.0 .+ (gv .* (1.0 .- av)) ./ max.(Cp, 1e-30)

    for tau in 1:n_months
        P_tau = Pm[tau, :]
        C_tau = Cm[tau, :]
        for i in 1:n
            # revealed demand: C / (1 + eps * (P-Pb)/Pb)
            denom = 1.0 + eps_vec[i] * (P_tau[i] - Pb[i]) / (Pb[i] + 1e-30)
            denom = max(denom, 0.1)  # Stability clamp
            Chat[i] += C_tau[i] / denom
        end
    end
    return Chat
end

# -----------------------------------------------------------------------------
function infer_growth(C_hat, C_prev)
    Chat_v  = _v(C_hat)
    Cprev_v = _v(C_prev)
    # Equation (17): G_hat = max(0, (Chat - C_prev) / C_prev)
    res = zeros(length(Chat_v))
    for i in eachindex(Chat_v)
        denom = max(Cprev_v[i], 1e-30)
        res[i] = max(0.0, (Chat_v[i] - Cprev_v[i]) / denom)
    end
    return res
end


# -----------------------------------------------------------------------------
function compute_investment(G_hat, A_bar, B, C_prev, G_vec, g_step, c_step; k::Int=20)
    G_hat_v  = _v(G_hat)
    C_prev_v = _v(C_prev)
    G_vec_v  = _v(G_vec)
    g        = Float64(g_step)

    # -- B operator -----------------------------------------------------------
    is_B_1d = B isa AbstractArray ? ndims(B) == 1 : pyconvert(Int, B.ndim) == 1
    B_m = is_B_1d ? nothing : _m(B)
    B_v = is_B_1d ? _v(B)   : nothing

    A_m = _m(A_bar)

    B_apply = v -> is_B_1d ? B_v .* v : B_m * v

    # 1. Consumption growth term: (MC + (MC)^2) C_{t-1}
    gC      = max.(G_hat_v, 0)
    Gv      = gC .* C_prev_v
    
    # term1_C = B @ neumann(A, Gv)
    term1_C = B_apply(neumann_apply(A_m, Gv, k))
    # term2_C = B @ neumann(A, gC * term1_C)
    term2_C = B_apply(neumann_apply(A_m, gC .* term1_C, k))
    
    # 2. Government growth term: (Mg + (Mg)^2) G_t
    G_v_g   = g .* G_vec_v
    term1_G = B_apply(neumann_apply(A_m, G_v_g, k))
    term2_G = B_apply(neumann_apply(A_m, g .* term1_G, k))

    return term1_C .+ term2_C .+ term1_G .+ term2_G
end

function solve_planner(alpha, A_bar, B, l_tilde, dK, K, L_total::Real, G_vec, gamma;
                       k::Int=20,
                       tol::Float64=1e-4,
                       tol_p::Float64=tol, tol_d::Float64=tol,
                       eta_K::Float64=0.25, eta_L::Float64=0.35,
                       max_iter::Int=2000)

    alpha_v   = _v(alpha)
    A_m       = _m(A_bar)
    eps_val   = 1e-15

    # -- B operator -----------------------------------------------------------
    is_B_1d = B isa AbstractArray ? ndims(B) == 1 : pyconvert(Int, B.ndim) == 1
    B_v = is_B_1d ? _v(B)   : nothing
    B_m = is_B_1d ? nothing : _m(B)

    l_tilde_v = _v(l_tilde)
    dK_v      = _v(dK)
    G_v       = _v(G_vec)
    K_v       = _v(K)
    gamma_v   = _v(gamma)
    L_total_f = Float64(L_total)
    n         = length(alpha_v)

    local Bt, BtT
    if is_B_1d
        Bt  = v -> B_v .* neumann_apply(A_m, v, k)
        BtT = w -> neumann_apply(A_m', B_v .* w, k)
    else
        Bt  = v -> B_m * neumann_apply(A_m, v, k)
        BtT = w -> neumann_apply(A_m', B_m' * w, k)
    end

    # -- Effective resources --------------------------------------------------
    K_eff = K_v .- Bt(dK_v)
    L_eff = L_total_f - dot(l_tilde_v, dK_v)

    # -- Active sector mask (relative threshold) ------------------------------
    pos_Keff  = K_eff[K_eff .> 0]
    K_mean    = isempty(pos_Keff) ? 1.0 : mean(pos_Keff)
    active    = K_eff .> 1e-6 * K_mean

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

    # -- Cold start heuristic -------------------------------------------------
    # From stationarity: π_i = α_i / (C_i − γ_i) and π ≈ BtT(λ_K) + λ_L l̃.
    # We want λ_K_i ≈ α_i / K_eff_i (capital is the binding constraint at start).
    alpha_mean = sum(alpha_act) / max(n_act, 1)
    Kv_mean    = sum(Kv)        / max(n_act, 1)
    lam_K      = fill(alpha_mean / max(Kv_mean, eps_val), n_act)
    lam_L      = alpha_mean / max(L_eff, eps_val)

    # -- Nesterov state -------------------------------------------------------
    # y is the look-ahead point; initialised to lam.
    log_y_K    = log.(lam_K)
    log_y_L    = log(lam_L)
    log_lam_K  = copy(log_y_K)
    log_lam_L  = log_y_L

    # Polyak averaging accumulators (second half of run)
    avg_start  = max(max_iter ÷ 2, 1)
    lam_K_sum  = zeros(n_act)
    lam_L_sum  = 0.0
    avg_count  = 0

    C_act     = zeros(n_act)
    converged = false
    opt_iter  = 0

    for iter in 1:max_iter
        opt_iter = iter

        # -- Step 1: Gradient at look-ahead y ---------------------------------
        y_K    = exp.(log_y_K)
        y_L    = exp(log_y_L)
        pi_vec = BtT_active(y_K) .+ y_L .* l_tilde_act
        C_act  = gamma_act .+ alpha_act ./ max.(pi_vec, eps_val)

        s_K = Kv   .- Bt_active(C_act)
        s_L = L_eff - dot(l_tilde_act, C_act)

        # -- Step 2: KKT check ------------------------------------------------
        # Primal feasibility checked at the look-ahead y (trial point).
        # Complementary slackness checked at the TRUE iterate lam (not y),
        # because y overshoots due to Nesterov momentum.
        primal_ok_K = all(s_K ./ (Kv .+ eps_val) .>= -tol_p)
        primal_ok_L = (s_L / (abs(L_eff) + eps_val)) >= -tol_p
        lam_K_cur   = exp.(log_lam_K)
        lam_L_cur   = exp(log_lam_L)
        comp_ok_K   = all(abs.(lam_K_cur .* s_K) .<= tol_d)
        comp_ok_L   = abs(lam_L_cur  *  s_L) <= tol_d

        if primal_ok_K && primal_ok_L && comp_ok_K && comp_ok_L
            converged = true
            break
        end

        # -- Step 3: Multiplicative ascent step from y ------------------------
        # Clamp [-3, 3]: exp(3) ≈ 20× per step.
        # Fast enough for cold-start approach, safe enough to avoid overshooting.
        step_K = clamp.(eta_K .* (.-s_K) ./ (Kv .+ eps_val), -3.0, 3.0)
        step_L = clamp(eta_L  *  (-s_L)  / (abs(L_eff) + eps_val), -3.0, 3.0)

        log_lam_K_new = log_y_K .+ step_K
        log_lam_L_new = log_y_L  + step_L

        # -- Step 4: Nesterov extrapolation -----------------------------------
        # β_t = (t−1)/(t+2) grows from 0 toward 1, giving increasing momentum.
        # In log space: log y_{t+1} = log lam_{t+1} + β·(log lam_{t+1} − log lam_t)
        beta_t = (iter - 1.0) / (iter + 2.0)

        log_y_K    = log_lam_K_new .+ beta_t .* (log_lam_K_new .- log_lam_K)
        log_y_L    = log_lam_L_new  + beta_t *  (log_lam_L_new  - log_lam_L)

        log_lam_K .= log_lam_K_new
        log_lam_L  = log_lam_L_new

        # Polyak averaging
        if iter >= avg_start
            lam_K_sum .+= exp.(log_lam_K)
            lam_L_sum  += exp(log_lam_L)
            avg_count  += 1
        end
    end

    # -- Fallback: Polyak-averaged duals --------------------------------------
    lam_K = exp.(log_lam_K)
    lam_L = exp(log_lam_L)

    if !converged && avg_count > 0
        lam_K  = lam_K_sum ./ avg_count
        lam_L  = lam_L_sum  / avg_count
        pi_vec = BtT_active(lam_K) .+ lam_L .* l_tilde_act
        C_act  = gamma_act .+ alpha_act ./ max.(pi_vec, eps_val)
    end

    # -- Final primal recovery ------------------------------------------------
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
# -----------------------------------------------------------------------------
function compute_income(v, X_star, pi, A)
    X_v  = _v(X_star)
    pi_v = _v(pi)
    A_m  = _m(A)

    AX         = A_m * X_v
    net_output = X_v .- AX
    deflator   = dot(pi_v, net_output)
    total_X = sum(X_v)
    return deflator > 0 ? v * total_X / deflator : v * total_X
end

# -----------------------------------------------------------------------------
function fast_loop(P_base, C_plan, alpha_fast_start, alpha_slow, rng::AbstractRNG, drift_fast::Real, kappa_fast::Real, Y::Real, gamma, n_months::Int=3; step_size::Float64=0.25)
    P_base_v     = _v(P_base)
    C_plan_v     = _v(C_plan)
    gamma_v      = _v(gamma)
    alpha_s_v    = _v(alpha_slow)
    Y_f          = Float64(Y)

    C_m = C_plan_v ./ n_months
    gamma_m = gamma_v ./ n_months
    Y_m = Y_f / n_months
    P   = copy(P_base_v)

    n         = length(P_base_v)
    C_monthly = zeros(n_months, n)
    P_monthly = zeros(n_months, n)

    monthly_drifts = zeros(n_months)
    monthly_residual_Y = zeros(n_months)
    
    a_f = copy(_v(alpha_fast_start))
    active = a_f .> 0

    for tau in 1:n_months
        # 1. Evolve alpha_fast towards alpha_slow (monthly transit)
        log_f = zeros(n)
        log_s = zeros(n)
        for i in 1:n
            if active[i]
                log_f[i] = log(max(a_f[i], 1e-30))
                log_s[i] = log(max(alpha_s_v[i], 1e-30))
            end
        end
        shocks_f = zeros(n)
        for i in 1:n
            active[i] && (shocks_f[i] = drift_fast * randn(rng))
        end
        log_f_new = log_f .+ kappa_fast .* (log_s .- log_f) .+ shocks_f
        
        a_f_new = zeros(n)
        for i in 1:n
            if active[i]
                a_f_new[i] = exp(log_f_new[i])
            end
        end
        a_f = a_f_new ./ sum(a_f_new)

        # 2. Compute Tatonnement metrics
        P_s  = [p > 0 ? p : 1e-30 for p in P]
        C_ms = [c > 0 ? c : 1e-30 for c in C_m]
        
        # LES demand using the freshly evolved a_f
        resid_Y = max(0.0, Y_m - dot(P_s, gamma_m))
        C_d  = gamma_m .+ a_f .* resid_Y ./ P_s
        
        monthly_residual_Y[tau] = resid_Y
        
        # Equation (22): P* = P * (1 - (1/eps) * (C-Cm)/C)
        # We add damping (step_size) to ensure stability when starting far from equilibrium
        C_d_safe = [c > 1e-30 ? c : 1e-30 for c in C_d]
        P   .= P_s .* (1.0 .+ step_size .* (C_d .- C_ms) ./ C_d_safe)
        # Safety: ensure prices remain positive
        P   .= [p > 1e-30 ? p : 1e-30 for p in P]
        
        C_monthly[tau, :] .= C_d
        P_monthly[tau, :] .= P
        
        monthly_drifts[tau] = mean(P ./ P_base_v .- 1.0)
    end

    active_consumer = alpha_s_v .> 1e-12
    # Use a safe denominator for relative drift calculation
    P_base_v_safe = [p > 1e-30 ? p : 1e-30 for p in P_base_v]
    
    if any(active_consumer)
        signed_drift = mean((P ./ P_base_v_safe .- 1.0)[active_consumer])
        rel_deviations = abs.(P ./ P_base_v_safe .- 1.0)[active_consumer]
        drift = mean(rel_deviations)
    else
        signed_drift = 0.0
        drift = 0.0
    end

    return (C_monthly=C_monthly, P_monthly=P_monthly, P_final=P, 
            price_drift=drift, signed_drift=signed_drift, 
            monthly_drifts=monthly_drifts, monthly_residual_Y=monthly_residual_Y,
            alpha_true_final=a_f)
end

end  # module ModelCore
