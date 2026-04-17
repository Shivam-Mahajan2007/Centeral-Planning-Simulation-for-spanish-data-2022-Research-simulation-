module ModelCore

using LinearAlgebra
using SparseArrays
using Random
using Statistics
using PythonCall: pyconvert


export neumann_apply, evolve_true_alpha, revealed_demand, infer_growth,
       compute_investment, solve_planner, solve_planner_unclamped, compute_income, fast_loop

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
function evolve_true_alpha(alpha_true, rng::AbstractRNG, alpha_bar,
                           drift::Real=0.012, kappa::Real=0.15)
    alpha_true_v = _v(alpha_true)
    alpha_bar_v  = _v(alpha_bar)
    sigma        = Float64(drift)

    active  = alpha_true_v .> 0
    log_a   = zeros(length(alpha_true_v))
    log_bar = zeros(length(alpha_true_v))
    for i in eachindex(alpha_true_v)
        if active[i]
            log_a[i]   = log(max(alpha_true_v[i], 1e-30))
            log_bar[i] = log(max(alpha_bar_v[i],  1e-30))
        end
    end

    shocks = zeros(length(alpha_true_v))
    for i in eachindex(alpha_true_v)
        active[i] && (shocks[i] = sigma * randn(rng))
    end

    log_new = log_a .+ kappa .* (log_bar .- log_a) .+ shocks
    a       = zeros(length(alpha_true_v))
    for i in eachindex(alpha_true_v)
        active[i] && (a[i] = exp(log_new[i]))
    end

    return a ./ sum(a)
end

# -----------------------------------------------------------------------------
function revealed_demand(C_monthly, P_monthly, P_base, C_plan)
    Cm   = _m(C_monthly)   # Monthly measured consumption (Supply realized)
    Pm   = _m(P_monthly)   # Monthly market prices
    Pb   = _v(P_base)      # Base prices
    eps_val = -1.0         # Price elasticity

    n_months = size(Pm, 1)
    n        = length(Pb)
    Chat     = zeros(n)

    for tau in 1:n_months
        P_tau = Pm[tau, :]
        C_tau = Cm[tau, :]
        for i in 1:n
            # Equation (16): Chat = sum( C_m / (1 + eps * (P_tau - P_base)/P_base) )
            # With eps = -1, denom = 2 - P_tau/P_base
            denom = 1.0 + eps_val * (P_tau[i] - Pb[i]) / (Pb[i] + 1e-30)
            denom = max(denom, 0.1) # Safety floor to prevent division by zero
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
function compute_investment(G_hat, A_bar, kappa, C_prev, G_vec, g_step, c_step; k::Int=20)
    G_hat_v  = _v(G_hat)
    kappa_v  = _v(kappa)
    C_prev_v = _v(C_prev)
    G_vec_v  = _v(G_vec)
    g        = Float64(g_step)

    # 1. Consumption growth term: (MC + (MC)^2) C_{t-1}
    # Enforce a minimum capacity expansion target (c_step) to proactively relieve bottlenecks
    gC      = max.(G_hat_v, 0)
    Gv      = gC .* C_prev_v
    term1_C = kappa_v .* neumann_apply(A_bar, Gv, k)
    term2_C = kappa_v .* neumann_apply(A_bar, gC .* term1_C, k)
    
    # 2. Government growth term: (Mg + (Mg)^2) G_t
    # Since g is a scalar growth rate, (Mg) G_t = g * (M G_t)
    # and (Mg)^2 G_t = g^2 * (M (M G_t))
    G_v_g   = g .* G_vec_v
    term1_G = kappa_v .* neumann_apply(A_bar, G_v_g, k)
    term2_G = kappa_v .* neumann_apply(A_bar, g .* term1_G, k)

    return term1_C .+ term2_C .+ term1_G .+ term2_G
end

# -----------------------------------------------------------------------------
# Multiplicative Dual Ascent Planner
#
# Problem (active sectors only):
#   max  sum_i alpha_i * log(C_i)
#   s.t. Bt(C) <= K_eff      (capital constraint,  lam_K >= 0)
#        l_tilde . C <= L_eff (labour constraint,   lam_L >= 0)
#
# Algorithm per iteration:
#   0. Guess lam_K, lam_L  (warm-start from previous quarter if available)
#
#   1. Compute shadow prices and consumption via stationarity (dL/dC = 0):
#        pi_vec = BtT(lam_K) + lam_L * l_tilde
#        C      = alpha / pi_vec
#
#   2. Compute slack for each constraint
#      (positive = feasible/unused, negative = violated):
#        s_K = K_eff - Bt(C)
#        s_L = L_eff - l_tilde . C
#
#   3. Check KKT convergence (all conditions must hold):
#        Primal feasibility:         s_K / K_eff >= -tol_p  and  s_L / L_eff >= -tol_p
#        Complementary slackness:    lam_K * s_K / K_eff <= tol_d
#                                    lam_L * s_L / L_eff <= tol_d
#      Normalising by constraint size makes the tolerance scale-free across
#      sectors with very different capital endowments.
#
#   4. Multiplicative dual update (fixed step eta, keeps lam > 0 by construction):
#        lam_K <- max( lam_K * exp( eta_K * (-s_K) / K_eff ),  eps )
#        lam_L <- max( lam_L * exp( eta_L * (-s_L) / L_eff ),  eps )
#
#      Interpretation: -s is the violation signal.
#        -s > 0  (constraint violated) -> exp > 1 -> lam rises
#        -s < 0  (constraint slack)    -> exp < 1 -> lam falls
#        -s = 0  (exactly binding)     -> exp = 1 -> lam unchanged
#      Dividing by the constraint value keeps the signal scale-free.
# -----------------------------------------------------------------------------
function solve_planner(alpha, A_bar, B, l_vec, l_tilde, dK, K, L_total::Real, G_vec;
                       C_prev=nothing, lambda_K_prev=nothing, lambda_L_prev=nothing,
                       k::Int=20,
                       tol::Float64=1e-4,
                       tol_p::Float64=tol, tol_d::Float64=tol,
                       eta_K::Float64=0.4, eta_L::Float64=0.4,
                       max_iter::Int=2000)

    alpha_v   = _v(alpha)
    A_m       = _m(A_bar)
    eps_val   = 1e-15

    # -- B operator (handles both vector and matrix B) -------------------------
    is_B_1d = B isa AbstractArray ? ndims(B) == 1 : pyconvert(Int, B.ndim) == 1
    B_v = is_B_1d ? _v(B)   : nothing
    B_m = is_B_1d ? nothing : _m(B)

    l_tilde_v = _v(l_tilde)
    dK_v      = _v(dK)
    G_v       = _v(G_vec)
    K_v       = _v(K)
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

    # -- Effective resources (net of committed investment dK) ------------------
    K_eff = K_v .- Bt(dK_v)
    L_eff = L_total_f - dot(l_tilde_v, dK_v)

    # Only sectors with strictly positive net capital are optimised.
    # Inactive sectors receive zero consumption in the final plan.
    active      = K_eff .> eps_val
    Kv          = K_eff[active]
    alpha_act   = alpha_v[active]
    l_tilde_act = l_tilde_v[active]
    n_act       = length(Kv)

    # -- Active-sector projections of Bt / BtT --------------------------------
    Bt_active = v_act -> begin
        v_full = zeros(n); v_full[active] .= v_act
        Bt(v_full)[active]
    end
    BtT_active = w_act -> begin
        w_full = zeros(n); w_full[active] .= w_act
        BtT(w_full)[active]
    end

    # Refined cold-start: lam_k = ((I - A_bar.T) (alpha/C_0))/kappa
    if C_prev !== nothing
        C_prev_v = _v(C_prev)
        p_guess  = alpha_v ./ max.(C_prev_v, eps_val)
        rhs      = p_guess .- A_m' * p_guess
        lam_K    = max.(rhs[active] ./ max.(B_v[active], eps_val), eps_val)
    else
        alpha_mean = sum(alpha_act) / max(n_act, 1)
        Kv_mean    = sum(Kv)        / max(n_act, 1)
        lam_K      = fill(alpha_mean / max(Kv_mean, eps_val), n_act)
    end
    lam_L = (sum(alpha_act) / max(n_act, 1)) / max(L_eff, eps_val)

    local_eta_K = fill(1.0, n_act)
    prev_s_K    = zeros(n_act)
    local_eta_L = 1.0
    prev_s_L    = 0.0

    C_act     = zeros(n_act)
    converged = false
    opt_iter  = 0

    for iter in 1:max_iter
        opt_iter = iter

        # -- Step 1: Compute pi_vec and C (stationarity: dL/dC = 0 -> C = alpha/pi) --
        pi_vec = BtT_active(lam_K) .+ lam_L .* l_tilde_act
        C_act  = alpha_act ./ max.(pi_vec, eps_val)

        # -- Step 2: Compute slack ---------------------------------------------
        # s > 0 -> constraint has room (feasible)
        # s < 0 -> constraint is violated
        s_K = Kv   .- Bt_active(C_act)
        s_L = L_eff - dot(l_tilde_act, C_act)

        # -- Step 3: Check KKT convergence ------------------------------------
        # Primal feasibility: normalised slack must not be more negative than -tol_p.
        # Since lam_K > 0, lam_K * s_K < 0 implies s_K < 0 (violation), so the
        # primal check is essential to catch violated constraints where the dual
        # has not yet risen enough to push lam * slack back above zero.
        primal_ok_K = all(s_K ./ (Kv .+ eps_val) .>= -tol_p)
        primal_ok_L = (s_L / (abs(L_eff) + eps_val)) >= -tol_p

        # Complementary slackness (Strict KKT: lam * slack = 0):
        # Since alpha weights sum to 1, lam * s is already normalized to the
        # total expenditure "value", making this check scale-free across sectors.
        comp_ok_K = all(abs.(lam_K .* s_K) .<= tol_d)
        comp_ok_L = abs(lam_L * s_L) <= tol_d

        if primal_ok_K && primal_ok_L && comp_ok_K && comp_ok_L
            converged = true
            break
        end

        # -- Step 4: Adaptive Multiplicative update ---------------------------
        # If violation direction is constant, accelerate (x1.1).
        # If it flips (oscillation), decelerate (x0.5).
        if iter > 1
            # Update local eta for capital constraints
            for i in 1:n_act
                if s_K[i] * prev_s_K[i] > 0
                    local_eta_K[i] = min(local_eta_K[i] * 1.1, 5.0)
                else
                    local_eta_K[i] = max(local_eta_K[i] * 0.5, 0.1)
                end
            end
            # Update local eta for labor constraint
            if s_L * prev_s_L > 0
                local_eta_L = min(local_eta_L * 1.1, 5.0)
            else
                local_eta_L = max(local_eta_L * 0.5, 0.1)
            end
        end
        prev_s_K .= s_K
        prev_s_L  = s_L

        exp_K = clamp.(eta_K .* local_eta_K .* (.-s_K) ./ (Kv .+ eps_val),      -20.0, 20.0)
        exp_L = clamp(eta_L  *  local_eta_L *  (-s_L)   / (abs(L_eff) + eps_val), -20.0, 20.0)
        lam_K = max.(lam_K .* exp.(exp_K), eps_val)
        lam_L = max(lam_L   *  exp(exp_L),  eps_val)
    end

    # -- Final recovery --------------------------------------------------------
    # Build full-length lam_K and recompute shadow prices over all n sectors.
    lam_K_full         = zeros(n)
    lam_K_full[active] .= lam_K

    pi_vec_star = BtT(lam_K_full) .+ lam_L .* l_tilde_v

    # Assign consumption only to active sectors; inactive sectors get zero.
    C_star         = zeros(n)
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
    v_v  = _v(v)
    X_v  = _v(X_star)
    pi_v = _v(pi)
    A_m  = _m(A)

    AX         = A_m * X_v
    net_output = X_v .- AX
    deflator   = dot(pi_v, net_output)
    return deflator > 0 ? dot(v_v, X_v) / deflator : dot(v_v, X_v)
end

# -----------------------------------------------------------------------------
function fast_loop(P_base, C_plan, alpha_true, Y::Real, n_months::Int=3)
    P_base_v     = _v(P_base)
    C_plan_v     = _v(C_plan)
    alpha_true_v = _v(alpha_true)
    Y_f          = Float64(Y)

    C_m = C_plan_v ./ n_months
    Y_m = Y_f / n_months
    P   = copy(P_base_v)

    n         = length(P_base_v)
    C_monthly = zeros(n_months, n)
    P_monthly = zeros(n_months, n)

    for tau in 1:n_months
        P_s  = [p > 0 ? p : 1e-30 for p in P]
        C_ms = [c > 0 ? c : 1e-30 for c in C_m]
        C_d  = alpha_true_v .* Y_m ./ P_s
        # Equation (22): P* = P * (1 - (1/eps) * (C-Cm)/C)
        # With eps = -1 for alpha-demand, P* = P * (1 + (C-Cm)/C)
        C_d_safe = [c > 1e-30 ? c : 1e-30 for c in C_d]
        P   .= P_s .* (1.0 .+ (C_d .- C_ms) ./ C_d_safe)
        # Safety: ensure prices remain positive
        P   .= [p > 1e-30 ? p : 1e-30 for p in P]
        
        C_monthly[tau, :] .= C_d
        P_monthly[tau, :] .= P
    end

    rel_deviations = abs.(P ./ P_base_v .- 1.0)
    drift = mean(rel_deviations)

    return (C_monthly=C_monthly, P_monthly=P_monthly, P_final=P, price_drift=drift)
end

# -----------------------------------------------------------------------------
# Diagnostic variant: duals are NOT clamped to >= 0 so they can go negative.
# This lets you see whether iteration-count spikes come from the clamp bouncing
# lam off zero, or from genuine ill-conditioning of the problem.
#
# The KKT loop structure is identical to solve_planner; only the dual update
# drops the max(0, .) floor. Do not use for production runs -- negative
# multipliers have no economic meaning and C = alpha/pi can blow up if pi -> 0.
# Guard rails kept: pi is still clamped to eps_val so C stays finite.
# -----------------------------------------------------------------------------
function solve_planner_unclamped(alpha, A_bar, B, l_vec, l_tilde, dK, K, L_total::Real;
                                 C_prev=nothing, lambda_K_prev=nothing, lambda_L_prev=nothing,
                                 k::Int=20,
                                 tol_p::Float64=1e-4, tol_d::Float64=1e-4,
                                 eta_K::Float64=0.15, eta_L::Float64=0.15,
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
    K_v       = _v(K)
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
    K_eff  = K_v .- Bt(dK_v)
    L_eff  = L_total_f - dot(l_tilde_v, dK_v)
    active = K_eff .> eps_val
    Kv     = K_eff[active]
    n_act  = length(Kv)

    norm_Lv = abs(L_eff)

    Bt_active  = v        -> Bt(v)[active]
    BtT_active = w_active -> begin
        w_full = zeros(n); w_full[active] .= w_active; BtT(w_full)
    end

    # Refined cold-start: lam_k = ((I - A_bar.T) (alpha/C_0))/kappa
    if C_prev !== nothing
        C_prev_v = _v(C_prev)
        p_guess  = alpha_v ./ max.(C_prev_v, eps_val)
        rhs      = p_guess .- A_m' * p_guess
        lam_K    = rhs[active] ./ max.(B_v[active], eps_val)
    else
        alpha_mean = sum(alpha_v) / max(n, 1)
        Kv_mean    = sum(Kv)      / max(n_act, 1)
        lam_K      = fill(alpha_mean / max(Kv_mean, eps_val), n_act)
    end

    if false # lambda_L_prev !== nothing
        lam_L = Float64(pyconvert(Float64, lambda_L_prev))
    else
        lam_L = alpha_mean / max(L_eff, eps_val)
    end

    C         = zeros(n)
    converged = false
    opt_iter  = 0

    # Track dual trajectory for diagnostics
    lam_K_hist = Vector{Vector{Float64}}()
    lam_L_hist = Vector{Float64}()

    for iter in 1:max_iter
        opt_iter = iter

        push!(lam_K_hist, copy(lam_K))
        push!(lam_L_hist, lam_L)

        # -- Step 1: Find C ---------------------------------------------------
        p = BtT_active(lam_K) .+ lam_L .* l_tilde_v
        p = max.(p, eps_val)    # keep C finite even if lam goes negative

        C = alpha_v ./ p

        # -- Step 2: Find slack -----------------------------------------------
        s_K = K_eff .- Bt_active(C)
        s_L = L_eff - dot(l_tilde_v, C)

        # -- Step 3: Check KKT conditions -------------------------------------
        # Primal feasibility and complementary slackness, both normalised by
        # constraint size so tol_p and tol_d are scale-free.
        viol_K = maximum(abs.(min.(0.0, s_K)) ./ (1.0 .+ Kv))
        viol_L = abs(min(0.0, s_L))           / (1.0 + norm_Lv)
        comp_K = maximum(abs.(lam_K .* s_K))
        comp_L = abs(lam_L * s_L)

        if viol_K <= tol_p && viol_L <= tol_p && comp_K <= tol_d && comp_L <= tol_d
            converged = true
            break
        end

        # -- Step 4: Update lam -- NO non-negativity clamp --------------------
        rel_s_K = s_K ./ (1.0 .+ Kv)
        rel_s_L = s_L  / (1.0  + norm_Lv)

        lam_K_scale = max.(abs.(lam_K), alpha_mean / max(Kv_mean, eps_val))
        lam_L_scale = max(abs(lam_L),   alpha_mean / max(L_eff,   eps_val))

        lam_K = lam_K .- eta_K .* lam_K_scale .* rel_s_K    # unclamped
        lam_L = lam_L -  eta_L *  lam_L_scale *  rel_s_L    # unclamped
    end

    # -- Post-loop ------------------------------------------------------------
    C_star = C
    # X* = (I-A)^-1 * (C + dK + G)
    # X* = (I-A)^-1 * (C + dK + G)
    X_star = neumann_apply(A_m, C_star .+ dK_v .+ G_v, k)
    p_star = BtT_active(lam_K) .+ lam_L .* l_tilde_v

    lam_K_full = zeros(n)
    lam_K_full[active] .= lam_K

    return (
        C_star        = C_star,
        X_star        = X_star,
        pi_star       = p_star,
        success       = converged,
        lambda_K      = lam_K_full,
        lambda_L      = lam_L,
        iterations    = opt_iter,
        lambda_K_hist = lam_K_hist,
        lambda_L_hist = lam_L_hist,
    )
end

end  # module ModelCore
