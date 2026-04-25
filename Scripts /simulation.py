import numpy as np
import logging
import pickle
import time
from pathlib import Path
from calibration import ModelState
from julia_bridge import (
    CORE, _to_dense, evolve_structural_alpha, revealed_demand, infer_growth,
    compute_investment, solve_planner, compute_income,
    fast_loop,
)

logger = logging.getLogger(__name__)


class SimulationError(Exception):
    """Raised when the simulation hits a terminal numerical state (e.g. NaNs)."""
    pass



def compute_government_quantity(V: np.ndarray, A, P: np.ndarray) -> np.ndarray:
    """Compute government expenditure in quantity terms.

    Uses the cost-push price vector (derived from v_per_unit) to deflate
    nominal government expenditure into physical quantities:
        G_qty = (I - A^T)^{-1} v_per_unit / P

    Parameters
    ----------
    V : np.ndarray
        v_per_unit vector (value added per physical unit, EUR/unit).
        Replaces the old 'v = V_total / X' ratio that was formerly passed here.
    A : sparse or dense matrix
        Technical-coefficient matrix.
    P : np.ndarray
        Current price vector.
    """
    import scipy.sparse as sp
    n = V.shape[0]
    if sp.issparse(A):
        A_dense = A.toarray()
    else:
        A_dense = np.asarray(A)
    M = np.linalg.inv(np.eye(n) - A_dense.T)
    G_qty = M @ V / np.where(P > 1e-30, P, 1e-30)
    return G_qty


def run_quarter(state: ModelState) -> ModelState:
    t    = state.t
    slim = getattr(state, "slim_history", False)
    logger.info(f"\n-- Q{t+1} {'-'*50}")
    
    # 0. True preference drift (unseen by planner - MOVING TARGET OU PROCESS)
    gamma = getattr(state, "habit_persistence", 0.7) # DSGE-standard quarterly habit (CEE 2005)
    
    if t == 0:
        state.alpha_habit = state.alpha_bar.copy()
    else:
        # What consumers actually received and valued last quarter
        expenditures     = state.P_monthly * state.C_monthly
        subsistence_exp  = state.P_monthly * state.gamma
        discretionary_exp = np.maximum(expenditures - subsistence_exp, 0.0)
        
        discretionary_total = discretionary_exp.sum(axis=1, keepdims=True)
        discretionary_total = np.maximum(discretionary_total, 1e-30)
        
        # Marginal budget shares (alpha) derived from discretionary spending
        realized_shares = (discretionary_exp / discretionary_total).mean(axis=0)
        realized_shares /= realized_shares.sum()  # Enforce normalization
        
        # Continuously update the habit target (EMA tracking)
        state.alpha_habit = gamma * getattr(state, "alpha_habit", state.alpha_bar) + (1 - gamma) * realized_shares
        state.alpha_habit = np.maximum(state.alpha_habit, 1e-30)
        state.alpha_habit /= state.alpha_habit.sum()

    # Preferences: Structural component evolves quarterly
    # Removed OU process; now directly set to habit
    state.alpha_slow = state.alpha_habit.copy()
    
    if t == 0:
        C_hat = state.C.copy()
        G_hat = state.G_hat_init.copy()
        G_hat_bare = state.G_hat_init.copy()  # Initialize for first quarter
        state.G_hat_lagged = G_hat_bare.copy()
    else:
        # Use G_hat_bare computed from previous quarter's tatonnement
        G_hat_bare = getattr(state, "G_hat_bare_prev", state.G_hat_init)
        
        # SMA(2) smoothing: average of previous and two-quarters-ago signals
        G_hat_prev = getattr(state, "G_hat_lagged", G_hat_bare)
        G_hat = (G_hat_bare + G_hat_prev) / 2.0
        
        # Update lag for next quarter
        state.G_hat_lagged = G_hat_bare.copy()
        
    logger.info(f"   G_hat  mean={G_hat.mean()*100:.3f}%  max={G_hat.max()*100:.3f}%")
    logger.info(f"   G_hat_bare (clipped)  mean={G_hat_bare.mean()*100:.3f}%  max={G_hat_bare.max()*100:.3f}%")

    # --- Government expenditure (grows at g_step per quarter) ---------------
    state.G = state.G * (1.0 + state.g_step)
    G_vec   = state.G
    logger.info(f"   G_gov = {G_vec.sum()/1e9:.2f} B EUR  (g_step={state.g_step*100:.2f}%)")

    # 3. Investment (Eq. 19) - use smoothed G_hat per Eq. (18), not raw G_hat_bare
    dK = compute_investment(G_hat, state.A_bar, state.B, state.C, G_vec,
                            state.g_step, state.c_step, k=state.neumann_k)
    logger.info(f"   dK = {dK.sum()/1e9:.2f} B EUR")

    # (I-A_bar)^{-1} * G
    Nc    = np.asarray(CORE.neumann_apply(_to_dense(state.A_bar), np.asarray(G_vec, dtype=np.float64), state.neumann_k))

    # K_eff = K - B @ Nc   (govt pre-empts capital)
    K_eff = state.K - state.B @ Nc

    # L_eff = L - l^T·(I−A)⁻¹·G       (govt absorbs upstream labour)
    L_eff = float(state.L_total - state.l_vec @ Nc)

    # -- Diagnostics ---------------------------------------------------------
    M_G        = state.B @ Nc
    n_negative = int((K_eff <= 0).sum())
    frac_K_rem = float(K_eff.sum()) / max(float(state.K.sum()), 1e-30)
    logger.info(f"   [DIAG] G_raw total      = {G_vec.sum()/1e9:.2f} B EUR")
    logger.info(f"   [DIAG] M·G (cap req)    = {M_G.sum()/1e9:.2f} B EUR  "
                f"(K used by govt: {(1-frac_K_rem)*100:.1f}%)")
    logger.info(f"   [DIAG] K_eff total      = {K_eff.sum()/1e9:.2f} B EUR  "
                f"({frac_K_rem*100:.1f}% of K remaining)")
    logger.info(f"   [DIAG] Sectors K_eff<=0 = {n_negative}")
    logger.info(f"   [DIAG] L_eff            = {L_eff/1e9:.2f} B hrs  "
                f"({L_eff/max(state.L_total, 1e-30)*100:.1f}% of L remaining)")

    # 4. Planner optimisation
    opt    = solve_planner(state.alpha, state.A_bar, state.B,
                           state.l_tilde, dK, K_eff, L_eff, G_vec, state.gamma, state.C,
                           k=state.neumann_k,
                           tol_p=state.primal_tol, tol_d=state.dual_tol,
                           eta_K=state.eta_K, eta_L=state.eta_L,
                           max_iter=state.max_iter)
    
    logger.debug(f"solve_planner keys: {list(opt.keys())}")
    for k in ["lam_K", "lam_L", "C_star"]:
        if opt.get(k) is None:
            logger.error(f"   [CRITICAL] opt['{k}'] is None!")
    
    C_star = opt["C_star"]
    X_star = opt["X_star"]
    pi     = opt["pi_star"]
    logger.info(f"   [DIAG] C_star total     = {C_star.sum()/1e9:.2f} B EUR  "
                f"({'converged' if opt['success'] else 'NOT CONVERGED'}, "
                f"{opt.get('iterations',-1)} iters)")
    logger.info(f"   [DIAG] Sectors C_star=0 = {int((C_star==0).sum())}")
    if not opt["success"]:
        logger.warning(f"[Q{t+1}] solve_planner did not converge -- "
                       f"results for this quarter use best iterate. "
                       f"Resetting dual warm-start to avoid poisoning Q{t+2}.")
        opt["lambda_K"] = None
        opt["lambda_L"] = None

    # Strict NaN check per user request for debugging
    if np.isnan(C_star).any() or np.isnan(X_star).any() or np.isnan(pi).any():
        raise SimulationError(f"Q{t+1}: Solver produced NaNs")

    # 5/6. Capital update
    # Re-calculate X* to exactly match physical output requirements: X* = (I - A_bar)^{-1}(C* + G + dK)
    X_star = np.asarray(CORE.neumann_apply(_to_dense(state.A_bar), np.asarray(C_star + G_vec + dK, dtype=np.float64), state.neumann_k))

    I_gross = X_star - (state.A @ X_star + C_star + G_vec)
    # Check for NaN in investment/output recalculation
    if np.isnan(X_star).any() or np.isnan(I_gross).any():
        raise SimulationError(f"Q{t+1}: Recalculation produced NaNs")

    I_gross = np.maximum(I_gross, 0.0)
    I_vec = I_gross
    
    # -- Firm Capital Update (Sector-Proportional Investment Rule) ----------
    # Rule: I_firm,i = I_total,i * K_firm,i / K_total,i
    #   where i is the sector/capital-type index,
    #         I_total,i  = I_vec[i]  (aggregate investment of type i),
    #         K_firm,i   = K_firms[f, i]  (firm f's stock of capital type i),
    #         K_total,i  = sum over all firms of K_firms[:, i].
    #
    # Evolution: K_firm,t+1 = (1 - delta) * K_firm,t + I_firm,t

    # K_total,i: total capital of each type i across all firms  (N,)
    K_total_i = state.K_firms.sum(axis=0)

    # Firm-level investment share: K_firms[f, i] / K_total_i[i]
    # Use safe division; fall back to uniform share when K_total_i[i] is zero.
    safe_denom = np.where(K_total_i > 1e-12, K_total_i, 1.0)
    firm_capital_share = state.K_firms / safe_denom[None, :]          # (5, N)
    # Where K_total_i was zero, apply uniform share 1/5
    uniform_share = 1.0 / 5
    firm_capital_share = np.where(
        K_total_i[None, :] > 1e-12,
        firm_capital_share,
        uniform_share,
    )

    # I_firms[f, i] = I_vec[i] * firm_capital_share[f, i]                (5, N)
    I_firms = firm_capital_share * I_vec[None, :]

    # Capital law of motion for each firm
    K_firms_new = (1 - state.delta) * state.K_firms + I_firms

    # Aggregate capital is DERIVED from firm layer (accounting identity: K ≡ Σ K_firms)
    K_new = K_firms_new.sum(axis=0)

    if np.isnan(K_new).any():
        raise SimulationError(f"Q{t+1}: K_new contains NaNs")

    # 7. Income (Chained Laspeyres Index)
    # We set Y such that the previous quarter's basket (state.C) remains 
    # affordable at current shadow prices (pi): sum(P_new * C_prev) = sum(P_prev * C_prev)
    if t == 0:
        # Initialize the fixed-price base for Real GDP metrics
        state.pi_0_fixed    = pi.copy()
        state.dual_weight_0 = pi - state.A.T @ pi
        state.C_0_init      = C_star.copy()
        
        # Robust fallback for failures
        lK = opt.get("lam_K")
        state.lambda_K_0 = lK.copy() if lK is not None else np.zeros_like(C_star)
        
        lL = opt.get("lam_L")
        state.lambda_L_0 = float(lL) if lL is not None else 0.0
        
        state.real_scale_factor = state.Y_0_init / max(float(pi @ C_star), 1e-30)
        state.VAL_0_Laspeyres = float((pi * state.C_0_init).sum())

    # 7. Nominal Income indexing (LES form) with Firm Layer
    # v_eff = (lambda_L,0 * l + M.T * Lambda_K,0) / [pi_t * (C* + dK_t + G_t)]
    # where M.T = (I - A).T^-1 @ B.T
    
    # Numerator part 2: (I - A).T^-1 @ (B.T @ Lambda_K,0)
    BT_lamK = (state.B.T @ state.lambda_K_0).astype(np.float64)
    # Ensure dense array for Julia bridge
    A_dense_T = state.A_bar.T.toarray().astype(np.float64)
    M_T_lamK = np.asarray(CORE.neumann_apply(A_dense_T, BT_lamK, int(state.neumann_k)))
    
    W1 = state.lambda_L_0 * state.l_vec + M_T_lamK
    
    # Denominator: pi_t @ (C* + G + dK) (Total Shadow Value)
    # The user specifies a sectoral objective weighted by total demand value.
    # We use the scalar dot product for the denominator as is standard for indexing.
    sector_demand = C_star + G_vec + dK
    shadow_net_product = float(pi @ sector_demand) 
    
    # v_eff: Effective unit value (N,)
    v_eff = W1 / max(shadow_net_product, 1e-30)
    
    resource_value = W1 @ X_star
    
    # We use v_eff for the firm allocation LP objective
    v_MIP = v_eff 

    # -- Firm Production Allocation (Linear Programming) ---------------------
    # Convex polytope optimization without Lagrangian duality
    #
    # For each sector j, solve:
    #   Maximize: sum_f v_MIP[j] * X_f[j,f]  (firm profits)
    #   Subject to:
    #     sum_f X_f[j,f] = X_star[j]          (meet sector target)
    #     0 <= X_f[j,f] <= cap_caps[f]       (capacity bounds)
    #
    # This replaces dual ascent with direct primal LP optimization.

    X_f = np.zeros((state.n, 5))
    B_dense = state.B.toarray()
    total_cap_all_sectors = np.zeros(state.n)

    for j in range(state.n):
        # Per-firm capital caps: max X_f[j,f] s.t. B[:,j] * X_f <= K_firms[f,:]
        col_B  = B_dense[:, j]
        nz_idx = col_B > 1e-12
        cap_caps = np.zeros(5)
        for f in range(5):
            if nz_idx.any():
                cap_caps[f] = np.min(state.K_firms[f, nz_idx] / col_B[nz_idx])
            else:
                cap_caps[f] = 1e30  # unconstrained sector
        
        total_cap = cap_caps.sum()
        total_cap_all_sectors[j] = total_cap
        target = X_star[j]
        if target <= 1e-12:
            X_f[j, :] = 0.0
            continue

        total_cap = cap_caps.sum()

        # Capital-constrained case: total capacity cannot meet target.
        # Fallback: all firms produce at full capacity (no scaling up).
        if total_cap <= target + 1e-10:
            X_f[j, :] = cap_caps
            continue

        # Linear Programming formulation
        from scipy.optimize import linprog

        # Objective: Maximize sum(v_MIP[j] * X_f)
        c = -v_MIP[j] * np.ones(5)  # negative for maximization

        # Equality constraint: sum(X_f) = target
        A_eq = np.ones((1, 5))
        b_eq = np.array([target])

        # Bounds: 0 <= X_f <= cap_caps
        bounds = [(0, cap_caps[f]) for f in range(5)]

        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if res.success:
            X_f[j, :] = res.x
        else:
            # Fallback: proportional allocation
            X_f[j, :] = cap_caps * (target / max(total_cap, 1e-30))

    # Check that sum_f X_f[j,f] = X_star[j] for each sector j
    X_actual = X_f.sum(axis=1)
    relative_deviations = (X_actual - X_star) / np.maximum(X_star, 1e-12)
    max_rel_dev = np.max(np.abs(relative_deviations))

    # Check for primal violations (target > capacity)
    X_star_cap_violation = np.maximum(X_star - total_cap_all_sectors, 0.0)
    denom_cap = np.maximum(total_cap_all_sectors, 1e-12)
    max_violation = np.max(X_star_cap_violation / denom_cap)
    
    if max_violation > state.primal_tol:
        logger.warning(f"   [WARNING] Planner target exceeds sectoral capacity by {max_violation*100:.2f}%")

    if max_rel_dev > 1e-5:
        logger.warning(f"   [WARNING] Firm allocation sum does not match X_star (dev: {max_rel_dev:.2e})")

    # Compute relative Mean Absolute Deviation between X_actual and X_star
    mad_relative = np.mean(np.abs(relative_deviations))
            
    # Firm Income: Y_f = v_eff * X_f  (all dynamics at firm level)
    Y_f_mat = v_eff[:, None] * X_f  # (N, 5)
    Y_f = Y_f_mat.sum(axis=0)       # (5,) per-firm total income

    # Income scale: calibrated once at Q1 to anchor nominal level
    if t == 0:
        state.income_scale = state.Y_0_init / max(float(Y_f.sum()), 1e-30)

    # Y = sum of firm incomes (scaled for nominal calibration)
    Y = float(Y_f.sum()) * state.income_scale

    # Y_planner: accounting variable — planner's target-based income (for comparison)
    Y_planner = float(v_eff @ X_star) * state.income_scale
    
    # 8. Prices (LES form): P = (Y * pi) / (1 + pi . gamma)
    P_new = (Y * pi) / (1.0 + np.dot(pi, state.gamma))

    # 8b. Inflation (Laspeyres Price Index)
    # denominator = sum(P_t-1 * C_t-1), numerator = sum(P_t * C_t-1)
    denom_inf = np.dot(state.P, state.C)
    num_inf   = np.dot(P_new,   state.C)
    inflation_link = (num_inf / max(denom_inf, 1e-30))
    inflation_rate = inflation_link - 1.0
    
    if t > 0:
        state.CPI_chained *= inflation_link
    else:
        state.CPI_chained = 1.0 # Anchor Q1 at 1.0


    opt_iter = opt.get("iterations", -1)
    logger.info(f"   C* = {(P_new * C_star).sum()/1e9:.1f} B EUR  "
                f"X_actual = {(P_new * X_actual).sum()/1e9:.1f} B EUR  "
                f"(X* target: {(P_new * X_star).sum()/1e9:.1f} B EUR)  "
                f"{'(OK)' if opt['success'] else '(WARNING)'}  ({opt_iter} iters)")
    logger.info(f"   Y_firm = {Y/1e9:.2f} B EUR  Y_planner = {Y_planner/1e9:.2f} B EUR  "
                f"(gap = {abs(Y - Y_planner)/max(Y, 1e-30)*100:.3f}%)")

    # 9. Fast tatonnement loop with intra-quarter preference evolution
    fast_res = fast_loop(
        P_new, C_star, state.alpha_true, state.alpha_slow, state.rng,
        state.pref_drift_rho, state.pref_drift_sigma, state.pref_noise_sigma,
        Y, state.gamma, K_eff,
        theta_drift=state.theta_drift
    )
    state.alpha_true = fast_res["alpha_true_final"]
    
    alpha_gap = float(np.linalg.norm(state.alpha_true - state.alpha))
    alpha_gap_linf = float(np.abs(state.alpha_true - state.alpha).max())
    logger.info(f"   alpha_true gap: norm={alpha_gap:.4f}  inf={alpha_gap_linf:.4f}")

    # 10. Update simulation state
    state.P = fast_res["P_final"]
    state.C = fast_res["C_monthly"].mean(axis=0) * 3
    state.C_monthly = fast_res["C_monthly"]
    state.P_monthly = fast_res["P_monthly"]
    state.price_drift = fast_res["price_drift"]

    # 10. Preference update
    # In LES, surplus expenditure P_i(C_i - gamma_i/3) = alpha_i * (Monthly Residual Income).
    # We use gamma/3 for monthly minimum thresholds.
    gamma_m      = state.gamma / 3.0
    surplus_exp  = fast_res["P_monthly"] * (fast_res["C_monthly"] - gamma_m)
    net_income   = surplus_exp.sum(axis=1, keepdims=True)
    net_income   = np.maximum(net_income, 1e-30)
    alpha_est    = surplus_exp / net_income
    
    alpha_new    = alpha_est.mean(axis=0)
    alpha_new    = np.maximum(alpha_new, 0)
    alpha_new    = (alpha_new / alpha_new.sum()
                    if alpha_new.sum() > 0 else state.alpha.copy())

    # User Requested: 50/50 smoothing 
    # alpha_t+1 = 1/2*(alpha_t + Expenditure share update)
    alpha_new = 0.5 * state.alpha + 0.5 * alpha_new
    alpha_new = alpha_new / alpha_new.sum()

    # -- Constant price (real) metrics (using X_actual, not X_star) --------
    # Use real_scale_factor to ensure Nominal == Real at t=0
    # Real GDP = value added = (I - A)^T pi * X_actual (actual production)
    Real_GDP          = state.real_scale_factor * float((state.dual_weight_0 * X_actual).sum())
    # Y_ratio = Y / Real_GDP (indicator of nominal vs real decoupling)
    Y_ratio = Y / max(Real_GDP, 1e-30)
    # Gross Output (actual) at Q1 prices
    Gross_Output_Real = state.real_scale_factor * float((X_actual * state.pi_0_fixed).sum())
    C_real = state.real_scale_factor * float((C_star  * state.pi_0_fixed).sum())
    C_realized = state.real_scale_factor * float((fast_res["C_monthly"].sum(axis=0) * state.pi_0_fixed).sum())
    I_gross_real = state.real_scale_factor * float((I_vec   * state.pi_0_fixed).sum())
    I_net_real   = state.real_scale_factor * float((dK      * state.pi_0_fixed).sum())
    G_real = state.real_scale_factor * float((G_vec   * state.pi_0_fixed).sum())
    Real_AD = C_real + I_gross_real + G_real
    Real_AD_realized = C_realized + I_gross_real + G_real

    # -- Capital utilisation in absolute monetary terms at Q1 prices ---------
    # Q1 prices = state.P_0 (set on first quarter; zeros before that).
    # cap_used[i]  = (B @ X_actual)_i  (capital actually committed to production)
    cap_used  = state.B @ X_actual                           # physical units
    cap_slack = state.K - cap_used                           # physical slack (units)

    # P_0 available after Q1 initialisation; use zeros before that (Q1 itself)
    P_0_prices = state.P_0 if len(state.P_0) == state.n else np.zeros(state.n)

    K_val_Q1     = float((state.K  * P_0_prices).sum())     # total K in EUR (Q1 prices)
    slack_val_Q1 = float((cap_slack * P_0_prices).sum())     # idle K in EUR (Q1 prices)
    used_val_Q1  = K_val_Q1 - slack_val_Q1                   # committed K in EUR

    logger.info(f"   Real GDP (ann.) = {Real_GDP*4/1e9:.1f} B EUR  "
                f"Real AD (ann.) = {Real_AD*4/1e9:.1f} B EUR")
    logger.info(f"   (C = {C_real*4/1e9:.1f}B, I = {I_gross_real*4/1e9:.1f}B, "
                f"G = {G_real*4/1e9:.1f}B  — annualised)")
    logger.info(f"   Capital (Q1 prices): K_total = {K_val_Q1/1e9:.1f} B EUR  "
                f"slack = {slack_val_Q1/1e9:.1f} B EUR  "
                f"({slack_val_Q1/max(K_val_Q1,1e-30)*100:.1f}% idle)")

    # -- Labour slack --------------------------------------------------------
    labor_used  = float((state.l_vec * X_actual).sum())
    labor_slack = (state.L_total - labor_used) / max(state.L_total, 1e-12)
    logger.info(f"   Labor utilization: {(1-labor_slack)*100:.2f}%")

    # -- Invariants ----------------------------------------------------------
    assert abs(alpha_new.sum() - 1.0) < 1e-8, "alpha no longer sums to 1"
    assert (K_new >= 0).all(), "Negative capital stock"
    if (C_star < 0).any():
        logger.warning("C_star has strictly negative elements (constraint violation)")

    # -- Excess demand -------------------------------------------------------
    ed_nom_val  = 0.0
    exp_nom_val = 0.0
    for tau in range(3):
        p_tau     = fast_res["P_monthly"][tau]
        c_rev_tau = fast_res["C_monthly"][tau]
        # Change to Demand - Supply (Z = C - Cm)
        ed_nom_val  += float(((c_rev_tau - C_star / 3.0) * p_tau).sum())
        exp_nom_val += float((c_rev_tau * p_tau).sum())

    logger.info(f"   [DIAG] Price drift from planner baseline: mean={fast_res['price_drift']*100:.2f}%  signed={fast_res['signed_drift']*100:.2f}%")
    # -- History record ------------------------------------------------------
    # Investment as % of GDP (annualised ratio; ×4 cancels from numerator & denominator)
    I_pct_GDP = (I_gross_real / max(Real_GDP, 1e-30)) * 100.0

    if slim:
        state.history.append(dict(
            t              = t + 1,
            GDP            = Real_GDP,
            Real_AD        = Real_AD,
            Y              = Y,
            C_total        = float((state.P * C_star).sum()),
            X_actual_total = float((state.P * X_actual).sum()),  # ACTUAL production
            X_star_total   = float((state.P * X_star).sum()),    # Planner target (diagnostic)
            I_total        = float((state.P * I_vec).sum()),
            G_total        = float((state.P * state.G).sum()),
            K_total        = float((state.P * state.K).sum()),
            G_mean         = float(G_hat.mean()),
            G_max          = float(G_hat.max()),
            alpha_gap      = alpha_gap,
            alpha_gap_linf = alpha_gap_linf,
            price_drift    = float(fast_res["price_drift"]),
            monthly_resid_Y = fast_res["monthly_resid_Y"],
            resource_value = resource_value,
            shadow_net_product = shadow_net_product,
            Y_ratio        = Y_ratio,
            lambda_K_mean  = float(opt["lam_K"].mean()) if opt["lam_K"] is not None else 0.0,
            lambda_K_max   = float(opt["lam_K"].max()) if opt["lam_K"] is not None else 0.0,
            lambda_L       = float(opt["lam_L"]) if opt["lam_L"] is not None else 0.0,
            C_rev          = fast_res["C_monthly"].mean(axis=0),
            ED_nom         = ed_nom_val,
            Exp_nom        = exp_nom_val,
            Inflation      = inflation_rate if t > 0 else 0.0,
            Gross_Output_Real = Gross_Output_Real,
            C_real         = C_real,
            C_realized     = C_realized,
            I_gross_real   = I_gross_real,
            I_net_real     = I_net_real,
            G_real         = G_real,
            Real_AD_realized = Real_AD_realized,
            # Capital slack in absolute monetary terms (EUR, Q1 prices)
            K_val_Q1       = K_val_Q1,
            slack_val_Q1   = slack_val_Q1,
            used_val_Q1    = used_val_Q1,
            labor_slack    = labor_slack,
            I_pct_GDP      = I_pct_GDP,
            iterations     = opt.get("iterations", 0),
            sector_short   = state.sector_short.copy(),
            v_MIP          = v_MIP.copy(),
            X_f            = X_f.copy(),
            MAD_relative   = mad_relative,
            Y_f            = Y_f.copy(),
            Y_planner      = Y_planner,
        ))
    else:
        state.history.append(dict(
            t              = t + 1,
            GDP            = Real_GDP,
            Real_AD        = Real_AD,
            Y              = Y,
            C_star         = C_star.copy(),
            X_actual       = X_actual.copy(),   # ACTUAL firm production
            X_star         = X_star.copy(),     # Planner target (diagnostic)
            I_vec          = I_vec.copy(),
            G_vec          = state.G.copy(),
            K              = state.K.copy(),
            pi_star        = pi.copy(),
            alpha          = state.alpha.copy(),
            alpha_true     = state.alpha_true.copy(),
            alpha_gap      = alpha_gap,
            alpha_gap_linf = alpha_gap_linf,
            price_drift    = float(fast_res["price_drift"]),
            shadow_net_product = shadow_net_product,
            Y_ratio        = Y_ratio,
            lambda_K_mean  = float(opt["lam_K"].mean()) if opt["lam_K"] is not None else 0.0,
            lambda_K_max   = float(opt["lam_K"].max()) if opt["lam_K"] is not None else 0.0,
            lambda_L       = float(opt["lam_L"]) if opt["lam_L"] is not None else 0.0,
            C_rev          = fast_res["C_monthly"].mean(axis=0),
            ED_nom         = ed_nom_val,
            Exp_nom        = exp_nom_val,
            Inflation      = inflation_rate if t > 0 else 0.0,
            Gross_Output_Real = Gross_Output_Real,
            C_real         = C_real,
            C_realized     = C_realized,
            I_gross_real   = I_gross_real,
            I_net_real     = I_net_real,
            G_real         = G_real,
            Real_AD_realized = Real_AD_realized,
            # Capital slack in absolute monetary terms (EUR, Q1 prices)
            K_val_Q1       = K_val_Q1,
            slack_val_Q1   = slack_val_Q1,
            used_val_Q1    = used_val_Q1,
            labor_slack    = labor_slack,
            G_hat_mean     = float(G_hat.mean()),
            CPI            = state.CPI_chained,
            I_pct_GDP      = I_pct_GDP,
            iterations     = opt.get("iterations", 0),
            sector_short   = state.sector_short.copy(),
            v_MIP          = v_MIP.copy(),
            X_f            = X_f.copy(),
            MAD_relative   = mad_relative,
            Y_f            = Y_f.copy(),
            Y_planner      = Y_planner,
            K_firms        = state.K_firms.copy(),
        ))

    # -- State update --------------------------------------------------------
    state.t         = t + 1
    state.K_firms   = K_firms_new
    state.K         = K_new  # = K_firms_new.sum(axis=0), accounting identity

    # Verify accounting identity: K ≡ Σ K_firms (within machine precision)
    assert np.allclose(state.K, state.K_firms.sum(axis=0), rtol=1e-5), \
        "K ≠ Σ K_firms — accounting identity violated"
    state.alpha     = alpha_new
    state.P         = P_new
    state.C         = C_star
    state.X         = X_actual    # UPDATE: use actual firm production, not planner target
    state.X_star    = X_star      # KEEP: planner target for diagnostics
    state.X_f       = X_f.copy()  # KEEP: firm-level outputs
    state.Y         = Y
    state.C_monthly = fast_res["C_monthly"]
    state.P_monthly = fast_res["P_monthly"]
    state.G_hat_bare_prev = fast_res["G_hat_bare"]  # Store for next quarter's investment calculation
    state.lambda_K  = opt["lam_K"]
    state.lambda_L  = opt["lam_L"]
    return state


def _print_investment_gdp_summary(history: list, start_year: int = 2022) -> None:
    """Print annual Investment/GDP ratios and a 5-year rolling average table.

    Quarters are grouped into annual buckets (4 quarters each).  The 5-year
    rolling average is computed over completed years; partial-year windows are
    labelled accordingly.
    """
    n_quarters = len(history)
    n_years    = n_quarters // 4
    leftover   = n_quarters % 4   # partial final year

    annual_ratios = []          # mean I/GDP (%) for each full year
    annual_years  = []          # calendar year labels

    for y in range(n_years):
        bucket = history[y * 4 : y * 4 + 4]
        avg    = np.mean([h["I_pct_GDP"] for h in bucket])
        annual_ratios.append(avg)
        annual_years.append(start_year + y)

    # Partial final year (if any)
    partial_label = None
    partial_avg   = None
    if leftover > 0:
        bucket      = history[n_years * 4 :]
        partial_avg = np.mean([h["I_pct_GDP"] for h in bucket])
        partial_label = f"{start_year + n_years} (partial, {leftover}Q)"

    logger.info("")
    logger.info("  Investment / GDP  —  Annual Averages & 5-Year Rolling Mean")
    logger.info("  " + "-" * 54)
    logger.info(f"  {'Year':<20}  {'I / GDP (%)':>12}  {'5-yr Avg (%)':>14}")
    logger.info("  " + "-" * 54)

    for i, (yr, ratio) in enumerate(zip(annual_years, annual_ratios)):
        # 5-year rolling average up to and including year i
        window     = annual_ratios[max(0, i - 4) : i + 1]
        rolling5   = np.mean(window)
        win_label  = f"({len(window)}-yr avg)" if len(window) < 5 else "(5-yr avg)"
        logger.info(f"  {str(yr):<20}  {ratio:>12.2f}  {rolling5:>10.2f}  {win_label}")

    if partial_label and partial_avg is not None:
        logger.info(f"  {partial_label:<20}  {partial_avg:>12.2f}")

    logger.info("  " + "-" * 54)

    # Final 5-year summary
    all_ratios = annual_ratios + ([partial_avg] if partial_avg is not None else [])
    if len(all_ratios) >= 5:
        final_5yr = np.mean(all_ratios[-5:])
        logger.info(f"  5-Year Average (last 5 obs):        {final_5yr:>8.2f} %")
    elif len(all_ratios) > 0:
        overall = np.mean(all_ratios)
        logger.info(f"  Overall Average ({len(all_ratios)} years):       {overall:>8.2f} %")
    logger.info("")


def run_simulation(state: ModelState, n_quarters: int = 20,
                   checkpoint_dir: Path = None, checkpoint_every: int = 5) -> ModelState:
    slim = getattr(state, "slim_history", False)
    logger.info(f"\n{'#'*60}")
    logger.info(f"  Cybernetic In-Kind Planning Simulation")
    logger.info(f"  n = {state.n:,}   T = {n_quarters}   delta = {state.delta}"
                f"   {'slim history' if slim else 'full history'}")
    logger.info(f"{'#'*60}")

    step_times = []
    for i in range(n_quarters):
        t0 = time.time()
        try:
            run_quarter(state)
        except SimulationError as e:
            logger.error(f"   [Simulation] TERMINATED at Q{state.t+1}: {e}")
            break
        t1 = time.time()

        step_duration = t1 - t0
        step_times.append(step_duration)
        logger.info(f"   [Step completed in {step_duration:.2f}s]")

        if checkpoint_dir and (i + 1) % checkpoint_every == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            chk_path = checkpoint_dir / f"checkpoint_q{state.t}.pkl"
            with open(chk_path, "wb") as f:
                pickle.dump(state, f)
            logger.info(f"  Saved checkpoint to {chk_path}")

    g0 = state.history[0]["GDP"]
    gT = state.history[-1]["GDP"]
    # Report annualised GDP (×4)
    logger.info(f"\n{'#'*60}")
    logger.info(f"  Done.  Annualised GDP: {g0*4/1e9:.1f}B → {gT*4/1e9:.1f}B EUR  "
                f"({(gT/g0-1)*100:+.1f}% cumulative)")
    if len(step_times) > 1:
        avg_time = sum(step_times[1:]) / (len(step_times) - 1)
        logger.info(f"  Avg time per quarter (excluding Q1 compilation): {avg_time:.2f}s")
    elif len(step_times) == 1:
        logger.info(f"  Time for Q1 (includes compilation): {step_times[0]:.2f}s")
    logger.info(f"{'#'*60}")

    # Investment / GDP subsystem summary
    _print_investment_gdp_summary(state.history)

    return state