import numpy as np
import logging
import pickle
import time
from pathlib import Path
from calibration import ModelState
from julia_bridge import (
    neumann_apply, evolve_true_alpha, revealed_demand, infer_growth,
    compute_investment, solve_planner, compute_income,
    fast_loop,
)

logger = logging.getLogger(__name__)


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
    gamma = getattr(state, "habit_persistence", 0.8) # How slowly the habit adapts
    
    if t == 0:
        state.alpha_habit = state.alpha_bar.copy()
    else:
        # What consumers actually received and valued last quarter
        expenditures  = state.P_monthly * state.C_monthly
        period_totals = expenditures.sum(axis=1, keepdims=True)
        period_totals = np.maximum(period_totals, 1e-30)
        period_shares = expenditures / period_totals
        realized_shares = period_shares.mean(axis=0)
        
        # Continuously update the habit target (EMA tracking)
        state.alpha_habit = gamma * getattr(state, "alpha_habit", state.alpha_bar) + (1 - gamma) * realized_shares
        state.alpha_habit = np.maximum(state.alpha_habit, 1e-30)
        state.alpha_habit /= state.alpha_habit.sum()

    # Preferences undergo OU random walk around the moving target
    state.alpha_true = evolve_true_alpha(state.alpha_true, state.rng, state.alpha_habit,
                                         drift=state.drift, kappa=state.kappa_ou)
    if t == 0:
        C_hat = state.C.copy()
        G_hat = state.G_hat_init.copy()
    else:
        C_hat = revealed_demand(state.C_monthly, state.P_monthly, state.P, state.C)
        G_hat_raw = infer_growth(C_hat, state.C)
        # 2-period rolling average (temporal smoothing)
        G_hat = (G_hat_raw + getattr(state, 'G_hat_prev', G_hat_raw)) / 2.0
        state.G_hat_prev = G_hat.copy()
        
    logger.info(f"   G_hat  mean={G_hat.mean()*100:.3f}%  max={G_hat.max()*100:.3f}%")

    # --- Government expenditure (grows at g_step per quarter) ---------------
    state.G = state.G * (1.0 + state.g_step)
    G_vec   = state.G
    logger.info(f"   G_gov = {G_vec.sum()/1e9:.2f} B EUR  (g_step={state.g_step*100:.2f}%)")

    # 3. Investment (Eq. 14)
    dK = compute_investment(G_hat, state.A_bar, state.B, state.C, G_vec,
                            state.g_step, state.c_step, k=state.neumann_k)
    logger.info(f"   dK = {dK.sum()/1e9:.2f} B EUR")

    # (I-A_bar)^{-1} * G
    Nc    = neumann_apply(state.A_bar, G_vec, k=state.neumann_k)

    # K_eff = K - diag(κ)·(I−A)⁻¹·G   (govt pre-empts capital)
    K_eff = state.K - state.B * Nc

    # L_eff = L - l^T·(I−A)⁻¹·G       (govt absorbs upstream labour)
    L_eff = float(state.L_total - state.l_vec @ Nc)

    # -- Diagnostics ---------------------------------------------------------
    M_G        = state.B * Nc
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
                           state.l_vec, state.l_tilde,
                           dK, K_eff, L_eff, G_vec,
                           C_prev=state.C, k=state.neumann_k,
                           tol_p=state.primal_tol, tol_d=state.dual_tol,
                           lambda_K_prev=getattr(state, "lambda_K", None),
                           lambda_L_prev=getattr(state, "lambda_L", None),
                           eta_K=getattr(state, "eta_K", 0.15),
                           eta_L=getattr(state, "eta_L", 0.15),
                           max_iter=getattr(state, "max_iter", 2000))
    C_star = opt["C_star"]
    X_star = opt["X_star"]
    pi     = opt["pi_star"]
    logger.info(f"   [DIAG] C_star total     = {C_star.sum()/1e9:.2f} B EUR  "
                f"({'converged' if opt['success'] else 'NOT CONVERGED'}, "
                f"{opt.get('iterations',-1)} iters)")
    logger.info(f"   [DIAG] Sectors C_star=0 = {int((C_star==0).sum())}")
    if not opt["success"]:
        logger.warning(f"[Q{t+1}] solve_planner did not converge -- "
                       f"results for this quarter use best iterate")

    # Strict NaN check per user request for debugging
    assert not np.isnan(C_star).any(), f"Q{t+1}: C_star contains NaN"
    assert not np.isnan(X_star).any(), f"Q{t+1}: X_star contains NaN"
    assert not np.isnan(pi).any(), f"Q{t+1}: pi_star contains NaN"

    # 5/6. Capital update
    # Re-calculate X* to exactly match physical output requirements: X* = (I - A_bar)^{-1}(C* + G + dK)
    X_star = neumann_apply(state.A_bar, C_star + G_vec + dK, k=state.neumann_k)

    I_gross = X_star - (state.A @ X_star + C_star + G_vec)
    # Check for NaN in investment/output recalculation
    assert not np.isnan(X_star).any(), f"Q{t+1}: X_star (re-calc) contains NaN"
    assert not np.isnan(I_gross).any(), f"Q{t+1}: I_gross contains NaN"

    I_gross = np.maximum(I_gross, 0.0)
    K_new = (1 - state.delta) * state.K + I_gross
    I_vec = I_gross

    assert not np.isnan(K_new).any(), f"Q{t+1}: K_new contains NaN"

    # 7. Income (Laspeyres Real Income Index)
    if t == 0:
        state.C_0_init   = C_star.copy()
        state.pi_0_fixed = pi.copy()
        state.P_0        = state.Y_0_init * pi.copy()

        state.dual_weight_0 = state.pi_0_fixed - state.A.T @ state.pi_0_fixed

        state.VAL_0_Laspeyres = float((pi * state.C_0_init).sum())
        logger.info(f"   [Simulation] Anchored macro weights to Q1: "
                    f"VAL_0={state.VAL_0_Laspeyres:.4f}")

    numerator   = state.VAL_0_Laspeyres
    denominator = max(float((pi * state.C_0_init).sum()), 1e-30)
    Y = state.Y_0_init * (numerator / denominator)

    # 8. Prices
    P_new = Y * pi

    # 8b. Inflation (Laspeyres Price Index formula provided by user)
    # denominator = sum(P_t-1 * C_t-1), numerator = sum(P_t * C_t-1)
    denom_inf = np.dot(state.P, state.C)
    num_inf   = np.dot(P_new,   state.C)
    inflation_rate = (num_inf / max(denom_inf, 1e-30) - 1.0) * 100.0


    opt_iter = opt.get("iterations", -1)
    logger.info(f"   C* = {(P_new * C_star).sum()/1e9:.1f} B EUR  "
                f"X* = {(P_new * X_star).sum()/1e9:.1f} B EUR  "
                f"{'(OK)' if opt['success'] else '(WARNING)'}  ({opt_iter} iters)")

    # 9. Fast tatonnement loop
    fast      = fast_loop(P_new, C_star, state.alpha_true, Y)
    alpha_gap = float(np.linalg.norm(state.alpha_true - state.alpha))
    alpha_gap_linf = float(np.abs(state.alpha_true - state.alpha).max())
    logger.info(f"   Y = {Y/1e9:.1f} B EUR  drift = {fast['price_drift']*100:.2f}%  "
                f"alpha_gap = {alpha_gap*100:.2f} pp")

    # 10. Preference update
    # Use actual demand signal (intended consumption) from fast loop for learning.
    # P_m * C_d = alpha_true * Y_m (unfiltered preferences).
    expenditures  = fast["P_monthly"] * fast["C_monthly"]
    period_totals = expenditures.sum(axis=1, keepdims=True)
    period_totals = np.maximum(period_totals, 1e-30)
    period_shares = expenditures / period_totals
    alpha_new     = period_shares.mean(axis=0)
    alpha_new     = np.maximum(alpha_new, 0)
    alpha_new     = (alpha_new / alpha_new.sum()
                     if alpha_new.sum() > 0 else state.alpha.copy())

    # -- Constant price (real) metrics ---------------------------------------
    Real_GDP          = state.Y_0_init * float((state.dual_weight_0 * X_star).sum())
    Gross_Output_Real = state.Y_0_init * float((X_star * state.pi_0_fixed).sum())
    C_real = state.Y_0_init * float((C_star  * state.pi_0_fixed).sum())
    I_gross_real = state.Y_0_init * float((I_vec   * state.pi_0_fixed).sum())
    I_net_real   = state.Y_0_init * float((dK      * state.pi_0_fixed).sum())
    G_real = state.Y_0_init * float((G_vec   * state.pi_0_fixed).sum())
    Real_AD = C_real + I_gross_real + G_real

    # -- Capital utilisation in absolute monetary terms at Q1 prices ---------
    # Q1 prices = state.P_0 (set on first quarter; zeros before that).
    # cap_used[i]  = kappa[i] * X_star[i]  (capital committed to production)
    # slack[i]     = K[i] - cap_used[i]    (idle capital, physical units)
    # Monetary value at Q1 prices: multiply by P_0 (cost-push, EUR/unit).
    cap_used  = state.B * X_star                             # physical units
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
    labor_used  = float((state.l_vec * X_star).sum())
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
        p_tau     = fast["P_monthly"][tau]
        c_rev_tau = fast["C_monthly"][tau]
        # Change to Demand - Supply (Z = C - Cm)
        ed_nom_val  += float(((c_rev_tau - C_star / 3.0) * p_tau).sum())
        exp_nom_val += float((c_rev_tau * p_tau).sum())

    # -- History record ------------------------------------------------------
    if slim:
        state.history.append(dict(
            t              = t + 1,
            GDP            = Real_GDP,
            Real_AD        = Real_AD,
            Y              = Y,
            C_total        = float((state.P * C_star).sum()),
            X_total        = float((state.P * X_star).sum()),
            I_total        = float((state.P * I_vec).sum()),
            G_total        = float((state.P * state.G).sum()),
            K_total        = float((state.P * state.K).sum()),
            G_mean         = float(G_hat.mean()),
            G_max          = float(G_hat.max()),
            alpha_gap      = alpha_gap,
            alpha_gap_linf = alpha_gap_linf,
            price_drift    = fast["price_drift"],
            C_rev          = fast["C_monthly"].mean(axis=0),
            ED_nom         = ed_nom_val,
            Exp_nom        = exp_nom_val,
            Inflation      = inflation_rate if t > 0 else 0.0,
            Gross_Output_Real = Gross_Output_Real,
            C_real         = C_real,
            I_gross_real   = I_gross_real,
            I_net_real     = I_net_real,
            G_real         = G_real,
            # Capital slack in absolute monetary terms (EUR, Q1 prices)
            K_val_Q1       = K_val_Q1,
            slack_val_Q1   = slack_val_Q1,
            used_val_Q1    = used_val_Q1,
            labor_slack    = labor_slack,
        ))
    else:
        state.history.append(dict(
            t              = t + 1,
            GDP            = Real_GDP,
            Real_AD        = Real_AD,
            Y              = Y,
            C_star         = C_star.copy(),
            X_star         = X_star.copy(),
            I_vec          = I_vec.copy(),
            G_vec          = state.G.copy(),
            K              = state.K.copy(),
            pi_star        = pi.copy(),
            alpha          = state.alpha.copy(),
            alpha_true     = state.alpha_true.copy(),
            alpha_gap      = alpha_gap,
            alpha_gap_linf = alpha_gap_linf,
            price_drift    = fast["price_drift"],
            C_rev          = fast["C_monthly"].mean(axis=0),
            ED_nom         = ed_nom_val,
            Exp_nom        = exp_nom_val,
            Inflation      = inflation_rate if t > 0 else 0.0,
            Gross_Output_Real = Gross_Output_Real,
            C_real         = C_real,
            I_gross_real   = I_gross_real,
            I_net_real     = I_net_real,
            G_real         = G_real,
            # Capital slack in absolute monetary terms (EUR, Q1 prices)
            K_val_Q1       = K_val_Q1,
            slack_val_Q1   = slack_val_Q1,
            used_val_Q1    = used_val_Q1,
            labor_slack    = labor_slack,
            G_hat_mean     = float(G_hat.mean()),
            CPI            = float((pi * state.C_0_init).sum()) / max(state.VAL_0_Laspeyres, 1e-30),
        ))

    # -- State update --------------------------------------------------------
    state.t         = t + 1
    state.K         = K_new
    state.alpha     = alpha_new
    state.P         = P_new
    state.C         = C_star
    state.X         = X_star
    state.Y         = Y
    state.C_monthly = fast["C_monthly"]
    state.P_monthly = fast["P_monthly"]
    state.lambda_K  = opt["lambda_K"]
    state.lambda_L  = opt["lambda_L"]
    return state


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
        run_quarter(state)
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
    return state
