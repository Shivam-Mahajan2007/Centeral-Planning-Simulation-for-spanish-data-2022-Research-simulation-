import numpy as np
import logging
from pathlib import Path
import scipy.sparse as sp

logger = logging.getLogger(__name__)

from juliacall import Main as jl

# -- Load model ---------------------------------------------------------------
JULIA_CORE = Path(__file__).parent / "model_core.jl"

def _load_core():
    jl.seval("""
        if isdefined(Main, :ModelCore)
            Core.eval(Main, :(ModelCore = nothing))
        end
    """)
    jl.include(str(JULIA_CORE))
    core = jl.ModelCore
    jl.seval("using Random")
    return core

try:
    CORE = _load_core()
    logger.info("Julia Core Engine loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Julia Core Engine: {e}")
    raise

# -- Helpers ------------------------------------------------------------------

def _to_dense(A):
    if sp.issparse(A):
        return np.asarray(A.toarray(), dtype=np.float64, order="C")
    return np.asarray(A, dtype=np.float64, order="C")

# -- Public API ---------------------------------------------------------------

def solve_planner(alpha, A, B, l_tilde, dK, K, L_total, G_vec, gamma, C_prev,
                  k=25, tol_p=1e-3, tol_d=1e-4,
                  eta_K=0.15, eta_L=0.15, max_iter=2000):
    a_jl   = np.asarray(alpha,   dtype=np.float64)
    A_jl   = _to_dense(A)
    B_jl   = _to_dense(B)
    lt_jl  = np.asarray(l_tilde, dtype=np.float64)
    dk_jl  = np.asarray(dK,      dtype=np.float64)
    K_jl   = np.asarray(K,       dtype=np.float64)
    Gv_jl  = np.asarray(G_vec,   dtype=np.float64)
    ga_jl  = np.asarray(gamma,   dtype=np.float64)
    c0_jl  = np.asarray(C_prev,  dtype=np.float64)

    res = CORE.solve_planner(
        a_jl, A_jl, B_jl, lt_jl, dk_jl, K_jl, float(L_total), Gv_jl, ga_jl, c0_jl,
        k=int(k), tol_p=float(tol_p), tol_d=float(tol_d),
        eta_K=float(eta_K), eta_L=float(eta_L), max_iter=int(max_iter)
    )

    # Convert Julia NamedTuple to Python dict using attribute access
    return {
        "C_star":     np.array(res.C_star),
        "X_star":     np.array(res.X_star),
        "pi_star":    np.array(res.pi_star),
        "success":    bool(res.success),
        "lam_K":      np.array(res.lambda_K),
        "lam_L":      float(res.lambda_L),
        "iterations": int(res.iterations),
    }

def fast_loop(P_base, C_plan, alpha_true, alpha_slow, rng,
              drift_rho, drift_sigma, noise_sigma, Y, gamma, K_v,
              theta_drift=0.1):
    pb_jl = np.asarray(P_base,    dtype=np.float64)
    cp_jl = np.asarray(C_plan,    dtype=np.float64)
    at_jl = np.asarray(alpha_true, dtype=np.float64)
    as_jl = np.asarray(alpha_slow, dtype=np.float64)
    ga_jl = np.asarray(gamma,      dtype=np.float64)
    kv_jl = np.asarray(K_v,        dtype=np.float64)
    
    seed   = int(rng.integers(0, 2**32))
    jl_rng = jl.Random.MersenneTwister(seed)

    res = CORE.fast_loop(
        pb_jl, cp_jl, at_jl, as_jl, jl_rng,
        float(drift_rho), float(drift_sigma), float(noise_sigma),
        float(Y), ga_jl, kv_jl, 3,
        theta_drift=float(theta_drift)
    )
    return {
        "C_monthly":        np.asarray(res.C_monthly),
        "P_monthly":        np.asarray(res.P_monthly),
        "P_final":          np.asarray(res.P_final),
        "price_drift":      float(res.price_drift),
        "signed_drift":     float(res.signed_drift),
        "monthly_drifts":   np.asarray(res.monthly_drifts),
        "monthly_resid_Y":  np.asarray(res.monthly_resid_Y),
        "alpha_true_final": np.asarray(res.alpha_true_final),
        "C_hat":            np.asarray(res.C_hat),
        "G_hat_bare":       np.asarray(res.G_hat_bare),
    }

def compute_income(v, X_star, pi, A):
    res = CORE.compute_income(
        float(v),
        np.asarray(X_star, dtype=np.float64),
        np.asarray(pi,     dtype=np.float64),
        _to_dense(A),
    )
    return float(res)

def evolve_structural_alpha(as_low, rng, ah_abit, drift_slow, kappa_slow):
    seed   = int(rng.integers(0, 2**32))
    jl_rng = jl.Random.MersenneTwister(seed)
    res = CORE.evolve_structural_alpha(
        np.asarray(as_low, dtype=np.float64),
        jl_rng,
        np.asarray(ah_abit, dtype=np.float64),
        float(drift_slow), float(kappa_slow)
    )
    return np.array(res)

def revealed_demand(*args, **kwargs):
    # Dummy to satisfy old imports if any exist
    return np.array([0.0])

def infer_growth(C_hat, C_prev):
    return (np.asarray(C_hat) / np.maximum(np.asarray(C_prev), 1e-12)) - 1.0

def compute_investment(G_hat, A_bar, B, C_prev, G_vec, g_step, c_step, k=25):
    res = CORE.compute_investment(
        np.asarray(G_hat,  dtype=np.float64),
        _to_dense(A_bar),
        _to_dense(B),
        np.asarray(C_prev, dtype=np.float64),
        np.asarray(G_vec,  dtype=np.float64),
        float(g_step),
        float(c_step),
        k=int(k),
    )
    return np.array(res)