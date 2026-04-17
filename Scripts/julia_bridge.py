import numpy as np
import logging
from pathlib import Path
import scipy.sparse as sp

logger = logging.getLogger(__name__)

from juliacall import Main as jl

# -- Load model ---------------------------------------------------------------
JULIA_CORE = Path(__file__).parent / "model_core.jl"

def _load_core():
    """
    Include model_core.jl and return the ModelCore module.

    Forces a clean re-evaluation every time this Python process starts:
      1. If Main.ModelCore already exists from a prior include() in the same
         session (e.g. an interactive REPL), un-define it so the next include()
         starts from a blank slate.
      2. include() the file -- Julia re-parses and re-compiles it.
      3. Sanity-check that solve_planner is actually defined at the top level of
         Main.ModelCore (not buried in a nested sub-module).
    """
    # Step 1: evict any stale definition from a prior include in this session
    jl.seval("""
        if isdefined(Main, :ModelCore)
            # Replace the binding with nothing so the old module is GC-eligible
            # and the next include() creates a fresh one.
            Core.eval(Main, :(ModelCore = nothing))
        end
    """)

    # Step 2: parse and compile the file
    jl.include(str(JULIA_CORE))

    core = jl.ModelCore

    # Step 3: verify that the key function landed in the right place
    expected = ["neumann_apply", "evolve_true_alpha", "revealed_demand",
                "infer_growth", "compute_investment", "solve_planner",
                "compute_income", "fast_loop"]
    missing = [f for f in expected if not jl.seval(f"isdefined(ModelCore, :{f})")]
    if missing:
        raise RuntimeError(
            f"model_core.jl loaded but these names are missing from ModelCore: "
            f"{missing}. "
            f"Check for nested 'module ModelCore' blocks or typos in the .jl file."
        )

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
    """Convert scipy sparse or any array-like to a contiguous float64 ndarray."""
    if sp.issparse(A):
        return np.asarray(A.toarray(), dtype=np.float64, order="C")
    return np.asarray(A, dtype=np.float64, order="C")


def _warmup() -> None:
    import time
    t0 = time.perf_counter()
    logger.info("Julia warmup: pre-compiling PythonCall specialisations...")
    n       = 4
    rng_w   = np.random.default_rng(0)
    A       = np.full((n, n), 0.05, dtype=np.float64)
    np.fill_diagonal(A, 0.0)
    alpha   = np.full(n, 1.0 / n, dtype=np.float64)
    B_vec   = np.ones(n,          dtype=np.float64)
    l_vec   = np.ones(n,          dtype=np.float64)
    l_tilde = np.ones(n,          dtype=np.float64)
    K       = np.full(n, 10.0,    dtype=np.float64)
    dK      = np.full(n,  0.1,    dtype=np.float64)
    C_prev  = np.full(n,  1.0,    dtype=np.float64)
    v       = np.ones(n,          dtype=np.float64)
    P_base  = np.ones(n,          dtype=np.float64)
    L_total = float(n * 10)
    C_monthly = np.ones((3, n),   dtype=np.float64)
    P_monthly = np.ones((3, n),   dtype=np.float64)

    seed_w   = int(rng_w.integers(0, 2**32))
    jl_rng_w = jl.Random.MersenneTwister(seed_w)

    def _try(name, fn):
        try:
            fn()
        except Exception as exc:
            logger.warning(f"Julia warmup: {name} failed (continuing): {exc}")

    _try("neumann_apply",      lambda: CORE.neumann_apply(A, v, 3))
    _try("evolve_true_alpha",  lambda: CORE.evolve_true_alpha(alpha, jl_rng_w, alpha, 0.012, 0.15))
    _try("revealed_demand",    lambda: CORE.revealed_demand(C_monthly, P_monthly, P_base, C_prev))

    _try("infer_growth",       lambda: CORE.infer_growth(C_prev, C_prev))
    _try("compute_investment", lambda: CORE.compute_investment(C_prev, A, B_vec, C_prev, C_prev, 0.005, 0.015, k=3))
    _try("solve_planner",      lambda: CORE.solve_planner(
                                    alpha, A, B_vec, l_vec, l_tilde,
                                    dK, K, L_total, B_vec,
                                    C_prev=C_prev, k=3,
                                    tol_p=1e-4, tol_d=1e-4,
                                    eta_K=0.15, eta_L=0.15, max_iter=50))
    _try("compute_income",     lambda: CORE.compute_income(1.0, 1.0, alpha, C_prev, A))
    _try("fast_loop",          lambda: CORE.fast_loop(P_base, C_prev, alpha, 1.0, 3))

    elapsed = time.perf_counter() - t0
    logger.info(f"Julia warmup complete ({elapsed:.1f}s).")

_warmup()


# -- Public API ---------------------------------------------------------------

def neumann_apply(A_bar, v, k=20):
    res = CORE.neumann_apply(_to_dense(A_bar), np.asarray(v, dtype=np.float64), int(k))
    return np.array(res)


def evolve_true_alpha(alpha_true, rng, alpha_bar, drift=0.012, kappa=0.15):
    at_jl  = np.asarray(alpha_true, dtype=np.float64)
    ab_jl  = np.asarray(alpha_bar,  dtype=np.float64)
    seed   = int(rng.integers(0, 2**32))
    jl_rng = jl.Random.MersenneTwister(seed)
    res    = CORE.evolve_true_alpha(at_jl, jl_rng, ab_jl, float(drift), float(kappa))
    return np.array(res)


def revealed_demand(C_monthly, P_monthly, P_base, C_plan):
    res = CORE.revealed_demand(
        np.asarray(C_monthly, dtype=np.float64),
        np.asarray(P_monthly, dtype=np.float64),
        np.asarray(P_base,    dtype=np.float64),
        np.asarray(C_plan,    dtype=np.float64),
    )
    return np.array(res)


def infer_growth(C_hat, C_prev):
    res = CORE.infer_growth(
        np.asarray(C_hat,  dtype=np.float64),
        np.asarray(C_prev, dtype=np.float64),
    )
    return np.array(res)


def compute_investment(G_hat, A_bar, kappa, C_prev, G_vec, g_step, c_step, k=20):
    res = CORE.compute_investment(
        np.asarray(G_hat,  dtype=np.float64),
        _to_dense(A_bar),
        np.asarray(kappa,  dtype=np.float64),
        np.asarray(C_prev, dtype=np.float64),
        np.asarray(G_vec,  dtype=np.float64),
        float(g_step),
        float(c_step),
        k=int(k),
    )
    return np.array(res)


def solve_planner(alpha, A, B, l, l_tilde, dK, K, L_total, G_vec,
                  C_prev=None, k=20, tol_p=1e-4, tol_d=1e-4,
                  lambda_K_prev=None, lambda_L_prev=None,
                  eta_K=0.15, eta_L=0.15, max_iter=2000):
    a_jl  = np.asarray(alpha,   dtype=np.float64)
    A_jl  = _to_dense(A)
    B_jl  = _to_dense(B)
    l_jl  = np.asarray(l,       dtype=np.float64)
    lt_jl = np.asarray(l_tilde, dtype=np.float64)
    dk_jl = np.asarray(dK,      dtype=np.float64)
    K_jl  = np.asarray(K,       dtype=np.float64)
    cp_jl = np.asarray(C_prev,  dtype=np.float64) if C_prev is not None else None

    lam_K_jl = np.asarray(lambda_K_prev, dtype=np.float64) if lambda_K_prev is not None else None
    lam_L_jl = float(lambda_L_prev) if lambda_L_prev is not None else None

    res = CORE.solve_planner(
        a_jl, A_jl, B_jl, l_jl, lt_jl, dk_jl, K_jl, float(L_total), np.asarray(G_vec, dtype=np.float64),
        C_prev=cp_jl, lambda_K_prev=lam_K_jl, lambda_L_prev=lam_L_jl,
        k=int(k), tol_p=float(tol_p), tol_d=float(tol_d),
        eta_K=float(eta_K), eta_L=float(eta_L), max_iter=int(max_iter),
    )

    return dict(
        C_star     = np.array(res.C_star),
        X_star     = np.array(res.X_star),
        pi_star    = np.array(res.pi_star),
        success    = bool(res.success),
        lambda_K   = np.array(res.lambda_K),
        lambda_L   = float(res.lambda_L),
        iterations = int(res.iterations),
    )


def compute_income(Y_0_init, VAL_0, pi, X_star, A):
    res = CORE.compute_income(
        float(Y_0_init),
        float(VAL_0),
        np.asarray(pi,     dtype=np.float64),
        np.asarray(X_star, dtype=np.float64),
        _to_dense(A),
    )
    return float(res)


def fast_loop(P_base, C_plan, alpha_true, Y, n_months=3):
    res = CORE.fast_loop(
        np.asarray(P_base,      dtype=np.float64),
        np.asarray(C_plan,      dtype=np.float64),
        np.asarray(alpha_true,  dtype=np.float64),
        float(Y),
        int(n_months),
    )
    return dict(
        C_monthly   = np.array(res.C_monthly),
        P_monthly   = np.array(res.P_monthly),
        P_final     = np.array(res.P_final),
        price_drift = float(res.price_drift),
    )
