import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs as sp_eigs
import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)

from julia_bridge import neumann_apply


# --- Sector capital intensity ------------------------------------------------

def _kappa(sector_names: List[str]) -> np.ndarray:
    """
    Return the diagonal of B (capital-coefficient matrix) as a 1-D array.

    kappa[i] = units of capital required per unit of output in sector i.
    Unrecognised sectors default to kappa = 0.8.
    """
    k = np.ones(len(sector_names)) * 0.8
    for i, name in enumerate(sector_names):
        nm = name.lower()
        if "real estate" in nm or "imputed rent" in nm:              k[i] = 5.0
        elif any(x in nm for x in ["electricity", "gas", "steam"]):  k[i] = 3.5
        elif any(x in nm for x in ["mining", "quarrying"]):          k[i] = 3.0
        elif "petroleum" in nm or "coke" in nm:                      k[i] = 2.5
        elif any(x in nm for x in ["water supply", "natural water"]): k[i] = 2.0
        elif any(x in nm for x in ["sewerage", "waste"]):            k[i] = 1.5
        elif any(x in nm for x in ["agricultur", "forestr", "fish"]): k[i] = 1.5
        elif any(x in nm for x in ["chemical", "pharmaceutical",
                                    "motor vehicle"]):               k[i] = 1.2
        elif any(x in nm for x in ["food", "textile", "wood", "paper",
                                    "rubber", "plastic", "fabricated metal",
                                    "electrical", "machinery",
                                    "basic metal"]):                 k[i] = 1.0
        elif "construct" in nm:                                      k[i] = 0.5
        elif any(x in nm for x in ["financial", "insurance"]):       k[i] = 0.5
        elif any(x in nm for x in ["telecommunication",
                                    "computer programming"]):        k[i] = 0.6
    return k


# --- Value added per unit of output ------------------------------------------

def _v_per_unit(sector_names: List[str]) -> np.ndarray:
    """
    Return the value-added content per unit of physical output for each sector.

    v_per_unit[i] is interpreted as: EUR of value added generated per one
    physical unit of sector i's output.  These are theoretically motivated
    structural parameters -- analogous to kappa -- estimated from typical
    sectoral value-added shares for a Western European economy.

    X_real is then recovered from observed value-added data V as:
        X_real = V / v_per_unit
    which gives physical output in units/quarter.

    Default (unrecognised sectors): 0.40
    """
    v = np.ones(len(sector_names)) * 0.60

    for i, name in enumerate(sector_names):
        nm = name.lower()

        # Primary sectors -- moderate VA share, significant material inputs
        if any(x in nm for x in ["agricultur", "forestr", "fish"]):
            v[i] = 0.50

        # Capital-intensive extraction & utilities -- low VA share per output unit
        elif any(x in nm for x in ["mining", "quarrying"]):
            v[i] = 0.45
        elif "petroleum" in nm or "coke" in nm:
            v[i] = 0.25    # very high material throughput, thin VA margin
        elif any(x in nm for x in ["electricity", "gas", "steam"]):
            v[i] = 0.40
        elif any(x in nm for x in ["water supply", "natural water"]):
            v[i] = 0.65
        elif any(x in nm for x in ["sewerage", "waste"]):
            v[i] = 0.70

        # Manufacturing -- broad band depending on processing intensity
        elif any(x in nm for x in ["food", "beverage", "tobacco"]):
            v[i] = 0.30    # high raw material cost
        elif any(x in nm for x in ["textile", "leather", "apparel"]):
            v[i] = 0.50
        elif any(x in nm for x in ["wood", "paper", "printing"]):
            v[i] = 0.50
        elif any(x in nm for x in ["chemical", "pharmaceutical"]):
            v[i] = 0.60
        elif any(x in nm for x in ["rubber", "plastic"]):
            v[i] = 0.50
        elif any(x in nm for x in ["basic metal", "fabricated metal"]):
            v[i] = 0.45
        elif any(x in nm for x in ["computer, elec", "electrical equip"]):
            v[i] = 0.50
        elif any(x in nm for x in ["machinery", "motor vehicle",
                                    "transport equip"]):
            v[i] = 0.45
        elif any(x in nm for x in ["furniture", "repair and install",
                                    "mineral"]):
            v[i] = 0.60

        # Construction -- substantial material inputs but meaningful labour VA
        elif "construct" in nm:
            v[i] = 0.65

        # Trade & Transport -- moderate-to-high VA (margins, logistics)
        elif any(x in nm for x in ["wholesale", "retail", "trade"]):
            v[i] = 0.70
        elif any(x in nm for x in ["land transport", "water transport",
                                    "air transport", "warehousing", "postal"]):
            v[i] = 0.65
        elif "accommodation" in nm:
            v[i] = 0.75

        # Finance & Real Estate -- very high VA share
        elif any(x in nm for x in ["financial", "insurance",
                                    "auxiliary to financial"]):
            v[i] = 0.85
        elif any(x in nm for x in ["real estate", "imputed rent"]):
            v[i] = 0.90    # mostly gross operating surplus, minimal inputs

        # ICT & Business Services -- high human-capital content
        elif any(x in nm for x in ["telecommunication"]):
            v[i] = 0.75
        elif any(x in nm for x in ["computer programming", "publishing",
                                    "motion picture"]):
            v[i] = 0.80
        elif any(x in nm for x in ["legal", "architectural",
                                    "scientific research", "advertising",
                                    "professional"]):
            v[i] = 0.85
        elif any(x in nm for x in ["rental and leas", "employment",
                                    "travel agency", "security",
                                    "services auxiliary"]):
            v[i] = 0.75

        # Public & Social -- predominantly labour value added
        elif any(x in nm for x in ["public admin", "defence",
                                    "compulsory social"]):
            v[i] = 0.90
        elif any(x in nm for x in ["education"]):
            v[i] = 0.90
        elif any(x in nm for x in ["health", "social work"]):
            v[i] = 0.85
        elif any(x in nm for x in ["arts", "entertainment",
                                    "recreation", "sport"]):
            v[i] = 0.75
        elif any(x in nm for x in ["membership organisation",
                                    "household employ",
                                    "extraterritorial"]):
            v[i] = 0.90

    return v


# --- Model state -------------------------------------------------------------

@dataclass
class ModelState:
    """
    All large matrices are sparse CSR; B is stored as a 1-D kappa vector.

    Notes on key fields
    -------------------
    v_per_unit  – structural value-added per unit of output (EUR/unit).
                  Used to recover X_real from observed V via X_real = V / v_per_unit.
                  Also acts as the labour-input proxy in l_vec.
    K           – capital stock (EUR, at model-consistent units).
    P_0         – Q1 cost-push prices, stored for capital slack valuation.
    """
    # -- static ----------------------------------------------------------------
    n:            int
    A:            object          # scipy.sparse.csr_matrix
    A_bar:        object          # scipy.sparse.csr_matrix  (A + delta*diag(kappa))
    B:            np.ndarray      # 1-D kappa vector  (diagonal of capital matrix)
    l_tilde:      np.ndarray      # (I - A_bar)^(-T) @ l,  precomputed at calibration
    v_per_unit:   np.ndarray      # value-added per physical unit (EUR/unit)
    l_vec:        np.ndarray
    L_total:      float
    delta:        float
    drift:        float
    kappa_ou:     float           # Preference mean-reversion speed
    wage_rate:    float           # Homogeneous wage rate (EUR/hour)
    neumann_k:    int             # Number of Neumann iterations
    primal_tol:   float           # Solver primal tolerance
    dual_tol:     float           # Solver dual tolerance
    eta_K:        float           # Dual ascent step size for capital constraint
    eta_L:        float           # Dual ascent step size for labour constraint
    max_iter:     int             # Maximum dual ascent iterations per quarter
    sector_names: List[str]
    sector_short: List[str]
    # -- dynamic ---------------------------------------------------------------
    t:            int           = 0
    K:            np.ndarray    = field(default_factory=lambda: np.array([]))
    G:            np.ndarray    = field(default_factory=lambda: np.array([]))
    g_step:       float         = 0.0    # per-quarter growth rate
    c_step:       float         = 0.015  # per-quarter minimum capacity growth target
    alpha:        np.ndarray    = field(default_factory=lambda: np.array([]))
    alpha_true:   np.ndarray    = field(default_factory=lambda: np.array([]))
    alpha_bar:    np.ndarray    = field(default_factory=lambda: np.array([]))
    P:            np.ndarray    = field(default_factory=lambda: np.array([]))
    C:            np.ndarray    = field(default_factory=lambda: np.array([]))
    X:            np.ndarray    = field(default_factory=lambda: np.array([]))
    Y:            float         = 0.0
    C_monthly:    np.ndarray    = field(default_factory=lambda: np.zeros((3, 1)))
    P_monthly:    np.ndarray    = field(default_factory=lambda: np.zeros((3, 1)))
    rng:          object        = field(default_factory=lambda: np.random.default_rng(42))
    G_hat_init:   np.ndarray    = field(default_factory=lambda: np.array([]))
    G_hat_prev:   np.ndarray    = field(default_factory=lambda: np.array([]))
    dK_0:         np.ndarray    = field(default_factory=lambda: np.array([]))
    history:      list          = field(default_factory=list)
    slim_history: bool          = False
    P_0:          np.ndarray    = field(default_factory=lambda: np.array([]))
    VAL_0:        float         = 1.0
    Y_0_init:     float         = 0.0
    VAL_0_Laspeyres: float      = 1.0
    pi_0_fixed:   np.ndarray    = field(default_factory=lambda: np.array([]))
    dual_weight_0: np.ndarray   = field(default_factory=lambda: np.array([]))
    C_0_init:     np.ndarray    = field(default_factory=lambda: np.array([]))
    # Backward-compat alias: expose v_per_unit also as .v for Julia bridge
    @property
    def v(self) -> np.ndarray:
        return self.v_per_unit


# --- Calibration -------------------------------------------------------------

def calibrate(data: dict,
              delta:        float = 0.0125,
              drift:        float = 0.012,
              neumann_k:    int   = 20,
              kappa_factor: float = 1.0,
              kappa_ou:     float = 0.15,
              L_total:      float = 9.75e9,
              wage_rate:    float = 11.2,
              labor_mult:   float = 1.0,
              primal_tol:   float = 1e-4,
              dual_tol:     float = 1e-4,
              eta_K:        float = 0.15,
              eta_L:        float = 0.15,
              max_iter:     int   = 2000,
              g_step:       float = 0.0,
              c_step:       float = 0.015,
              slim_history: bool  = None) -> ModelState:
    """
    Build and return a fully calibrated ModelState from the IO data dict.

    All monetary inputs in `data` are expected in EUR/quarter (as produced by
    data_loader after its M-EUR→EUR and annual→quarterly conversions).

    X_real computation
    ------------------
    Physical output X_real is no longer taken directly from the IO table's
    gross-output column.  Instead it is derived from observed sectoral value
    added (V) and a structural parameter v_per_unit:

        X_real[i] = V[i] / v_per_unit[i]

    v_per_unit[i] is the EUR of value added embodied in one physical unit of
    sector i's output -- a sector-specific constant analogous to kappa, set
    via the _v_per_unit() lookup table.  Value added (V) is thus the only IO
    variable used to pin down physical production levels; the gross-output
    column from the IO table is retained only for dead-sector detection.
    """
    A_in         = data["A"]
    # V  = sectoral value added, EUR/quarter  (used to derive X_real)
    V            = np.asarray(data["V_total"], dtype=float).copy()
    C_hh         = np.asarray(data["C"],       dtype=float).copy()
    X_data       = np.asarray(data["X"],       dtype=float).copy()   # IO gross output (EUR/q)
    sector_names = list(data["sector_names"])
    sector_short = list(data["sector_short"])
    n            = len(sector_names)

    # -- Convert A to sparse CSR (no-op if already sparse) -------------------
    if sp.issparse(A_in):
        A = A_in.tocsr().astype(float)
    else:
        A = sp.csr_matrix(np.asarray(A_in, dtype=float))
        sparsity = 1.0 - A.nnz / n**2
        if n > 1000:
            logger.info(f"[calibration] Dense A converted to CSR  "
                        f"(sparsity={sparsity*100:.1f}%, nnz={A.nnz:,})")

    # -- Dead sector detection -----------------------------------------------
    # A sector is dead if both the IO gross-output column and value-added are
    # essentially zero.  Threshold: 1 M EUR/quarter  (= 4 M EUR/year, a tiny
    # firm by any measure).
    _dead_threshold = 1e6   # 1 M EUR/quarter
    dead = np.where((X_data < _dead_threshold) & (V < _dead_threshold))[0]
    if len(dead):
        logger.info(f"[calibration] Zeroing {len(dead)} dead sector(s)")
        A = A.tolil()
        for i in dead:
            A[i, :] = 0;  A[:, i] = 0
            X_data[i] = 0;  V[i] = 0;  C_hh[i] = 0
        A = A.tocsr()
        A.eliminate_zeros()

    # -- v_per_unit: value-added per physical unit (structural parameter) ----
    # Analogous to kappa; derived from a sector-name lookup table, NOT from
    # dividing V by X.  V is only used below to compute X_real.
    v_per_unit = np.asarray(data["v_per_unit"], dtype=float) \
                 if "v_per_unit" in data else _v_per_unit(sector_names)

    # -- X_real: physical output recovered from value-added data -------------
    # X_real[i] = V[i] / v_per_unit[i]
    # For dead sectors (V = 0) this naturally yields X_real = 0.
    X_real = V / np.maximum(v_per_unit, 1e-12)

    # -- kappa (diagonal of B) -----------------------------------------------
    kappa = np.asarray(data["kappa"], dtype=float) if "kappa" in data \
            else _kappa(sector_names)
    kappa = kappa * kappa_factor

    # -- Eq. 6: A_bar = A + delta * diag(kappa) ------------------------------
    A_bar = A + delta * sp.diags(kappa, format="csr")
    A_bar.eliminate_zeros()

    # Spectral radius check
    try:
        eigvals_top = sp_eigs(A_bar, k=1, which="LM", return_eigenvectors=False,
                              maxiter=10 * n, tol=1e-4)
        rho = float(np.abs(eigvals_top).max())
        if rho < 1:
            logger.info(f"[calibration] rho(A_bar) = {rho:.4f}  (OK)")
        else:
            logger.warning(f"[calibration] rho(A_bar) = {rho:.4f}  (WARNING) -- Neumann will diverge!")
    except Exception:
        row_sums = np.asarray(np.abs(A_bar).sum(axis=1)).ravel()
        rho = row_sums.max()
        if rho < 1:
            logger.info(f"[calibration] Gershgorin rho(A_bar) <= {rho:.4f}  (stable)")
        else:
            logger.warning(f"[calibration] Gershgorin rho(A_bar) <= {rho:.4f}  (check stability!)")

    # -- Initial cost-push prices P_real -------------------------------------
    P_real = neumann_apply(A.T, v_per_unit, k=neumann_k)
    P_0_raw = P_real  # Alias for compatibility if needed

    # -- Nominal anchor ------------------------------------------------------
    Y_0 = 201.75e9   # 201.75 B EUR

    # -- Initial physical demand C_0 and G_0 ---------------------------------
    C_0 = C_hh / np.where(P_real > 1e-30, P_real, 1e-30)
    C_0[dead] = 0

    G_raw_cal = np.asarray(data.get("G_raw", np.zeros(n)), dtype=float)
    G_0 = G_raw_cal / np.where(P_real > 1e-30, P_real, 1e-30)

    # -- dK_0 for capital dynamics -------------------------------------------
    g_init = 0.01
    v1 = g_init * C_0 + g_step * G_0
    v2 = (g_init**2) * C_0 + (g_step**2) * G_0

    term1 = kappa * neumann_apply(A_bar, v1, k=neumann_k)
    inner_v2 = neumann_apply(A_bar, v2, k=neumann_k)
    term2 = kappa * neumann_apply(A_bar, kappa * inner_v2, k=neumann_k)
    dK_0 = term1 + term2

    # -- Initial Capital -----------------------------------------------------
    # Ensuring K_0 supports both X_real and the specific requirements for G and dK
    K_0 = kappa * neumann_apply(A_bar,C_0 + G_0 + dK_0, k=neumann_k)

    # -- Preferences and base basket -----------------------------------------
    exp = np.where(P_real * C_0 > 0, P_real * C_0, 0.0)
    alpha_0 = exp / max(exp.sum(), 1e-10)

    P_0 = np.where(C_0 > 1e-12, Y_0 * alpha_0 / C_0, 0.0)

    logger.info(f"[calibration] Initial Prices anchored "
                f"(implied price level: {(P_0.mean() / max(P_real.mean(), 1e-30)):.4f})")

    # -- Labour proxy --------------------------------------------------------
    l_vec = (v_per_unit.copy() / wage_rate) * labor_mult
    l_tilde = neumann_apply(A_bar.T, l_vec, k=neumann_k)

    # -- Initial demand push -------------------------------------------------
    G_hat_init = np.where(C_0 > 0, g_init, 0.0)

    mem_A_MB = (A_bar.data.nbytes + A_bar.indices.nbytes
                + A_bar.indptr.nbytes) / 1e6
    logger.info(f"[calibration] n={n:,}  nnz={A_bar.nnz:,}  "
                f"A_bar mem={mem_A_MB:.1f} MB")
    logger.info(f"[calibration] Y0={Y_0/1e9:.1f}B EUR  "
                f"K0={K_0.sum()/1e9:.1f}B units  "
                f"dK_0={dK_0.sum()/1e9:.1f}B units  "
                f"C0={C_0.sum()/1e9:.1f}B units/quarter  "
                f"delta={delta}")
    logger.info(f"[calibration] X_real (from V/v_per_unit): "
                f"total={X_real.sum()/1e9:.1f}B EUR-equiv units/quarter")

    state = ModelState(
        n=n, A=A, A_bar=A_bar, B=kappa,
        l_tilde=l_tilde,
        v_per_unit=v_per_unit,
        l_vec=l_vec, L_total=L_total, delta=delta, drift=drift,
        kappa_ou=kappa_ou, wage_rate=wage_rate,
        neumann_k=neumann_k,
        primal_tol=primal_tol, dual_tol=dual_tol,
        eta_K=eta_K, eta_L=eta_L, max_iter=max_iter,
        sector_names=sector_names, sector_short=sector_short,
        t=0,
        K=K_0.copy(),
        G=G_0.copy(),
        g_step=float(g_step),
        c_step=float(c_step),
        alpha=alpha_0.copy(), alpha_true=alpha_0.copy(),
        alpha_bar=alpha_0.copy(),
        P=P_0.copy(), C=C_0.copy(), X=X_real.copy(), Y=Y_0,
        C_monthly=np.tile(C_0 / 3, (3, 1)),
        P_monthly=np.tile(P_0,     (3, 1)),
        G_hat_init=G_hat_init,
        G_hat_prev=G_hat_init.copy(),
        dK_0=dK_0.copy(),
    )
    if slim_history is None:
        slim_history = (n > 5000)
        if slim_history:
            logger.info("[calibration] Auto-enabled slim_history for n > 5000")
    state.slim_history = slim_history

    # -- Initial VAL_0 for the income formula --------------------------------
    pi_0 = np.where(C_0 > 1e-12, alpha_0 / C_0, 0.0)

    state.P_0             = P_0.copy()
    state.C_0_init        = C_0.copy()
    state.VAL_0_Laspeyres = 1.0
    state.pi_0_fixed      = np.array([])
    state.dual_weight_0   = np.array([])
    state.Y_0_init        = float(Y_0)
    state.B               = kappa.copy()

    return state
