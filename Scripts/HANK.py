"""
HANK model: 65-sector Spanish IO, 1000 households with idiosyncratic
income risk, borrowing constraints, NK Phillips curve, Taylor rule.
Outputs written to ./hank_results/.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigs as sp_eigs
from pathlib import Path
import logging
import os
import time
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 0.  Paths
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "Results" / "HANK"
FIGURES_DIR = RESULTS_DIR / "Figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1.  Data loading  (uses shared data_loader.py)
# ---------------------------------------------------------------------------

try:
    from data_loader import load_data as _load_data_shared
    _HAS_SHARED_LOADER = True
except ImportError:
    _HAS_SHARED_LOADER = False


def load_spanish_data(data_dir: Path = DATA_DIR) -> dict:
    """Load Spanish 2022 IO data, adapting keys for HANK."""
    if _HAS_SHARED_LOADER:
        log.info("Loading Spanish 2022 IO data via shared data_loader …")
        raw = _load_data_shared(data_dir)
        # Adapt keys: data_loader uses V_total, G_raw; HANK expects V, G
        out = dict(
            A            = raw["A"],
            V            = raw["V_total"],
            C            = raw["C"],
            G            = raw.get("G_raw", np.zeros_like(raw["C"])),
            X            = raw["X"],
            I_gross      = raw["I_gross"],
            sector_names = raw["sector_names"],
        )
    else:
        log.info("Loading Spanish 2022 IO data (standalone) …")
        df_a       = pd.read_excel(data_dir / "Spanish A-matrix.xlsx", index_col=0)
        A_df       = df_a.iloc[1:, :]
        sector_names = list(A_df.columns)
        A          = A_df.values.astype(float)
        df_v   = pd.read_excel(data_dir / "Value added.xlsx", header=None)
        V      = df_v.iloc[0, :].values.astype(float) * 1e6 / 4.0
        df_cp  = pd.read_excel(data_dir / "Consumption and total production.xlsx",
                               header=None)
        C      = df_cp.iloc[2:, 0].values.astype(float) * 1e6 / 4.0
        I_gros = np.clip(df_cp.iloc[2:, 2].values.astype(float), 0, None) * 1e6 / 4.0
        G      = df_cp.iloc[2:, 4].values.astype(float) * 1e6 / 4.0
        X      = df_cp.iloc[2:, 6].values.astype(float) * 1e6 / 4.0
        out = dict(A=A, V=V, C=C, G=G, X=X, I_gross=I_gros,
                   sector_names=sector_names)

    log.info(f"  Annual GDP  : {out['V'].sum()*4/1e9:,.1f} B EUR")
    log.info(f"  Annual C    : {out['C'].sum()*4/1e9:,.1f} B EUR")
    log.info(f"  Annual G    : {out['G'].sum()*4/1e9:,.1f} B EUR")
    log.info(f"  Sectors     : {len(out['sector_names'])}")
    return out


# ---------------------------------------------------------------------------
# 2.  Capital-intensity vector  (mirrors calibration._kappa)
# ---------------------------------------------------------------------------

def _kappa(sector_names):
    k = np.full(len(sector_names), 0.50)
    mappings = [
        (["real estate", "imputed rent"],                               3.10),
        (["electricity", "gas", "steam"],                               2.17),
        (["mining", "quarrying"],                                       1.86),
        (["petroleum", "coke"],                                         1.55),
        (["water supply", "natural water"],                             1.24),
        (["sewerage", "waste", "agricultur", "forestr", "fish"],        0.93),
        (["chemical", "pharmaceutical", "motor vehicle"],               0.74),
        (["food", "textile", "wood", "paper", "rubber", "plastic",
          "fabricated metal", "electrical", "machinery", "basic metal"],0.62),
        (["construct", "financial", "insurance"],                       0.31),
        (["telecommunication", "computer programming"],                 0.37),
    ]
    for i, name in enumerate(sector_names):
        nm = name.lower()
        for keys, val in mappings:
            if any(k_ in nm for k_ in keys):
                k[i] = val; break
    return k


def _v_per_unit(sector_names):
    """Value-added per unit of gross output (EUR/EUR)."""
    v = np.ones(len(sector_names)) * 0.60
    subs = [
        (["agricultur", "forestr", "fish"],   0.50),
        (["mining", "quarrying"],             0.45),
        (["petroleum", "coke"],               0.25),
        (["electricity", "gas", "steam"],     0.40),
        (["food", "beverage", "tobacco"],     0.30),
        (["textile", "leather", "apparel"],   0.50),
        (["chemical", "pharmaceutical"],      0.60),
        (["rubber", "plastic"],               0.50),
        (["basic metal", "fabricated metal"], 0.45),
        (["construct"],                       0.65),
        (["wholesale", "retail", "trade"],    0.70),
        (["financial", "insurance"],          0.85),
        (["real estate", "imputed rent"],     0.90),
        (["telecommunication"],               0.75),
        (["public admin", "defence"],         0.90),
        (["education"],                       0.90),
        (["health", "social work"],           0.85),
    ]
    for i, name in enumerate(sector_names):
        nm = name.lower()
        for keys, val in subs:
            if any(k_ in nm for k_ in keys):
                v[i] = val; break
    return v


# ---------------------------------------------------------------------------
# 3.  Neumann-series Leontief inverse  (no Julia dependency)
# ---------------------------------------------------------------------------

def neumann_inverse_vec(A_bar, v, K=27):
    """Compute (I - A_bar)^{-1} v via Neumann series to order K."""
    x = v.copy()
    term = v.copy()
    for _ in range(K):
        term = A_bar @ term
        x    = x + term
    return x


def neumann_inverse_mat(A_bar, F, K=27):
    """Column-wise Neumann inverse for matrix F."""
    X = F.copy()
    T = F.copy()
    for _ in range(K):
        T = A_bar @ T
        X = X + T
    return X


# ---------------------------------------------------------------------------
# 4.  HANK calibration
# ---------------------------------------------------------------------------

def build_kappa_B(n, sector_names, kappa_factor=4.0, rng=None):
    """Build sparse capital-requirement matrix B (same structure as planner)."""
    if rng is None:
        rng = np.random.default_rng(42)
    diag_kappa = _kappa(sector_names)
    B = sp.diags(diag_kappa, format="csr", dtype=float)
    noise_density = 0.05
    n_noise = int(n * n * noise_density)
    rows = rng.integers(0, n, size=n_noise)
    cols = rng.integers(0, n, size=n_noise)
    vals = rng.uniform(0.15, 0.2, size=n_noise)
    B_noise = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    B_full  = (B + B_noise).tolil()
    B_full.setdiag(diag_kappa)
    B_full  = B_full.tocsr()
    B_full.eliminate_zeros()
    return B_full * kappa_factor


# ---------------------------------------------------------------------------
# 5.  Household block
# ---------------------------------------------------------------------------

class HANKHouseholdBlock:
    """
    HANK block: 1000 households with idiosyncratic labour income risk,
    borrowing constraint, endogenous wealth distribution, and
    buffer-stock MPCs. No fixed HtM/Saver split.
    """

    N_HH        = 1000
    PHI_BORROW  = 0.0    # natural borrowing limit (no debt)
    BETA_SPREAD = 0.02   # discount factor heterogeneity range

    def __init__(self, n_sectors, sector_names, alpha_0, rng, hh_dispersion=0.05, 
                 n_firms=5, firm_shares=None, phi_matrix=None):
        self.n  = n_sectors
        self.nh = self.N_HH
        self.use_phi = (phi_matrix is not None)

        # ---- Idiosyncratic income process: AR(1) in logs ----
        # Floden-Linde calibration: rho=0.97, sigma_eps=0.15
        self.rho_y   = 0.97
        self.sig_eps = 0.15
        sig_stat     = self.sig_eps / np.sqrt(1.0 - self.rho_y ** 2)
        self.log_y_h = rng.normal(0.0, sig_stat, self.nh)
        self.y_h     = np.exp(self.log_y_h)
        self.y_h    /= self.y_h.mean()   # normalise mean = 1

        # ---- Wealth distribution: log-normal / Pareto tail ----
        # Calibrated to Spain 2022 HFCS Gini(wealth) ≈ 0.67
        raw_a        = rng.pareto(1.3, self.nh) + 1.0
        raw_a       /= raw_a.mean()
        self.a_h     = raw_a.copy()   # (nh,) asset holdings

        # ---- Heterogeneous discount factors (Krusell-Smith) ----
        beta_mean    = 0.99
        self.beta_h  = np.linspace(beta_mean - self.BETA_SPREAD / 2,
                                    beta_mean + self.BETA_SPREAD / 2,
                                    self.nh)
        rng.shuffle(self.beta_h)

        # ---- CRRA parameters ----
        self.sigma_h = np.ones(self.nh) * 1.0
        self.rho_h   = 1.0 / self.beta_h - 1.0

        # ---- Income shares (wealth-weighted OR Matrix-weighted) ----
        if phi_matrix is not None and firm_shares is not None:
            # Standardize distribution using the Phi matrix (Matrix Y_firms)
            # Match the logic in the planner: Y_h = W_ownership @ Y_f
            hh_income_raw = phi_matrix @ firm_shares
            self.income_share = hh_income_raw / hh_income_raw.sum()
        else:
            income_raw        = self.y_h * (1.0 + 0.5 * self.a_h)
            income_raw       /= income_raw.sum()
            self.income_share = income_raw   # (nh,)

        # ---- Buffer-stock MPCs (Carroll 2006) ----
        # MPC_h ≈ 1 - beta_h * (1+r)^{1/sigma}; constrained agents get MPC=1
        r_ss            = 0.005
        mpc_raw         = 1.0 - self.beta_h * ((1.0 + r_ss) ** (1.0 / self.sigma_h))
        at_constraint   = self.a_h < 0.05
        self.mpc_h      = np.where(at_constraint, 1.0, np.clip(mpc_raw, 0.02, 0.99))

        # ---- Preference heterogeneity ----
        # Use fixed seed 123 to match calibration.py for identical initial preferences
        rng_hh = np.random.default_rng(123)
        alpha_h = np.zeros((self.nh, n_sectors))
        for h in range(self.nh):
            noise      = rng_hh.normal(0.0, hh_dispersion, n_sectors)
            a          = alpha_0 * np.exp(noise)
            a          = np.maximum(a, 1e-30)
            a         /= a.sum()
            alpha_h[h] = a
        self.alpha_h      = alpha_h
        self.alpha_true_h = alpha_h.copy()

        self.pref_rho   = 0.9
        self.pref_sigma = 0.02
        self.C_h        = None

    def update_idiosyncratic_income(self, rng):
        """Advance idiosyncratic income via AR(1); recompute income shares."""
        eps            = rng.normal(0.0, self.sig_eps, self.nh)
        self.log_y_h   = self.rho_y * self.log_y_h + eps
        self.y_h       = np.exp(self.log_y_h)
        self.y_h      /= self.y_h.mean()
        
        if not self.use_phi:
            income_raw     = self.y_h * (1.0 + 0.5 * self.a_h)
            income_raw    /= income_raw.sum()
            self.income_share = income_raw

    def update_wealth(self, r_real, C_h_agg_scalar=None):
        """Budget-constraint asset update with borrowing constraint."""
        c_h      = self.mpc_h * (self.y_h + r_real * self.a_h)
        a_new    = (1.0 + r_real) * self.a_h + self.y_h - c_h
        self.a_h = np.maximum(a_new, self.PHI_BORROW)
        # Recompute MPCs from updated wealth
        r_ss          = 0.005
        mpc_raw       = 1.0 - self.beta_h * ((1.0 + r_ss) ** (1.0 / self.sigma_h))
        at_constraint = self.a_h < 0.05
        self.mpc_h    = np.where(at_constraint, 1.0, np.clip(mpc_raw, 0.02, 0.99))

    def update_preferences(self, rng):
        """AR(1) preference drift each quarter."""
        shock = rng.normal(0.0, self.pref_sigma, (self.nh, self.n))
        log_a = np.log(np.maximum(self.alpha_true_h, 1e-30))
        log_a = (self.pref_rho * log_a
                 + (1 - self.pref_rho) * np.log(np.maximum(self.alpha_h, 1e-30))
                 + shock)
        new_a = np.exp(log_a)
        new_a = np.maximum(new_a, 1e-30)
        new_a /= new_a.sum(axis=1, keepdims=True)
        self.alpha_true_h = new_a

    def aggregate_alpha(self, sigma_vec):
        """Income-weighted CES aggregate preference (Eq. 16)."""
        w     = self.income_share
        a_agg = np.zeros(self.n)
        for i in range(self.n):
            s         = sigma_vec[i]
            wa        = np.maximum(w * self.alpha_true_h[:, i], 1e-50)
            a_agg[i]  = max(np.sum(wa ** (1.0 / s)) ** s, 1e-50)
        a_agg  = np.maximum(a_agg, 1e-30)
        a_agg /= a_agg.sum()
        return a_agg

    def demand(self, P, Y_h, sigma_vec, mu):
        """CRRA Marshallian demand; returns (nh, n) array."""
        P_safe = np.maximum(P, 1e-15)[None, :]
        a      = self.alpha_true_h
        w      = self.income_share[:, None]
        s      = sigma_vec[None, :]
        return (w * a / (mu * P_safe)) ** (1.0 / s)

    def aggregate_demand(self, P, Y_total, sigma_vec, mu):
        """Aggregate demand across all households; returns (n,) array."""
        return self.demand(P, None, sigma_vec, mu).sum(axis=0)


# ---------------------------------------------------------------------------
# 6.  Firm / price-setting block (Rotemberg / Calvo linearised)
# ---------------------------------------------------------------------------

class FirmBlock:
    """
    Sector-level NK pricing via log-linearised Rotemberg Phillips curve.
    Adaptive expectations: E[pi_{t+1}] ≈ pi_{t-1}.
    """

    def __init__(self, A, l_vec, wage_rate,
                 beta=0.99, kappa_nk=0.15, calvo_theta=0.75,
                 eps_demand=10.0):
        self.A         = A            # (n, n) tech coefficients (sparse)
        self.l_vec     = l_vec        # (n,) labour per unit output
        self.wage      = wage_rate
        self.beta      = beta
        self.kappa_nk  = kappa_nk
        self.calvo     = calvo_theta
        # Steady-state price-cost margin: pi = 0 when mc = mc_ss = 1/markup
        # Standard Dixit-Stiglitz: markup = eps/(eps-1), mc_ss = (eps-1)/eps
        self.mc_ss     = (eps_demand - 1.0) / eps_demand   # ~0.9 for eps=10
        self.pi_prev   = None

    def marginal_cost(self, P):
        """Compute real marginal cost per sector."""
        P_safe     = np.maximum(P, 1e-10)
        labor_mc   = self.wage * self.l_vec / P_safe
        input_mc   = np.asarray(self.A.T @ P) / P_safe
        mc         = labor_mc + input_mc
        # Dead sectors (P≈0) get mc=1 (no distortion)
        mc         = np.where(P < 1e-9, 1.0, mc)
        return mc   # (n,)

    def inflation(self, P, mc, E_pi_next=None):
        """Sector-level inflation from NK Phillips curve, clipped to ±5% per quarter."""
        if E_pi_next is None:
            # Adaptive expectations: expect inflation to be what it was last quarter
            E_pi_next = np.zeros(len(P)) if self.pi_prev is None else self.pi_prev
            
        pi = self.beta * E_pi_next + self.kappa_nk * (mc - self.mc_ss)
        # Zero-out dead sectors; clip live sectors to prevent divergence
        pi = np.where(P < 1e-9, 0.0, pi)
        pi = np.clip(pi, -0.05, 0.05)
        self.pi_prev = pi.copy()
        return pi   # (n,)

    def update_prices(self, P, pi):
        P_new = P * np.exp(pi)
        return np.where(P < 1e-9, P, P_new)   # keep dead sectors dead


# ---------------------------------------------------------------------------
# 7.  Monetary policy (Taylor rule)
# ---------------------------------------------------------------------------

class MonetaryPolicy:
    """Inertial Taylor rule; Fisher real rate approximation."""

    def __init__(self, r_ss=0.005, phi_pi=1.5, phi_y=0.5,
                 rho_i=0.8, pi_star=0.005):
        self.r_ss    = r_ss      # quarterly steady-state real rate
        self.phi_pi  = phi_pi
        self.phi_y   = phi_y
        self.rho_i   = rho_i
        self.pi_star = pi_star
        # Initialize nominal rate at steady state (Fisher equation approx)
        self.i_prev  = r_ss + pi_star

    def nominal_rate(self, pi_agg, y_gap):
        i = (self.rho_i * self.i_prev +
             (1 - self.rho_i) * (self.r_ss
                                  + self.phi_pi * (pi_agg - self.pi_star)
                                  + self.phi_y  * y_gap))
        self.i_prev = i
        return i

    def real_rate(self, i, E_pi_next):
        return i - E_pi_next


# ---------------------------------------------------------------------------
# 8.  Capital accumulation
# ---------------------------------------------------------------------------

def update_capital(K, I_net, delta):
    """K_{t+1} = (1 - delta) * K_t + I_t"""
    return (1.0 - delta) * K + I_net


# ---------------------------------------------------------------------------
# 9.  Main HANK simulation
# ---------------------------------------------------------------------------

class HANKModel:
    """
    20-quarter HANK simulation: 65-sector IO, 1000 households with
    idiosyncratic income risk and borrowing constraints.
    State per quarter: X, C, P, K, G (n-vectors), Y (scalar).
    """

    def __init__(self, data, config=None, rng_seed=None, n_households=1000,
                 hh_dispersion=0.05, n_quarters=20,
                 n_firms=5, firm_shares=None, phi_matrix=None):
        cfg = config or {}
        self.n_quarters   = n_quarters
        self.n_firms      = n_firms
        self.firm_shares  = firm_shares
        self.phi_matrix   = phi_matrix
        self.delta       = cfg.get("delta", 0.01)
        self.g_step      = cfg.get("g_step", 0.01)
        self.kappa_factor= cfg.get("kappa_factor", 4.0)
        self.neumann_k   = cfg.get("neumann_k", 27)
        self.wage_rate   = cfg.get("wage_rate", 16.9)
        self.sigma_val   = cfg.get("sigma_val", 1.0)
        
        # Determine discount factor from target inflation and steady state real rate
        self.pi_star     = cfg.get("pi_star", 0.005)  # target inflation (default 0.5% per quarter)
        self.r_ss        = cfg.get("r_ss", 0.005)
        # Using Euler equation steady state, beta = 1/(1+r_ss)
        self.beta        = 1.0 / (1.0 + self.r_ss)

        self.rng = np.random.default_rng(rng_seed)

        # ---- IO structure ----
        A_raw        = np.asarray(data["A"], dtype=float)
        self.n       = A_raw.shape[0]
        n            = self.n
        self.sector_names = data["sector_names"]
        self.A       = sp.csr_matrix(A_raw)

        # Value-added coefficients
        self.v_per_unit = _v_per_unit(self.sector_names)

        # Capital requirement matrix: Use fixed seed 42 to match calibration.py
        self.B = build_kappa_B(n, self.sector_names, self.kappa_factor, None)

        # Augmented matrix A_bar = A + delta * B
        self.A_bar = (self.A + self.delta * self.B).tocsr()
        self.A_bar.eliminate_zeros()

        # Check spectral radius
        try:
            ev = sp_eigs(self.A_bar, k=1, which="LM",
                         return_eigenvectors=False, tol=1e-4)
            rho = float(np.abs(ev).max())
            log.info(f"  rho(A_bar) = {rho:.4f}")
        except Exception:
            pass

        # ---- Spectral bound for revealed demand: rho(M) = rho(B @ L_inv) ----
        # L = I - A
        L = np.eye(n) - self.A.toarray()
        try:
            L_inv = np.linalg.inv(L)
            M = self.B.toarray() @ L_inv
            ev_M = np.linalg.eigvals(M)
            self.rho_M = float(np.abs(ev_M).max())
            log.info(f"  rho(M) = rho(B @ L_inv) = {self.rho_M:.4f}")
        except Exception:
            self.rho_M = 0.5
            log.warning("Could not compute spectral bound rho(M); using default 0.5")

        # ---- Prices (cost-push from value-added) ----
        P_real = neumann_inverse_vec(
            np.asarray(self.A.T.toarray()),
            self.v_per_unit, K=self.neumann_k)

        # ---- Initial conditions from data ----
        C_data   = np.asarray(data["C"],       dtype=float)
        G_data   = np.asarray(data["G"],       dtype=float)
        X_data   = np.asarray(data["X"],       dtype=float)
        V_data   = np.asarray(data["V"],       dtype=float)

        # Physical quantities via value-added deflator
        X_real   = V_data / np.maximum(self.v_per_unit, 1e-12)

        # Nominal consumption for income anchoring
        # Target nominal aggregate household income (annualised -> quarterly)
        nom_cons_ann = cfg.get("nominal_consumption_annual", 807e9)
        Y_0 = nom_cons_ann / 4.0
        
        # Determine nominal scaling factor relative to raw IO data (where P=1)
        va_annual_base  = float(V_data.sum()) * 4.0
        nom_scale = nom_cons_ann / max(va_annual_base, 1e-30)

        # Real consumption
        C_0  = C_data / np.maximum(P_real, 1e-30)
        C_0  = np.maximum(C_0, 0.0)

        # Expenditure shares → initial alpha
        exp_shares = np.maximum(P_real * C_0, 0.0)
        exp_shares = np.maximum(exp_shares, 1e-30)
        alpha_0    = exp_shares / exp_shares.sum()

        # Anchor prices to market (nominal)
        P_0 = Y_0 * alpha_0 / np.where(C_0 > 1e-12, C_0, 1e-12)

        # Government spending
        G_0  = G_data / np.maximum(P_real, 1e-30)

        # Labour coefficients (using base wage for physical units)
        l_vec   = self.v_per_unit / self.wage_rate
        # Now scale nominal wage for the NK firms (matching scaled P_0)
        self.wage_rate *= nom_scale
        l_tilde = neumann_inverse_vec(
            np.asarray(self.A_bar.T.toarray()),
            l_vec, K=self.neumann_k)

        # Initial capital stock  K_0 = B (I - A_bar)^{-1} (C + G)
        demand_0 = C_0 + G_0
        X_0      = neumann_inverse_vec(
            np.asarray(self.A_bar.toarray()), demand_0, K=self.neumann_k)
        K_0      = np.asarray(self.B @ X_0)

        # Initial marginal utility of income
        sigma_vec = np.full(n, self.sigma_val)
        pi_0      = alpha_0 / np.maximum(C_0 ** sigma_vec, 1e-30)
        mu_0      = float(np.dot(pi_0, C_0) / max(Y_0, 1e-30))

        # ---- Sub-blocks ----
        self.sigma_vec = sigma_vec
        self.households = HANKHouseholdBlock(n, self.sector_names, alpha_0,
                                             self.rng, hh_dispersion,
                                             n_firms=n_firms, 
                                             firm_shares=firm_shares, 
                                             phi_matrix=phi_matrix)
        self.households.C_h = (self.households.income_share[:, None] *
                                C_0[None, :] / len(self.households.income_share))

        self.firms  = FirmBlock(self.A, l_vec, self.wage_rate,
                                beta=self.beta, kappa_nk=0.15, eps_demand=10.0)
        self.taylor = MonetaryPolicy(r_ss=self.r_ss, phi_pi=1.5, phi_y=0.5,
                                     rho_i=0.8, pi_star=self.pi_star)

        # ---- State ----
        self.P          = P_0.copy()
        self.C          = C_0.copy()
        self.X          = X_0.copy()
        self.K          = K_0.copy()
        self.G          = G_0.copy()
        self.Y          = float(Y_0)
        self.mu         = mu_0
        self.alpha      = alpha_0.copy()    # planner's estimate
        self.alpha_true = self.households.aggregate_alpha(sigma_vec)
        self.l_tilde    = l_tilde
        self.l_vec      = l_vec
        self.P_real_0   = P_real.copy()
        self.C_0        = C_0.copy()
        self.Y_0        = Y_0
        self.X_0        = X_0.copy()
        self.i_nominal  = self.r_ss + self.pi_star   # initial nominal rate
        self.apc_0      = float(np.dot(self.P, self.C)) / max(self.Y_0, 1e-30)

        # ---- Real GDP deflation (constant Q1 prices) ----
        # P_base = Q1 prices used as constant deflator
        self.P_base     = P_0.copy()
        # Value-added weights at Q1 prices: va_i = P_i - sum_j A_ji * P_j
        self.va_weight_0 = P_0 - np.asarray(self.A.T @ P_0).ravel()
        # Real scale factor: anchors real GDP = nominal GDP at Q1
        demand_0_total  = C_0 + G_0
        self.gdp_real_0 = float(np.dot(self.va_weight_0, X_0))
        # Fallback: if va_weight based approach gives odd values, use expenditure
        self.gdp_real_exp_0 = float(np.dot(P_0, demand_0_total))

        # ---- History ----
        self.history    = []

    # ------------------------------------------------------------------
    # Quarter step
    # ------------------------------------------------------------------

    def _solve_output(self, C, G, dK):
        """Gross output from material balance via Neumann."""
        F = C + G + dK
        X = neumann_inverse_vec(
            np.asarray(self.A_bar.toarray()), F, K=self.neumann_k)
        return np.maximum(X, 0.0)

    def _investment(self, C, G, G_hat):
        """
        Investment feedback via truncated Neumann (Eq. 26):
        dK = M*G_hat*C + M²*G_hat²*C + M³*G_hat³*C, M = B(I-A_bar)^{-1}.
        """
        A_bar_d = np.asarray(self.A_bar.toarray())
        B_d     = np.asarray(self.B.toarray())

        # M v = B (I - A_bar)^{-1} v
        def M_times(v):
            return B_d @ neumann_inverse_vec(A_bar_d, v, K=self.neumann_k)

        v1 = G_hat * C + self.g_step * G
        v2 = (G_hat**2) * C + (self.g_step**2) * G
        v3 = (G_hat**3) * C + (self.g_step**3) * G

        dK = M_times(v1) + M_times(M_times(v2)) + M_times(M_times(M_times(v3)))
        return np.maximum(dK, 0.0)

    def _tatonnement(self, C_planned, G_hat_vec, n_iter=8):
        """
        Fast tâtonnement: 3 monthly sub-periods, Newton price adjustment (Eq. 31).
        Returns monthly consumption (3, n) and prices (3, n).
        """
        P_t   = self.P.copy()
        C_m   = np.tile(C_planned / 3.0, (3, 1))
        P_m   = np.zeros((3, self.n))

        for tau in range(3):
            for _ in range(n_iter):
                # Aggregate demand at current prices
                mu_t    = self._compute_mu(P_t, C_m[tau])
                C_dem   = self.households.aggregate_demand(
                    P_t, self.Y, self.sigma_vec, mu_t)
                C_dem   = np.maximum(C_dem, 0.0)
                # Excess demand
                Z       = C_dem - C_m[tau]
                # Newton step (Eq. 31): eps_i = -1/sigma_i
                eps     = -1.0 / np.maximum(self.sigma_vec, 1e-6)
                dP      = -P_t / eps * Z / np.maximum(C_m[tau], 1e-15)
                P_t     = np.maximum(P_t + dP, 1e-15)
            C_m[tau] = C_dem
            P_m[tau] = P_t.copy()

        return C_m, P_m

    def _compute_mu(self, P, C):
        """Marginal utility of income from budget constraint."""
        a_safe  = np.maximum(self.alpha, 1e-30)
        C_safe  = np.maximum(C, 1e-30)
        P_safe  = np.maximum(P, 1e-30)
        pi_vec  = a_safe / (C_safe ** self.sigma_vec)
        mu      = float(np.dot(pi_vec, C_safe) / max(self.Y / 3.0, 1e-30))
        return max(mu, 1e-30)

    def _update_alpha(self, C_m, P_m):
        """Preference update rule (Eq. 18): average over monthly sub-periods."""
        alpha_new = np.zeros(self.n)
        for tau in range(3):
            P_   = np.maximum(P_m[tau], 1e-15)
            C_   = np.maximum(C_m[tau], 1e-15)
            num  = P_ * (C_ ** self.sigma_vec)
            denom = num.sum()
            alpha_new += num / max(denom, 1e-30)
        alpha_new /= 3.0
        alpha_new  = np.maximum(alpha_new, 1e-5)
        alpha_new /= alpha_new.sum()
        return alpha_new

    def _revealed_demand(self, C_m, P_m):
        """Elasticity-corrected revealed demand (Eq. 24)."""
        P_mean = P_m.mean(axis=0)
        C_hat  = np.zeros(self.n)
        for tau in range(3):
            dP  = P_m[tau] - P_mean
            eps = -1.0 / np.maximum(self.sigma_vec, 1e-6)
            s   = eps * dP / np.maximum(P_mean, 1e-15)
            # Remove s_max cap; only use safety floor for denom
            C_hat += C_m[tau] / np.maximum(1.0 - s, 1e-6)
        return C_hat

    def step(self):
        """Advance one quarter."""
        t = len(self.history) + 1

        # 1.  Update household preferences (AR(1) drift) + idiosyncratic income
        self.households.update_preferences(self.rng)
        self.households.update_idiosyncratic_income(self.rng)
        self.alpha_true = self.households.aggregate_alpha(self.sigma_vec)

        # 2.  Government spending path
        self.G = self.G * (1.0 + self.g_step)

        # 3.  Revealed demand and growth inference
        if t == 1:
            C_hat   = self.C.copy()
            G_hat_i = np.zeros(self.n)   # no prior history
        else:
            prev = self.history[-1]
            C_hat   = self._revealed_demand(
                np.array(prev["C_m"]), np.array(prev["P_m"]))
            G_hat_i = np.clip(
                (C_hat - prev["C"]) / np.maximum(prev["C"], 1e-30),
                -0.1, 0.3)

        # 4.  Investment feedback
        dK      = self._investment(self.C, self.G, G_hat_i)

        # 5.  Solve gross output
        X_new   = self._solve_output(self.C, self.G, dK)

        # 6.  Marginal cost and NK price inflation
        mc      = self.firms.marginal_cost(self.P)
        pi_sec  = self.firms.inflation(self.P, mc)

        # 7.  Aggregate inflation (Laspeyres)
        pi_agg  = float(np.dot(self.P, pi_sec) / max(np.sum(self.P), 1e-30))

        # 8.  Taylor rule: nominal and real rates
        # Y_potential grows at g_step to prevent trend inflation fight
        Y_potential = self.Y_0 * ((1.0 + self.g_step) ** (t - 1))
        Y_gap   = float((self.Y - Y_potential) / max(Y_potential, 1e-30))
        i_nom   = self.taylor.nominal_rate(pi_agg, Y_gap)
        r_real  = self.taylor.real_rate(i_nom, pi_agg)   # Fisher approximation
        self.i_nominal = i_nom

        # 9. HANK Euler Equation: income-weighted average across all households
        #    Each household h: log C_growth_h = (r_real - rho_h) / sigma_h
        #    Aggregate via income-share weighting
        log_C_growth_h   = (r_real - self.households.rho_h) / self.households.sigma_h
        C_growth_h       = np.exp(np.clip(log_C_growth_h, -0.1, 0.1))
        # Income-weighted aggregate consumption growth
        C_agg_growth     = float(np.dot(self.households.income_share, C_growth_h))
        # Dampen to prevent explosive dynamics
        C_agg_growth_damped = 1.0 + 0.5 * (C_agg_growth - 1.0)

        # 10. Update prices
        P_new   = self.firms.update_prices(self.P, pi_sec)
        P_new   = np.maximum(P_new, 1e-15)

        # 11. Nominal income path: buffer-stock aggregate (HANK)
        t_q     = len(self.history)   # quarters elapsed (0-indexed)
        Y_new   = self.Y_0 * ((1.0 + self.g_step) ** t_q) * C_agg_growth_damped

        # 12. Laspeyres mu update (Eq. 30)
        mu_new  = self.mu * float(
            np.dot(P_new, self.C) / max(np.dot(self.P, self.C), 1e-30))
        mu_new  = max(mu_new, 1e-30)

        # 13. Optimal consumption from planner FOC (Eq. 37)
        C_star  = (np.maximum(self.alpha, 1e-30) /
                   (mu_new * np.maximum(P_new, 1e-15))) ** (1.0 / self.sigma_vec)
        C_star  = np.maximum(C_star, 0.0)

        # Target nominal consumption: Baseline APC × income
        C_nom_target = self.apc_0 * Y_new

        C_nom_actual = float(np.dot(P_new, C_star))
        if C_nom_actual > 1e-30:
            C_star = C_star * (C_nom_target / C_nom_actual)

        # Update household wealth distribution
        self.households.update_wealth(r_real)

        # 14.  Tâtonnement (fast loop) — 3 monthly sub-periods
        C_m, P_m = self._tatonnement(C_star, G_hat_i, n_iter=8)
        C_realised = C_m.sum(axis=0)

        # Nominal GDP (expenditure approach, realised)
        Y_exp  = (float(np.dot(P_new, C_realised)) +
                  float(np.dot(P_new, dK)) +
                  float(np.dot(P_new, self.G)))
        # Keep Y_new from the smooth income path (prevents explosive multiplier)
        # but cross-check with expenditure for reasonableness
        Y_new  = max(Y_new, 0.5 * Y_exp)   # floor: cannot fall below 50% of expenditure

        # 15.  Capital update
        K_new   = update_capital(self.K, dK, self.delta)

        # 16.  Preference update (Eq. 18)
        alpha_new = self._update_alpha(C_m, P_m)

        # 17.  Alpha gap (L2 norm)
        alpha_gap = float(np.linalg.norm(alpha_new - self.alpha_true))

        # 18.  Real GDP (constant Q1 prices, expenditure approach)
        #      Uses P_base (Q1 prices) to deflate current quantities
        gdp_real = float(np.dot(self.P_base, C_realised) +
                         np.dot(self.P_base, self.G) +
                         np.dot(self.P_base, dK))
        # Also compute via value-added approach for cross-check
        gdp_real_va = float(np.dot(self.va_weight_0, X_new))

        nominal_gdp = float(np.dot(P_new, C_realised) +
                             np.dot(P_new, self.G) +
                             np.dot(P_new, dK))

        # ---- Per-household consumption (for welfare comparison) ----
        C_h_realised = self.households.demand(
            P_new, None, self.sigma_vec, mu_new)  # (nh, n)
        # Scale so aggregate matches realised
        C_h_total_per_sector = C_h_realised.sum(axis=0)
        scale = np.where(C_h_total_per_sector > 1e-30,
                         C_realised / C_h_total_per_sector, 1.0)
        C_h_realised = C_h_realised * scale[None, :]

        # ---- Annualise all flow quantities (x4) for output ----
        # Note: Welfare comparison uses quarterly values (Planner is quarterly).
        # Annualised vectors are used for output only —
        # internal state (self.C, self.X, ...) stays quarterly for dynamics.
        C_ann           = C_realised   * 4.0
        X_ann           = X_new        * 4.0
        G_ann           = self.G       * 4.0
        dK_ann          = dK           * 4.0
        gdp_nom_ann     = nominal_gdp  * 4.0
        gdp_real_ann    = gdp_real     * 4.0
        gdp_real_va_ann = gdp_real_va  * 4.0

        # ---- Record (all flows in annualised EUR) ----
        rec = dict(
            t           = t,
            gdp_nom     = gdp_nom_ann,
            gdp_real    = gdp_real_ann,
            gdp_real_va = gdp_real_va_ann,
            pi_agg      = pi_agg,
            r_real      = r_real,
            i_nom       = i_nom,
            C_agg       = float(np.dot(P_new, C_ann)),
            C_real      = float(np.dot(self.P_base, C_ann)),
            I_agg       = float(np.dot(P_new, dK_ann)),
            I_real      = float(np.dot(self.P_base, dK_ann)),
            G_agg       = float(np.dot(P_new, G_ann)),
            G_real      = float(np.dot(self.P_base, G_ann)),
            alpha_gap   = alpha_gap,
            C         = C_ann,                           # (n,)  annualised
            C_h         = C_h_realised,                    # (nh,n) quarterly (for welfare)
            P           = P_new.copy(),
            X           = X_ann,                           # (n,)  annualised
            K_agg       = float(K_new.sum()),
            C_m         = (np.array(C_m) * 4.0).tolist(), # monthly -> annual
            P_m         = P_m.tolist(),
            mu          = mu_new,
            Y           = Y_new * 4.0,                    # annualised nominal
        )
        self.history.append(rec)

        # ---- Advance state ----
        self.P     = P_new
        self.C     = C_realised
        self.X     = X_new
        self.K     = K_new
        self.Y     = Y_new
        self.mu    = mu_new
        self.alpha = alpha_new

        return rec

    def run(self):
        log.info(f"Running HANK model for {self.n_quarters} quarters …")
        for q in range(1, self.n_quarters + 1):
            rec = self.step()
            if q % 5 == 0 or q == 1:
                log.info(f"  Q{q:02d}  GDP={rec['gdp_nom']/1e9:.2f}B  "
                         f"pi={rec['pi_agg']*100:.3f}%  "
                         f"alpha_gap={rec['alpha_gap']:.4f}")
        return self.history


# ---------------------------------------------------------------------------
# 10.  Welfare comparison with planner
# ---------------------------------------------------------------------------

def load_planner_consumption(results_root: Path = None) -> dict | None:
    """Load per-household consumption from the latest planner checkpoint. Returns None if not found."""
    if results_root is None:
        results_root = Path(__file__).parent.parent / "Results"
    if not results_root.exists():
        log.warning("No Results/ directory found — skipping planner import.")
        return None

    # Find all timestamped run directories, pick the latest
    run_dirs = sorted(
        [d for d in results_root.iterdir() if d.is_dir() and d.name[0].isdigit()],
        key=lambda d: d.name, reverse=True)
    if not run_dirs:
        log.warning("No planner run directories found.")
        return None

    import pickle
    for run_dir in run_dirs:
        # Look for the highest-numbered checkpoint
        chks = sorted(run_dir.glob("checkpoint_q*.pkl"),
                       key=lambda p: int(p.stem.split("q")[1]), reverse=True)
        if not chks:
            continue
        try:
            with open(chks[0], "rb") as f:
                state = pickle.load(f)
            log.info(f"Loaded planner state from {chks[0]} "
                     f"({len(state.history)} quarters)")
            return {
                "history":     state.history,
                "sigma_vec":   state.sigma_vec,
                "n_households": state.n_households,
                "n":           state.n,
                "alpha_true_h": state.alpha_true_h,
                "w_h":         state.w_h,
            }
        except Exception as e:
            log.warning(f"Could not load {chks[0]}: {e}")
            continue

    log.warning("No usable planner checkpoints found.")
    return None


def _crra_utility(C_h, sigma_vec):
    """CRRA felicity per household; log utility when sigma_i == 1. Returns (nh,)."""
    C_safe = np.maximum(C_h, 1e-30)
    s = sigma_vec[None, :]   # (1, n)
    is_log = np.abs(s - 1.0) < 1e-8
    u_components = np.where(is_log,
                            np.log(C_safe),
                            C_safe ** (1.0 - s) / (1.0 - s))
    return u_components.sum(axis=1)   # (nh,)


def _ev_fraction(U_plan, U_hank, sigma_vec, C_h_hank):
    """
    Equivalent variation as a consumption fraction (EV%).
    For CES/CRRA: EV_h = (U_plan/U_hank)^{1/(1-sigma_bar)} - 1
    for the homothetic case; we use the log-linear approximation
    EV_h ≈ (U_plan - U_hank) / |dU/dlnC|, where the denominator
    is mean marginal utility times mean consumption.
    Sign-safe and scale-invariant regardless of utility level.
    """
    sigma_bar = float(sigma_vec.mean())
    # Mean marginal utility weight: E[alpha C^{-sigma}] * mean(C)
    # approximated as |U_hank| * (1 - sigma_bar) for sigma != 1,
    # or 1.0 for log utility (sigma=1), giving dU/dlnC = 1 per good.
    n = C_h_hank.shape[1]
    if abs(sigma_bar - 1.0) < 0.05:
        # Log utility: dU/dlnC_h = n (one per good, weight 1)
        scale = float(n)
    else:
        # CRRA: dU/dC * C = (1-sigma) * C^{1-sigma} / (1-sigma) = C^{1-sigma}
        # sum over goods ≈ |U_hank| * (1 - sigma_bar)  (signed, so take abs)
        scale = max(abs(float(U_hank)) * abs(1.0 - sigma_bar), 1e-30)
    return (U_plan - U_hank) / scale


def compute_welfare_comparison(hank_hist, planner_data, sigma_vec,
                               hank_households=None):
    """
    Apples-to-apples CRRA welfare comparison: planner vs HANK.

    Unit convention: HANK C_h is quarterly.
    Planner C_star / C_h_star is also quarterly. Utility is evaluated on
    these quarterly real-quantity vectors so both sides are comparable.

    Fallback distribution: uses HANK income shares (not uniform) so
    Jensen's inequality does not spuriously favour the planner.

    Per-quintile: households sorted by asset wealth (a_h).
    """
    if planner_data is None:
        return None

    planner_hist = planner_data["history"]
    n_compare = min(len(hank_hist), len(planner_hist))
    if n_compare == 0:
        return None

    results = []
    for q in range(n_compare):
        h_rec = hank_hist[q]
        p_rec = planner_hist[q]

        # --- HANK side (C_h is quarterly from step()) ---
        C_h_hank = h_rec.get("C_h")
        if C_h_hank is None:
            continue
        C_h_hank = np.maximum(np.asarray(C_h_hank, dtype=float), 1e-30)
        nh = C_h_hank.shape[0]

        # --- Planner side ---
        # Prefer stored per-household matrix (quarterly in planner)
        C_h_planner = p_rec.get("C_h_star")
        if C_h_planner is not None:
            C_h_planner = np.asarray(C_h_planner, dtype=float)
            if C_h_planner.shape[0] != nh:
                C_h_planner = None  # shape mismatch — use aggregate fallback

        if C_h_planner is None:
            # Aggregate planner consumption (quarterly, real units)
            C_star_agg = p_rec.get("C_star") or p_rec.get("C_rev")
            if C_star_agg is None:
                log.debug(f"Q{q+1}: no planner C_star found, skipping.")
                continue
            C_star_agg = np.asarray(C_star_agg, dtype=float)

            # Distribute using HANK income shares so the assumption is
            # identical on both sides (NOT uniform — uniform inflates planner
            # utility via Jensen's inequality on the concave CRRA function).
            if hank_households is not None:
                inc = hank_households.income_share[:, None]   # (nh, 1)
            else:
                # Fallback: derive shares from HANK C_h sums this quarter
                log.warning(f"Q{q+1}: hank_households not supplied; "
                            f"using C_h-derived shares (may slightly bias results).")
                c_tot = C_h_hank.sum(axis=1)                  # (nh,)
                inc   = (c_tot / max(c_tot.sum(), 1e-30))[:, None]

            C_h_planner = inc * C_star_agg[None, :]           # (nh, n) quarterly

        C_h_planner = np.maximum(np.asarray(C_h_planner, dtype=float), 1e-30)

        if C_h_planner.shape != C_h_hank.shape:
            log.warning(f"Q{q+1}: shape mismatch {C_h_planner.shape} vs "
                        f"{C_h_hank.shape}, skipping.")
            continue

        # --- CRRA utility ---
        U_hank    = _crra_utility(C_h_hank,    sigma_vec)   # (nh,)
        U_planner = _crra_utility(C_h_planner, sigma_vec)   # (nh,)

        mean_U_hank    = float(U_hank.mean())
        mean_U_planner = float(U_planner.mean())
        welfare_gain   = ((mean_U_planner - mean_U_hank)
                          / max(abs(mean_U_hank), 1e-30))

        # --- Per-wealth-quintile breakdown ---
        # Sort by asset holdings so quintiles reflect true wealth rank
        if hank_households is not None:
            wealth_rank = np.argsort(hank_households.a_h)   # ascending
        else:
            wealth_rank = np.argsort(C_h_hank.sum(axis=1)) # fallback: by C

        quintile_size = nh // 5
        type_gains = []
        for wt in range(5):
            lo  = wt * quintile_size
            hi  = lo + quintile_size if wt < 4 else nh
            idx = wealth_rank[lo:hi]
            if len(idx) == 0:
                type_gains.append(0.0)
                continue
            ug_h = float(U_hank[idx].mean())
            ug_p = float(U_planner[idx].mean())
            type_gains.append((ug_p - ug_h) / max(abs(ug_h), 1e-30))

        results.append(dict(
            t              = q + 1,
            U_hank_mean    = mean_U_hank,
            U_planner_mean = mean_U_planner,
            welfare_gain   = welfare_gain,
            type_gains     = type_gains,
        ))

    return results if results else None


# ---------------------------------------------------------------------------
# 11.  Monte Carlo runner
# ---------------------------------------------------------------------------

def run_monte_carlo(data, config, n_runs=10, n_quarters=20, n_households=1000):
    """Run `n_runs` HANK simulations with independent random seeds."""
    log.info(f"Starting {n_runs}-run Monte Carlo ensemble …")
    gdp_paths  = np.zeros((n_runs, n_quarters))
    pi_paths   = np.zeros((n_runs, n_quarters))
    c_paths    = np.zeros((n_runs, n_quarters))
    alpha_paths= np.zeros((n_runs, n_quarters))

    t0 = time.time()
    for run in range(n_runs):
        print(f"\r  HANK MC Progress: [{run+1:03d}/{n_runs}]  (Elapsed: {time.time()-t0:.1f}s)", end="", flush=True)
        model = HANKModel(data, config=config,
                          rng_seed=1000 + run,
                          n_households=n_households,
                          n_quarters=n_quarters)
        hist  = model.run()

        base_gdp = hist[0]["gdp_nom"]
        for q, rec in enumerate(hist):
            gdp_paths  [run, q] = rec["gdp_nom"]   / max(base_gdp, 1e-30) * 100
            pi_paths   [run, q] = rec["pi_agg"] * 100        # in %
            c_paths    [run, q] = rec["C_agg"]
            alpha_paths[run, q] = rec["alpha_gap"]

    print(f"\n  Monte Carlo complete in {time.time()-t0:.1f}s")

    def quantiles(arr):
        qs = [5, 15, 25, 50, 75, 85, 95]
        return {f"p{q}": np.percentile(arr, q, axis=0) for q in qs}

    return dict(
        gdp  = quantiles(gdp_paths),
        pi   = quantiles(pi_paths),
        c    = quantiles(c_paths),
        alpha= quantiles(alpha_paths),
    )


# ---------------------------------------------------------------------------
# 12.  Output / plotting
# ---------------------------------------------------------------------------

def save_results(det_hist, mc_results, sector_names, n_quarters,
                 welfare_results=None):
    log.info("Saving results …")
    quarters = list(range(1, n_quarters + 1))

    # Deterministic run GDP (nominal + real)
    base_gdp     = det_hist[0]["gdp_nom"]
    base_gdp_r   = det_hist[0]["gdp_real"]
    gdp_idx      = [r["gdp_nom"]  / base_gdp   * 100 for r in det_hist]
    gdp_real_idx = [r["gdp_real"] / base_gdp_r * 100 for r in det_hist]
    gdp_real_bn  = [r["gdp_real"] * 4 / 1e9          for r in det_hist]  # annualised
    pi_vals  = [r["pi_agg"] * 100   for r in det_hist]
    c_vals   = [r["C_agg"] / 1e9    for r in det_hist]
    ag_vals  = [r["alpha_gap"]       for r in det_hist]
    r_vals   = [r["r_real"] * 100   for r in det_hist]
    i_vals   = [r["i_nom"]  * 100   for r in det_hist]

    pd.DataFrame({"Quarter": quarters,
                  "GDP_Nom_Index": gdp_idx,
                  "GDP_Real_Index": gdp_real_idx,
                  "GDP_Real_Ann_BnEUR": gdp_real_bn}).to_csv(
        RESULTS_DIR / "hank_gdp.csv", index=False)
    pd.DataFrame({"Quarter": quarters, "Inflation_pct": pi_vals,
                  "NomRate_pct": i_vals, "RealRate_pct": r_vals}).to_csv(
        RESULTS_DIR / "hank_inflation.csv", index=False)
    pd.DataFrame({"Quarter": quarters, "Consumption_BnEUR": c_vals}).to_csv(
        RESULTS_DIR / "hank_consumption.csv", index=False)
    pd.DataFrame({"Quarter": quarters, "Alpha_Gap_L2": ag_vals}).to_csv(
        RESULTS_DIR / "hank_alpha_gap.csv", index=False)

    # Sector output
    X_mat = np.stack([r["X"] for r in det_hist])    # (T, n)
    df_x  = pd.DataFrame(X_mat, columns=sector_names)
    df_x.insert(0, "Quarter", quarters)
    df_x.to_csv(RESULTS_DIR / "hank_sector_output.csv", index=False)

    # Summary
    summary = {
        "Cumulative_Nom_GDP_Growth_%": round(gdp_idx[-1] - 100, 3),
        "Cumulative_Real_GDP_Growth_%": round(gdp_real_idx[-1] - 100, 3),
        "Mean_Quarterly_Inflation_%": round(float(np.mean(pi_vals)), 4),
        "Mean_RealRate_%": round(float(np.mean(r_vals)), 4),
        "Final_Alpha_Gap": round(float(ag_vals[-1]), 5),
        "Final_Consumption_BnEUR": round(float(c_vals[-1]), 3),
    }
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "hank_summary.csv", index=False)

    # MC quantiles
    q_labels = ["p5", "p15", "p25", "p50", "p75", "p85", "p95"]
    mc_gdp_df = pd.DataFrame(
        {q: mc_results["gdp"][q] for q in q_labels},
        index=range(1, n_quarters + 1))
    mc_gdp_df.index.name = "Quarter"
    mc_gdp_df.to_csv(RESULTS_DIR / "hank_mc_gdp.csv")

    mc_pi_df = pd.DataFrame(
        {q: mc_results["pi"][q] for q in q_labels},
        index=range(1, n_quarters + 1))
    mc_pi_df.index.name = "Quarter"
    mc_pi_df.to_csv(RESULTS_DIR / "hank_mc_inflation.csv")

    mc_alpha_df = pd.DataFrame(
        {q: mc_results["alpha"][q] for q in q_labels},
        index=range(1, n_quarters + 1))
    mc_alpha_df.index.name = "Quarter"
    mc_alpha_df.to_csv(RESULTS_DIR / "hank_mc_alpha_gap.csv")

    # Welfare comparison (if available)
    if welfare_results is not None:
        wf_df = pd.DataFrame(welfare_results)
        # Expand per-type gains into separate columns
        for wt in range(5):
            wf_df[f"Type{wt}_Welfare_Gain"] = [
                r["type_gains"][wt] for r in welfare_results]
        wf_df.drop(columns=["type_gains"], inplace=True)
        wf_df.to_csv(RESULTS_DIR / "hank_welfare_comparison.csv", index=False)
        log.info(f"  Welfare comparison: {len(welfare_results)} quarters")
        log.info(f"  Mean welfare gain (planner vs HANK): "
                 f"{np.mean([r['welfare_gain'] for r in welfare_results])*100:.3f}%")

    log.info(f"Results written to {RESULTS_DIR}/")


def plot_results(det_hist, mc_results, n_quarters, welfare_results=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        log.warning("matplotlib not available — skipping plots")
        return

    quarters = np.arange(1, n_quarters + 1)
    base_gdp = det_hist[0]["gdp_nom"]
    gdp_idx  = np.array([r["gdp_nom"] / base_gdp * 100 for r in det_hist])
    pi_vals  = np.array([r["pi_agg"] * 100 for r in det_hist])
    ag_vals  = np.array([r["alpha_gap"] for r in det_hist])
    i_vals   = np.array([r["i_nom"]  * 100 for r in det_hist])
    r_vals   = np.array([r["r_real"] * 100 for r in det_hist])

    BLUE  = "#1f6aa5"
    LBLUE = "#a8dadc"
    GREY  = "#aaaaaa"
    DARK  = "#264653"

    # ----- Figure 1: Monte Carlo GDP -----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for lo, hi, alpha in [("p5","p95",0.15),("p15","p85",0.25),("p25","p75",0.35)]:
        ax.fill_between(quarters,
                        mc_results["gdp"][lo], mc_results["gdp"][hi],
                        color=LBLUE, alpha=alpha)
    ax.plot(quarters, mc_results["gdp"]["p50"], color=BLUE, lw=2.5, label="HANK Median")
    ax.plot(quarters, gdp_idx, color=DARK, lw=1.5, ls="--", label="Deterministic")
    ax.axhline(100, color=GREY, lw=0.8, ls=":")
    ax.set_xlabel("Quarter"); ax.set_ylabel("Index (Q1 = 100)")
    ax.set_title("Real GDP Level: Monte Carlo Ensemble", fontsize=13)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "hank_mc_gdp.png", dpi=150)
    plt.close(fig)

    # ----- Figure 2: Monte Carlo Inflation -----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for lo, hi, alpha in [("p5","p95",0.15),("p15","p85",0.25),("p25","p75",0.35)]:
        ax.fill_between(quarters,
                        mc_results["pi"][lo], mc_results["pi"][hi],
                        color="#457b9d", alpha=alpha)
    ax.plot(quarters, mc_results["pi"]["p50"], color="#1d3557", lw=2, label="Median")
    ax.plot(quarters, pi_vals, color=DARK, lw=1.5, ls="--", label="Deterministic")
    ax.axhline(0, color=GREY, lw=0.8, ls=":")
    ax.set_xlabel("Quarter"); ax.set_ylabel("Quarterly Inflation (%)")
    ax.set_title("Inflation Rate: Monte Carlo Ensemble", fontsize=13)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "hank_mc_inflation.png", dpi=150)
    plt.close(fig)

    # ----- Figure 3: Alpha Gap -----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for lo, hi, a_ in [("p5","p95",0.15),("p15","p85",0.20),("p25","p75",0.25)]:
        ax.fill_between(quarters,
                        mc_results["alpha"][lo], mc_results["alpha"][hi],
                        color="#6a040f", alpha=a_)
    ax.plot(quarters, mc_results["alpha"]["p50"], color="#370617",
            lw=2, label="Median")
    ax.plot(quarters, ag_vals, color=DARK, lw=1.5, ls="--", label="Deterministic")
    ax.set_xlabel("Quarter"); ax.set_ylabel("L2-norm of Preference Gap")
    ax.set_title("Preference Drift: Monte Carlo Ensemble", fontsize=13)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "hank_alpha_gap.png", dpi=150)
    plt.close(fig)

    # ----- Figure 4: Interest rates (deterministic) -----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(quarters, i_vals, color=BLUE,  lw=2, label="Nominal rate (Taylor)")
    ax.plot(quarters, r_vals, color="#2a9d8f", lw=2, label="Real rate (Fisher)")
    ax.plot(quarters, pi_vals, color=DARK, lw=1.5, ls="--", label="Inflation")
    ax.axhline(0, color=GREY, lw=0.8, ls=":")
    ax.set_xlabel("Quarter"); ax.set_ylabel("Rate (%)")
    ax.set_title("Interest Rates and Inflation", fontsize=13)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "hank_rates.png", dpi=150)
    plt.close(fig)

    log.info(f"Figures saved to {FIGURES_DIR}/")

    # ----- Figure 5: Welfare comparison (if available) -----
    if welfare_results is not None:
        wq = np.array([r["t"] for r in welfare_results])
        wg = np.array([r["welfare_gain"] * 100 for r in welfare_results])
        type_labels = ["Low-wealth", "Mid-low", "Middle", "Mid-high", "High-wealth"]
        # Standardized palette
        type_colors = [BLUE, "#457b9d", "#a8dadc", "#2a9d8f", DARK]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: aggregate welfare gain
        ax1.bar(wq, wg, color=BLUE, alpha=0.6, edgecolor=BLUE)
        ax1.axhline(0, color="black", lw=0.8)
        ax1.set_xlabel("Quarter")
        ax1.set_ylabel("Welfare Gain (%)")
        ax1.set_title("Planner vs HANK: Aggregate Gain", fontsize=12)

        # Right: per-type welfare gain (final quarter)
        final = welfare_results[-1]
        tg = [g * 100 for g in final["type_gains"]]
        bars = ax2.bar(type_labels, tg, color=type_colors, alpha=0.8,
                       edgecolor="white")
        ax2.axhline(0, color=GREY, lw=0.8, ls=":")
        ax2.set_ylabel("Welfare Gain (%)")
        ax2.set_title(f"Per-Type Welfare Gain (Q{final['t']})")
        ax2.tick_params(axis="x", rotation=15)

        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "hank_welfare_comparison.png", dpi=150)
        plt.close(fig)
        log.info(f"Welfare comparison figure saved.")


# ---------------------------------------------------------------------------
# 13.  Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    # Load config
    cfg_path = DATA_DIR / "config.json"
    with open(cfg_path) as f:
        raw_cfg = json.load(f)
    config = {k: v for k, v in raw_cfg.items() if not k.startswith("//")}

    N_QUARTERS   = int(config.get("n_quarters", 20))
    # Terminal input for runs
    try:
        val = input(f"\nEnter number of HANK Monte Carlo runs (config default: {config.get('n_runs', 250)}): ").strip()
        N_RUNS = int(val) if val else int(config.get("n_runs", 250))
    except ValueError:
        print("Invalid input, using config default.")
        N_RUNS = int(config.get("n_runs", 10))

    N_HOUSEHOLDS = int(config.get("n_households", 1000))
    HH_DISP      = float(config.get("hh_dispersion", 0.05))
    SIGMA_VAL    = float(config.get("sigma_val", 1.0))

    # Load data
    data = load_spanish_data(DATA_DIR)

    # ---- Deterministic run ----
    log.info("=== Deterministic HANK run ===")
    det_model = HANKModel(data, config=config,
                          rng_seed=42,
                          n_households=N_HOUSEHOLDS,
                          hh_dispersion=HH_DISP,
                          n_quarters=N_QUARTERS)
    det_hist = det_model.run()

    base_gdp = det_hist[0]["gdp_nom"]
    base_gdp_r = det_hist[0]["gdp_real"]
    log.info(f"Deterministic summary:")
    log.info(f"  Nominal GDP growth : {det_hist[-1]['gdp_nom']/base_gdp*100-100:.2f}%")
    log.info(f"  Real GDP growth    : {det_hist[-1]['gdp_real']/base_gdp_r*100-100:.2f}%")
    log.info(f"  Mean quarterly infl.  : {np.mean([r['pi_agg'] for r in det_hist])*100:.4f}%")
    log.info(f"  Final alpha gap (L2)  : {det_hist[-1]['alpha_gap']:.5f}")

    # ---- Welfare comparison with planner ----
    log.info("\n=== Loading planner consumption data ===")
    planner_data    = load_planner_consumption()
    sigma_vec       = np.full(det_model.n, SIGMA_VAL)
    welfare_results = compute_welfare_comparison(det_hist, planner_data, sigma_vec,
                                                  hank_households=det_model.households)

    if welfare_results:
        log.info(f"Welfare comparison computed for {len(welfare_results)} quarters.")
        final_wg = welfare_results[-1]["welfare_gain"]
        log.info(f"  Final-quarter aggregate welfare gain (planner vs HANK): "
                 f"{final_wg*100:+.3f}%")
    else:
        log.info("  No planner data available — welfare comparison skipped.")

    # ---- Monte Carlo ensemble ----
    print(f"--- Running HANK Monte Carlo Ensemble ({N_RUNS} runs) ---")
    mc = run_monte_carlo(data, config,
                         n_runs=N_RUNS,
                         n_quarters=N_QUARTERS,
                         n_households=N_HOUSEHOLDS)


    # ---- Save + plot ----
    save_results(det_hist, mc, data["sector_names"], N_QUARTERS,
                 welfare_results=welfare_results)
    plot_results(det_hist, mc, N_QUARTERS,
                 welfare_results=welfare_results)

    # ---- Real GDP comparison plot (Planner vs HANK) ----
    if planner_data is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            planner_hist = planner_data["history"]
            n_compare = min(len(det_hist), len(planner_hist))
            quarters = np.arange(1, n_compare + 1)

            # HANK real GDP: already annualised in rec (x4 applied in step())
            hank_gdp = np.array([det_hist[q]["gdp_real"] / 1e9
                                 for q in range(n_compare)])

            # Planner real GDP: try common key names; planner stores quarterly EUR
            def _planner_gdp(rec):
                for k in ("gdp_real", "GDP_real", "GDP", "real_gdp"):
                    if k in rec:
                        # Multiply by 4.0 to annualise for the plot
                        return float(rec[k]) * 4.0 / 1e9
                return None

            planner_gdp_raw = [_planner_gdp(planner_hist[q]) for q in range(n_compare)]
            if any(v is None for v in planner_gdp_raw):
                log.warning("Planner history missing GDP key — skipping comparison plot.")
                raise ValueError("no planner GDP")
            planner_gdp = np.array(planner_gdp_raw, dtype=float)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

            # Left: absolute levels
            ax1.plot(quarters, planner_gdp, color="#2e8b57", lw=2.5,
                     marker="o", ms=4, label="Planner (Real GDP)")
            ax1.plot(quarters, hank_gdp, color="#1f6aa5", lw=2.5,
                     marker="s", ms=4, label="HANK (Real GDP)")
            ax1.set_xlabel("Quarter")
            ax1.set_ylabel("Annualised Real GDP (B EUR)")
            ax1.set_title("Real GDP: Planner vs HANK")
            ax1.legend(fontsize=10)
            ax1.grid(alpha=0.25, ls="--")

            # Right: indexed (Q1 = 100)
            hank_idx = hank_gdp / hank_gdp[0] * 100
            plan_idx = planner_gdp / planner_gdp[0] * 100
            ax2.plot(quarters, plan_idx, color="#2e8b57", lw=2.5,
                     marker="o", ms=4, label="Planner")
            ax2.plot(quarters, hank_idx, color="#1f6aa5", lw=2.5,
                     marker="s", ms=4, label="HANK")
            ax2.axhline(100, color="#aaaaaa", lw=0.8, ls=":")
            ax2.set_xlabel("Quarter")
            ax2.set_ylabel("Index (Q1 = 100)")
            ax2.set_title("Real GDP Index: Planner vs HANK")
            ax2.legend(fontsize=10)
            ax2.grid(alpha=0.25, ls="--")

            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "hank_vs_planner_real_gdp.png", dpi=200)
            plt.close(fig)

            # Also save the comparison data as CSV
            pd.DataFrame({
                "Quarter": list(range(1, n_compare + 1)),
                "HANK_Real_GDP_Ann_BnEUR": hank_gdp.tolist(),
                "Planner_Real_GDP_Ann_BnEUR": planner_gdp.tolist(),
                "HANK_Index": hank_idx.tolist(),
                "Planner_Index": plan_idx.tolist(),
            }).to_csv(RESULTS_DIR / "hank_vs_planner_gdp.csv", index=False)

            log.info(f"\n=== Real GDP Comparison ===")
            log.info(f"  Planner Q1: {planner_gdp[0]:.1f} B EUR  "
                     f"Q{n_compare}: {planner_gdp[-1]:.1f} B EUR  "
                     f"({plan_idx[-1]-100:+.2f}%)")
            log.info(f"  HANK    Q1: {hank_gdp[0]:.1f} B EUR  "
                     f"Q{n_compare}: {hank_gdp[-1]:.1f} B EUR  "
                     f"({hank_idx[-1]-100:+.2f}%)")
            log.info(f"  Figure saved to {FIGURES_DIR / 'hank_vs_planner_real_gdp.png'}")

        except Exception as e:
            log.warning(f"Could not generate GDP comparison plot: {e}")
    else:
        log.info("No planner data — skipping GDP comparison plot.")

    log.info("Done.")