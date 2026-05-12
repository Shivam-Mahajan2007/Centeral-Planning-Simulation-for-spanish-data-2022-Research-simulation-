"""
ma_calibration.py
-----------------
Multi-Agent CGE calibration module.

Creates MACGEState with all necessary parameters for the MA-CGE simulation.
Reuses calibration anchors from the existing CIKP system for fair comparison.
"""

import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass, field
from typing import List
import logging

logger = logging.getLogger(__name__)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import load_data
from planner_scripts.calibration import _kappa, _v_per_unit, _get_structural_b_matrix
from julia_bridge import CORE, _to_dense


@dataclass
class MACGEState:
    """State container for Multi-Agent CGE simulation."""
    
    # ---- Shared with CIKP (same calibration anchors) ----
    n: int
    A: sp.csr_matrix
    B: sp.csr_matrix
    sector_names: List[str]
    sector_short: List[str]
    l_vec: np.ndarray        # labour coefficients
    v_per_unit: np.ndarray
    delta: float
    wage_rate: float
    L_total: float
    n_households: int        # = 4
    n_firms: int            # = 5
    W_ownership: np.ndarray  # (n_hh, n_firms) identical seed to CIKP
    sigma_vec: np.ndarray   # CRRA exponents
    neumann_k: int

    # ---- MA-CGE specific ----
    # Firm-level state
    gamma_vec: np.ndarray   # (n,) CES capital share
    rho_ces: np.ndarray     # (n,) CES ρ parameter
    markup: np.ndarray      # (n,) current markup μ_i
    markup_max: float       # cap on markup (default 0.5)
    eta_markup: float       # markup adjustment speed
    r_vec: np.ndarray       # (n,) rental rate of capital by sector
    phi_invest: np.ndarray  # (n,) investment sensitivity to q
    K: np.ndarray           # (n,) aggregate capital (= K_firms.sum(0))
    K_firms: np.ndarray     # (n_firms, n) firm-level capital
    X: np.ndarray           # (n,) gross output
    P: np.ndarray           # (n,) price vector

    # Household state (multi-HH, matching CIKP structure)
    alpha_h: np.ndarray      # (n_hh, n)
    alpha_slow_h: np.ndarray # (n_hh, n)
    alpha_habit_h: np.ndarray # (n_hh, n)
    alpha_true_h: np.ndarray # (n_hh, n) revealed preferences
    w_h: np.ndarray         # (n_hh,) welfare weights
    C: np.ndarray           # (n,) aggregate consumption
    C_monthly: np.ndarray   # (3, n)
    P_monthly: np.ndarray   # (3, n)
    Y: float                # aggregate income

    # Preference drift params (identical to CIKP)
    pref_drift_rho: float
    pref_drift_sigma: float
    pref_noise_sigma: float
    theta_drift: float
    habit_persistence: float

    # Simulation control
    t: int
    rng: np.random.Generator
    price_tol: float
    max_price_iter: int
    g_step: float
    G: np.ndarray
    history: list
    slim_history: bool

    # Metrics
    CPI_chained: float
    rho_M: float             # Perron-Frobenius scalar (passed to fast_loop)


def calibrate_ma(data_dir=None, seed=42, **kwargs):
    """
    Calibrate MA-CGE model from IO data.
    
    Parameters:
    -----------
    data_dir : str or Path, optional
        Directory containing IO data files
    seed : int, default 42
        Random seed for reproducible calibration
    **kwargs : dict
        Additional calibration parameters
        
    Returns:
    --------
    MACGEState : calibrated state object
    """
    
    # Step 1 — Load shared IO data
    data = load_data()
    A = sp.csr_matrix(data["A"])
    V = np.asarray(data["V_total"], dtype=float)
    sector_names = data["sector_names"]
    n = len(sector_names)
    
    # Step 2 — Recover base quantities (identical to CIKP)
    v_per_unit = _v_per_unit(sector_names)
    X_real = V / v_per_unit
    B = _get_structural_b_matrix(n, sector_names, data)
    
    # Standard calibration parameters
    delta = kwargs.get('delta', 0.025)  # 2.5% quarterly depreciation
    wage_rate = kwargs.get('wage_rate', 1.0)
    L_total = kwargs.get('L_total', 100.0)
    n_households = 4
    n_firms = 5
    neumann_k = kwargs.get('neumann_k', 20)
    
    # Labour coefficients from data
    l_vec = np.asarray(data.get("l_vec", np.ones(n) * 0.5), dtype=float)
    
    # Step 3 — Calibrate CES parameters
    kappa = _kappa(sector_names)
    gamma_vec = kappa / (kappa + v_per_unit)  # Capital share
    sigma_ces = np.full(n, kwargs.get('sigma_ces_default', 0.5))
    rho_ces = (sigma_ces - 1.0) / sigma_ces
    
    # Step 4 — Anchor prices (Leontief cost-push, same as CIKP)
    P_0 = CORE.neumann_apply(A.T.toarray(), v_per_unit, neumann_k)
    
    # Step 5 — Initial capital and rental rate
    K_0 = np.asarray(data.get("K_0", X_real * 0.8), dtype=float)  # Initial capital stock
    
    # r_i = P_i * MPK_i where MPK_i at calibration = gamma_i * (X_i/K_i)^(1-rho_i)
    MPK_0 = gamma_vec * (X_real / np.maximum(K_0, 1e-12)) ** (1.0 - rho_ces)
    r_vec = P_0 * MPK_0
    
    # Step 6 — Initial markup
    MC_0 = wage_rate * l_vec + r_vec * kappa + A.T @ P_0
    markup_max = kwargs.get('markup_max', 0.5)
    markup_0 = np.clip(P_0 / np.maximum(MC_0, 1e-12) - 1.0, 0.0, markup_max)
    
    # Step 7 — Multi-household initialisation (identical to CIKP)
    rng = np.random.default_rng(seed)
    
    # Ownership matrix (n_hh x n_firms)
    W_ownership = rng.dirichlet(np.ones(n_firms), size=n_households)
    
    # Household CRRA parameters
    sigma_vec = np.full(n, kwargs.get('crra_sigma', 2.0))
    
    # Preference weights (alpha_h)
    alpha_base = np.ones((n_households, n)) / n
    alpha_noise = rng.lognormal(mean=0.0, sigma=0.2, size=(n_households, n))
    alpha_h = alpha_base * alpha_noise
    alpha_h = alpha_h / alpha_h.sum(axis=1, keepdims=True)
    
    # Welfare weights
    w_h = np.ones(n_households) / n_households
    
    # Step 8 — Investment sensitivity
    phi_invest = np.full(n, kwargs.get('phi_invest_default', 0.1))
    
    # Step 9 — Preference drift parameters (identical to CIKP)
    pref_drift_rho = kwargs.get('pref_drift_rho', 0.95)
    pref_drift_sigma = kwargs.get('pref_drift_sigma', 0.01)
    pref_noise_sigma = kwargs.get('pref_noise_sigma', 0.05)
    theta_drift = kwargs.get('theta_drift', 0.1)
    habit_persistence = kwargs.get('habit_persistence', 0.8)
    
    # Government expenditure
    G_vec = kwargs.get('G_vec', V * 0.2)  # 20% of GDP as government spending
    g_step = kwargs.get('g_step', 0.005)   # 0.5% quarterly growth
    
    # Simulation control
    price_tol = kwargs.get('price_tol', 0.005)
    max_price_iter = kwargs.get('max_price_iter', 50)
    slim_history = kwargs.get('slim_history', False)
    
    # Initialize household preference tracking
    alpha_slow_h = alpha_h.copy()
    alpha_habit_h = alpha_h.copy()
    alpha_true_h = alpha_h.copy()
    
    # Aggregate consumption initialization
    C = np.zeros(n)
    C_monthly = np.zeros((3, n))
    P_monthly = np.zeros((3, n))
    
    # Aggregate income
    Y = wage_rate * L_total
    
    # Metrics
    CPI_chained = 1.0
    rho_M = np.max(np.linalg.eigvals(A.todense()).real)
    
    # Create state object
    state = MACGEState(
        # Shared fields
        n=n, A=A, B=B, sector_names=sector_names, sector_short=sector_names,
        l_vec=l_vec, v_per_unit=v_per_unit, delta=delta, wage_rate=wage_rate,
        L_total=L_total, n_households=n_households, n_firms=n_firms,
        W_ownership=W_ownership, sigma_vec=sigma_vec, neumann_k=neumann_k,
        
        # MA-CGE specific
        gamma_vec=gamma_vec, rho_ces=rho_ces, markup=markup_0,
        markup_max=markup_max, eta_markup=kwargs.get('eta_markup', 0.1),
        r_vec=r_vec, phi_invest=phi_invest, K=K_0.copy(),
        K_firms=np.tile(K_0 / n_firms, (n_firms, 1)),
        X=X_real.copy(), P=P_0.copy(),
        
        # Household state
        alpha_h=alpha_h, alpha_slow_h=alpha_slow_h, alpha_habit_h=alpha_habit_h,
        alpha_true_h=alpha_true_h, w_h=w_h, C=C, C_monthly=C_monthly,
        P_monthly=P_monthly, Y=Y,
        
        # Preference drift params
        pref_drift_rho=pref_drift_rho, pref_drift_sigma=pref_drift_sigma,
        pref_noise_sigma=pref_noise_sigma, theta_drift=theta_drift,
        habit_persistence=habit_persistence,
        
        # Simulation control
        t=0, rng=rng, price_tol=price_tol, max_price_iter=max_price_iter,
        g_step=g_step, G=G_vec.copy(), history=[], slim_history=slim_history,
        
        # Metrics
        CPI_chained=CPI_chained, rho_M=rho_M
    )
    
    logger.info(f"MA-CGE calibration complete: {n} sectors, {n_households} households, {n_firms} firms")
    return state
