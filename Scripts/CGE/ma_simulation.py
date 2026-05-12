"""
ma_simulation.py
-----------------
Multi-Agent CGE simulation engine.

Implements the main simulation loop following the 10-step quarterly execution
order specified in the implementation plan.
"""

import numpy as np
import scipy.sparse as sp
import logging
from typing import Dict, Any, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CGE.ma_calibration import MACGEState
from CGE.agents import Firm, Household
from julia_bridge import CORE, MACGE_CORE, _to_dense

logger = logging.getLogger(__name__)


def run_quarter_ma(state: MACGEState) -> MACGEState:
    """
    Execute one quarter of the MA-CGE simulation.
    
    Follows the 10-step execution order from the implementation plan.
    
    Parameters:
    -----------
    state : MACGEState
        Current simulation state
        
    Returns:
    --------
    state : MACGEState
        Updated simulation state
    """
    
    # Step 1: HABIT FORMATION
    state.alpha_habit_h = (state.habit_persistence * state.alpha_habit_h + 
                          (1 - state.habit_persistence) * state.alpha_true_h)
    state.alpha_slow_h = state.alpha_habit_h.copy()
    
    # Step 2: GOVERNMENT EXPENDITURE UPDATE
    state.G = state.G * (1 + state.g_step)
    
    # Step 3: SUPPLY-SIDE: CES PRODUCTION
    # This will be handled by the Julia equilibrium solver
    
    # Step 4: MARKUP UPDATE (pre-tatonnement)
    # This will be handled by the Julia equilibrium solver
    
    # Step 5: MARGINAL COST & PRICE PROPOSAL
    # This will be handled by the Julia equilibrium solver
    
    # Step 6: HOUSEHOLD INCOME
    # Compute firm profits from previous period
    profits = np.zeros(state.n_firms)
    for f in range(state.n_firms):
        # Simplified profit calculation
        revenue = np.dot(state.P, state.K_firms[f, :])
        costs = (state.wage_rate * np.dot(state.l_vec, state.K_firms[f, :]) +
                np.dot(state.r_vec, state.K_firms[f, :]))
        profits[f] = max(revenue - costs, 0)
    
    # Update household incomes
    for h in range(state.n_households):
        labor_income = state.w_h[h] * state.L_total / state.n_households
        capital_income = np.dot(state.W_ownership[h, :], profits)
        state.Y = labor_income + capital_income
    
    # Step 7: WALRASIAN TATONNEMENT (Julia: solve_market_equilibrium)
    # Call Julia equilibrium solver
    P_new, X_new, K_new, markup_new = MACGE_CORE.solve_market_equilibrium(
        state.P,
        state.K_firms,
        state.K,
        _to_dense(state.A),
        _to_dense(state.B),
        state.l_vec,
        state.gamma_vec,
        state.rho_ces,
        state.sigma_vec,
        state.markup,
        state.alpha_h,
        np.full(state.n_households, state.Y / state.n_households),
        state.w_h,
        state.G,
        state.delta,
        state.phi_invest,
        state.wage_rate,
        state.r_vec
    )
    
    # Update state with equilibrium results
    state.P = P_new
    state.X = X_new
    state.K = K_new
    state.markup = markup_new
    
    # Step 8: PREFERENCE DRIFT (Julia: fast_loop reused)
    # Use existing fast_loop from julia_bridge for preference evolution
    alpha_evolved, C_monthly, P_monthly = CORE.fast_loop(
        state.alpha_h,
        state.alpha_slow_h,
        state.alpha_habit_h,
        state.P,
        state.Y / state.n_households,
        state.w_h,
        state.pref_drift_rho,
        state.pref_drift_sigma,
        state.pref_noise_sigma,
        state.theta_drift,
        state.rho_M,
        state.t
    )
    
    state.alpha_h = alpha_evolved
    state.C_monthly = C_monthly
    state.P_monthly = P_monthly
    
    # Step 9: INVESTMENT & CAPITAL UPDATE
    # Compute Tobin's q and investment
    q_vec = np.zeros(state.n)
    I_vec = np.zeros(state.n)
    
    for i in range(state.n):
        # Marginal product of capital
        if state.K[i] > 0:
            MPK_i = state.gamma_vec[i] * (state.X[i] / state.K[i]) ** (1.0 - state.rho_ces[i])
            q_vec[i] = (state.P[i] * MPK_i) / (state.r_vec[i] + state.delta)
        else:
            q_vec[i] = 1.0
        
        # Investment
        I_vec[i] = (state.delta * state.K[i] + 
                   state.phi_invest[i] * state.K[i] * max(q_vec[i] - 1.0, 0.0))
    
    # Update capital (simplified - proportional distribution across firms)
    total_investment = np.sum(I_vec)
    savings = state.Y - np.dot(state.P, state.C) - np.dot(state.P, state.G)
    total_investment = min(total_investment, max(savings, 0))
    
    # Distribute investment proportionally
    if total_investment > 0:
        investment_shares = I_vec / total_investment
        for f in range(state.n_firms):
            state.K_firms[f, :] = ((1 - state.delta) * state.K_firms[f, :] + 
                                  investment_shares * total_investment / state.n_firms)
    
    state.K = np.sum(state.K_firms, axis=0)
    
    # Step 10: METRICS & HISTORY
    # Compute aggregate consumption
    state.C = np.sum([Household(h, state.Y, state.alpha_h[h, :], state.sigma_vec)
                     .demand(state.P, state.Y / state.n_households) 
                     for h in range(state.n_households)], axis=0)
    
    # Real GDP (using Q1 prices as deflator)
    P_Q1 = state.history[0]['P'] if state.history else state.P
    real_gdp = np.dot(P_Q1, state.C + I_vec + state.G)
    
    # Inflation
    if state.t > 0:
        prev_C = state.history[-1]['C']
        inflation = (np.dot(state.P, prev_C) / np.dot(state.history[-1]['P'], prev_C) - 1)
    else:
        inflation = 0.0
    
    # Update CPI
    if state.t > 0:
        state.CPI_chained *= (1 + inflation)
    
    # Calculate additional metrics
    avg_markup = np.mean(state.markup)
    markup_dispersion = np.std(state.markup)
    tobin_q_mean = np.mean(q_vec)
    
    # Record history
    history_entry = {
        't': state.t,
        'GDP': real_gdp,
        'Real_AD': np.dot(P_Q1, state.C),
        'Y': state.Y,
        'Inflation': inflation,
        'Inflation_Geom': state.CPI_chained - 1,
        'Gross_Output_Real': np.dot(P_Q1, state.X),
        'C_real': np.dot(P_Q1, state.C),
        'C_realized': np.dot(state.P, state.C),
        'I_gross_real': np.dot(P_Q1, I_vec),
        'I_net_real': np.dot(P_Q1, I_vec - state.delta * state.K),
        'G_real': np.dot(P_Q1, state.G),
        'K_val_Q1': np.dot(P_Q1, state.K),
        'slack_val_Q1': np.dot(P_Q1, state.K) - np.dot(P_Q1, state.X),
        'labor_slack': state.L_total - np.dot(state.l_vec, state.X),
        'I_pct_GDP': np.dot(P_Q1, I_vec) / real_gdp,
        'alpha_gap': np.linalg.norm(state.alpha_h - state.alpha_true_h),
        'alpha_gap_linf': np.max(np.abs(state.alpha_h - state.alpha_true_h)),
        'price_drift': np.std(np.diff(state.P)) if state.t > 0 else 0.0,
        'CPI': state.CPI_chained,
        # MA-CGE specific additions
        'avg_markup': avg_markup,
        'markup_dispersion': markup_dispersion,
        'tobin_q_mean': tobin_q_mean,
        'P': state.P.copy(),
        'C': state.C.copy(),
        'X': state.X.copy(),
        'K': state.K.copy()
    }
    
    state.history.append(history_entry)
    
    # Increment time
    state.t += 1
    
    logger.info(f"Quarter {state.t-1} completed: GDP={real_gdp:.2f}, Inflation={inflation:.4f}")
    
    return state


def run_simulation_ma(state: MACGEState, n_quarters: int, 
                     checkpoint_freq: Optional[int] = None) -> MACGEState:
    """
    Run the MA-CGE simulation for multiple quarters.
    
    Parameters:
    -----------
    state : MACGEState
        Initial calibrated state
    n_quarters : int
        Number of quarters to simulate
    checkpoint_freq : int, optional
        Frequency of checkpointing (every N quarters)
        
    Returns:
    --------
    state : MACGEState
        Final simulation state
    """
    
    logger.info(f"Starting MA-CGE simulation for {n_quarters} quarters")
    logger.info(f"Initial state: {state.n} sectors, {state.n_households} households, {state.n_firms} firms")
    
    for quarter in range(n_quarters):
        state = run_quarter_ma(state)
        
        # Checkpointing
        if checkpoint_freq and (quarter + 1) % checkpoint_freq == 0:
            logger.info(f"Checkpoint at quarter {quarter + 1}")
            # Could add checkpoint saving here
    
    logger.info(f"MA-CGE simulation completed: {len(state.history)} quarters simulated")
    return state


def run_montecarlo_ma(n_runs: int, n_quarters: int, data_dir=None, seed_base=1000):
    """
    Run Monte Carlo simulations of the MA-CGE model.
    
    Parameters:
    -----------
    n_runs : int
        Number of Monte Carlo runs
    n_quarters : int
        Quarters per run
    data_dir : str or Path, optional
        Data directory
    seed_base : int
        Base seed for reproducibility
        
    Returns:
    --------
    results : list
        List of simulation results
    """
    
    from CGE.ma_calibration import calibrate_ma
    
    results = []
    
    for run in range(n_runs):
        seed = seed_base + run
        logger.info(f"Monte Carlo run {run + 1}/{n_runs} with seed {seed}")
        
        # Calibrate
        state = calibrate_ma(data_dir=data_dir, seed=seed)
        
        # Run simulation
        state = run_simulation_ma(state, n_quarters)
        
        results.append(state.history)
    
    logger.info(f"Monte Carlo study completed: {n_runs} runs")
    return results
