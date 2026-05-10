import os
import json
import logging
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Mitigate signal collisions
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

from data_loader import load_data
from calibration import calibrate
from simulation import run_simulation
from HANK import HANKModel, _crra_utility, load_spanish_data

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Force silence on all sub-components for clean terminal output
logging.getLogger("simulation").setLevel(logging.ERROR)
logging.getLogger("HANK").setLevel(logging.ERROR)
logging.getLogger("julia_bridge").setLevel(logging.ERROR)
logging.getLogger("data_loader").setLevel(logging.ERROR)
logging.getLogger("calibration").setLevel(logging.ERROR)

def run_mc_compare(n_runs=None, n_quarters=20):
    # Prompt for user input if not provided
    if n_runs is None:
        try:
            val = input("\nEnter number of Monte Carlo runs (e.g. 100): ").strip()
            n_runs = int(val) if val else 100
        except ValueError:
            print("Invalid input, defaulting to 10 runs.")
            n_runs = 10

    script_dir = Path(__file__).parent
    data_dir   = script_dir.parent / "Data"
    
    data = load_data(data_dir)
    
    with open(data_dir / "config.json", "r") as f:
        config = json.load(f)
    
    SIGMA_VAL = float(config.get("sigma_val", 1.0))
    sigma_vec = np.ones(len(data["sector_names"])) * SIGMA_VAL

    n_households = config.get("n_households", 1000)
    hh_disp = config.get("hh_dispersion", 0.05)
    n_firms = config.get("n_firms", 250)
    
    # ---------------------------------------------------------
    # 0. Global Distribution Structures (Stochastic Alignment)
    # ---------------------------------------------------------
    # Generate same sharing structures for both models to ensure 
    # identical initial household distribution.
    rng_dist = np.random.default_rng(12345)
    
    # a. Firm shares of global output (proportional scaling)
    firm_shares = np.abs(rng_dist.normal(1.0/n_firms, 0.02, n_firms))
    firm_shares /= firm_shares.sum()
    
    # b. Ownership Matrix Phi (n_households, n_firms)
    # Each column sums to 1 (distribution of one firm's income)
    phi_matrix = rng_dist.dirichlet(np.ones(n_households), size=n_firms).T # (n_h, n_f)
    
    print(f"\n--- Starting Joint Monte-Carlo Comparison ({n_runs} runs) ---")
    
    welfare_gains_agg = []
    welfare_ts = np.zeros((n_runs, n_quarters))
    
    start_total = time.time()
    
    # ---------------------------------------------------------
    # 1. HANK (TANK) Ensemble
    # ---------------------------------------------------------
    tank_data = load_spanish_data()  
    tank_hists = []
    for i in range(n_runs):
        print(f"\r  Progress [1/2]: HANK Ensemble execution... [{i+1:03d}/{n_runs}]", end="", flush=True)
        seed = 1000 + i
        model = HANKModel(tank_data, config=config, rng_seed=seed, 
                          n_households=n_households, hh_dispersion=hh_disp, 
                          n_quarters=n_quarters,
                          n_firms=n_firms,
                          firm_shares=firm_shares,
                          phi_matrix=phi_matrix)
        tank_hists.append(model.run())
    print("\n  HANK Ensemble complete.")

    # ---------------------------------------------------------
    # 2. Central Planner Ensemble
    # ---------------------------------------------------------
    planner_hists = []
    base_kwargs = {
        "delta": config.get("delta", 0.01),
        "pref_drift_rho": config.get("pref_drift_rho", 0.9),
        "pref_drift_sigma": config.get("pref_drift_sigma", 0.02),
        "pref_noise_sigma": config.get("pref_noise_sigma", 0.05),
        "theta_drift": config.get("theta_drift", 0.03),
        "epsilon": config.get("epsilon", 0.5),
        "neumann_k": config.get("neumann_k", 25),
        "kappa_factor": config.get("kappa_factor", 4.0),
        "L_total": config.get("L_total", 39e9),
        "wage_rate": config.get("wage_rate", 16.9),
        "primal_tol": config.get("primal_tol", 1e-3),
        "dual_tol": config.get("dual_tol", 1e-5),
        "eta_K": config.get("eta_K", 0.2),
        "eta_L": config.get("eta_L", 0.15),
        "max_iter": config.get("max_iter", 2000),
        "g_step": config.get("g_step", 0.01),
        "c_step": config.get("c_step", 0.01),
        "habit_persistence": config.get("habit_persistence", 0.9),
        "n_households": n_households,
        "hh_dispersion": hh_disp,
        "n_firms": n_firms,
        "cybernetic_k_sigma": config.get("cybernetic_k_sigma", 1.0),
        "nominal_consumption_annual": config.get("nominal_consumption_annual", 807e9),
        "firm_shares": firm_shares,
        "phi_matrix": phi_matrix
    }

    for i in range(n_runs):
        print(f"\r  Progress [2/2]: Planner Ensemble execution... [{i+1:03d}/{n_runs}]", end="", flush=True)
        seed = 1000 + i
        state = calibrate(data, **base_kwargs)
        state.rng = np.random.default_rng(seed)
        state.slim_history = True
        state.verbose = False
        
        state = run_simulation(state, n_quarters=n_quarters)
        planner_hists.append(state.history)
    print("\n  Planner Ensemble complete.")
    
    # ---------------------------------------------------------
    # 3. Post-processing Metrics
    # ---------------------------------------------------------
    print("  Computing final comparative statistics...")
    for i in range(n_runs):
        p_hist = planner_hists[i]
        h_hist = tank_hists[i]
        
        run_gains = []
        for q in range(min(n_quarters, len(p_hist), len(h_hist))):
            h_rec = h_hist[q]
            p_rec = p_hist[q]

            C_h_hank = h_rec.get("C_h")
            C_h_planner = p_rec.get("C_h_star")
            
            if C_h_hank is None or C_h_planner is None:
                continue
                
            U_hank = _crra_utility(C_h_hank, sigma_vec)
            U_planner = _crra_utility(C_h_planner, sigma_vec)

            mU_h = float(U_hank.mean())
            mU_p = float(U_planner.mean())
            
            gain = (mU_p - mU_h) / max(abs(mU_h), 1e-30) * 100
            welfare_ts[i, q] = gain
            run_gains.append(gain)
            
        welfare_gains_agg.append(np.mean(run_gains) if run_gains else 0.0)

    print(f"\nTotal comparison complete in {time.time()-start_total:.1f}s.")
    
    # ---------------------------------------------------------
    # 4. Generate Research Plots
    # ---------------------------------------------------------
    out_dir = script_dir.parent / "Results" / "MC_Welfare"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    BLUE_LINE = "#1f6aa5"
    BLUE_FAN  = "#a8dadc"
    GREY_GRID = "#e5e5e5"
    
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.grid": True,
        "grid.color": GREY_GRID,
        "axes.spines.top": False,
        "axes.spines.right": False
    })

    # Plot 1: DistributionTANK
    plt.figure(figsize=(7, 5))
    plt.hist(welfare_gains_agg, bins=max(12, n_runs//8), color=BLUE_LINE, alpha=0.7, edgecolor='white')
    plt.axvline(np.mean(welfare_gains_agg), color='red', linestyle='--', lw=2, label=f'Mean: {np.mean(welfare_gains_agg):.3f}%')
    plt.title('Aggregate Welfare Gains\n(Planner over HANK Ensemble)', fontsize=13)
    plt.xlabel('Average Quarterly Welfare Gain (%)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "welfare_distribution.png", dpi=300)
    plt.close()

    # Plot 2: Fan Chart
    plt.figure(figsize=(8, 5))
    quarters = np.arange(1, n_quarters + 1)
    w_mean = np.mean(welfare_ts, axis=0)
    w_p05 = np.percentile(welfare_ts, 5, axis=0)
    w_p95 = np.percentile(welfare_ts, 95, axis=0)
    w_p25 = np.percentile(welfare_ts, 25, axis=0)
    w_p75 = np.percentile(welfare_ts, 75, axis=0)
    
    plt.fill_between(quarters, w_p05, w_p95, color=BLUE_FAN, alpha=0.3, label='5th-95th Percentile')
    plt.fill_between(quarters, w_p25, w_p75, color=BLUE_FAN, alpha=0.6, label='25th-75th Percentile')
    plt.plot(quarters, w_mean, color=BLUE_LINE, lw=2.5, label='Mean Gain')
    plt.axhline(0, color='black', lw=1)
    plt.title('Welfare Gain Profile (20 Quarters)', fontsize=13)
    plt.xlabel('Quarter')
    plt.ylabel('Welfare Gain (%)')
    plt.legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "welfare_evolution_fan.png", dpi=300)
    plt.close()

    # Plot 3: Boxplot
    plt.figure(figsize=(9, 5.5))
    bp = plt.boxplot([welfare_ts[:, q] for q in range(n_quarters)], tick_labels=quarters, 
                patch_artist=True, showfliers=False)
    for box in bp['boxes']:
        box.set(facecolor=BLUE_FAN, alpha=0.7)
    for median in bp['medians']:
        median.set(color=BLUE_LINE, lw=2)
        
    plt.axhline(0, color='black', lw=1)
    plt.title('Quarterly Welfare Distributions', fontsize=13)
    plt.xlabel('Quarter')
    plt.ylabel('Welfare Gain (%)')
    plt.tight_layout()
    plt.savefig(out_dir / "welfare_quarterly_boxplot.png", dpi=300)
    plt.close()

    print(f"Results saved to {out_dir}")
    
    stats_df = pd.DataFrame({
        "Quarter": quarters,
        "Mean": w_mean,
        "Std": np.std(welfare_ts, axis=0),
        "p05": w_p05,
        "p25": w_p25,
        "p50": np.median(welfare_ts, axis=0),
        "p75": w_p75,
        "p95": w_p95
    })
    stats_df.to_csv(out_dir / "welfare_time_series_stats.csv", index=False)

if __name__ == "__main__":
    run_mc_compare()
