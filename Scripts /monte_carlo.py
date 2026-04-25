"""
monte_carlo.py
--------------
Runs the macroeconomic simulation N times with different random stochastic seeds. 
Collects solver iterations, inflation, investment/GDP, and GDP growth from each 
quarter to generate Monte-Carlo distribution graphs.
"""

import numpy as np
import logging
import json
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import skew, hmean, gmean, trim_mean

from data_loader import load_data, sector_groups
from calibration import calibrate
from simulation import run_simulation

# Reduce logging noise from the main loop
logging.basicConfig(level=logging.WARNING, format="%(message)s")


def plot_iteration_histogram(data, out_path):
    """Plot a histogram of solver iterations with convergence statistics."""
    from scipy.stats import mode as sp_mode
    
    # Flatten the 2D array (runs, quarters) into a 1D sample pool
    samples = data.flatten()
    
    mean_val = trim_mean(samples, 0.05)
    med_val  = np.median(samples)
    p95_val  = np.percentile(samples, 95)
    mode_res = sp_mode(samples, keepdims=True)
    mode_val = float(mode_res.mode[0])
    
    # Tail Spread Ratio (TS): P95 / Median
    ts_ratio = p95_val / max(med_val, 1.0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    # Using more bins for iterations as they are integers
    bins = np.arange(0, max(samples) + 10, 10)
    ax.hist(samples, bins=bins, color="#2c3e50", edgecolor="#34495e", alpha=0.8)
    
    ax.axvline(mean_val, color="#e74c3c", linestyle="dashed", linewidth=2, 
               label=f"Mean: {mean_val:.1f}")
    ax.axvline(med_val, color="#8e44ad", linestyle="dashed", linewidth=2, 
               label=f"Median: {med_val:.1f}")
    ax.axvline(p95_val, color="#f39c12", linestyle="dotted", linewidth=2, 
               label=f"95th %ile: {p95_val:.1f}")
    
    ax.set_title("Solver Convergence: Iterations per Quarter", fontsize=14, fontweight='bold')
    ax.set_xlabel("Number of Iterations")
    ax.set_ylabel("Frequency (Count of Quarters)")
    
    textstr = "\n".join((
        f"Mean:   {mean_val:.1f}",
        f"Mode:   {mode_val:.1f}",
        f"Median: {med_val:.1f}",
        f"P95:    {p95_val:.1f}",
        f"Tail Spread (P95/Med): {ts_ratio:.2f}x"
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#bdc3c7')
    ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, family='monospace')
    
    ax.legend(loc="upper right", frameon=False, bbox_to_anchor=(0.95, 0.65))
    ax.grid(True, alpha=0.15, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    
    print(f"\n--- Convergence Statistics ---")
    print(f"  Mean:       {mean_val:.2f}")
    print(f"  Mode:       {mode_val:.2f}")
    print(f"  Median:     {med_val:.2f}")
    print(f"  95th %ile:  {p95_val:.2f}")
    print(f"  Tail Spread: {ts_ratio:.2f}x (Ratio P95/Med)")
    print(f"  Saved histogram: {out_path}")


def plot_fan_chart(data, title, ylabel, out_path, start_year=2022, exclude_outliers=True):
    """Plot a Monte-Carlo fan chart with shaded confidence intervals.
    
    If exclude_outliers is True, we filter out trajectories that end in the 
    top/bottom 5% to prevent extreme paths from squashing the scale.
    """
    n_runs_orig, n_q = data.shape
    x = np.arange(1, n_q + 1)

    filtered_data = data
    if exclude_outliers and n_runs_orig > 20:
        # Sort by final value
        final_vals = data[:, -1]
        p_low  = np.percentile(final_vals, 5)
        p_high = np.percentile(final_vals, 95)
        mask = (final_vals >= p_low) & (final_vals <= p_high)
        filtered_data = data[mask]
        n_runs = filtered_data.shape[0]
        print(f"  Chart '{title}': Excluded {n_runs_orig - n_runs} outlier trajectories.")
    else:
        n_runs = n_runs_orig
    
    # Calculate statistics at each quarter on FILTERED data
    mean_val = np.array([gmean(1 + filtered_data[:, q]) - 1 if np.all(1 + filtered_data[:, q] > 0) else trim_mean(filtered_data[:, q], 0.05) for q in range(n_q)])
    med_val  = np.median(filtered_data, axis=0)
    std_val  = np.std(filtered_data, axis=0)
    p5_val   = np.percentile(filtered_data, 5, axis=0)
    p25_val  = np.percentile(filtered_data, 25, axis=0)
    p75_val  = np.percentile(filtered_data, 75, axis=0)
    p95_val  = np.percentile(filtered_data, 95, axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Shaded regions (Fan)
    ax.fill_between(x, p5_val, p95_val, color="#3498db", alpha=0.15, label="90% CI (p5-p95)")
    ax.fill_between(x, p25_val, p75_val, color="#3498db", alpha=0.30, label="IQR (p25-p75)")
    ax.fill_between(x, mean_val - std_val, mean_val + std_val, color="#3498db", alpha=0.10, label="Mean ± 1σ")
    
    # Central lines
    ax.plot(x, mean_val, color="#2c3e50", linewidth=2.5, label="Mean")
    
    # Individual trajectories (light grey for a few samples)
    for i in range(min(15, n_runs)):
        ax.plot(x, filtered_data[i, :], color="#bdc3c7", linewidth=0.5, alpha=0.4)
        
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Quarter")
    
    # Dates on X-axis (e.g. 2022 Q1)
    q_labels = []
    for i in x:
        yr = start_year + (i-1)//4
        qtr = ((i-1)%4) + 1
        q_labels.append(f"{yr} Q{qtr}")
    
    ax.set_xticks(x[::2])
    ax.set_xticklabels(q_labels[::2], rotation=45)
    
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(True, alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved fan chart: {out_path}")


def main():
    with open("Data/config.json", "r") as f:
        config = json.load(f)
    
    n_runs = 100
    n_q = 20
    config["n_quarters"] = n_q
       # Trajectory collectors (rows=runs, cols=quarters)
    traj_iterations = np.zeros((n_runs, n_q))
    traj_inflation  = np.zeros((n_runs, n_q))
    traj_ipct_gdp   = np.zeros((n_runs, n_q))
    traj_gdp_growth = np.zeros((n_runs, n_q - 1))
    traj_gdp_level  = np.zeros((n_runs, n_q))
    traj_cpi        = np.zeros((n_runs, n_q))
    traj_mad        = np.zeros((n_runs, n_q))
    
    # New metrics
    traj_alpha_gap   = np.zeros((n_runs, n_q))
    traj_price_drift = np.zeros((n_runs, n_q))
    traj_labor_slack = np.zeros((n_runs, n_q))
    traj_cap_slack   = np.zeros((n_runs, n_q))
    traj_lambda_K    = np.zeros((n_runs, n_q))
    traj_lambda_L    = np.zeros((n_runs, n_q))
    
    print(f"Running Monte-Carlo Simulation ({n_runs} runs of {n_q} quarters)...")
    
    for i in range(n_runs):
        if (i+1) % 5 == 0:
            print(f"  Completed {i+1} runs...")
        
        config["seed"] = 1000 + i
        data = load_data(Path("Data"))
        state = calibrate(
            data,
            delta=config.get("delta", 0.015),
            pref_drift_rho=config.get("pref_drift_rho", 0.95),
            pref_drift_sigma=config.get("pref_drift_sigma", 0.01),
            pref_noise_sigma=config.get("pref_noise_sigma", 0.001),
            theta_drift=config.get("theta_drift", 0.075),
            epsilon=config.get("epsilon", 0.5),
            neumann_k=config.get("neumann_k", 25),
            kappa_factor=config.get("kappa_factor", 4.0),
            L_total=config.get("L_total", 39e9),
            wage_rate=config.get("wage_rate", 21.0),
            primal_tol=config.get("primal_tol", 1e-3),
            dual_tol=config.get("dual_tol", 1e-4),
            eta_K=config.get("eta_K", 0.15),
            eta_L=config.get("eta_L", 0.15),
            max_iter=config.get("max_iter", 2000),
            g_step=config.get("g_step", 0.01),
            c_step=config.get("c_step", 0.01),
            habit_persistence=config.get("habit_persistence", 0.7),
            nominal_consumption_annual=config.get("nominal_consumption_annual", 807e9),
            labor_mult=config.get("labor_mult", 1.0),
        )
        state.slim_history = True
        
        # Mute logging
        logging.getLogger("simulation").setLevel(logging.ERROR)
        logging.getLogger("calibration").setLevel(logging.ERROR)
        
        state = run_simulation(state, n_quarters=config["n_quarters"])
        hist = state.history
        
        gdp_q1 = hist[0]["GDP"]
        
        for q in range(len(hist)):
            h = hist[q]
            traj_iterations[i, q] = h.get("iterations", 0)
            traj_inflation[i, q]  = h.get("Inflation", 0.0)
            traj_ipct_gdp[i, q]   = h.get("I_pct_GDP", 0.0)
            traj_gdp_level[i, q]  = (h["GDP"] / gdp_q1) * 100.0
            traj_cpi[i, q]        = traj_cpi[i, q-1] * (1.0 + traj_inflation[i, q]) if q > 0 else 1.0
            traj_mad[i, q]        = h.get("MAD_relative", 0.0)
            
            # Record new metrics
            traj_alpha_gap[i, q]   = h.get("alpha_gap", 0.0)
            traj_price_drift[i, q] = h.get("price_drift", 0.0)
            traj_labor_slack[i, q] = h.get("labor_slack", 0.0)
            traj_cap_slack[i, q]   = (h.get("slack_val_Q1", 0.0) / h.get("K_val_Q1", 1.0)) * 100.0
            traj_lambda_K[i, q]    = h.get("lambda_K_mean", 0.0)
            traj_lambda_L[i, q]    = h.get("lambda_L", 0.0)

            if q > 0:
                g_now = h["GDP"]
                g_prev = hist[q-1]["GDP"]
                traj_gdp_growth[i, q-1] = (g_now / g_prev - 1.0) * 100.0

    out_dir = Path("Results/MonteCarlo")
    out_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n--- Generating Fan Charts ---")
    
    charts = [
        (traj_gdp_level, "Real GDP Level (Q1=100)", "Index", "gdp_level"),
        (traj_gdp_growth, "Q-o-Q GDP Growth Rate", "Growth (%)", "gdp_growth"),
        (traj_inflation * 100, "Quarterly Inflation Rate", "Inflation (%)", "inflation"),
        (traj_ipct_gdp, "Investment / GDP Ratio", "Share (%)", "investment"),
        (traj_alpha_gap, "Preference Tracking Error (Alpha Gap)", "L2 Norm", "alpha_gap"),
        (traj_price_drift * 100, "Planner vs Market Price Drift", "Deviation (%)", "price_drift"),
        (traj_labor_slack * 100, "Labor Resource Slack", "Unused (%)", "labor_slack"),
        (traj_cap_slack, "Capital Capacity Slack", "Unused (%)", "capital_slack"),
        (traj_lambda_K, "Avg Shadow Price of Capital (Lambda K)", "Shadow Value", "lambda_k"),
        (traj_lambda_L, "Shadow Price of Labor (Lambda L)", "Shadow Value", "lambda_l"),
    ]

    for data, title, ylabel, fname in charts:
        plot_fan_chart(data, f"Monte Carlo: {title}", ylabel, out_dir / f"fan_{fname}.png")

    plot_iteration_histogram(traj_iterations, out_dir / "hist_iterations.png")

    print("\nMonte-Carlo trajectory analysis complete.")
    send_notification("Monte-Carlo Complete", f"Finished {n_runs} runs. Plots saved to {out_dir}")


def send_notification(title, message):
    import subprocess
    try:
        subprocess.run(["notify-send", title, message], check=True)
    except Exception:
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo(title, message)
            root.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    main()
