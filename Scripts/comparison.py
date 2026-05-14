import os
import sys
import numpy as np
import logging
import time
from pathlib import Path
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# Handle Julia/Python signal collisions
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

# Setup Paths
SCRIPTS_DIR = Path(__file__).parent
sys.path.append(str(SCRIPTS_DIR))

from data.data_loader import load_data
from data.calibration import calibrate
from engine.simulation import run_quarter, run_simulation

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)

class WelfareTrajectoryCollector:
    def __init__(self, n_runs, n_q):
        self.n_runs = n_runs
        self.n_q = n_q
        self.welfare_oracle = np.full((n_runs, n_q), np.nan)
        self.welfare_learn = np.full((n_runs, n_q), np.nan)
        self.welfare_loss = np.full((n_runs, n_q), np.nan)

    def compute_loss(self):
        mask = (~np.isnan(self.welfare_oracle)) & (~np.isnan(self.welfare_learn))
        # Welfare Loss (%) = (Oracle - Learning) / |Oracle| * 100
        # Positive value = Oracle is better (expected).
        self.welfare_loss[mask] = (self.welfare_oracle[mask] - self.welfare_learn[mask]) / np.abs(self.welfare_oracle[mask]) * 100.0

class ProfessionalPlotter:
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def plot_fan(self, data, title, ylabel, filename):
        valid = data[~np.isnan(data).any(axis=1)]
        if valid.size == 0: return
        x = np.arange(1, valid.shape[1] + 1)
        pcts = [np.percentile(valid, p, axis=0) for p in [5, 15, 25, 75, 85, 95]]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.fill_between(x, pcts[0], pcts[5], color="#f0d5d5", alpha=0.9, label='90% CI')
        ax.fill_between(x, pcts[1], pcts[4], color="#cd8585", alpha=0.9, label='70% CI')
        ax.fill_between(x, pcts[2], pcts[3], color="#600b0b", alpha=0.9, label='50% CI')
        
        ax.set_title(f"Welfare Loss Ensemble: {title}", fontweight="bold", fontsize=14)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Quarter")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(self.out_dir / f"fan_{filename}.png", dpi=300)
        plt.savefig(self.out_dir / f"fan_{filename}.pdf")
        plt.close()

def calculate_welfare_from_history(h_record, sigma_vec):
    """
    Computes aggregate household welfare from a standardized history record.
    Using the pre-saved C_h_star ensures we use the correct mu-scale and 
    production constraints realization for that specific quarter.
    """
    a_h      = h_record["alpha_true_h"]  # realized household preferences
    w_h      = h_record["w_h"]           # income-share weights
    C_h_star = h_record["C_h_star"]      # realized consumption (already scaled/mu-chained)
    sig      = sigma_vec

    C_safe = np.maximum(C_h_star, 1e-12)
    is_log = np.abs(sig - 1.0) < 1e-6
    not_log = ~is_log
    
    u_mat = np.zeros_like(C_h_star)
    if np.any(not_log):
        s_vals = sig[not_log]
        u_mat[:, not_log] = a_h[:, not_log] * (C_safe[:, not_log]**(1.0 - s_vals)) / (1.0 - s_vals)
    if np.any(is_log):
        u_mat[:, is_log] = a_h[:, is_log] * np.log(C_safe[:, is_log])
        
    return np.sum(w_h[:, None] * u_mat)

def get_oracle_belief(state):
    n = state.n
    a_h = state.alpha_true_h
    w_h = state.w_h
    sig = state.sigma_vec
    a_agg = np.zeros(n)
    for i in range(n):
        a_agg[i] = (np.sum((w_h * a_h[:, i])**(1.0/sig[i])))**sig[i]
    a_agg /= a_agg.sum()
    return a_agg

def run_ensemble(n_runs=20, n_quarters=20):
    DATA_DIR = SCRIPTS_DIR.parent / "Data"
    data = load_data(DATA_DIR)
    
    config = {
        "delta": 0.0125,
        "primal_tol": 1e-3,
        "dual_tol": 1e-4,
        "n_households": 1000, 
        "n_firms": 250,
    }
    
    out_dir = SCRIPTS_DIR.parent / "Results" / "Comparison" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    collector = WelfareTrajectoryCollector(n_runs, n_quarters)
    plotter = ProfessionalPlotter(out_dir)
    
    start_time = time.time()
    
    # Silence logger for clean ensemble output
    logging.getLogger("engine.simulation").setLevel(logging.WARNING)
    
    # Phase 1: Learning Planner
    # Must use run_simulation to benefit from optimized vectorized kernels
    print(f"--- Phase 1: Learning Planner Ensemble ({n_runs} runs) ---")
    for i in range(n_runs):
        run_start = time.time()
        try:
            state = calibrate(data, **config)
            state.rng = np.random.default_rng(2000 + i)
            state.slim_history = True 
            
            # Execute full simulation
            state = run_simulation(state, n_quarters=n_quarters)
            
            # Extract welfare strictly from history records
            w_traj = [calculate_welfare_from_history(h, state.sigma_vec) for h in state.history]
            collector.welfare_learn[i, :] = w_traj
            
            elapsed = time.time() - run_start
            if (i+1) % 10 == 0 or i == 0:
                print(f"  Learning Run {i+1}/{n_runs} complete ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  Learning Run {i+1} failed: {e}")

    # Phase 2: Oracle Planner
    # Oracle uses the same logic but injects true preference belief at each step.
    print(f"\n--- Phase 2: Oracle Planner Ensemble ({n_runs} runs) ---")
    for i in range(n_runs):
        run_start = time.time()
        try:
            state_oracle = calibrate(data, **config)
            state_oracle.rng = np.random.default_rng(2000 + i)
            state_oracle.slim_history = True
            
            for q in range(n_quarters):
                # Endow planner with perfect preference knowledge
                state_oracle.alpha = get_oracle_belief(state_oracle)
                run_quarter(state_oracle)
            
            # Extract welfare strictly from history records (same path as Phase 1)
            w_traj = [calculate_welfare_from_history(h, state_oracle.sigma_vec) for h in state_oracle.history]
            collector.welfare_oracle[i, :] = w_traj
            
            elapsed = time.time() - run_start
            if (i+1) % 10 == 0 or i == 0:
                print(f"  Oracle Run {i+1}/{n_runs} complete ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  Oracle Run {i+1} failed: {e}")

    # Phase 3: Analysis
    print("\n--- Phase 3: Comparative Analysis ---")
    collector.compute_loss()
    plotter.plot_fan(collector.welfare_loss, "Efficiency Loss vs Oracle", "Loss (%)", "welfare_loss")
    
    avg_final = np.nanmean(collector.welfare_loss[:, -1])
    total_time = time.time() - start_time
    print(f"\nEnsemble Complete. Total Time: {total_time/60:.1f} mins")
    print(f"Average Final Welfare Loss: {avg_final:.4f}%")
    print(f"Results saved to {out_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--quarters", type=int, default=20)
    args = parser.parse_args()
    
    run_ensemble(n_runs=args.runs, n_quarters=args.quarters)
