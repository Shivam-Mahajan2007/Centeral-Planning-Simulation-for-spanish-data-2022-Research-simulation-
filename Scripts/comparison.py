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
from engine.simulation import run_quarter

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)

class WelfareTrajectoryCollector:
    def __init__(self, n_runs, n_q):
        self.n_runs = n_runs
        self.n_q = n_q
        self.success_count = 0
        self.welfare_oracle = np.full((n_runs, n_q), np.nan)
        self.welfare_learn = np.full((n_runs, n_q), np.nan)
        self.welfare_loss = np.full((n_runs, n_q), np.nan)

    def add_run(self, idx, oracle_traj, learn_traj):
        self.success_count += 1
        self.welfare_oracle[idx, :] = oracle_traj
        self.welfare_learn[idx, :] = learn_traj
        # Welfare loss %: (Oracle - Learning) / abs(Oracle) * 100
        self.welfare_loss[idx, :] = (oracle_traj - learn_traj) / np.abs(oracle_traj) * 100.0

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

def calculate_welfare(state, alpha_true_h=None):
    n_h = state.n_households
    sig = state.sigma_vec
    w_h = state.w_h
    a_h = alpha_true_h if alpha_true_h is not None else state.alpha_true_h
    P = state.P
    mu = state.mu_planner * 3.0
    
    C_h_star = (w_h[:, None] * a_h / (mu * P[None, :])) ** (1.0 / sig[None, :])
    
    welfare = 0.0
    for h in range(n_h):
        # Contribution per household
        h_welfare = 0.0
        for i in range(state.n):
            c_val = max(C_h_star[h, i], 1e-10)
            if abs(sig[i] - 1.0) < 1e-6:
                val = a_h[h, i] * np.log(c_val)
            else:
                val = a_h[h, i] * (c_val ** (1.0 - sig[i])) / (1.0 - sig[i])
            h_welfare += val
        welfare += w_h[h] * h_welfare
    return welfare

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

def run_ensemble(n_runs=20, n_quarters=8):
    DATA_DIR = SCRIPTS_DIR.parent / "Data"
    data = load_data(DATA_DIR)
    
    config = {
        "delta": 0.0125,
        "primal_tol": 1e-3,
        "dual_tol": 1e-4,
        "n_households": 100, 
        "n_firms": 50,
    }
    
    out_dir = SCRIPTS_DIR.parent / "Results" / "Comparison" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    collector = WelfareTrajectoryCollector(n_runs, n_quarters)
    plotter = ProfessionalPlotter(out_dir)
    
    print(f"Starting Welfare Comparison Ensemble: {n_runs} runs.")
    start_time = time.time()
    
    for i in range(n_runs):
        run_start = time.time()
        try:
            # Calibrate base
            state_base = calibrate(data, **config)
            
            # Learning Run
            state_learn = deepcopy(state_base)
            state_learn.rng = np.random.default_rng(2000 + i)
            w_learn = []
            for q in range(n_quarters):
                run_quarter(state_learn)
                w_learn.append(calculate_welfare(state_learn))
            
            # Oracle Run
            state_oracle = deepcopy(state_base)
            state_oracle.rng = np.random.default_rng(2000 + i)
            w_oracle = []
            for q in range(n_quarters):
                state_oracle.alpha = get_oracle_belief(state_oracle)
                run_quarter(state_oracle)
                w_oracle.append(calculate_welfare(state_oracle))
            
            collector.add_run(i, np.array(w_oracle), np.array(w_learn))
            
            elapsed = time.time() - run_start
            print(f"  Run {i+1}/{n_runs} complete ({elapsed:.1f}s). Loss Q{n_quarters}: {collector.welfare_loss[i, -1]:.4f}%")
            
            if (i+1) % 5 == 0:
                plotter.plot_fan(collector.welfare_loss, "Efficiency Loss vs Oracle", "Loss (%)", "welfare_loss")

        except Exception as e:
            print(f"  Run {i+1} failed: {e}")
            import traceback
            traceback.print_exc()

    plotter.plot_fan(collector.welfare_loss, "Efficiency Loss vs Oracle", "Loss (%)", "welfare_loss")
    
    avg_final = np.nanmean(collector.welfare_loss[:, -1])
    print(f"\nEnsemble Complete. Average Final Welfare Loss: {avg_final:.4f}%")
    print(f"Results saved to {out_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--quarters", type=int, default=8)
    args = parser.parse_args()
    
    run_ensemble(n_runs=args.runs, n_quarters=args.quarters)
