import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

from data_loader import load_data, sector_groups, annualise
from calibration import calibrate
from simulation import run_simulation

DATA_DIR      = Path(__file__).parent.parent / "Data"
SCENARIOS_DIR = Path(__file__).parent.parent / "Results" / "scenarios"
SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)

_base_config = {
    "n_quarters": 20,
    "neumann_k": 20,
    "rng_seed": 42,
    "kappa_factor": 1.0,
    "L_total": 33e9,
    "wage_rate": 16.9,
    "primal_tol": 1e-4,
    "dual_tol": 1e-4,
    "eta_K": 0.15,
    "eta_L": 0.15,
    "max_iter": 2000,
    "checkpoint_every": 5,
}
CONFIG_PATH = DATA_DIR / "config.json"
if CONFIG_PATH.exists():
    try:
        with open(CONFIG_PATH, "r") as f:
            _base_config.update(json.load(f))
        logger.info(f"[scenarios] Loaded config from {CONFIG_PATH}")
    except Exception as e:
        logger.warning(f"[scenarios] Failed to load config.json: {e}. Using defaults.")

DELTAS     = _base_config.get("scenario_deltas", [0.010, 0.0125, 0.015])
DRIFTS     = _base_config.get("scenario_drifts", [0.005, 0.012, 0.020])
N_QUARTERS = _base_config["n_quarters"]


def main():
    data    = load_data(DATA_DIR)
    summary = []

    for delta in DELTAS:
        for drift in DRIFTS:
            logger.info(f"\n{'='*60}\nRunning scenario: delta={delta:.4f}, drift={drift:.4f}\n{'='*60}")
            state = calibrate(
                data, delta=delta,
                neumann_k=_base_config["neumann_k"],
                kappa_factor=_base_config["kappa_factor"],
                L_total=_base_config["L_total"],
                wage_rate=_base_config["wage_rate"],
                labor_mult=_base_config.get("labor_mult", 1.0),
                primal_tol=_base_config["primal_tol"],
                dual_tol=_base_config["dual_tol"],
                eta_K=_base_config["eta_K"],
                eta_L=_base_config["eta_L"],
                max_iter=_base_config["max_iter"],
            )
            state.rng = np.random.default_rng(_base_config.get("rng_seed", 42))
            state = run_simulation(state, n_quarters=N_QUARTERS,
                                   checkpoint_every=_base_config["checkpoint_every"])

            g0           = annualise(state.history[0]["GDP"])
            gT           = annualise(state.history[-1]["GDP"])
            total_growth = (gT / g0 - 1) * 100
            mean_alpha_err = np.mean([h["alpha_gap"] for h in state.history])

            summary.append({
                "delta":             delta,
                "drift":             drift,
                "GDP_initial_B_EUR": g0 / 1e9,
                "GDP_final_B_EUR":   gT / 1e9,
                "Total_Growth_Pct":  total_growth,
                "Mean_Alpha_Gap":    mean_alpha_err,
            })

    df = pd.DataFrame(summary)
    summary_path = SCENARIOS_DIR / "summary.csv"
    df.to_csv(summary_path, index=False)
    logger.info(f"\nSaved overview to {summary_path}")

    fig, ax = plt.subplots(figsize=(10, 6))
    for delta in DELTAS:
        sub = df[df["delta"] == delta]
        ax.plot(sub["drift"], sub["Total_Growth_Pct"], marker="o", label=f"delta={delta:.4f}")
    ax.set_xlabel("Preference Drift (sigma)")
    ax.set_ylabel("Total GDP Growth over Simulation (%)")
    ax.set_title("Scenario Sensitivity: GDP Growth vs. Preference Drift")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(SCENARIOS_DIR / "growth_sensitivity.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
