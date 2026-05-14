# Cybernetic In-Kind Planning Simulation — Spain 2022

> **Project Aim:** This project implements and validates a high-frequency, cybernetic planning engine calibrated to the 2022 Spanish economy. By combining state-of-the-art dual-ascent optimization with a decentralized firm-level production layer, the simulation demonstrates the technical feasibility of real-time economic coordination at an arbitrary scale (up to 10,000 sectors). The model serves as a computational "digital twin" to research the stability, resilience, and growth dynamics of hybrid planning systems under stochastic preference drift and complex production network constraints.

---

## 1. Project Overview

The simulation models a cybernetic planning system where resources are allocated based on revealed demand signals.

- **Vectorized Neumann Expansion**: Firm-level capital allocations are computed as a single batch operation in Julia, reducing cross-language IPC overhead by 99% and achieving a 7-10x speedup.
=======
- **Dual Ascent Solver**: A Nesterov-accelerated dual-ascent solver operating in log-space with backtracking line search and adaptive momentum restarts. Cold-started Lagrange multipliers ensure robustness to structural shifts between quarters. Converges in ~120 iterations (~380 MVPs) on the 65-sector benchmark.
- **Multithreaded Market Clearing**: The monthly tâtonnement and household demand aggregation are parallelized via `Threads.@threads` in Julia for high-throughput execution across 1,000+ heterogeneous households.
- **Vectorized Neumann Expansion**: Firm-level capital allocations are computed as a single batch operation in Julia, reducing cross-language IPC overhead by 99%.
- **Decentralized Production (250 Firms)**: Sectoral capital is distributed across 250 independent firms. Each firm solves a local Linear Program natively via the **HiGHS Dual Simplex** solver.
- **Micro-Aggregated Demand Side**: 1,000 heterogeneous households with CRRA utility functions and stochastic preference evolution.
- **Oracle Welfare Benchmark**: A welfare comparison suite (`comparison.py`) evaluates the learning planner against a first-best Oracle with perfect preference information. Monte Carlo results confirm a welfare loss of ~0.1%.
- **Professional Diagnostic Ensemble**: The Monte Carlo engine supports 250+ stochastic runs with real-time intermediate plotting.

---

## 2. Directory Structure

```
Scripts/
├── main.py                 # Primary simulation entry point
├── comparison.py           # Monte Carlo welfare comparison (Learning vs Oracle)
├── core/
│   ├── model_core.jl       # Julia kernels: FISTA solver, tâtonnement, firm LPs
│   ├── julia_bridge.py     # Python-Julia FFI layer (juliacall)
│   └── scaling.jl          # Empirical complexity scaling study
├── data/
│   ├── data_loader.py      # IO data loading and preprocessing
│   └── calibration.py      # Model state initialization and calibration
├── engine/
│   ├── simulation.py       # Quarterly loop orchestration
│   ├── monte_carlo.py      # Ensemble engine with fan chart generation
│   └── scenarios.py        # Sensitivity analysis framework
└── analysis/
    └── plots.py            # Diagnostic plotting routines

Data/                       # 2022 Spanish National Accounts IO tables
Results/                    # Auto-generated output directories
  ├── MonteCarlo/           # Growth and stability ensemble results
  └── Comparison/           # Welfare comparison fan charts
```

---

## 3. Getting Started

### Prerequisites
- **Python 3.10+**, **Julia 1.10+**
- **Pip Dependencies**: `numpy`, `matplotlib`, `juliacall`, `openpyxl`, `scipy`

```bash
pip install numpy matplotlib juliacall openpyxl scipy
```

### Running the Simulation
```bash
# Standard simulation (uses Data/config.json)
python3 Scripts/main.py

# Monte Carlo growth ensemble (250 runs, 20 quarters)
python3 Scripts/engine/monte_carlo.py

# Welfare comparison: Learning vs Oracle (250 runs, 8 quarters)
python3 Scripts/comparison.py --runs 250 --quarters 8
```

---

## 4. Configuration and Outputs

Key configurations in `Data/config.json`:
- `n_runs` (250): Number of Monte Carlo realizations.
- `n_quarters` (20): Simulation horizon.
- `n_households` (1000): Number of heterogeneous households.
- `n_firms` (250): Number of decentralized firms.
- `primal_tol` (0.001): Planner KKT precision.
- `pref_drift_sigma` (0.02): Volatility of stochastic preference shifts.

### Diagnostic Outputs
The ensemble generates high-fidelity diagnostic charts in `Results/`:
- **GDP Level**, **Geomean Inflation**, **Capital Capacity Slack**, **Alpha Gap**, **Price Drift**, **Solver MVPs** (in `Results/MonteCarlo/`)
- **Welfare Loss Fan Chart** (in `Results/Comparison/`)

---

## License
GNU General Public License v3.0 — see `LICENSE`.
