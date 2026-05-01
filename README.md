# Cybernetic In-Kind Planning Simulation — Spain 2022

A high-performance computational model of a centrally planned macroeconomic system calibrated on the **2022 Spanish Input-Output tables** (65 sectors, ~€1.25 trillion GDP). The simulation features a hybrid **Python-Julia architecture** where a central planner optimizes household utility subject to physical constraints, while modeling decentralized production across 250 autonomous firms.

---

## 1. Project Overview

The simulation models a cybernetic planning system where resources are allocated based on revealed demand signals.

- **High-Performance Hybrid Backend**: Orchestration and calibration are handled in Python, while computationally intensive inner loops (Dual Ascent solver, Neumann expansion, and monthly tâtonnement) are offloaded to a specialized Julia core (`ModelCore.jl`).
- **Vectorized Neumann Expansion**: Firm-level capital allocations are computed as a single batch operation in Julia, reducing cross-language IPC overhead by 99% and achieving a 7-10x speedup.
- **Decentralized Production (250 Firms)**: Sectoral capital is distributed across 250 independent firms. Each firm solves a local Linear Program natively via the **HiGHS Dual Simplex** solver.
- **Micro-Aggregated Demand Side**: 1,000 heterogeneous households with Linear Expenditure System (LES) utility functions.
- **Professional Diagnostic Ensemble**: The Monte Carlo engine supports 250+ stochastic runs with real-time intermediate plotting and a 100% success rate.

---

## 2. Directory Structure and Components

### [Scripts/](./Scripts) — Kernels and Orchestration
- **`model_core.jl`**: Native Julia implementation of the Nesterov-accelerated Dual Ascent solver, HiGHS-based decentralized LPs, and Walrasian tâtonnement.
- **`julia_bridge.py`**: FFI layer using `juliacall` for low-latency data exchange.
- **`simulation.py`**: Orchestrates the quarterly loop and physical accounting identities.
- **`monte_carlo.py`**: High-performance ensemble engine with real-time fan chart generation.
- **`api.py`**: FastAPI server for dashboard telemetry.

### [App/](./App) — Interactive Research Dashboard
- **Real-Time Telemetry**: Uses Server-Sent Events (SSE) to stream simulation results from Python.
- **Dynamic Charting**: Responsive grid of charts visualizing GDP, Inflation, and Shadow Prices.
- **Parameter Control**: Interface to modify `config.json` parameters before launching runs.

### [Data/](./Data) — Benchmark Datasets
- **`Spanish_A-matrix.xlsx`**: Technical coefficients (A) for 65 sectors.
- **`Value_added.xlsx` / `Consumption_and_total_production.xlsx`**: Benchmark totals for 2022 Spanish National Accounts.
- **`config.json`**: Central runtime configuration (tolerances, shock sigmas, etc.).

---

## 3. Getting Started

### Prerequisites
- **Python 3.10+**, **Julia 1.10+**, **Node.js 18+** (for Dashboard)
- **Pip Dependencies**: `numpy`, `matplotlib`, `juliacall`, `openpyxl`, `fastapi`, `uvicorn`

```bash
# Install dependencies
pip install numpy matplotlib juliacall openpyxl fastapi uvicorn sse-starlette
```

### Running the Ensemble (CLI)
```bash
python3 Scripts/monte_carlo.py
```

### Running the Dashboard
1. Start Backend: `uvicorn Scripts.api:app --reload`
2. Start Frontend: `cd App && npm run dev`
Open `http://localhost:5174` in your browser.

---

## 4. Configuration and Outputs

Key configurations in `Data/config.json`:
- `n_runs` (250): Number of Monte Carlo realizations.
- `primal_tol` (0.001): Planner math precision.
- `pref_drift_sigma` (0.02): Volatility of stochastic preference shifts.

The ensemble generates 6 high-fidelity diagnostic charts in `Results/MonteCarlo/`:
- **GDP Level**, **Geomean Inflation**, **Capital Capacity Slack**, **Alpha Gap**, **Price Drift**, and **Solver Iterations**.

---

## License
GNU General Public License v3.0 — see `LICENSE`.
