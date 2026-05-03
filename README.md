# Cybernetic In-Kind Planning Simulation — Spain 2022

> **Project Aim:** This project implements and validates a high-frequency, cybernetic planning engine calibrated to the 2022 Spanish economy. By combining state-of-the-art dual-ascent optimization with a decentralized firm-level production layer, the simulation demonstrates the technical feasibility of real-time economic coordination at an arbitrary scale (up to 10,000 sectors). The model serves as a computational "digital twin" to research the stability, resilience, and growth dynamics of hybrid planning systems under stochastic preference drift and complex production network constraints.

---

## 1. Project Overview

The simulation models a cybernetic planning system where resources are allocated based on revealed demand signals.

- **Stable Preconditioned Dual Ascent**: A state-of-the-art dual-ascent solver using **Jacobi Preconditioned Barzilai-Borwein (PBB)** step sizes and **FISTA (Beck-Teboulle)** momentum. It achieves high-precision convergence ($10^{-5}$ KKT) in ~120 iterations.
- **Vectorized Neumann Expansion**: Firm-level capital allocations are computed as a single batch operation in Julia, reducing cross-language IPC overhead by 99% and achieving a 7-10x speedup.
- **Decentralized Production (250 Firms)**: Sectoral capital is distributed across 250 independent firms. Each firm solves a local Linear Program natively via the **HiGHS Dual Simplex** solver.
- **Micro-Aggregated Demand Side**: 1,000 heterogeneous households with Linear Expenditure System (LES) utility functions.
- **Professional Diagnostic Ensemble**: The Monte Carlo engine supports 250+ stochastic runs with real-time intermediate plotting and a 100% success rate.

---

## 2. Directory Structure and Components

### [Scripts/](./Scripts) — Kernels and Orchestration
- **`model_core.jl`**: Native Julia implementation of the **Stable PBB** dual ascent solver, HiGHS-based decentralized LPs, and Walrasian tâtonnement.
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
- **GDP Level**, **Geomean Inflation**, **Capital Capacity Slack**, **Alpha Gap**, **Price Drift**, and **Solver Iterations (avg 121)**.

---

## License
GNU General Public License v3.0 — see `LICENSE`.
