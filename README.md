# Cybernetic In-Kind Planning Simulation — Spain 2022

A computational model of a centrally planned macroeconomic, cybernetic system calibrated on the **2022 Spanish Input-Output tables** (65 sectors, ~€1.25 trillion GDP). The simulation features a hybrid architecture where the planner solves a material-balance optimization problem each quarter using a computationally robust Adaptive Dual Ascent algorithm in Julia, while mimicking deep firm-level microeconomic dynamics in Python.

---

## 1. Project Overview

The simulation models a cybernetic planning system where a central planner optimizes household utility subject to physical constraints (Leontief I-O structure), sectoral capital availability, and macroeconomic labor pools. It extends deeply into both macroeconomic orchestration and firm-level micro-heterogeneity. Key features include:

- **Micro-Firm Production Layer**: Instead of operating on monolithic sectors, capital and production are distributed across individual micro-firms. Within each quarter, Direct Primal Linear Programming is used to allocate production targets optimally across firms subject to firm-specific capital constraints, ensuring realistic output dispersion and organic bottlenecks.
- **Linear Expenditure System (LES)**: Household utility is modeled via a Linear Expenditure System, defining strict baseline subsistence consumption thresholds and dynamically evolving preference drifts for surplus income. Preference habit-persistence and moving-target OU (Ornstein-Uhlenbeck) processes evolve household behavior continuously tracking actual consumption constraints.
- **Adaptive Dual Ascent Optimization & Fast Tatonnement**: A bespoke solver engineered in Julia tackles the enormous dimensionality of real-time material balancing for the macro-planner. It features coordinate-wise adaptive learning rates and dual variable warm-starting.
- **Quarterly Price & Capital Dynamics**: A high-fidelity price discovery and chaining index (CPI/Laspeyres) responds to true demand gaps (Demand-Supply). Capital dynamically depreciates and investment is apportioned precisely to bottlenecked micro-firms via sector-proportional rules.
- **Monte-Carlo Simulation**: Natively supports hundreds of stochastic trajectory runs to generate probability density functions, statistical fan-charts, and Tail Spread Ratios to validate systemic robustness.
- **Sparse-Matrix Operations**: Capable of massive throughput and scaling by leveraging fully sparse matrix operations via `scipy` and Julia native mathematical array operations.

## 2. Directory Structure

- **[`Data/`](./Data)**: Contains the calibrated Input-Output tables (`.xlsx`) and the definitive simulation parameter configuration (`config.json`).
- **[`Scripts/`](./Scripts)**: The core Python engine and Julia kernal orchestration layer.
    - `main.py`: Primary entry point for a baseline single-run deterministic or stochastic simulation loop.
    - `monte_carlo.py`: Generates stochastic Monte Carlo trajectories of macroeconomic indicators over N runs.
    - `simulation.py`: Core orchestration of the quarterly time-step loop, updating the capital layer, LES preferences, macro-planner solver, and firm production allocations.
    - `calibration.py`: Model state initialization and dense/sparse data preparation.
    - `julia_bridge.py`: FFI boundary managing cross-language data serialization securely to the fast Julia compute node.
    - `model_core.jl`: High-performance Julia kernels (Neumann series, Adaptive Dual Ascent model optimization, and rapid fast-tatonnement loops).
    - `plots.py`: Generates multi-panel diagnostic data visualization charts.
    - `scaling.py` / `data_loader.py` / `scenarios.py`: Support matrices loading, processing, output shaping, and varied parametric sweep designs.
- **[`Results/`](./Results)**: Results directory. Automatically stores detailed PDF/PNG data visualizations, terminal checkpoints, and `MonteCarlo/` aggregation chart renderings.

## 3. Getting Started

### Prerequisites
- **Python 3.8+**
- **Julia 1.10+** — ensure `julia` is correctly accessible on your system PATH.

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd "Centeral Planning Simulation"

# 2. Install dependencies via pip
pip install numpy scipy matplotlib pandas juliacall openpyxl

# Note: The PyJulia/JuliaCall package will automatically bootstrap 
# necessary Julia environments and Julia packages dynamically on initialization.
```

## 4. Running the Simulation

**Standard Baseline Run:**
```bash
cd Scripts
python main.py
```
*(Executes a standard deterministic cycle over the designated quarter amount and saves rich visual reports under Results/.*)

**Monte-Carlo Stochastic Envelope Run:**
```bash
cd Scripts
python monte_carlo.py
```
*(Executes extensive randomized trajectory tests using stochastic solver preference volatility, saving aggregate trimmed-mean reports and statistical fan margins representing probability curves over iterations, growth, and inflation.)*

## 5. Configuration

All major constraints, behavioral hyperparameters, and solver convergence variables are maintained in `Data/config.json`:

| Key Name | Example Baseline | Description |
|---|---|---|
| `n_quarters` | 20 | Total simulated quarters (20 = 5 years span). |
| `primal_tol` / `dual_tol` | 1e-3 / 1e-4 | Precision parameters for the mathematical bounds of physical constraints. |
| `habit_persistence` | 0.90 | Moving target tracking constant tracking consumption shifts in behavior. |
| `drift_fast` | 0.018 | Intrinsic volatility (sigma) of consumer preference parameters mapping cyclic shifts. |
| `kappa_factor` | 4.0 | Multiplier on structural capital-intensity requirement. |
| `delta` | 0.0125 | Model-wide quarterly fixed depreciation rate. |
| `g_step` | 0.0125 | Baseline external government expenditure expansion bounds per quarter. |

## 6. Outputs

The simulation automatically compiles the state variables and physical constraints into distinct views per run:
- **Macroeconomic Metrics**: Real GDP, chained CPI index metrics, capital formation (Investment/GDP), and output aggregates.
- **Constraint Shadows & Slack**: Multiplier tracking to visualize the shadow-prices of absolute resource barriers and resulting idle physical factors / capital slack.
- **Consumer Behavior Gap**: Visual divergence graphs tracking intrinsic household desires (`alpha_true`) against the planner perception mechanisms (`alpha_slow`).
- **Fan Charts**: Stochastic representations charting probabilities of physical constraints generating inflation environments and capital crashes via the internal firm-layer thresholds.

## 7. License

This project is licensed under the **GNU General Public License v3.0**. See `LICENSE` for the full terms.