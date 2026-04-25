# Cybernetic In-Kind Planning Simulation — Spain 2022

A computational model of a centrally planned macroeconomic, cybernetic system calibrated on the **2022 Spanish Input-Output tables** (65 sectors, ~€1.3 trillion GDP). The simulation features a hybrid architecture where the planner solves a material-balance optimisation problem each quarter using a computationally robust Adaptive Dual Ascent algorithm in Julia, while modelling deep firm-level microeconomic dynamics in Python.

---

## 1. Project Overview

The simulation models a cybernetic planning system where a central planner optimises household utility subject to physical constraints (Leontief I-O structure), sectoral capital availability, and macroeconomic labour pools. Key features include:

- **Micro-Firm Production Layer**: Capital and production are distributed across five individual micro-firms. Within each quarter, Direct Primal Linear Programming (HiGHS solver via `scipy.optimize.linprog`) allocates production targets optimally across firms subject to firm-specific capital constraints, producing realistic output dispersion and organic bottlenecks.
- **Linear Expenditure System (LES)**: Household utility is modelled via a Linear Expenditure System with fixed baseline subsistence thresholds (`gamma`) and dynamically evolving marginal budget shares (`alpha`). Preference habit-persistence tracks a rolling EMA of realised expenditure shares, and intra-quarter preference drift follows a log-space OU process toward the habit target.
- **Adaptive Dual Ascent Optimisation & Fast Tâtonnement**: A bespoke solver in Julia tackles the material-balancing problem for the macro-planner, featuring Barzilai–Borwein adaptive step sizes, Nesterov momentum and Polyak iterate averaging. The fast tâtonnement loop resolves intra-quarter prices iteratively using LES demand functions.
- **Quarterly Price & Capital Dynamics**: A chained Laspeyres CPI (`CPI_chained`) responds to true demand gaps. Capital depreciates each quarter and investment is apportioned to micro-firms in proportion to their existing capital shares (sector-proportional rule).
- **Monte-Carlo Simulation**: Natively supports hundreds of stochastic trajectory runs to generate probability density functions, statistical fan charts, and Tail Spread Ratios to validate systemic robustness.
- **Sparse-Matrix Operations**: Leverages fully sparse matrix operations via `scipy.sparse` and Julia native arrays for large-scale throughput.

---

## 2. Directory Structure

- **[`Data/`](./Data)**: Contains the calibrated Input-Output tables (`.xlsx`) and the simulation parameter configuration (`config.json`).
- **[`Scripts/`](./Scripts)**: The core Python engine and Julia kernel orchestration layer.
  - `main.py`: Primary entry point. Runs a single deterministic or stochastic simulation, prints a quarterly GDP table, and saves all diagnostic charts to a timestamped subfolder under `Results/`.
  - `monte_carlo.py`: Runs N stochastic trajectories (default 100, 20 quarters each) with varied RNG seeds. Saves fan charts and a convergence histogram to `Results/MonteCarlo/`.
  - `simulation.py`: Core quarterly time-step loop. Orchestrates the capital update, LES preference evolution, macro-planner solver (`solve_planner`), firm LP allocation, fast tâtonnement, and history recording.
  - `calibration.py`: Constructs the initial `ModelState` from IO data. Derives physical output (`X_real = V / v_per_unit`), builds the capital matrix `B` (diagonal `kappa` + sparse off-diagonal noise), checks spectral stability of `A_bar`, and initialises firm capital shares.
  - `julia_bridge.py`: FFI boundary managing cross-language data serialisation between Python and Julia. Exposes `solve_planner`, `fast_loop`, `compute_investment`, and `evolve_structural_alpha`.
  - `model_core.jl`: High-performance Julia kernels — Neumann series inversion, Adaptive Dual Ascent optimiser (`solve_planner`), iterative tâtonnement price clearing, and stochastic preference evolution (`fast_loop`).
  - `plots.py`: Generates 20 multi-panel diagnostic charts including GDP, aggregate demand breakdown, shadow prices, capital slack, labour utilisation, inflation, firm income distribution, and solver convergence.
  - `data_loader.py`: Reads and validates the three Spanish IO Excel files (A-matrix, value added, consumption/production). Converts from millions-of-EUR annual to EUR quarterly, and caches results as a compressed pickle (`.data_cache.pkl`) validated against source file modification times.
  - `scaling.py`: Pure-Python fallback Neumann solver and `solve_planner` implementation. Used for testing and comparison against the Julia kernel.
  - `scenarios.py`: Parametric sweep runner. Iterates over a configurable grid of `delta` × `pref_drift_sigma` values, runs a full simulation for each combination, and saves a `summary.csv` plus a GDP growth sensitivity plot to `Results/scenarios/`.
- **[`Results/`](./Results)**: Auto-created on each run. Stores timestamped PDF/PNG diagnostic charts, `run_config.json` snapshots, pickle checkpoints, and `MonteCarlo/` or `scenarios/` subdirectories.

---

## 3. Getting Started

### Prerequisites
- **Python 3.8+**
- **Julia 1.10+** — ensure `julia` is correctly accessible on your system PATH.

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd "Centeral Planning Simulation"

# 2. Install Python dependencies
pip install numpy scipy matplotlib pandas juliacall openpyxl

# Note: JuliaCall will automatically bootstrap the necessary Julia
# packages (PythonCall, LinearAlgebra, SparseArrays) on first run.
```

---

## 4. Running the Simulation

All entry points must be run from inside the `Scripts/` directory:

**Standard Baseline Run:**
```bash
cd Scripts
python main.py
```
Executes a single deterministic simulation over the configured number of quarters and saves 20 diagnostic charts to `Results/<timestamp>/`.

**Monte-Carlo Stochastic Envelope Run:**
```bash
cd Scripts
python monte_carlo.py
```
Executes 100 randomised trajectory runs, saving fan charts and a convergence histogram to `Results/MonteCarlo/`.

**Parametric Scenario Sweep:**
```bash
cd Scripts
python scenarios.py
```
Runs a grid sweep over depreciation rates and preference drift sigmas, saving `Results/scenarios/summary.csv` and a growth sensitivity plot.

---

## 5. Configuration

All major constraints, behavioural hyperparameters, and solver convergence variables are set in `Data/config.json`. Values in that file override the in-code defaults shown below.

| Key Name | Default (in-code) | Description |
|---|---|---|
| `n_quarters` | `20` | Total simulated quarters (20 = 5-year span). |
| `delta` | `0.0125` | Quarterly capital depreciation rate. |
| `primal_tol` | `1e-4` | Primal feasibility tolerance for the Dual Ascent solver. |
| `dual_tol` | `1e-4` | Dual (complementary slackness) convergence tolerance. |
| `habit_persistence` | `0.7` | EMA smoothing constant for the household preference habit target (0 = no memory, 1 = fixed). |
| `pref_drift_sigma` | `0.04` | Intra-quarter log-space OU shock magnitude on consumer preferences. |
| `pref_drift_rho` | `0.95` | AR(1) persistence of the structural preference component. |
| `pref_noise_sigma` | `0.01` | Additional white-noise sigma layered on preference shocks. |
| `theta_drift` | `0.1` | Mean-reversion speed of the OU preference process toward `alpha_slow`. |
| `kappa_factor` | `1.0` | Scalar multiplier applied to all sectoral capital-intensity coefficients (`kappa`). |
| `kappa_ou` | `0.15` | (Monte-Carlo) OU mean-reversion speed for stochastic `kappa` variation. |
| `g_step` | `0.0` | Per-quarter growth rate of government expenditure. |
| `c_step` | `0.01` | Per-quarter minimum capacity expansion target used in investment computation. |
| `neumann_k` | `20` | Truncation order for the Neumann series approximation of `(I - A)^{-1}`. |
| `eta_K` | `0.15` | Initial step-size for capital dual variables in the Adaptive Dual Ascent solver. |
| `eta_L` | `0.15` | Initial step-size for the labour dual variable. |
| `max_iter` | `2000` | Maximum Dual Ascent iterations per quarter. |
| `L_total` | `33e9` | Total labour endowment (hours). |
| `wage_rate` | `16.9` | Wage rate (EUR/hour), used to derive the sectoral labour-input vector. |
| `rng_seed` | `42` | NumPy RNG seed for reproducibility (`null` for random). |
| `checkpoint_every` | `5` | Save a state pickle every N quarters. |
| `slim_history` | `null` | If `null`, auto-enables slim (memory-reduced) history when `n > 5000`. Set `true` to force. |
| `nominal_consumption_annual` | `807e9` | Annual nominal household consumption anchor (EUR) used to pin the income scale. |

---

## 6. Outputs

The simulation compiles state variables and physical constraints into distinct views per run:

- **Macroeconomic Metrics**: Real GDP (chained, at Q1 prices), chained CPI index, Investment/GDP ratio, and gross output aggregates. A quarterly GDP table (YoY and QoQ growth) is printed to the terminal.
- **Constraint Shadows & Slack**: Shadow-price vectors (`lambda_K`, `lambda_L`) tracking the binding tightness of capital and labour constraints, plus monetary capital slack at Q1 prices.
- **Consumer Behaviour Gap**: Divergence between the planner's perceived preferences (`alpha`) and the true evolving household preferences (`alpha_true`), measured in L2 and L∞ norms each quarter.
- **Firm-Level Distribution**: Per-firm output and income across the five micro-firms, visualised as a stacked distribution chart.
- **Fan Charts** *(Monte-Carlo only)*: Stochastic probability envelopes for GDP growth, inflation, investment share, capital slack, labour slack, shadow prices, and solver convergence iterations.

---

## 7. License

This project is licensed under the **GNU General Public License v3.0**. See `LICENSE` for the full terms.
