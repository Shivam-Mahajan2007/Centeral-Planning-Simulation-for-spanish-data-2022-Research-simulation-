# Cybernetic In-Kind Planning Simulation — Spain 2022

A computational model of a centrally planned macroeconomic, cybernetic system calibrated on the **2022 Spanish Input-Output tables** (65 sectors, ~€1.25 trillion GDP). The simulation features a hybrid architecture where the planner solves a material-balance optimisation problem each quarter using a computationally robust Adaptive Dual Ascent algorithm in Julia, while modelling deep firm-level microeconomic dynamics in Python.

---

## 1. Project Overview

The simulation models a cybernetic planning system where a central planner optimises household utility subject to physical constraints (Leontief I-O structure), sectoral capital availability, and macroeconomic labour pools. Key features include:

- **Micro-Firm Production Layer**: Capital and production are distributed across five individual micro-firms. Within each quarter, Direct Primal Linear Programming (HiGHS solver via `scipy.optimize.linprog`) allocates production targets optimally across firms subject to firm-specific capital constraints, producing realistic output dispersion and organic bottlenecks.
- **Linear Expenditure System (LES)**: Household utility is modelled via a Linear Expenditure System with fixed baseline subsistence thresholds (`gamma`) and dynamically evolving marginal budget shares (`alpha`). Preference habit-persistence tracks a rolling EMA of realised expenditure shares, and intra-quarter preference drift follows a log-space OU process toward the habit target.
- **Adaptive Dual Ascent Optimisation & Fast Tâtonnement**: A bespoke solver in Julia tackles the material-balancing problem for the macro-planner, featuring Barzilai–Borwein adaptive step sizes, Nesterov momentum, Polyak iterate averaging, and dual variable warm-starting. The fast tâtonnement loop resolves intra-quarter prices iteratively using LES demand functions.
- **Quarterly Price & Capital Dynamics**: A chained Laspeyres CPI (`CPI_chained`) responds to true demand gaps. Capital depreciates each quarter and investment is apportioned to micro-firms in proportion to their existing capital shares (sector-proportional rule).
- **Monte-Carlo Simulation**: Natively supports hundreds of stochastic trajectory runs to generate probability density functions, statistical fan charts, and Tail Spread Ratios to validate systemic robustness.
- **Sparse-Matrix Operations**: Leverages fully sparse matrix operations via `scipy.sparse` and Julia native arrays for large-scale throughput.

---

## Directory Structure

```
Data/
  Spanish_A-matrix.xlsx              — 65×65 technical-coefficient matrix
  Value_added.xlsx                   — Sectoral value added (M EUR/year)
  Consumption_and_total_production.xlsx
  config.json                        — Runtime parameters

Scripts /
  main.py          — Single baseline run
  monte_carlo.py   — 100-run stochastic ensemble with fan charts
  scenarios.py     — Parametric sweep over delta × drift
  simulation.py    — Quarterly loop orchestration
  calibration.py   — ModelState initialisation and IO calibration
  julia_bridge.py  — Python↔Julia FFI layer
  model_core.jl    — Julia kernels (solver, tatonnement, Neumann series)
  data_loader.py   — Excel parser with file-mtime cache
  plots.py         — Publication-quality diagnostic charts
  scaling.py       — Python-side solver and synthetic benchmark

Results/           — Auto-generated per run (timestamped subdirectories)
  MonteCarlo/      — Fan charts from monte_carlo.py
  scenarios/       — CSV summary and sensitivity plots
```

---

## Getting Started

**Prerequisites**: Python 3.8+, Julia 1.10+

```bash
pip install numpy scipy matplotlib pandas juliacall openpyxl
```

JuliaCall bootstraps the required Julia packages automatically on first run.

**Baseline run:**
```bash
cd "Scripts "
python main.py
```

**Monte-Carlo ensemble (100 runs):**
```bash
python monte_carlo.py
```

---

## Configuration

All parameters live in `Data/config.json`:

| Key | Default | Description |
|---|---|---|
| `n_quarters` | 20 | Quarters to simulate (20 = 5 years) |
| `delta` | 0.015 | Quarterly capital depreciation rate |
| `kappa_factor` | 4.0 | Multiplier on sector capital intensity |
| `g_step` | 0.01 | Per-quarter government expenditure growth |
| `habit_persistence` | 0.7 | EMA weight on previous habit target |
| `theta_drift` | 0.075 | Mean-reversion speed in intra-quarter OU |
| `pref_drift_sigma` | 0.01 | Preference shock volatility |
| `primal_tol` | 1e-3 | Planner primal feasibility tolerance |
| `dual_tol` | 1e-4 | Planner complementary slackness tolerance |
| `max_iter` | 2000 | Maximum dual-ascent iterations |
| `neumann_k` | 25 | Neumann series truncation depth |
| `L_total` | 39e9 | Total labour supply (hours/quarter) |

---

## Outputs

Each run writes to `Results/<timestamp>/`:

| File | Content |
|---|---|
| `01_gdp.png` | Real GDP and aggregate demand (annualised) |
| `02_output_consumption.png` | Gross output and consumption by sector group |
| `03_investment.png` | Gross investment by sector group |
| `04_shadow_prices.png` | Shadow prices `π` per sector group |
| `05_capital.png` | Capital stock at Q1 prices |
| `06_alpha_learning.png` | True vs estimated preferences |
| `07_alpha_error.png` | L2 preference estimation error |
| `08_capital_output_ratio.png` | K/Y ratio over time |
| `09_shadow_price_index.png` | CPI from chained Laspeyres |
| `10_capital_slack.png` | Committed vs idle capital (EUR, Q1 prices) |
| `11_labor_utilization.png` | Aggregate labour utilisation |
| `12_cybernetic_signals.png` | Price drift and nominal excess demand |
| `13_real_income_index.png` | Household purchasing power index |
| `14_labor_productivity.png` | Real GDP per labour hour |
| `15_growth_targets.png` | G_hat vs achieved GDP growth |
| `16_excess_demand.png` | Value-weighted sectoral excess demand |
| `17_inflation.png` | Quarterly Laspeyres inflation |
| `18_investment_gdp_ratio.png` | I/GDP with 5-year rolling average |
| `19_iterations.png` | Solver iterations per quarter |
| `20_firm_income.png` | Firm income distribution (top 10 sectors) |

---

## License

GNU General Public License v3.0 — see `LICENSE`.
