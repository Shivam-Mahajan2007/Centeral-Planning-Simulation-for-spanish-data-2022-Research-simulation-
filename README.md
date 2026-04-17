# Cybernetic In-Kind Planning Simulation — Spain 2022

A computational model of a centrally planned economy calibrated on the **2022 Spanish Input-Output tables** (65 sectors, ~€1.25 trillion GDP). The planner solves a material-balance optimization problem each quarter using a mathematically robust, high-performance **Adaptive Dual Ascent Algorithm**, implemented as a native **Julia** kernel called from a **Python** orchestration layer via [PythonCall / JuliaCall](https://github.com/JuliaPy/PythonCall.jl).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Repository Structure](#3-repository-structure)
4. [Prerequisites](#4-prerequisites)
5. [Installation](#5-installation)
6. [Running the Simulation](#6-running-the-simulation)
7. [Configuration](#7-configuration)
8. [Outputs](#8-outputs)
9. [License](#9-license)

---

## 1. Project Overview

The simulation models a cybernetic planning system where a central planner optimizes household utility subject to technological constraints (Input-Output matrix), capital availability, and labor supply. Key features include:

- **Adaptive Dual Ascent Optimization**: Uses a customized Dual Ascent solver featuring **coordinate-wise adaptive learning rates** for ultra-fast material balancing (typically < 100 iterations per quarter).
- **Hybrid KKT Convergence**: Gated by strict mathematical limits balancing macroeconomic precision with scaling issues, guaranteeing **< 0.1% physical violations** and **10⁻⁴ complementary slackness**.
- **Dual Variable Warm-Starting**: Seamlessly passes discovered shadow prices across simulation quarters to eliminate boundary search times.
- **Preference Learning**: Implements a Moving Target Ornstein-Uhlenbeck process to model shifting household preferences (true $\alpha$), where the underlying habit baseline continuously tracks actual consumption constraints.
- **Macroeconomic Anchoring**: Anchors the initial nominal household income to **€636 Billion** (Spanish 2022 baseline), providing a precise numéraire for all value-based aggregates.
- **Cybernetic Feedback**: Quarterly "tâtonnement" price discovery loops align planned production with revealed demand, tracking productivity-led deflation via a Shadow Price Index.
- **Research-Grade Analytics**: Generates 15+ publication-quality plots spanning growth, capital utilization, shadow prices, and preference gaps.

## 2. Directory Structure

The system is organized to cleanly separate orchestration, mathematics, and output generation:

- **[`Data/`](./Data)**: Contains the calibrated Input-Output tables (`.xlsx`) and the definitive simulation parameter file (`config.json`).
- **[`Scripts/`](./Scripts)**: The core Python engine and Julia kernel.
    - `main.py`: Primary entry point for baseline simulations.
    - `scenarios.py`: Sensitivity analysis and parameter sweeps.
    - `simulation.py`: Orchestrates the quarterly time-step loop.
    - `julia_bridge.py`: FFI boundary managing cross-language data serialization.
    - `model_core.jl`: High-performance Julia kernels (Neumann series, Dual Ascent solver).
- **[`Results/`](./Results)**: Output repository for timestamped simulation results, including PNGs, PDFs, and state checkpoints.

## 3. Getting Started

### Prerequisites

- **Python 3.8+**
- **Julia 1.10+** — download from [julialang.org](https://julialang.org/downloads/) and ensure `julia` is on your PATH.

### Installation

**Linux / macOS**

```bash
# 1. Clone the repository
git clone <repo-url>
cd "Planning simulation IPM version"

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Build the Julia system image — run this ONCE, takes ~5 minutes
#    This pre-compiles the Julia kernel so the simulation starts in < 1s
julia --project=. Scripts/build_sysimage.jl

# 4. Point the bridge at the compiled image — add this to your shell profile
#    (.bashrc / .zshrc) to make it permanent
export JULIA_SYSIMAGE="$(pwd)/Scripts/model_sysimage.so"
```

**Windows (PowerShell)**

```powershell
# 1. Clone the repository
git clone <repo-url>
cd "Planning simulation IPM version"

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Verify Julia is on your PATH
julia --version

# 4. Build the Julia system image — run this ONCE, takes ~5 minutes
julia --project=. Scripts/build_sysimage.jl

# 5. Point the bridge at the compiled image for this session
$env:JULIA_SYSIMAGE = "$PWD\Scripts\model_sysimage.dll"
```

> **Why the system image?** The simulation's hot path (the Adaptive Dual Ascent solver) runs as a compiled Julia kernel. Julia compiles functions to native code on first use, adding ~40s of dead time before Q1 if no image is present. The system image persists that compiled code across runs for instant execution.

---

## 4. Running the Simulation

### Baseline Run

```bash
cd Scripts
python main.py
```

The simulation will:
1. Load all parameters exactly as defined in `Data/config.json`.
2. Calibrate the structural representation of the economy.
3. Run the fast quarterly loop, solving for consumer planning targets and investment levels.
4. Export high-quality analytics to `Results/<timestamp>/`.

---

## 5. Configuration

All runtime constraints, economic shocks, and mathematical solver tolerances are strictly governed by `Data/config.json`. If a key is absent, the simulation uses an embedded fallback.

### Key Simulation & Economic Parameters

| Key | Default | Description |
|---|---|---|
| `n_quarters` | `20` | Number of quarters to simulate (20 = 5 years) |
| `delta` | `0.0125` | Quarterly capital depreciation rate |
| `drift` | `0.012` | Quarterly standard deviation (σ) for preference shocks |
| `kappa_ou` | `0.15` | Mean-reversion speed (θ) for consumer preference drift towards the habit target |
| `habit_persistence`| `0.80` | EMA decay parameter (γ) controlling how slowly the habit target tracks actual consumption |
| `wage_rate` | `16.9` | Baseline hourly wage scaling constant |

### Advanced Solver Hyperparameters

| Key | Default | Description |
|---|---|---|
| `primal_tol`| `1e-3` | Physical resource constraint threshold (e.g., 0.1% violation) |
| `dual_tol` | `1e-4` | Complementary slackness threshold for optimal resource matching |
| `eta_K` | `0.4` | Base coordinate ascent step size for the 65 capital constraints |
| `eta_L` | `0.6` | Base coordinate ascent step size for the global labor constraint |
| `max_iter` | `2000` | Absolute loop truncation budget per quarter |

**Note**: The solver implements a **Sign-Based Adaptive Mechanism**, dynamically bounding learning rates during execution to rapidly locate the equilibrium prices while suppressing resonance oscillations.

---

## 6. Outputs

Each simulation run generates automatic analytics within `Results/<timestamp>/`:

| Output File Name | Description |
|---|---|
| `01_gdp.png` | Real GDP and Real Final Demand tracked against true physical value. |
| `02_output_consumption.png` | Group-level structural mix between consumer allocations and total output. |
| `04_shadow_prices.png` | Dynamic dual multipliers generated by the Dual Ascent engine. |
| `06_alpha_learning.png` | The cybernetic gap isolating planner prediction error vs true preference drift. |
| `10_capital_slack.png` | Unused resource capacity highlighting utilization boundaries. |

Figures are created using standard, pre-configured `matplotlib` publication aesthetics.

---

## 7. License

This project is licensed under the **GNU General Public License v3.0**. See `LICENSE` for the full terms.