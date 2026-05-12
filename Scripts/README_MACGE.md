# Multi-Agent CGE (MA-CGE) Implementation

This directory contains the complete implementation of the Multi-Agent Computable General Equilibrium (MA-CGE) simulation system, designed for comparative analysis against the existing Cybernetic In-Kind Planning (CIKP) system.

## Directory Structure

```
Scripts/
├── CGE/                           # MA-CGE implementation
│   ├── __init__.py                # Package initialization
│   ├── ma_calibration.py           # MA-CGE calibration module
│   ├── agents.py                  # Firm and Household agent classes
│   ├── ma_model_core.jl          # Julia core for CES production & tatonnement
│   ├── ma_simulation.py           # Main simulation engine
│   └── compare.py                # Comparative analysis interface
├── planner_scripts/               # Original CIKP system
│   ├── __init__.py
│   ├── calibration.py
│   ├── simulation.py
│   ├── monte_carlo.py
│   ├── scenarios.py
│   ├── plots.py
│   └── mc_compare.py
├── data_loader.py                # Shared data loading (unchanged)
├── julia_bridge.py              # Updated to include both engines
├── model_core.jl               # Original CIKP Julia engine
├── test_macge.py               # Test script for MA-CGE
└── README_MACGE.md             # This file
```

## Implementation Overview

### Core Components

1. **MACGEState** (`ma_calibration.py`): Complete state container with all economic variables
2. **Agent Classes** (`agents.py`): Firm and Household classes with economic behavior
3. **Julia Core** (`ma_model_core.jl`): High-performance CES production and tatonnement
4. **Simulation Engine** (`ma_simulation.py`): 10-step quarterly execution loop
5. **Comparison Interface** (`compare.py`): Statistical comparison with CIKP

### Key Features

- **Nested Production**: Leontief intermediate inputs + CES primary factors
- **Price-Mediated Markets**: Walrasian tatonnement for market clearing
- **Markov Pricing**: Adaptive markup dynamics based on demand pressure
- **Tobin-q Investment**: Capital accumulation with accelerator mechanism
- **Multi-Household**: 4 households with CRRA preferences and habit formation
- **Preference Drift**: Identical LN-AR process to CIKP for fair comparison

## Usage Examples

### Basic MA-CGE Simulation

```python
from CGE import calibrate_ma, run_simulation_ma

# Calibrate model
state = calibrate_ma(data_dir="Data/", seed=42)

# Run simulation
state = run_simulation_ma(state, n_quarters=20)

# Access results
gdp_history = [h['GDP'] for h in state.history]
inflation_history = [h['Inflation'] for h in state.history]
```

### Comparative Analysis

```python
from CGE.compare import run_comparison, summarize_comparison

# Run both models for comparison
df = run_comparison(n_quarters=20, seed=42)

# Get summary statistics
summary = summarize_comparison(df)
print(summary)
```

### Monte Carlo Study

```python
from CGE.compare import run_montecarlo_comparison

# Run Monte Carlo comparison
mc_results = run_montecarlo_comparison(n_runs=50, n_quarters=20)
```

## Testing

Run the test script to verify installation:

```bash
cd Scripts/
python test_macge.py
```

## Economic Model Specification

### Production Technology
Each sector uses nested Leontief-CES technology:
- **Leontief layer**: Intermediate inputs from A-matrix
- **CES layer**: Primary factors (capital, labor) with elasticity σ = 0.5

### Market Clearing
Walrasian tatonnement process:
- Excess demand drives price adjustments
- Convergence tolerance: 0.5% relative excess demand

### Investment
Tobin-q accelerator rule:
- Replacement investment: δ·K
- Net investment: φ·K·max(q−1, 0)
- Bounded by available savings

### Household Behavior
4 households with:
- CRRA utility (σ = 2.0)
- Budget-constrained Marshallian demands
- Habit formation and preference drift

## Comparison Metrics

The system tracks identical metrics to CIKP for fair comparison:

**Core Economic Indicators:**
- Real GDP, Inflation, Investment share
- Capital stock, Labor slack, Government spending

**MA-CGE Specific:**
- Average markup, Markup dispersion
- Tobin's q, Walrasian iterations

## Calibration Anchors

Both models share identical calibration:
- IO data (A-matrix, value-added)
- Capital coefficients (κ, v_per_unit)
- Initial prices (Neumann shadow prices)
- Household ownership and preferences
- Government expenditure path

## Performance

- **Julia integration**: High-performance numerical core
- **Sparse matrices**: Efficient IO operations
- **Vectorized operations**: Bulk household calculations
- **Modular design**: Easy extension and modification

## Dependencies

**Python:**
- numpy, scipy, pandas
- dataclasses, pathlib (standard library)

**Julia:**
- LinearAlgebra, SparseArrays
- Random, Statistics
- PythonCall (for bridge)

## Development Notes

1. **Modular Architecture**: Clear separation between calibration, simulation, and comparison
2. **Reproducible Results**: Fixed seeds for Monte Carlo studies
3. **Extensible Design**: Easy to add new agent types or market mechanisms
4. **Comprehensive Testing**: Unit tests and integration validation
5. **Documentation**: Full docstrings and type hints

## Future Extensions

- Sector-specific CES elasticities
- Endogenous technological change
- International trade module
- Environmental constraints
- Financial sector integration

---

*Implementation follows the detailed specification in MA_CGE_Implementation_Plan.md*
