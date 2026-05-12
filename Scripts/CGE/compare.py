"""
compare.py
-----------
Comparative analysis interface for CIKP vs MA-CGE models.

Runs both engines on identical calibrated data and returns results for
statistical comparison and visualization.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional

from data_loader import load_data
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planner_scripts.calibration import calibrate as calibrate_cikp
from planner_scripts.simulation import run_simulation as run_simulation_cikp
from CGE.ma_calibration import calibrate_ma
from CGE.ma_simulation import run_simulation_ma

logger = logging.getLogger(__name__)

# Common metrics for comparison
COMMON_METRICS = [
    'GDP', 'Real_AD', 'Y', 'Inflation', 'Inflation_Geom',
    'Gross_Output_Real', 'C_real', 'C_realized',
    'I_gross_real', 'I_net_real', 'G_real',
    'K_val_Q1', 'slack_val_Q1', 'labor_slack',
    'I_pct_GDP', 'alpha_gap', 'alpha_gap_linf',
    'price_drift', 'CPI'
]

# MA-CGE specific metrics
MACGE_METRICS = [
    'avg_markup', 'markup_dispersion', 'tobin_q_mean'
]


def run_comparison(n_quarters: int = 20,
                   data_dir: Optional[Path] = None,
                   seed: int = 42,
                   **kwargs) -> pd.DataFrame:
    """
    Run both CIKP and MA-CGE engines on identical calibrated data.
    
    Parameters:
    -----------
    n_quarters : int, default 20
        Number of quarters to simulate
    data_dir : Path or str, optional
        Directory containing IO data
    seed : int, default 42
        Random seed for reproducibility
    **kwargs : dict
        Additional calibration parameters
        
    Returns:
    --------
    df : pd.DataFrame
        Tidy DataFrame with results for statistical comparison
    """
    
    logger.info(f"Starting comparative study: {n_quarters} quarters, seed={seed}")
    
    # Load data
    data = load_data(data_dir)
    
    # CIKP engine
    logger.info("Calibrating CIKP model...")
    state_cikp = calibrate_cikp(data_dir=data_dir, seed=seed, **kwargs)
    logger.info("Running CIKP simulation...")
    state_cikp = run_simulation_cikp(state_cikp, n_quarters)
    
    # MA-CGE engine
    logger.info("Calibrating MA-CGE model...")
    state_macge = calibrate_ma(data_dir=data_dir, seed=seed, **kwargs)
    logger.info("Running MA-CGE simulation...")
    state_macge = run_simulation_ma(state_macge, n_quarters)
    
    # Build comparison DataFrame
    rows = []
    for t in range(n_quarters):
        h_c = state_cikp.history[t]
        h_m = state_macge.history[t]
        
        # Common metrics
        for key in COMMON_METRICS:
            if key in h_c and key in h_m:
                rows.append(dict(
                    quarter=t+1, metric=key,
                    CIKP=h_c[key], MACGE=h_m[key],
                    difference=h_m[key] - h_c[key],
                    pct_diff=(h_m[key] / h_c[key] - 1) * 100 if h_c[key] != 0 else np.nan
                ))
        
        # MA-CGE specific metrics
        for key in MACGE_METRICS:
            if key in h_m:
                rows.append(dict(
                    quarter=t+1, metric=key,
                    CIKP=np.nan, MACGE=h_m[key],
                    difference=np.nan, pct_diff=np.nan
                ))
    
    df = pd.DataFrame(rows)
    logger.info(f"Comparison complete: {len(df)} observations generated")
    
    return df


def run_montecarlo_comparison(n_runs: int = 20,
                              n_quarters: int = 20,
                              data_dir: Optional[Path] = None,
                              seed_base: int = 1000,
                              **kwargs) -> pd.DataFrame:
    """
    Run Monte Carlo comparison of both models.
    
    Parameters:
    -----------
    n_runs : int, default 20
        Number of Monte Carlo runs
    n_quarters : int, default 20
        Quarters per run
    data_dir : Path or str, optional
        Directory containing IO data
    seed_base : int, default 1000
        Base seed for reproducibility
    **kwargs : dict
        Additional calibration parameters
        
    Returns:
    --------
    df : pd.DataFrame
        Tidy DataFrame with Monte Carlo results
    """
    
    logger.info(f"Starting Monte Carlo comparison: {n_runs} runs, {n_quarters} quarters each")
    
    all_results = []
    
    for run in range(n_runs):
        seed = seed_base + run
        logger.info(f"Monte Carlo run {run + 1}/{n_runs} with seed {seed}")
        
        try:
            df_run = run_comparison(n_quarters=n_quarters, data_dir=data_dir, 
                                  seed=seed, **kwargs)
            df_run['run'] = run + 1
            all_results.append(df_run)
        except Exception as e:
            logger.error(f"Run {run + 1} failed: {e}")
            continue
    
    if all_results:
        df = pd.concat(all_results, ignore_index=True)
        logger.info(f"Monte Carlo comparison complete: {len(df)} total observations")
        return df
    else:
        raise RuntimeError("All Monte Carlo runs failed")


def summarize_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize comparison results with statistics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Comparison DataFrame from run_comparison
        
    Returns:
    --------
    summary : pd.DataFrame
        Summary statistics by metric
    """
    
    # Filter out MA-CGE specific metrics for difference calculations
    common_metrics = [m for m in COMMON_METRICS if m in df['metric'].unique()]
    
    summaries = []
    
    for metric in common_metrics:
        metric_data = df[df['metric'] == metric].copy()
        
        # Remove NaN values
        metric_data = metric_data.dropna(subset=['CIKP', 'MACGE'])
        
        if len(metric_data) == 0:
            continue
        
        summary = {
            'metric': metric,
            'n_observations': len(metric_data),
            'mean_cikp': metric_data['CIKP'].mean(),
            'mean_macge': metric_data['MACGE'].mean(),
            'mean_difference': metric_data['difference'].mean(),
            'mean_pct_diff': metric_data['pct_diff'].mean(),
            'std_cikp': metric_data['CIKP'].std(),
            'std_macge': metric_data['MACGE'].std(),
            'std_difference': metric_data['difference'].std(),
            'correlation': metric_data['CIKP'].corr(metric_data['MACGE']),
            'rmse': np.sqrt((metric_data['difference']**2).mean()),
            'mae': np.abs(metric_data['difference']).mean()
        }
        
        summaries.append(summary)
    
    summary_df = pd.DataFrame(summaries)
    
    # Add MA-CGE specific metrics
    for metric in MACGE_METRICS:
        if metric in df['metric'].unique():
            metric_data = df[df['metric'] == metric]['MACGE'].dropna()
            if len(metric_data) > 0:
                summary = {
                    'metric': metric,
                    'n_observations': len(metric_data),
                    'mean_cikp': np.nan,
                    'mean_macge': metric_data.mean(),
                    'mean_difference': np.nan,
                    'mean_pct_diff': np.nan,
                    'std_cikp': np.nan,
                    'std_macge': metric_data.std(),
                    'std_difference': np.nan,
                    'correlation': np.nan,
                    'rmse': np.nan,
                    'mae': np.nan
                }
                summary_df = pd.concat([summary_df, pd.DataFrame([summary])], 
                                      ignore_index=True)
    
    return summary_df


def export_comparison_results(df: pd.DataFrame, output_path: Path):
    """
    Export comparison results to CSV and Excel formats.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Comparison DataFrame
    output_path : Path
        Output directory for results
    """
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Export detailed results
    df.to_csv(output_path / "comparison_detailed.csv", index=False)
    df.to_excel(output_path / "comparison_detailed.xlsx", index=False)
    
    # Export summary
    summary = summarize_comparison(df)
    summary.to_csv(output_path / "comparison_summary.csv", index=False)
    summary.to_excel(output_path / "comparison_summary.xlsx", index=False)
    
    logger.info(f"Results exported to {output_path}")


def plot_comparison_metrics(df: pd.DataFrame, metrics: Optional[List[str]] = None,
                           output_path: Optional[Path] = None):
    """
    Create comparison plots for key metrics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Comparison DataFrame
    metrics : list of str, optional
        Metrics to plot (default: key economic indicators)
    output_path : Path, optional
        Path to save plots
    """
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if metrics is None:
            metrics = ['GDP', 'Inflation', 'I_pct_GDP', 'CPI', 'labor_slack']
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in df['metric'].unique()]
        
        if not available_metrics:
            logger.warning("No specified metrics available for plotting")
            return
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(available_metrics[:6]):
            ax = axes[i]
            metric_data = df[df['metric'] == metric].dropna(subset=['CIKP', 'MACGE'])
            
            if len(metric_data) == 0:
                ax.text(0.5, 0.5, f'No data for {metric}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(metric)
                continue
            
            # Time series plot
            ax.plot(metric_data['quarter'], metric_data['CIKP'], 
                   label='CIKP', marker='o', alpha=0.7)
            ax.plot(metric_data['quarter'], metric_data['MACGE'], 
                   label='MA-CGE', marker='s', alpha=0.7)
            ax.set_title(f'{metric} Over Time')
            ax.set_xlabel('Quarter')
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(available_metrics), 6):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path / "comparison_plots.png", dpi=300, bbox_inches='tight')
            plt.savefig(output_path / "comparison_plots.pdf", bbox_inches='tight')
            logger.info(f"Plots saved to {output_path}")
        
        plt.show()
        
    except ImportError:
        logger.warning("Matplotlib/Seaborn not available for plotting")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Run comparison
    df = run_comparison(n_quarters=20, seed=42)
    
    # Summarize results
    summary = summarize_comparison(df)
    print("Comparison Summary:")
    print(summary.to_string())
    
    # Export results
    export_comparison_results(df, Path("Results/Comparison"))
    
    # Create plots
    plot_comparison_metrics(df, output_path=Path("Results/Comparison"))
