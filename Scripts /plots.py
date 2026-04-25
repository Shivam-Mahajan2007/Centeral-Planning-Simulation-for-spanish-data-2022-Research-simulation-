"""
plots.py
--------
Publication-quality plotting functions for the cybernetic planning simulation.
All functions write both a .png and a .pdf to the supplied save_path.

Unit convention
---------------
All monetary values stored in history are in EUR.
All plots display values in *Billions of EUR* (dividing by 1e9).
Annualised values are obtained by multiplying quarterly figures by 4.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

GROUP_COLORS = [
    "#27ae60", "#e67e22", "#2980b9", "#e74c3c",
    "#8e44ad", "#16a085", "#f39c12", "#2c3e50",
]

_B_EUR = 1e9   # divisor: EUR → Billions EUR


# --- Shared helpers ----------------------------------------------------------

def qlabels(n: int, start_year: int = 2022):
    return [f"{start_year + i//4} Q{i%4+1}" for i in range(n)]

def group_agg(history, key, groups, prices=None):
    if prices is None:
        return {g: np.array([h[key][idx].sum() for h in history])
                for g, idx in groups.items()}
    return {g: np.array([(h[key] * prices)[idx].sum() for h in history])
            for g, idx in groups.items()}

def _xticks(ax, ts, ql, step=2):
    ax.set_xticks(ts[::step])
    ax.set_xticklabels(ql[::step], rotation=40, ha="right")

def savefig(fig, path):
    import matplotlib.pyplot as plt
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    logger.info(f"[plots] {path}")


# --- Figures -----------------------------------------------------------------

def plot_gdp(history, save_path, P_initial=None, A=None, real_scale_factor=None):
    import matplotlib.pyplot as plt
    ts  = np.arange(len(history))
    ql  = qlabels(len(history))

    # Unified GDP (C + I + G) stored in history
    gdp = np.array([h["GDP"] for h in history]) * 4 / _B_EUR
    # Target Aggregate Demand (linked to GDP in simulation.py)
    ad  = np.array([h["Real_AD"] for h in history]) * 4 / _B_EUR

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ts, gdp, marker="o", label="Real GDP — (C* + dK + G) (annualised)")
    ax.plot(ts, ad,  marker="s", linestyle="--", label="Real AD — Planned Total (annualised)")   
    ax.set_ylabel("Billions EUR (annualised)")
    ax.set_title("Macroeconomic Aggregates")
    _xticks(ax, ts, ql)
    ax.legend()
    ax.grid()
    savefig(fig, save_path)


def plot_aggregate_demand_breakdown(history, save_path):
    import matplotlib.pyplot as plt
    T  = len(history)
    ts = np.arange(T)
    ql = qlabels(T)

    # Annualise (×4) and convert to B EUR
    C_real = np.array([h["C_real"] for h in history]) * 4 / _B_EUR
    I_gross_real = np.array([h["I_gross_real"] for h in history]) * 4 / _B_EUR
    G_real = np.array([h["G_real"] for h in history]) * 4 / _B_EUR

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.stackplot(ts, C_real, I_gross_real, G_real,
                 labels=["Household Consumption", "Gross Investment", "Government Expenditure"],
                 colors=["#2ecc71", "#3498db", "#e74c3c"], alpha=0.8)

    ax.set_ylabel("Billions EUR (annualised, base prices)")
    ax.set_title("Aggregate Demand Breakdown")
    _xticks(ax, ts, ql)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    savefig(fig, save_path)


def plot_output_consumption(history, groups, save_path, P_0=None):
    import matplotlib.pyplot as plt
    T  = len(history)
    ts = np.arange(T)
    ql = qlabels(T)

    def valued_grps(key):
        prices_to_use = P_0 if P_0 is not None else history[0]["P"]
        return {g: np.array([(h[key] * prices_to_use)[idx].sum() for h in history])
                for g, idx in groups.items()}

    # Use X_actual (actual firm production) instead of X_star (planner target)
    X_val_grps = valued_grps("X_actual")
    X_val_tot  = np.array([sum(v[t] for v in X_val_grps.values()) for t in range(T)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    bot = np.zeros(T)
    for j, g in enumerate(groups):
        sh = X_val_grps[g] / (X_val_tot + 1e-10) * 100
        ax1.fill_between(ts, bot, bot + sh, color=GROUP_COLORS[j], label=g)
        bot += sh
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("% of gross output value")
    ax1.set_title("Actual Gross Output (value shares)")
    _xticks(ax1, ts, ql)
    ax1.legend()
    ax1.grid()

    C_val_grps = valued_grps("C_star")
    bot = np.zeros(T)
    for j, g in enumerate(groups):
        ax2.fill_between(ts, bot / _B_EUR, (bot + C_val_grps[g]) / _B_EUR,
                         color=GROUP_COLORS[j], label=g)
        bot += C_val_grps[g]
    ax2.set_ylabel("Billions EUR")
    ax2.set_title("Actual Household Consumption (value, quarterly)")
    _xticks(ax2, ts, ql)
    ax2.legend()
    ax2.grid()
    savefig(fig, save_path)


def plot_investment(history, groups, save_path, P_0=None):
    import matplotlib.pyplot as plt
    T  = len(history)
    ts = np.arange(T)
    ql = qlabels(T)
    prices_to_use = P_0 if P_0 is not None else history[0]["P"]
    I_val_grps = {
        g: np.array([(h["I_vec"] * prices_to_use)[idx].sum() for h in history])
        for g, idx in groups.items()
    }
    fig, ax = plt.subplots(figsize=(12, 7))
    bot = np.zeros(T)
    for j, g in enumerate(groups):
        ax.fill_between(ts, bot / _B_EUR, (bot + I_val_grps[g]) / _B_EUR,
                        color=GROUP_COLORS[j], label=g)
        bot += I_val_grps[g]
    ax.set_ylabel("Billions EUR (quarterly)")
    ax.set_title("Gross Investment by Sector Group (quarterly, base prices)")
    _xticks(ax, ts, ql)
    ax.legend()
    ax.grid()
    savefig(fig, save_path)


def plot_shadow_prices(history, sector_short, groups, save_path):
    import matplotlib.pyplot as plt
    T      = len(history)
    ts     = np.arange(T)
    ql     = qlabels(T)
    pi_mat = np.array([h["pi_star"] for h in history])
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    axes = axes.flatten()
    for j, gname in enumerate(groups):
        ax  = axes[j]
        idx = groups[gname]
        for i in idx:
            ax.plot(ts, pi_mat[:, i])
        ax.set_title(gname)
        _xticks(ax, ts, ql, 4)
        ax.grid()
    savefig(fig, save_path)


def plot_capital(history, groups, save_path, P_0=None):
    import matplotlib.pyplot as plt
    T  = len(history)
    ts = np.arange(T)
    ql = qlabels(T)
    K_val_grps = (group_agg(history, "K", groups, prices=P_0)
                  if P_0 is not None else group_agg(history, "K", groups))
    fig, ax = plt.subplots(figsize=(12, 7))
    bot = np.zeros(T)
    for j, g in enumerate(groups):
        ax.fill_between(ts, bot / _B_EUR, (bot + K_val_grps[g]) / _B_EUR,
                        color=GROUP_COLORS[j], label=g)
        bot += K_val_grps[g]
    ax.set_ylabel("Billions EUR (Q1 cost-push prices)")
    ax.set_title("Capital Stock by Sector Group (valued at Q1 prices)")
    _xticks(ax, ts, ql)
    ax.legend()
    ax.grid()
    savefig(fig, save_path)


def plot_alpha(history, save_path):
    import matplotlib.pyplot as plt
    alpha      = np.array([h["alpha"]      for h in history])
    alpha_true = np.array([h["alpha_true"] for h in history])
    ts = np.arange(len(history))
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(alpha.shape[1]):
        ax.plot(ts, alpha_true[:, i], color="black", alpha=0.3)
        ax.plot(ts, alpha[:, i], linestyle="--", alpha=0.7)
    ax.set_title("Preference Learning (True vs Estimated)")
    ax.set_ylabel("alpha")
    ax.grid()
    savefig(fig, save_path)


def plot_alpha_gap(history, save_path):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    gap = np.array([h["alpha_gap"] for h in history])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gap, marker="o")
    ax.set_title("Preference Estimation Error")
    ax.set_ylabel("||alpha - alpha_true|| (pp)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x*100:.2f}"))
    ax.grid()
    savefig(fig, save_path)


def plot_capital_output_ratio(history, save_path):
    import matplotlib.pyplot as plt
    # Use monetary K and annualised GDP for the ratio
    K = np.array([h["K_val_Q1"] for h in history])
    Y = np.array([h["GDP"]      for h in history])*4.0   # annualise
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(K / np.maximum(Y, 1e-30), marker="o")
    ax.set_title("Capital-Output Ratio  K / Y  (Q1 prices, annualised GDP)")
    ax.set_ylabel("Ratio")
    ax.grid()
    savefig(fig, save_path)


def plot_capital_slack(history, save_path):
    """
    Capital utilisation in absolute monetary terms at Q1 prices.

    Shows two stacked areas:
      • Committed capital  (K - slack, i.e. κ·X valued at Q1 prices)
      • Idle capital slack (K - κ·X valued at Q1 prices)

    Both in Billions EUR.  This replaces the old relative-slack (%) chart.
    """
    import matplotlib.pyplot as plt
    ts = np.arange(len(history))
    ql = qlabels(len(history))

    K_total  = np.array([h["K_val_Q1"]     for h in history]) / _B_EUR
    slack    = np.array([h["slack_val_Q1"]  for h in history]) / _B_EUR
    used     = np.array([h["used_val_Q1"]   for h in history]) / _B_EUR

    fig, ax = plt.subplots(figsize=(12, 6))

    # Stacked: committed (bottom) + idle (top)
    ax.fill_between(ts, np.zeros(len(ts)), used,
                    color="#2980b9", alpha=0.75, label="Committed Capital (κ·X)")
    ax.fill_between(ts, used, K_total,
                    color="#e74c3c", alpha=0.45, label="Idle Capital Slack")

    ax.plot(ts, K_total, color="#2c3e50", linewidth=2.0, label="Total Capital Stock")

    ax.set_ylabel("Billions EUR (Q1 prices)")
    ax.set_title("Capital Stock vs. Idle Slack  —  Monetary Terms at Q1 Prices")
    _xticks(ax, ts, ql)
    ax.legend()
    ax.grid(alpha=0.3)
    savefig(fig, save_path)


def plot_labor_utilization(history, save_path):
    import matplotlib.pyplot as plt
    ts   = np.arange(len(history))
    ql   = qlabels(len(history))
    util = np.array([(1.0 - h["labor_slack"]) * 100 for h in history])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ts, util, marker="o", color="#c0392b", linewidth=2, label="Labor Utilization")
    ax.axhline(100, color="black", linestyle="--", alpha=0.5, label="Full Employment (100%)")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Utilization Rate (%)")
    ax.set_title("Global Resource Bound: Aggregate Labor Utilization")
    _xticks(ax, ts, ql)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    savefig(fig, save_path)


def plot_shadow_price_index(history, save_path):
    import matplotlib.pyplot as plt
    ts  = np.arange(len(history))
    ql  = qlabels(len(history))
    # Calculate CPI from Inflation: CPI_{t+1} = (1 + i_t) * CPI_t
    inflation = np.array([h.get("Inflation", 0.0) for h in history])
    cpi = np.ones(len(history))
    for i in range(1, len(history)):
        cpi[i] = cpi[i-1] * (1.0 + inflation[i])


    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ts, cpi, marker="o", color="#34495e", linewidth=2)
    ax.set_ylabel("Shadow Price Index (Q1 = 1.0)")
    ax.set_title("System-Wide Cost Level (Production-Led Deflation Tracking)")
    _xticks(ax, ts, ql)
    ax.grid(alpha=0.3)
    savefig(fig, save_path)


def plot_cybernetic_signals(history, save_path):
    import matplotlib.pyplot as plt
    ts    = np.arange(len(history))
    ql    = qlabels(len(history))
    drift = np.array([h["price_drift"] for h in history]) * 100
    ed    = np.array([h["ED_nom"]       for h in history]) / _B_EUR

    fig, ax1 = plt.subplots(figsize=(12, 7))

    color = "#8e44ad"
    ax1.set_xlabel("Quarter")
    ax1.set_ylabel("Price Drift (%)", color=color)
    ax1.plot(ts, drift, marker="o", color=color, label="Price Drift (tatonnement shift)")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = "#d35400"
    ax2.set_ylabel("Nominal Excess Demand (B EUR)", color=color)
    ax2.plot(ts, ed, marker="s", linestyle="--", color=color, label="Nominal Excess Demand")
    ax2.tick_params(axis='y', labelcolor=color)

    ax1.set_title("Cybernetic Feedback Signals")
    _xticks(ax1, ts, ql)
    fig.tight_layout()
    ax1.grid(alpha=0.2)
    savefig(fig, save_path)


def plot_real_income_index(history, save_path):
    import matplotlib.pyplot as plt
    ts = np.arange(len(history))
    ql = qlabels(len(history))
    Y  = np.array([h["Y"] for h in history]) / (history[0]["Y"] + 1e-30)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ts, Y, marker="o", color="#16a085", linewidth=2.5)
    ax.set_ylabel("Real Income Index (Q1 = 1.0)")
    ax.set_title("Household Real Purchasing Power")
    _xticks(ax, ts, ql)
    ax.grid(alpha=0.3)
    savefig(fig, save_path)


def plot_labor_productivity(history, save_path):
    import matplotlib.pyplot as plt
    ts    = np.arange(len(history))
    ql    = qlabels(len(history))
    # Annualise GDP (×4)
    gdp   = np.array([h["GDP"] for h in history]) * 4
    labor = np.array([max(1.0 - h["labor_slack"], 1e-10) for h in history])
    prod  = gdp / labor
    prod  = prod / (prod[0] + 1e-30)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ts, prod, marker="o", color="#2980b9", linewidth=2.5)
    ax.set_ylabel("Index (Q1 = 1.0)")
    ax.set_title("Aggregate Labor Productivity (Real GDP annualised / Labor Hour)")
    _xticks(ax, ts, ql)
    ax.grid(alpha=0.3)
    savefig(fig, save_path)


def plot_growth_targets(history, save_path):
    import matplotlib.pyplot as plt
    ts    = np.arange(len(history))
    ql    = qlabels(len(history))
    ghat  = np.array([h.get("G_hat_mean", 0.0) for h in history]) * 100
    gdp   = np.array([h["GDP"] for h in history])
    growth_achieved = np.zeros(len(history))
    for t in range(1, len(history)):
        growth_achieved[t] = (gdp[t] / (gdp[t-1] + 1e-30) - 1) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ts, ghat, marker="o", label="Target Consumption Growth (G_hat)")
    ax.plot(ts, growth_achieved, marker="x", linestyle="--", label="Achieved GDP Growth")
    ax.set_ylabel("Quarterly Growth (%)")
    ax.set_title("Planning Performance: Growth Targets vs. Achievement")
    _xticks(ax, ts, ql)
    ax.legend()
    ax.grid(alpha=0.3)
    savefig(fig, save_path)


def plot_excess_demand(history, save_path):
    import matplotlib.pyplot as plt
    ts = np.arange(len(history))
    ql = qlabels(len(history))

    ed_mean = []
    for h in history:
        pi         = h["pi_star"]
        y          = h["Y"]
        alpha_true = h["alpha_true"]
        c_star     = h["C_star"]
        i_vec      = h["I_vec"]
        g_vec      = h["G_vec"]

        p_vec      = y * pi
        demand     = alpha_true / np.where(pi > 1e-30, pi, 1e-30)
        nominal_ad = np.sum(p_vec * (c_star + i_vec + g_vec))
        denom      = max(nominal_ad, 1e-30)
        m          = p_vec * (c_star - demand) / denom
        ed_mean.append(np.abs(m).mean() * 100)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ts, ed_mean, marker="o", color="#c0392b", linewidth=2)
    ax.set_ylabel("Mean Absolute Sectoral Excess Demand (%)")
    ax.set_title("System Performance: Value-Weighted Excess Demand  P·(C* − D) / AD")
    _xticks(ax, ts, ql)
    ax.grid(alpha=0.3)
    savefig(fig, save_path)


def plot_inflation(history, save_path):
    import matplotlib.pyplot as plt
    ts  = np.arange(len(history))
    ql  = qlabels(len(history))
    inf = np.array([h.get("Inflation", 0.0) for h in history])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ts, inf, marker="o", color="#e67e22", linewidth=2.5)
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_ylabel("Quarterly Inflation Rate (%)")
    ax.set_title("Price Stability: Quarterly Inflation (Laspeyres Index)")
    _xticks(ax, ts, ql)
    ax.grid(alpha=0.3)
    savefig(fig, save_path)


def plot_investment_gdp_ratio(history, save_path, ref_pct: float = 19.8,
                              start_year: int = 2022):
    """Investment as % of GDP — quarterly series with annual-average bars.

    Parameters
    ----------
    history    : list of quarter dicts (must contain 'I_pct_GDP').
    save_path  : pathlib.Path for the .png output (.pdf is auto-created).
    ref_pct    : horizontal reference line (default 22% — Spanish hist. avg).
    start_year : calendar year of Q1 (default 2022).
    """
    import matplotlib.pyplot as plt

    T   = len(history)
    ts  = np.arange(T)
    ql  = qlabels(T, start_year=start_year)

    i_pct = np.array([h["I_pct_GDP"] for h in history])

    # Annual-average bars -------------------------------------------------------
    n_full_years = T // 4
    bar_xs, bar_hs, bar_lbls = [], [], []
    for y in range(n_full_years):
        idxs = np.arange(y * 4, y * 4 + 4)
        bar_xs.append(float(idxs.mean()))
        bar_hs.append(float(i_pct[idxs].mean()))
        bar_lbls.append(str(start_year + y))
    if T % 4 > 0:                                    # partial final year
        idxs = np.arange(n_full_years * 4, T)
        bar_xs.append(float(idxs.mean()))
        bar_hs.append(float(i_pct[idxs].mean()))
        bar_lbls.append(f"{start_year + n_full_years}*")

    # 5-year rolling average (on annual grid) -----------------------------------
    roll5_xs = bar_xs[:]
    roll5_ys = [float(np.mean(bar_hs[max(0, i - 4): i + 1]))
                for i in range(len(bar_hs))]

    # Plot ----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 7))

    ax.bar(bar_xs, bar_hs, width=3.5,
           color="#3498db", alpha=0.28, zorder=1, label="Annual Average")
    for bx, bh, bl in zip(bar_xs, bar_hs, bar_lbls):
        ax.text(bx, bh + 0.15, f"{bh:.1f}%",
                ha="center", va="bottom", fontsize=8, color="#2980b9")

    ax.plot(ts, i_pct, marker="o", markersize=4, linewidth=2,
            color="#2c3e50", zorder=3, label="Quarterly I / GDP (%)")

    ax.plot(roll5_xs, roll5_ys, marker="D", markersize=6, linewidth=2.5,
            linestyle="--", color="#e67e22", zorder=4,
            label="5-Year Rolling Average")

    ax.axhline(ref_pct, color="#c0392b", linestyle=":", linewidth=1.5,
               alpha=0.75, label=f"Reference {ref_pct:.0f}% (Spain hist. avg)")

    ax.set_ylabel("Investment / GDP (%)")
    ax.set_title("Gross Investment as % of Real GDP  —  Quarterly & Annual Averages")
    _xticks(ax, ts, ql)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    savefig(fig, save_path)


def plot_firm_income_distribution(hist, n, out_path):
    """Plot the income distribution across the 5 firms for the largest sectors."""
    import matplotlib.pyplot as plt
    h_final = hist[-1]
    v_MIP = h_final.get("v_MIP", np.zeros(n))
    X_f = h_final.get("X_f", np.zeros((n, 5)))
    Y_f_mat = v_MIP[:, None] * X_f # (N, 5)
    
    # Total income per sector
    sector_Y = Y_f_mat.sum(axis=1)
    top_10_idx = np.argsort(sector_Y)[-10:]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bottom = np.zeros(10)
    colors = ["#1abc9c", "#3498db", "#9b59b6", "#f1c40f", "#e67e22"]
    
    for f in range(5):
        vals = Y_f_mat[top_10_idx, f] / 1e9 # B EUR
        # Use short names 
        names = [h_final["sector_short"][i] for i in top_10_idx]
        ax.bar(names, vals, bottom=bottom, color=colors[f], alpha=0.9, label=f"Firm {f+1}")
        bottom += vals
        
    ax.set_title("Income Distribution Across Firms (Top 10 Sectors)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Production Income (B EUR nominal)")
    ax.legend(frameon=False)
    plt.xticks(rotation=45)
    plt.tight_layout()
    savefig(fig, out_path)


def plot_iterations(history, save_path):
    import matplotlib.pyplot as plt
    ts  = np.arange(len(history))
    ql  = qlabels(len(history))
    iters = np.array([h.get("iterations", 0) for h in history])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(ts, iters, color="#8e44ad", alpha=0.7)
    
    # Add a trend line or rolling average
    if len(iters) >= 4:
        # Simple moving average
        window_size = min(4, len(iters))
        weights = np.repeat(1.0, window_size) / window_size
        sma = np.convolve(iters, weights, 'valid')
        ax.plot(ts[window_size-1:], sma, color="#2980b9", lw=2, label=f"{window_size}-Qtr Moving Average")
        ax.legend()
        
    # Mark maxed out iterations (if using max_iter=2000)
    max_iter_mask = iters >= 2000
    if np.any(max_iter_mask):
        ax.scatter(ts[max_iter_mask], iters[max_iter_mask], color='red', 
                   s=100, zorder=5, label="Max Iterations Reached")
        ax.legend()

    ax.set_title("Solver Iterations per Quarter (Convergence Speed)")
    ax.set_ylabel("Number of Iterations")
    ax.grid(axis='y', alpha=0.3)
    _xticks(ax, ts, ql)
    
    fig.tight_layout()
    savefig(fig, save_path)


