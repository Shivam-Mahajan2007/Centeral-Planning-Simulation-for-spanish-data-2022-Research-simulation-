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

def plot_gdp(history, save_path):
    import matplotlib.pyplot as plt
    ts  = np.arange(len(history))
    ql  = qlabels(len(history))
    # Annualise quarterly GDP/AD by ×4 for display
    gdp = np.array([h["GDP"]     for h in history]) * 4 / _B_EUR
    ad  = np.array([h["Real_AD"] for h in history]) * 4 / _B_EUR

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ts, gdp, marker="o", label="Real GDP — Value Added (annualised)")
    ax.plot(ts, ad,  marker="s", linestyle="--", label="Real AD — Final Demand (annualised)")
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

    X_val_grps = valued_grps("X_star")
    X_val_tot  = np.array([sum(v[t] for v in X_val_grps.values()) for t in range(T)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    bot = np.zeros(T)
    for j, g in enumerate(groups):
        sh = X_val_grps[g] / (X_val_tot + 1e-10) * 100
        ax1.fill_between(ts, bot, bot + sh, color=GROUP_COLORS[j], label=g)
        bot += sh
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("% of gross output value")
    ax1.set_title("Gross Output (value shares)")
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
    ax2.set_title("Household Consumption (value, quarterly)")
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
    Y = np.array([h["GDP"]      for h in history])   # annualise
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
    cpi = np.array([h.get("CPI", 1.0) for h in history])

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

