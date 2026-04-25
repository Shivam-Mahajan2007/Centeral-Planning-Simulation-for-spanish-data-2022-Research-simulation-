"""
api.py — FastAPI backend for the Cybernetic Planning Simulation dashboard.
"""

import asyncio
import base64
import json
import logging
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).parent
DATA_DIR    = SCRIPTS_DIR.parent / "Data"
sys.path.insert(0, str(SCRIPTS_DIR))

app = FastAPI(title="Cybernetic Planning Simulation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Shared state (one simulation at a time) ───────────────────────────────────
_lock            = threading.Lock()
_run_state       = {"status": "idle", "progress": 0, "total": 0}
_run_queue: queue.Queue = queue.Queue()
_mc_state        = {"status": "idle", "progress": 0, "total": 0}
_mc_queue: queue.Queue  = queue.Queue()
_last_results_dir: Optional[Path] = None
_last_mc_dir:      Optional[Path] = None


# ── Config endpoints ──────────────────────────────────────────────────────────

@app.get("/config")
def get_config():
    cfg_path = DATA_DIR / "config.json"
    if not cfg_path.exists():
        return JSONResponse({})
    with open(cfg_path) as f:
        return JSONResponse(json.load(f))


@app.post("/config")
async def save_config(request_body: dict):
    cfg_path = DATA_DIR / "config.json"
    with open(cfg_path, "w") as f:
        json.dump(request_body, f, indent=4)
    return {"ok": True}


# ── Single run ────────────────────────────────────────────────────────────────

def _run_simulation_thread(config: dict):
    global _last_results_dir
    _run_queue.put({"type": "log", "msg": "Initializing core numerical libraries... (This may take a moment)"})
    # Heavy imports deferred to thread start
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    import numpy as np
    from data_loader import load_data, sector_groups
    from calibration import calibrate
    from simulation import run_quarter
    import plots as plt_module

    _run_state.update({"status": "running", "progress": 0, "total": config.get("n_quarters", 20)})
    _run_queue.put({"type": "status", "status": "running"})

    try:
        _run_queue.put({"type": "log", "msg": "Loading IO data..."})
        data  = load_data(DATA_DIR)
        state = calibrate(
            data,
            delta=config.get("delta", 0.015),
            pref_drift_rho=config.get("pref_drift_rho", 0.95),
            pref_drift_sigma=config.get("pref_drift_sigma", 0.04),
            pref_noise_sigma=config.get("pref_noise_sigma", 0.01),
            theta_drift=config.get("theta_drift", 0.1),
            epsilon=config.get("epsilon", 0.5),
            neumann_k=config.get("neumann_k", 25),
            kappa_factor=config.get("kappa_factor", 1.0),
            L_total=config.get("L_total", 39e9),
            wage_rate=config.get("wage_rate", 21.0),
            labor_mult=config.get("labor_mult", 1.0),
            primal_tol=config.get("primal_tol", 1e-3),
            dual_tol=config.get("dual_tol", 1e-4),
            eta_K=config.get("eta_K", 0.15),
            eta_L=config.get("eta_L", 0.15),
            max_iter=config.get("max_iter", 2000),
            g_step=config.get("g_step", 0.0),
            c_step=config.get("c_step", 0.01),
            habit_persistence=config.get("habit_persistence", 0.7),
            nominal_consumption_annual=config.get("nominal_consumption_annual", 807e9),
        )
        seed = config.get("rng_seed", 42)
        state.rng = np.random.default_rng(seed)
        _run_queue.put({"type": "log", "msg": "Calibration complete. Starting simulation..."})

        n_q = config.get("n_quarters", 20)
        for _ in range(n_q):
            run_quarter(state)
            q = state.t
            adv = f"{state.history[-1]['GDP']*4/1e9:.1f}B EUR ann. GDP"
            _run_queue.put({"type": "log",      "msg": f"Q{q} complete — {adv}"})
            _run_queue.put({"type": "progress", "progress": q, "total": n_q})
            _run_state.update({"progress": q, "total": n_q})

        # Generate charts
        from datetime import datetime
        ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = SCRIPTS_DIR.parent / "Results" / ts
        results_dir.mkdir(parents=True, exist_ok=True)
        _last_results_dir = results_dir
        _run_queue.put({"type": "log", "msg": "Generating charts..."})

        groups = sector_groups(data["sector_names"])
        P_0    = state.P_0
        _plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight"})

        chart_fns = [
            ("01_gdp",                   plt_module.plot_gdp,                   [state.history, results_dir / "01_gdp.png"],              {"P_initial": state.pi_0_fixed, "A": state.A, "real_scale_factor": state.real_scale_factor}),
            ("02_ad_breakdown",          plt_module.plot_aggregate_demand_breakdown, [state.history, results_dir / "02_ad_breakdown.png"], {}),
            ("03_output_consumption",    plt_module.plot_output_consumption,    [state.history, groups, results_dir / "03_output_consumption.png"], {"P_0": P_0}),
            ("04_investment",            plt_module.plot_investment,             [state.history, groups, results_dir / "04_investment.png"],  {"P_0": P_0}),
            ("05_capital",               plt_module.plot_capital,                [state.history, groups, results_dir / "05_capital.png"],    {"P_0": P_0}),
            ("06_alpha_learning",        plt_module.plot_alpha,                  [state.history, results_dir / "06_alpha_learning.png"],     {}),
            ("07_alpha_error",           plt_module.plot_alpha_gap,              [state.history, results_dir / "07_alpha_error.png"],        {}),
            ("08_capital_output_ratio",  plt_module.plot_capital_output_ratio,   [state.history, results_dir / "08_capital_output_ratio.png"],{}),
            ("09_capital_slack",         plt_module.plot_capital_slack,          [state.history, results_dir / "09_capital_slack.png"],      {}),
            ("10_labor_utilization",     plt_module.plot_labor_utilization,      [state.history, results_dir / "10_labor_utilization.png"],  {}),
            ("11_cybernetic_signals",    plt_module.plot_cybernetic_signals,     [state.history, results_dir / "11_cybernetic_signals.png"], {}),
            ("12_real_income_index",     plt_module.plot_real_income_index,      [state.history, results_dir / "12_real_income_index.png"],  {}),
            ("13_inflation",             plt_module.plot_inflation,              [state.history, results_dir / "13_inflation.png"],          {}),
            ("14_investment_gdp_ratio",  plt_module.plot_investment_gdp_ratio,   [state.history, results_dir / "14_investment_gdp_ratio.png"],{}),
            ("15_iterations",            plt_module.plot_iterations,             [state.history, results_dir / "15_iterations.png"],         {}),
            ("16_firm_income",           plt_module.plot_firm_income_distribution,[state.history, state.n, results_dir / "16_firm_income.png"],{}),
        ]

        generated = []
        for key, fn, args, kwargs in chart_fns:
            try:
                fn(*args, **kwargs)
                generated.append(key)
            except Exception as e:
                _run_queue.put({"type": "log", "msg": f"  [WARN] Chart {key} failed: {e}"})

        _run_state.update({"status": "done", "progress": n_q, "total": n_q, "charts": generated, "results_dir": str(results_dir)})
        _run_queue.put({"type": "done", "charts": generated})

    except Exception as e:
        _run_state.update({"status": "error", "error": str(e)})
        _run_queue.put({"type": "error", "msg": str(e)})


@app.post("/run/start")
async def start_run(config: dict):
    with _lock:
        if _run_state.get("status") == "running":
            return {"ok": False, "reason": "A run is already in progress"}
        while not _run_queue.empty():
            try: _run_queue.get_nowait()
            except: break
        _run_state.update({"status": "starting", "progress": 0})

    t = threading.Thread(target=_run_simulation_thread, args=(config,), daemon=True)
    t.start()
    return {"ok": True}


@app.get("/run/stream")
async def stream_run():
    async def generator():
        ping_ticks = 0
        while True:
            try:
                msg = _run_queue.get_nowait()
                yield {"data": json.dumps(msg)}
                if msg.get("type") in ("done", "error"):
                    break
            except queue.Empty:
                await asyncio.sleep(0.5)
                ping_ticks += 1
                if ping_ticks >= 20: # Emit ping every 10 secs
                    ping_ticks = 0
                    yield {"data": json.dumps({"type": "ping"})}
    return EventSourceResponse(generator())


@app.get("/run/status")
def run_status():
    return _run_state


# ── Chart serving ─────────────────────────────────────────────────────────────

@app.get("/charts/{chart_key}")
def get_chart(chart_key: str):
    rd = _run_state.get("results_dir")
    if not rd:
        return JSONResponse({"error": "no results yet"}, status_code=404)
    from fastapi.responses import FileResponse
    results_dir = Path(rd)
    matches = list(results_dir.glob(f"{chart_key}*.png"))
    if not matches:
        return JSONResponse({"error": "chart not found"}, status_code=404)
    return FileResponse(matches[0], media_type="image/png")


# ── Monte Carlo ───────────────────────────────────────────────────────────────

def _mc_thread(config: dict):
    global _last_mc_dir
    n_runs  = config.get("mc_runs", 50)
    n_q     = config.get("n_quarters", 20)
    _mc_state.update({"status": "running", "progress": 0, "total": n_runs})
    _mc_queue.put({"type": "status", "status": "running"})

    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        from data_loader import load_data
        from calibration import calibrate
        from simulation import run_simulation
        from monte_carlo import plot_fan_chart, plot_iteration_histogram

        data = load_data(DATA_DIR)
        traj_iterations = np.zeros((n_runs, n_q))
        traj_inflation  = np.zeros((n_runs, n_q))
        traj_ipct_gdp   = np.zeros((n_runs, n_q))
        traj_gdp_growth = np.zeros((n_runs, n_q - 1))
        traj_gdp_level  = np.zeros((n_runs, n_q))
        traj_alpha_gap  = np.zeros((n_runs, n_q))
        traj_price_drift = np.zeros((n_runs, n_q))
        traj_labor_slack = np.zeros((n_runs, n_q))
        traj_cap_slack  = np.zeros((n_runs, n_q))
        traj_lambda_K   = np.zeros((n_runs, n_q))
        traj_lambda_L   = np.zeros((n_runs, n_q))

        for i in range(n_runs):
            cfg_i = {**config, "seed": 1000 + i}
            state = calibrate(
                data,
                delta=cfg_i.get("delta", 0.015),
                pref_drift_rho=cfg_i.get("pref_drift_rho", 0.95),
                pref_drift_sigma=cfg_i.get("pref_drift_sigma", 0.01),
                pref_noise_sigma=cfg_i.get("pref_noise_sigma", 0.001),
                theta_drift=cfg_i.get("theta_drift", 0.075),
                epsilon=cfg_i.get("epsilon", 0.5),
                neumann_k=cfg_i.get("neumann_k", 25),
                kappa_factor=cfg_i.get("kappa_factor", 4.0),
                L_total=cfg_i.get("L_total", 39e9),
                wage_rate=cfg_i.get("wage_rate", 21.0),
                primal_tol=cfg_i.get("primal_tol", 1e-3),
                dual_tol=cfg_i.get("dual_tol", 1e-4),
                eta_K=cfg_i.get("eta_K", 0.15),
                eta_L=cfg_i.get("eta_L", 0.15),
                max_iter=cfg_i.get("max_iter", 2000),
                g_step=cfg_i.get("g_step", 0.01),
                c_step=cfg_i.get("c_step", 0.01),
                habit_persistence=cfg_i.get("habit_persistence", 0.7),
                nominal_consumption_annual=cfg_i.get("nominal_consumption_annual", 807e9),
                labor_mult=cfg_i.get("labor_mult", 1.0),
            )
            state.slim_history = True
            state.rng = np.random.default_rng(1000 + i)
            state = run_simulation(state, n_quarters=n_q)
            hist  = state.history
            gdp_q1 = hist[0]["GDP"]
            for q in range(len(hist)):
                h = hist[q]
                traj_iterations[i, q]  = h.get("iterations", 0)
                traj_inflation[i, q]   = h.get("Inflation", 0.0)
                traj_ipct_gdp[i, q]    = h.get("I_pct_GDP", 0.0)
                traj_gdp_level[i, q]   = (h["GDP"] / gdp_q1) * 100.0
                traj_alpha_gap[i, q]   = h.get("alpha_gap", 0.0)
                traj_price_drift[i, q] = h.get("price_drift", 0.0)
                traj_labor_slack[i, q] = h.get("labor_slack", 0.0)
                traj_cap_slack[i, q]   = (h.get("slack_val_Q1", 0.0) / max(h.get("K_val_Q1", 1.0), 1e-30)) * 100.0
                traj_lambda_K[i, q]    = h.get("lambda_K_mean", 0.0)
                traj_lambda_L[i, q]    = h.get("lambda_L", 0.0)
                if q > 0:
                    traj_gdp_growth[i, q-1] = (h["GDP"] / hist[q-1]["GDP"] - 1.0) * 100.0

            _mc_state.update({"progress": i + 1, "total": n_runs})
            _mc_queue.put({"type": "progress", "progress": i + 1, "total": n_runs,
                           "msg": f"Run {i+1}/{n_runs} complete"})

        from datetime import datetime
        mc_dir = SCRIPTS_DIR.parent / "Results" / "MonteCarlo" / datetime.now().strftime("%Y%m%d_%H%M%S")
        mc_dir.mkdir(parents=True, exist_ok=True)
        _last_mc_dir = mc_dir

        charts_mc = [
            (traj_gdp_level,         "Real GDP Level (Q1=100)",         "Index",        "gdp_level"),
            (traj_gdp_growth,        "Q-o-Q GDP Growth Rate",           "Growth (%)",   "gdp_growth"),
            (traj_inflation * 100,   "Quarterly Inflation Rate",         "Inflation (%)", "inflation"),
            (traj_ipct_gdp,          "Investment / GDP Ratio",           "Share (%)",    "investment"),
            (traj_alpha_gap,         "Preference Tracking Error",        "L2 Norm",      "alpha_gap"),
            (traj_price_drift * 100, "Price Drift",                     "Deviation (%)", "price_drift"),
            (traj_labor_slack * 100, "Labor Slack",                     "Unused (%)",   "labor_slack"),
            (traj_cap_slack,         "Capital Capacity Slack",           "Unused (%)",   "capital_slack"),
            (traj_lambda_K,          "Shadow Price of Capital",          "Shadow Value", "lambda_k"),
            (traj_lambda_L,          "Shadow Price of Labor",            "Shadow Value", "lambda_l"),
        ]
        generated = []
        for data_arr, title, ylabel, fname in charts_mc:
            try:
                plot_fan_chart(data_arr, f"Monte Carlo: {title}", ylabel, mc_dir / f"fan_{fname}.png")
                generated.append(fname)
            except Exception as e:
                _mc_queue.put({"type": "log", "msg": f"  [WARN] {fname}: {e}"})

        try:
            plot_iteration_histogram(traj_iterations, mc_dir / "iterations.png")
            generated.append("iterations")
        except Exception as e:
            _mc_queue.put({"type": "log", "msg": f"  [WARN] iterations: {e}"})

        _mc_state.update({"status": "done", "charts": generated, "mc_dir": str(mc_dir)})
        _mc_queue.put({"type": "done", "charts": generated})

    except Exception as e:
        _mc_state.update({"status": "error", "error": str(e)})
        _mc_queue.put({"type": "error", "msg": str(e)})


@app.post("/montecarlo/start")
async def start_mc(config: dict):
    with _lock:
        if _mc_state.get("status") == "running":
            return {"ok": False, "reason": "Monte Carlo is already running"}
        while not _mc_queue.empty():
            try: _mc_queue.get_nowait()
            except: break
        _mc_state.update({"status": "starting", "progress": 0})

    t = threading.Thread(target=_mc_thread, args=(config,), daemon=True)
    t.start()
    return {"ok": True}


@app.get("/montecarlo/stream")
async def stream_mc():
    async def generator():
        ping_ticks = 0
        while True:
            try:
                msg = _mc_queue.get_nowait()
                yield {"data": json.dumps(msg)}
                if msg.get("type") in ("done", "error"):
                    break
            except queue.Empty:
                await asyncio.sleep(0.5)
                ping_ticks += 1
                if ping_ticks >= 20:
                    ping_ticks = 0
                    yield {"data": json.dumps({"type": "ping"})}
    return EventSourceResponse(generator())


@app.get("/montecarlo/status")
def mc_status():
    return _mc_state


@app.get("/montecarlo/charts/{chart_key}")
def get_mc_chart(chart_key: str):
    mc_dir = _mc_state.get("mc_dir")
    if not mc_dir:
        return JSONResponse({"error": "no results yet"}, status_code=404)
    from fastapi.responses import FileResponse
    matches = list(Path(mc_dir).glob(f"*{chart_key}*.png"))
    if not matches:
        return JSONResponse({"error": "chart not found"}, status_code=404)
    return FileResponse(matches[0], media_type="image/png")


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
