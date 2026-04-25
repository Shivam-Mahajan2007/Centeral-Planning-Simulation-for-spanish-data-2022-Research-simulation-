"""
data_loader.py
--------------
Loads and validates the Spanish 2022 IO data from three Excel source files.

Raw files are published by INE in millions of EUR at annual frequency.
I apply two unit conversions once here so every downstream module sees clean units:
  1. Scale by 1e6  → values in EUR
  2. Divide by 4   → quarterly flows

Stocks (capital K) are not divided by 4; only flows are.
The inverse operation (quarterly → annual) is multiplication by 4.

A file-mtime-keyed pickle cache avoids re-parsing xlsx on unchanged runs.
Delete Data/.data_cache.pkl to force a full reload.
"""

import hashlib
import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "Data"

_A_NAMES  = ["Spanish_A-matrix.xlsx",  "Spanish A-matrix.xlsx"]
_V_NAMES  = ["Value_added.xlsx",        "Value added.xlsx"]
_CP_NAMES = ["Consumption_and_total_production.xlsx",
             "Consumption and total production.xlsx"]

_M_EUR_TO_EUR      = 1e6
_ANNUAL_TO_QUARTER = 4


def _find(data_dir: Path, candidates: list) -> Path:
    for name in candidates:
        p = data_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find any of {candidates} in {data_dir}.\n"
        f"Please place the Excel files in: {data_dir.resolve()}"
    )


def _fingerprint(data_dir: Path) -> str:
    paths = [_find(data_dir, n) for n in [_A_NAMES, _V_NAMES, _CP_NAMES]]
    sig   = "".join(str(p.stat().st_mtime_ns) for p in paths)
    return hashlib.md5(sig.encode()).hexdigest()


def _cache_path(data_dir: Path) -> Path:
    return data_dir / ".data_cache.pkl"


def annual_to_quarterly(arr: np.ndarray) -> np.ndarray:
    return arr / _ANNUAL_TO_QUARTER


def annualise(quarterly_value) -> float:
    return quarterly_value * _ANNUAL_TO_QUARTER


def _load_from_xlsx(data_dir: Path) -> dict:
    # A matrix (65×65 technical coefficients — pure ratios, no unit conversion)
    df_a = pd.read_excel(_find(data_dir, _A_NAMES), index_col=0)
    A_df = df_a.iloc[1:, :]
    sector_names = list(A_df.columns)
    A = A_df.values.astype(float)

    assert A.shape == (65, 65), f"Expected (65,65), got {A.shape}"
    assert not np.isnan(A).any(), "NaN values in A matrix"
    assert (A >= 0).all(), "Negative entries in A matrix"

    # Value added (1×65, annual M EUR → EUR/quarter)
    df_v    = pd.read_excel(_find(data_dir, _V_NAMES), header=None)
    V_total = annual_to_quarterly(df_v.iloc[0, :].values.astype(float) * _M_EUR_TO_EUR)
    assert len(V_total) == 65

    # Consumption & total production (65×7, annual M EUR → EUR/quarter)
    df_cp       = pd.read_excel(_find(data_dir, _CP_NAMES), header=None)
    C_raw       = df_cp.iloc[2:, 0].values.astype(float)
    I_gross_raw = np.clip(df_cp.iloc[2:, 2].values.astype(float), 0, None)
    G_raw_raw   = df_cp.iloc[2:, 4].values.astype(float)
    X_raw       = df_cp.iloc[2:, 6].values.astype(float)
    assert len(C_raw) == 65 and len(X_raw) == 65

    C       = annual_to_quarterly(C_raw       * _M_EUR_TO_EUR)
    I_gross = annual_to_quarterly(I_gross_raw * _M_EUR_TO_EUR)
    G_raw   = annual_to_quarterly(G_raw_raw   * _M_EUR_TO_EUR)
    X       = annual_to_quarterly(X_raw       * _M_EUR_TO_EUR)

    sector_short = [s[:35].strip() for s in sector_names]

    logger.info(f"[data_loader] Loaded {len(sector_names)} sectors.")
    logger.info(f"  Annualised GDP (sum of VA)  : {annualise(V_total.sum()) / 1e9:,.1f} B EUR")
    logger.info(f"  Annualised gross output     : {annualise(X.sum())        / 1e9:,.1f} B EUR")
    logger.info(f"  Annualised household C      : {annualise(C.sum())        / 1e9:,.1f} B EUR")

    return dict(A=A, V_total=V_total, C=C, I_gross=I_gross, X=X, G_raw=G_raw,
                sector_names=sector_names, sector_short=sector_short)


def load_data(data_dir: Path = DATA_DIR) -> dict:
    """Return parsed IO data, using a file-mtime cache to skip xlsx parsing when unchanged."""
    cache = _cache_path(data_dir)
    try:
        fp = _fingerprint(data_dir)
        if cache.exists():
            with open(cache, "rb") as f:
                cached = pickle.load(f)
            if cached.get("fingerprint") == fp:
                logger.info("[data_loader] Using cached data (xlsx files unchanged).")
                return cached["data"]
    except Exception as e:
        logger.warning(f"[data_loader] Cache check failed ({e}), reloading from xlsx.")

    data = _load_from_xlsx(data_dir)
    try:
        with open(cache, "wb") as f:
            pickle.dump({"fingerprint": fp, "data": data}, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"[data_loader] Cache written to {cache}.")
    except Exception as e:
        logger.warning(f"[data_loader] Could not write cache: {e}")

    return data


def sector_groups(sector_names: list) -> dict:
    """Classify 65 IO sectors into 8 broad groups for aggregate plotting."""
    groups = {
        "Agriculture & Fishing": [],
        "Mining & Energy":       [],
        "Manufacturing":         [],
        "Construction":          [],
        "Trade & Transport":     [],
        "Finance & Real Estate": [],
        "ICT & Business Svcs":   [],
        "Public & Social Svcs":  [],
    }

    for i, name in enumerate(sector_names):
        n = name.lower()
        if any(k in n for k in ["agricultur", "forestr", "fish"]):
            groups["Agriculture & Fishing"].append(i)
        elif any(k in n for k in ["mining", "quarrying", "petroleum", "electricity",
                                   "gas", "steam", "water treatment", "sewerage"]):
            groups["Mining & Energy"].append(i)
        elif any(k in n for k in ["food", "textile", "leather", "wood", "paper",
                                   "printing", "chemical", "pharmaceutical", "rubber",
                                   "plastic", "mineral", "metal", "computer, elec",
                                   "electrical equip", "machinery", "motor vehicle",
                                   "transport equip", "furniture", "repair and install"]):
            groups["Manufacturing"].append(i)
        elif "construct" in n:
            groups["Construction"].append(i)
        elif any(k in n for k in ["wholesale", "retail", "trade", "land transport",
                                   "water transport", "air transport", "warehousing",
                                   "postal", "accommodation"]):
            groups["Trade & Transport"].append(i)
        elif any(k in n for k in ["financial", "insurance", "real estate",
                                   "imputed rent", "auxiliary to financial"]):
            groups["Finance & Real Estate"].append(i)
        elif any(k in n for k in ["publishing", "motion picture", "telecommunication",
                                   "computer programming", "legal", "architectural",
                                   "scientific research", "advertising", "professional",
                                   "rental and leas", "employment", "travel agency",
                                   "security", "services auxiliary"]):
            groups["ICT & Business Svcs"].append(i)
        else:
            groups["Public & Social Svcs"].append(i)

    all_assigned = sum(len(v) for v in groups.values())
    if all_assigned != len(sector_names):
        logger.warning(f"[sector_groups] {len(sector_names)-all_assigned} unassigned sectors")

    return groups
