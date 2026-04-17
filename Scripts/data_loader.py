"""
data_loader.py
--------------
Loads and validates the Spanish 2022 IO data from the three Excel files.

Raw source files are published by INE in *millions of EUR* at *annual* frequency.

Unit pipeline (applied here once, so every downstream module sees clean units):
  Step 1 – Scale:    Multiply all monetary arrays by 1e6  →  values in EUR.
  Step 2 – Temporal: Divide all flow variables by 4       →  values are quarterly.

Stocks (e.g. capital K, computed in calibration from flow data) are NOT
divided by 4; only flows (output, consumption, investment, government spend,
value added) are.  The inverse operation — quarterly → annual — is simply
multiplication by 4, as per standard econometric convention.

Returns clean numpy arrays ready for calibration.

Performance note
----------------
The three Excel files are read once and cached as a compressed pickle in
Data/.data_cache.pkl.  On every subsequent run the cache is validated
against the source files' modification timestamps -- if none of the files
have changed, the cache is returned immediately (~5 ms vs ~2 s for xlsx).
Delete .data_cache.pkl to force a full reload.
"""

import hashlib
import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "Data"

# Filename map -- tries both underscore and space variants automatically
_A_NAMES  = ["Spanish_A-matrix.xlsx",  "Spanish A-matrix.xlsx"]
_V_NAMES  = ["Value_added.xlsx",        "Value added.xlsx"]
_CP_NAMES = ["Consumption_and_total_production.xlsx",
             "Consumption and total production.xlsx"]

# ---------------------------------------------------------------------- #
#  Unit conversion constants                                              #
# ---------------------------------------------------------------------- #
_M_EUR_TO_EUR      = 1e6   # source files are in millions of EUR
_ANNUAL_TO_QUARTER = 4     # IO table is annual; model runs quarterly


def _find(data_dir: Path, candidates: list) -> Path:
    """Return the first existing filename from candidates, or raise clearly."""
    for name in candidates:
        p = data_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find any of {candidates} in {data_dir}.\n"
        f"Please place the Excel files in: {data_dir.resolve()}"
    )


def _fingerprint(data_dir: Path) -> str:
    """Return an MD5 of the combined mtimes of all three source files."""
    paths = [
        _find(data_dir, _A_NAMES),
        _find(data_dir, _V_NAMES),
        _find(data_dir, _CP_NAMES),
    ]
    sig = "".join(str(p.stat().st_mtime_ns) for p in paths)
    return hashlib.md5(sig.encode()).hexdigest()


def _cache_path(data_dir: Path) -> Path:
    return data_dir / ".data_cache.pkl"


def annual_to_quarterly(arr: np.ndarray) -> np.ndarray:
    """
    Convert an annual flow (EUR/year) to a quarterly flow (EUR/quarter).
    Standard econometric convention: divide by 4.
    The inverse operation (quarterly → annual) is multiplication by 4.
    """
    return arr / _ANNUAL_TO_QUARTER


def annualise(quarterly_value) -> float:
    """
    Convert a quarterly flow to its annualised equivalent.
    Standard econometric convention: multiply by 4.
    Works for scalars and numpy arrays.
    """
    return quarterly_value * _ANNUAL_TO_QUARTER


def _load_from_xlsx(data_dir: Path) -> dict:
    """Parse all three Excel files and return a dict of clean arrays."""

    # ------------------------------------------------------------------ #
    #  A matrix  (65 x 65 technical coefficients)                         #
    #  Pure ratios -- no unit conversion needed.                          #
    # ------------------------------------------------------------------ #
    df_a = pd.read_excel(_find(data_dir, _A_NAMES), index_col=0)
    # Row 0 of the raw file contains sector-number labels -- skip it
    A_df = df_a.iloc[1:, :]
    sector_names = list(A_df.columns)
    A = A_df.values.astype(float)

    assert A.shape == (65, 65), f"Expected (65,65), got {A.shape}"
    assert not np.isnan(A).any(), "NaN values in A matrix"
    assert (A >= 0).all(), "Negative entries in A matrix"

    # ------------------------------------------------------------------ #
    #  Value added  (1 x 65 row -- total value added per sector)          #
    #  Raw: annual M EUR  →  Step 1: EUR  →  Step 2: EUR/quarter          #
    # ------------------------------------------------------------------ #
    df_v   = pd.read_excel(_find(data_dir, _V_NAMES), header=None)
    V_raw  = df_v.iloc[0, :].values.astype(float)             # M EUR/year
    V_total = annual_to_quarterly(V_raw * _M_EUR_TO_EUR)       # EUR/quarter

    assert len(V_total) == 65, f"Expected 65 value-added entries, got {len(V_total)}"

    # ------------------------------------------------------------------ #
    #  Consumption & total production  (65 sectors x 7 columns)           #
    #  Raw: annual M EUR  →  Step 1: EUR  →  Step 2: EUR/quarter          #
    # ------------------------------------------------------------------ #
    # Row 0: column headers, Row 1: empty, Rows 2-66: sector data
    # Col 0: household consumption
    # Col 2: gross capital formation
    # Col 4: public administration consumption
    # Col 6: total use (gross output)
    df_cp   = pd.read_excel(_find(data_dir, _CP_NAMES), header=None)

    C_raw       = df_cp.iloc[2:, 0].values.astype(float)   # M EUR/year
    I_gross_raw = df_cp.iloc[2:, 2].values.astype(float)   # M EUR/year
    G_raw_raw   = df_cp.iloc[2:, 4].values.astype(float)   # M EUR/year
    X_raw       = df_cp.iloc[2:, 6].values.astype(float)   # M EUR/year

    assert len(C_raw) == 65 and len(X_raw) == 65, "C/X length mismatch"

    # Clip negative capital formation to 0 (inventory draw-downs don't
    # reduce physical capital stock in this model)
    I_gross_raw = np.clip(I_gross_raw, 0, None)

    # Step 1: M EUR → EUR; Step 2: annual flow → quarterly flow
    C       = annual_to_quarterly(C_raw       * _M_EUR_TO_EUR)
    I_gross = annual_to_quarterly(I_gross_raw * _M_EUR_TO_EUR)
    G_raw   = annual_to_quarterly(G_raw_raw   * _M_EUR_TO_EUR)
    X       = annual_to_quarterly(X_raw       * _M_EUR_TO_EUR)

    # Short sector labels (first 35 chars) for axis ticks
    sector_short = [s[:35].strip() for s in sector_names]

    # Log annualised totals (×4) so numbers match published national accounts
    logger.info(f"[data_loader] Loaded {len(sector_names)} sectors.")
    logger.info(f"  Annualised GDP (sum of VA)  : {annualise(V_total.sum()) / 1e9:,.1f} B EUR")
    logger.info(f"  Annualised gross output     : {annualise(X.sum())        / 1e9:,.1f} B EUR")
    logger.info(f"  Annualised household C      : {annualise(C.sum())        / 1e9:,.1f} B EUR")
    logger.info(f"  (Quarterly values = annualised / 4)")

    return dict(
        A=A,
        V_total=V_total,
        C=C,
        I_gross=I_gross,
        X=X,
        G_raw=G_raw,
        sector_names=sector_names,
        sector_short=sector_short,
    )


def load_data(data_dir: Path = DATA_DIR) -> dict:
    """
    Return parsed IO data, using a file-mtime-keyed cache to skip xlsx
    parsing on runs where none of the source files have changed.
    """
    cache = _cache_path(data_dir)

    # Attempt to read and validate the cache
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

    # Cache miss or stale -- parse xlsx
    data = _load_from_xlsx(data_dir)

    # Write cache (silently skip if the Data dir is read-only)
    try:
        with open(cache, "wb") as f:
            pickle.dump({"fingerprint": fp, "data": data}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"[data_loader] Cache written to {cache}.")
    except Exception as e:
        logger.warning(f"[data_loader] Could not write cache: {e}")

    return data


def sector_groups(sector_names: list) -> dict:
    """
    Classify 65 IO sectors into 8 broad groups for aggregate plotting.
    Returns a dict mapping group_name -> list of sector indices.
    """
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

    # Sanity check
    all_assigned = sum(len(v) for v in groups.values())
    if all_assigned != len(sector_names):
        unassigned = [i for i in range(len(sector_names))
                      if not any(i in v for v in groups.values())]
        logger.warning(
            f"[sector_groups] Warning: {len(sector_names)-all_assigned} "
            f"unassigned sectors: {unassigned}"
        )

    return groups
