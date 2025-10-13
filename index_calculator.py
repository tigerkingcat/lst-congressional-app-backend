# index_calculator.py
# One input dict (IndexVector) -> one output dict {"HVI": float}
# Baseline is loaded once (at import) for performance; the incoming dict is NOT mixed into baseline.
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from fastapi import Body

# =========================
#  CONFIG: CHANGE THIS PATH
# =========================
BASELINE_PATH = Path(r"./model/representative_points_with_HVI.csv")
# If your file is elsewhere, update the path above.

# -------------------------
#  Load & Calibrate Baseline
# -------------------------
if not BASELINE_PATH.exists():
    raise FileNotFoundError(f"Baseline file not found: {BASELINE_PATH}")

_df_base = pd.read_csv(BASELINE_PATH)

# Build baseline peer labels (climate × urban)
_CLIMATE_COLS_BASE = [
    'climate_category_Arid (Cold)',
    'climate_category_Arid (Hot)',
    'climate_category_Mediterranean',
    'climate_category_Semi-Arid (Cold)',
]

def _baseline_climate_lbl(row: pd.Series) -> str:
    for c in _CLIMATE_COLS_BASE:
        if row.get(c, 0) == 1:
            return c.replace('climate_category_', '')
    return 'Unknown'

_df_base['climate_lbl'] = _df_base.apply(_baseline_climate_lbl, axis=1)
_df_base['urban_lbl'] = np.where(
    _df_base.get('urban_rural_classification_U', 0) == 1, 'Urban', 'NonUrban'
)
_df_base['peer'] = _df_base['climate_lbl'] + '|' + _df_base['urban_lbl']

# Fit per-peer regression: LST_Celsius ~ elev
_peer_regression: Dict[str, Tuple[float, float]] = {}  # peer -> (slope, intercept)
_peer_mean_lst: Dict[str, float] = {}                  # peer -> mean LST

for peer, sub in _df_base.groupby('peer'):
    sub_ok = sub[['LST_Celsius', 'elev']].dropna()
    if len(sub_ok) >= 2 and sub_ok['elev'].std() > 0:
        slope, intercept = np.polyfit(sub_ok['elev'].values, sub_ok['LST_Celsius'].values, 1)
        _peer_regression[peer] = (float(slope), float(intercept))
    _peer_mean_lst[peer] = float(sub['LST_Celsius'].mean())

# Compute baseline LST_resid (needed to know peer ranges for residuals)
_lst_resid_vals = np.zeros(len(_df_base), dtype=float)
for peer, idx in _df_base.groupby('peer').groups.items():
    sub = _df_base.loc[idx, ['LST_Celsius', 'elev']].dropna()
    if len(sub) >= 2 and sub['elev'].std() > 0:
        slope, intercept = _peer_regression.get(peer, (0.0, _peer_mean_lst.get(peer, 0.0)))
        resid = sub['LST_Celsius'] - (intercept + slope * sub['elev'])
        _lst_resid_vals[sub.index] = resid
    else:
        mean_val = _df_base.loc[idx, 'LST_Celsius'].mean()
        _lst_resid_vals[idx] = _df_base.loc[idx, 'LST_Celsius'] - mean_val
_df_base['LST_resid'] = _lst_resid_vals

# Build per-peer min/max for scaling
_cols_needed = [
    'LST_resid', 'impervious', 'Pv', 'NDWI',
    'Families_Below_Poverty', 'Median_Age', 'Total_Population',
    'Unemployment', 'High_School_Diploma_25plus', 'Bachelors_Degree_25plus',
    'Median_Household_Income', 'Per_Capita_Income', 'Median_Housing_Value',
    'Median_Gross_Rent', 'Renter_Occupied_Housing_Units'
]

def _peer_minmax(df: pd.DataFrame, col: str) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    if col not in df.columns:
        return out
    for peer, sub in df.groupby('peer'):
        s = pd.to_numeric(sub[col], errors='coerce').astype(float)
        mn = float(np.nanmin(s.values)) if np.isfinite(np.nanmin(s.values)) else np.nan
        mx = float(np.nanmax(s.values)) if np.isfinite(np.nanmax(s.values)) else np.nan
        out[peer] = (mn, mx)
    return out

_peer_minmax_map: Dict[str, Dict[str, Tuple[float, float]]] = {
    col: _peer_minmax(_df_base, col) for col in _cols_needed
}

# Payload climate one-hot (snake_case) -> label used in baseline peers
_CLIMATE_LABEL_MAP = {
    'climate_category_Arid_Cold':      'Arid (Cold)',
    'climate_category_Arid_Hot':       'Arid (Hot)',
    'climate_category_Mediterranean':  'Mediterranean',
    'climate_category_Semi_Arid_Cold': 'Semi-Arid (Cold)',
}

# -------------------------
#  Helpers for single input
# -------------------------
def _payload_peer(payload: dict) -> str:
    climate_lbl = 'Unknown'
    for k, label in _CLIMATE_LABEL_MAP.items():
        if int(payload.get(k, 0)) == 1:
            climate_lbl = label
            break
    urban_lbl = 'Urban' if int(payload.get('urban_rural_classification_U', 0)) == 1 else 'NonUrban'
    return f"{climate_lbl}|{urban_lbl}"

def _lst_residual(peer: str, lst_value: float, elev_value: float) -> float:
    slope, intercept = _peer_regression.get(peer, (0.0, _peer_mean_lst.get(peer, 0.0)))
    if peer in _peer_regression and elev_value is not None and not np.isnan(elev_value):
        return float(lst_value - (intercept + slope * elev_value))
    # fallback: center by peer mean LST
    return float(lst_value - _peer_mean_lst.get(peer, 0.0))

def _scale_peer_value(x: Optional[float], mn: float, mx: float, flip: bool=False) -> float:
    # robust 0-1 scaler with protective flip; neutral (0.0) on missing/degenerate ranges
    x = float(x) if x is not None else np.nan
    if any(np.isnan(v) for v in [x, mn, mx]) or mx == mn:
        val = 0.0
    else:
        val = (x - mn) / (mx - mn)
        val = float(np.clip(val, 0.0, 1.0))
    return 1.0 - val if flip else val

def _scale_single(col: str, peer: str, x: Optional[float], flip: bool=False) -> float:
    mn, mx = _peer_minmax_map.get(col, {}).get(peer, (np.nan, np.nan))
    return _scale_peer_value(x, mn, mx, flip=flip)

# -------------------------
#  Main exported function
# -------------------------
async def predictHVI(payload: dict = Body(...)) -> dict:
    """
    Input:  one dict (IndexVector) from your frontend
    Output: {"HVI": <float>}  -- final index only
    """
    # Build the peer for THIS single observation
    peer = _payload_peer(payload)

    # LST residual by elevation, using this peer's regression or mean-centering
    lst_val  = float(payload.get('lst', np.nan))
    elev_val = float(payload.get('elev', np.nan))
    lst_resid = _lst_residual(peer, lst_val, elev_val)

    # ---- Environmental (E) ----
    p_LST  = _scale_single('LST_resid', peer, lst_resid, flip=False)
    p_imp  = _scale_single('impervious', peer, payload.get('impervious'), flip=False)
    p_pv   = _scale_single('Pv', peer, payload.get('Pv'), flip=True)      # protective
    p_ndwi = _scale_single('NDWI', peer, payload.get('NDWI'), flip=True)  # protective
    E = 0.8 * p_LST + 0.2 * (p_imp + p_pv + p_ndwi) / 3.0

    # ---- Sensitivity (S) ----
    s_terms = [
        _scale_single('Families_Below_Poverty',     peer, payload.get('Families_Below_Poverty')),
        _scale_single('Median_Age',                 peer, payload.get('Median_Age')),
        _scale_single('Total_Population',           peer, payload.get('Total_Population')),
        _scale_single('Unemployment',               peer, payload.get('Unemployment')),
        _scale_single('High_School_Diploma_25plus', peer, payload.get('High_School_Diploma_25plus'), flip=True),
    ]
    # Optional: BA flips if present in baseline AND payload
    S = float(np.mean(s_terms))

    # ---- Adaptive Capacity (AC) ----
    ac_terms = [
        _scale_single('Median_Household_Income',     peer, payload.get('Median_Household_Income'), flip=True),
        _scale_single('Per_Capita_Income',           peer, payload.get('Per_Capita_Income'),       flip=True),
        _scale_single('Median_Housing_Value',        peer, payload.get('Median_Housing_Value'),    flip=True),
        _scale_single('Median_Gross_Rent',           peer, payload.get('Median_Gross_Rent'),       flip=True),
        _scale_single('Renter_Occupied_Housing_Units', peer, payload.get('Renter_Occupied_Housing_Units')),
    ]
    AC = float(np.mean(ac_terms))

    # ---- Final index ----
    HVI_raw = 0.5 * E + 0.3 * S + 0.2 * AC
    HVI_raw *= 100

    points = [59.42, 53.34, 49.20, 43.13]

    if HVI_raw > points[0]:
        var1 = 4
    elif points[0] >= HVI_raw > points[1]:
        var1 = 3
    elif points[1] >= HVI_raw > points[2]:
        var1 = 2
    elif points[2] >= HVI_raw > points[3]:
        var1 = 1
    else:
        var1 = 0

    return {"HVI": round(float(HVI_raw), 4), "level": var1}
