#!/usr/bin/env python3
"""
Count how many CSV points fall inside one polygon selected by a variable named `id`.

Files (change if needed):
  GEOJSON = "/mnt/data/tl_2023_06_bg_fc.json"
  CSV     = "/mnt/data/processed_env_data_20250708_191941.csv"

Edit the `id` variable below to target a different polygon.
"""

import pandas as pd
import geopandas as gpd

# -----------------------------
# User-adjustable variables
# -----------------------------
id = 2607  # <- CHANGE THIS to your target polygon identifier (int or string)
GEOJSON = r"./model/tl_2023_06_bg_fc.json"
CSV     = r"./model/processed_env_data_20250708_191941.csv"
LON_COL = "lon"
LAT_COL = "lat"
SAVE_MATCHING_TO = None   # e.g., "inside_poly.csv" or leave as None to skip saving
# -----------------------------

# (Optional) robust geometry fix
try:
    from shapely.validation import make_valid
    def fix_geom(g): return make_valid(g)
except Exception:
    def fix_geom(g):
        try: return g.buffer(0)
        except Exception: return g

def pick_polygon_row(polys: gpd.GeoDataFrame, target):
    """
    Pick polygon row by matching `id` column if present, else `GEOID` if present.
    Accepts numeric or string `target`.
    """
    str_target = str(target)

    if 'id' in polys.columns:
        row = polys[(polys['id'] == target) | (polys['id'].astype(str) == str_target)]
        if not row.empty:
            return row.iloc[0]

    if 'GEOID' in polys.columns:
        row = polys[(polys['GEOID'] == str_target)]
        if not row.empty:
            return row.iloc[0]

    # Nothing matched—helpful error
    sample_cols = ['id', 'GEOID']
    have_cols = [c for c in sample_cols if c in polys.columns]
    msg = f"No polygon found for id={target}. "
    if have_cols:
        msg += f"Available identifier columns in GeoJSON: {have_cols}. "
    if 'id' in polys.columns:
        msg += f"Example ids: {list(polys['id'].astype(str).head(5))}"
    elif 'GEOID' in polys.columns:
        msg += f"Example GEOIDs: {list(polys['GEOID'].astype(str).head(5))}"
    raise ValueError(msg)

def main():
    # Load polygons
    polys = gpd.read_file(GEOJSON)

    # Ensure CRS is geographic (EPSG:4326)
    if polys.crs is None:
        polys = polys.set_crs(epsg=4326, allow_override=True)
    else:
        polys = polys.to_crs(epsg=4326)

    poly_row = pick_polygon_row(polys, id)
    poly_geom = fix_geom(poly_row.geometry)

    # Load points
    df = pd.read_csv(CSV)
    if LON_COL not in df.columns or LAT_COL not in df.columns:
        raise KeyError(f"CSV must contain columns '{LON_COL}' and '{LAT_COL}'. Found: {list(df.columns)}")

    df = df.dropna(subset=[LON_COL, LAT_COL]).copy()
    df[LON_COL] = df[LON_COL].astype(float)
    df[LAT_COL] = df[LAT_COL].astype(float)

    points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[LON_COL], df[LAT_COL]),
        crs="EPSG:4326"
    )

    # Count points inside (include boundary as inside if available)
    try:
        mask = points.covered_by(poly_geom)  # Shapely 2.x
    except Exception:
        mask = points.within(poly_geom) | points.touches(poly_geom)

    inside = points[mask]
    count = len(inside)

    # Print results
    id_label = "id" if "id" in polys.columns else ("GEOID" if "GEOID" in polys.columns else "identifier")
    print(f"Selected polygon {id_label} = {id}")
    print(f"Points inside = {count}")

    # Optional save
    if SAVE_MATCHING_TO:
        inside.drop(columns="geometry", errors="ignore").to_csv(SAVE_MATCHING_TO, index=False)
        print(f"Saved matching rows to: {SAVE_MATCHING_TO}")

if __name__ == "__main__":
    main()
