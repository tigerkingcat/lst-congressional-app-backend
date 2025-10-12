#!/usr/bin/env python3
"""
Standalone script to select one representative point per polygon block group.

- Reads polygons from a GeoJSON file with an 'id' property.
- Reads point data (including lat/lon and other attributes) from a CSV.
- For each polygon:
    * Finds all points within that polygon.
    * Computes the distance to the polygon's centroid.
    * Selects the nearest point (if any).
    * Ensures no point is used more than once (drops used points).
- Records one row per polygon, with `None` values where no points were found.
- Saves the representative points (with original CSV fields and `polygon_id`) to a new CSV.
"""
import geopandas as gpd
import pandas as pd

# --- Configuration ----------------------------------------------------------
POLY_FILE = "tl_2023_06_bg_fc.json"         # GeoJSON with polygons, must have 'id'
POINT_CSV = "processed_env_data_20250708_191941.csv"  # CSV with point data, needs 'lat','lon'
OUTPUT_CSV = "representative_points_test.csv"

# --- Load data ---------------------------------------------------------------
print("Loading polygon GeoJSON...")
poly_gdf = gpd.read_file(POLY_FILE)
if 'id' not in poly_gdf.columns:
    raise ValueError("GeoJSON must have an 'id' property on each feature.")
poly_gdf.set_index('id', inplace=True)

print("Loading point CSV...")
csv_df = pd.read_csv(POINT_CSV)
# Validate lat/lon columns
for col in ('lat', 'lon'):
    if col not in csv_df.columns:
        raise ValueError(f"CSV must have a '{col}' column.")

# Prepare GeoDataFrame for points
df_columns = list(csv_df.columns)
pts_gdf = gpd.GeoDataFrame(
    csv_df,
    geometry=gpd.points_from_xy(csv_df.lon, csv_df.lat),
    crs="EPSG:4326"
)

# --- Select representative points -------------------------------------------
records = []
print("Selecting representative points...")
for poly_id, poly_row in poly_gdf.iterrows():
    centroid = poly_row.geometry.centroid

    # Points strictly inside the polygon
    pts_within = pts_gdf[pts_gdf.geometry.within(poly_row.geometry)]

    if pts_within.empty:
        # No points: record None for all CSV fields
        rec = {col: None for col in df_columns}
        rec['polygon_id'] = poly_id
        records.append(rec)
        print(f"No points in polygon {poly_id}, recorded None.")
        continue

    # Compute distance to centroid
    pts_within = pts_within.copy()
    pts_within['distance'] = pts_within.geometry.distance(centroid)
    print(f" Length of polygon {poly_id} is {len(pts_within)}")
    # Pick nearest point
    nearest = pts_within.loc[pts_within['distance'].idxmin()]

    # Record all CSV fields + polygon ID
    rec = nearest[df_columns].to_dict()
    rec['polygon_id'] = poly_id
    records.append(rec)

    # Remove the selected point so it's not reused
    pts_gdf = pts_gdf.drop(nearest.name)

# --- Save output -------------------------------------------------------------
out_df = pd.DataFrame(records)
print(f"Saving {len(out_df)} representative points to {OUTPUT_CSV}...")
out_df.to_csv(OUTPUT_CSV, index=False)
print("Done.")