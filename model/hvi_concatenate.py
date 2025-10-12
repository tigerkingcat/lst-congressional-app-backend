import pandas as pd
import numpy as np

points = pd.read_csv(r"C:\Users\aarav\App\backend\model\representative_points.csv")
hvi = pd.read_csv(r"C:\Users\aarav\App\backend\model\representative_points_with_HVI.csv")

merged_df = pd.merge(points, hvi[["polygon_id", "HVI_score", "HVI_level"]], on="polygon_id", how="left")
merged_df = merged_df.drop(columns="Bachelors_Degree_25plus")
merged_df.to_csv("final_hvi_data.csv", index=False)