#Test to see if index is practical
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv(r"C:\Users\aarav\App\backend\model\representative_points.csv")
print(df)
df = df.dropna().reset_index(drop=True)
df = df.drop(columns="Bachelors_Degree_25plus")
print(df)
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = df.copy()

# ---------------------------------------------------------------------
# 1) Build peer-group labels from climate + urban/rural classification
# ---------------------------------------------------------------------
climate_cols = [
    'climate_category_Arid (Cold)',
    'climate_category_Arid (Hot)',
    'climate_category_Mediterranean',
    'climate_category_Semi-Arid (Cold)'
]

def climate_label(row):
    for c in climate_cols:
        if row.get(c, 0) == 1:
            return c.replace('climate_category_', '')
    return 'Unknown'

df['climate_lbl'] = df.apply(climate_label, axis=1)

df['urban_lbl'] = np.where(
    df['urban_rural_classification_U'] == 1,
    'Urban',
    'NonUrban'
)

df['peer'] = df['climate_lbl'] + '|' + df['urban_lbl']
print(df)
# ---------------------------------------------------------------------
# 2) Helper function for groupwise MinMax scaling
# ---------------------------------------------------------------------
def group_minmax(series: pd.Series, groups: pd.Series, flip: bool=False) -> pd.Series:
    """Groupwise 0–1 scaling that returns a float Series aligned to the original index."""
    def _scale_group(s: pd.Series) -> np.ndarray:
        # if group is constant or empty -> zeros
        arr = s.values.reshape(-1, 1).astype(float)
        # Check constant group to avoid NaNs from MinMaxScaler
        if np.all(np.isnan(arr)):
            scaled = np.zeros_like(arr, dtype=float)
        else:
            # If all non-nan values are identical, MinMaxScaler returns zeros (fine)
            scaler = MinMaxScaler()
            # For safety, fill NaNs with the group's median before fitting
            # (MinMaxScaler can't handle NaN). Then put NaNs back after scaling.
            mask = np.isnan(arr)
            if mask.any():
                fill_value = np.nanmedian(arr)
                if np.isnan(fill_value):
                    # whole group is NaN
                    scaled = np.zeros_like(arr, dtype=float)
                else:
                    arr_filled = arr.copy()
                    arr_filled[mask] = fill_value
                    scaled = scaler.fit_transform(arr_filled)
                    scaled[mask] = 0.0  # neutral for missing
            else:
                scaled = scaler.fit_transform(arr)
        scaled = scaled.squeeze()
        if flip:
            scaled = 1 - scaled
        return scaled  # same length as s

    return series.groupby(groups).transform(_scale_group).astype(float)
# ---------------------------------------------------------------------
# 3) Residualize LST by elevation inside each peer group
# ---------------------------------------------------------------------
def lst_residual_by_peer(df, lst_col='LST_Celsius', elev_col='elev', group_col='peer'):
    out = np.zeros(len(df), dtype=float)
    for group_name, idx in df.groupby(group_col).groups.items():
        subset = df.loc[idx, [lst_col, elev_col]].dropna()
        if len(subset) >= 2 and subset[elev_col].std() > 0:
            # Fit simple regression: LST ~ a + b*elev
            slope, intercept = np.polyfit(subset[elev_col].values,
                                          subset[lst_col].values, 1)
            residuals = subset[lst_col] - (intercept + slope * subset[elev_col])
            out[subset.index] = residuals
        else:
            # Not enough variation, just center LST
            mean_val = df.loc[idx, lst_col].mean()
            out[idx] = df.loc[idx, lst_col] - mean_val
    return pd.Series(out, index=df.index, name='LST_resid')

df['LST_resid'] = lst_residual_by_peer(df)
print(df)
# ---------------------------------------------------------------------
# 4) Exposure (E)
# ---------------------------------------------------------------------
p_LST  = group_minmax(df['LST_resid'], df['peer'], flip=False)
p_imp  = group_minmax(df['impervious'], df['peer'], flip=False)
p_pv   = group_minmax(df['Pv'], df['peer'], flip=True)    # more vegetation = protective
p_ndwi = group_minmax(df['NDWI'], df['peer'], flip=True)  # more water = protective
print(df)
df['E'] = (
    0.8 * p_LST +
    0.2 * (p_imp + p_pv + p_ndwi) / 3.0
)

# ---------------------------------------------------------------------
# 5) Sensitivity (S)
# ---------------------------------------------------------------------
p_pov  = group_minmax(df['Families_Below_Poverty'], df['peer'])
p_age  = group_minmax(df['Median_Age'], df['peer'])
p_pop  = group_minmax(df['Total_Population'], df['peer'])
p_unem = group_minmax(df['Unemployment'], df['peer'])
p_hs   = group_minmax(df['High_School_Diploma_25plus'], df['peer'], flip=True)

df['S'] = (
    p_pov + p_age + p_pop + p_unem + p_hs
) / 5.0
print(df)
# ---------------------------------------------------------------------
# 6) Adaptive Capacity (AC)
# ---------------------------------------------------------------------
p_inc     = group_minmax(df['Median_Household_Income'], df['peer'], flip=True)
p_pci     = group_minmax(df['Per_Capita_Income'], df['peer'], flip=True)
p_home    = group_minmax(df['Median_Housing_Value'], df['peer'], flip=True)
p_rent    = group_minmax(df['Median_Gross_Rent'], df['peer'], flip=True)
p_renters = group_minmax(df['Renter_Occupied_Housing_Units'], df['peer'])

df['AC'] = (
    p_inc + p_pci + p_home + p_rent + p_renters
) / 5.0

# ---------------------------------------------------------------------
# 7) Final HVI
# ---------------------------------------------------------------------
df['HVI_raw'] = (
    0.5 * df['E'] +
    0.3 * df['S'] +
    0.2 * df['AC']
)

df['HVI_score'] = (100 * df['HVI_raw']).round(2)
print(df)
df['HVI_level'] = pd.qcut(
    df['HVI_raw'],
    5,
    labels=[0, 1, 2, 3, 4],
    duplicates='drop'
)
output_path = r"C:\Users\aarav\App\backend\model\representative_points_with_HVI.csv"
df.to_csv(output_path, index=False)

print(f"Exported to {output_path}")