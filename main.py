# main.py
from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from service import predictLST
from index_calculator import predictHVI
import pandas as pd

# --- Load your CSV (adjust path if needed) ---
df = pd.read_csv(r"./model/final_hvi_data.csv")

# --- Key mapping you already had ---
KEY_MAP = {
    "LST_Celsius":                      "lst_celsius",
    "impervious":                       "impervious",
    "Pv":                               "pv",
    "NDWI":                             "ndwi",
    "elev":                             "elev",
    "Median_Household_Income":          "median_household_income",
    "High_School_Diploma_25plus":       "high_school_diploma_25plus",
    "Unemployment":                     "unemployment",
    "Median_Housing_Value":             "median_housing_value",
    "Median_Gross_Rent":                "median_gross_rent",
    "Renter_Occupied_Housing_Units":    "renter_occupied_housing_units",
    "Total_Population":                 "total_population",
    "Median_Age":                       "median_age",
    "Per_Capita_Income":                "per_capita_income",
    "Families_Below_Poverty":           "families_below_poverty",
    "year_centered":                    "year_centered",
    "climate_category_Arid (Cold)":     "climate_category_Arid_Cold",
    "climate_category_Arid (Hot)":      "climate_category_Arid_Hot",
    "climate_category_Mediterranean":   "climate_category_Mediterranean",
    "climate_category_Semi-Arid (Cold)":"climate_category_Semi_Arid_Cold",
    "urban_rural_classification_U":     "urban_rural_classification_U",
    "urban_rural_classification_nan":   "urban_rural_classification_nan",
    "polygon_id":                       "polygon_id",
}

app = FastAPI()

# --- CORS: allow Vite dev server origins; enable credentials if you need cookies/auth ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,   # set True if you might send cookies/Authorization
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/hello")
async def hello():
    return {"status": "ok"}

class FeatureVector(BaseModel):
    impervious: float
    Pv: float
    NDWI: float
    elev: float
    climate_category_Arid_Cold: int
    climate_category_Arid_Hot: int
    climate_category_Mediterranean: int
    climate_category_Semi_Arid_Cold: int
    urban_rural_classification_U: int
    urban_rural_classification_nan: int
    Median_Household_Income: float
    High_School_Diploma_25plus: float
    Unemployment: float
    Median_Housing_Value: float
    Median_Gross_Rent: float
    Renter_Occupied_Housing_Units: float
    Total_Population: float
    Median_Age: float
    Per_Capita_Income: float
    Families_Below_Poverty: float
    year_centered: float

class IndexVector(BaseModel):
    lst: float
    impervious: float
    Pv: float
    NDWI: float
    elev: float
    climate_category_Arid_Cold: int
    climate_category_Arid_Hot: int
    climate_category_Mediterranean: int
    climate_category_Semi_Arid_Cold: int
    urban_rural_classification_U: int
    urban_rural_classification_nan: int
    Median_Household_Income: float
    High_School_Diploma_25plus: float
    Unemployment: float
    Median_Housing_Value: float
    Median_Gross_Rent: float
    Renter_Occupied_Housing_Units: float
    Total_Population: float
    Median_Age: float
    Per_Capita_Income: float
    Families_Below_Poverty: float
    year_centered: float

@app.post("/api/predict-hvi")
async def predict_hvi(payload: IndexVector):
    features = payload.dict()
    index = await predictHVI(features)
    return index


@app.post("/api/predict-lst")
async def predict_lst(payload: FeatureVector):
    features = payload.dict()
    lst = await predictLST(features)
    return lst

@app.post("/api/receive-data")
async def receive_data(payload: int = Body(...)):
    match = df[df["polygon_id"] == payload]
    if match.empty:
        raise HTTPException(status_code=404, detail="No data for that polygon_id")

    row = match.iloc[0]
    raw = row.to_dict()

    result = {}
    for orig_key, val in raw.items():
        new_key = KEY_MAP.get(orig_key, orig_key)
        result[new_key] = val

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
