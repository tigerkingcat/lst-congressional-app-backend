import pickle
from fastapi import Body
import pandas as pd
from pathlib import Path
import gc

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model3_rf_trained_model_1.pkl"

# Global model variable - starts as None
_model = None

def load_model():
    """Lazy load the model only when needed"""
    global _model
    if _model is None:
        print("Loading model for the first time...")
        with MODEL_PATH.open("rb") as f:
            _model = pickle.load(f)
        print("Model loaded successfully!")
        # Force garbage collection to free memory
        gc.collect()
    return _model

async def predictLST(payload: dict = Body(...)):
    # Load model only when prediction is requested
    model = load_model()

    COLUMN_MAP = {
            "climate_category_Arid_Cold": "climate_category_Arid (Cold)",
            "climate_category_Arid_Hot": "climate_category_Arid (Hot)",
            "climate_category_Semi_Arid_Cold": "climate_category_Semi-Arid (Cold)",
    }

    FEATURE_ORDER = [
        'impervious', 'Pv', 'NDWI', 'elev', 'climate_category_Arid (Cold)', 'climate_category_Arid (Hot)',
         'climate_category_Mediterranean', 'climate_category_Semi-Arid (Cold)', 'urban_rural_classification_U',
         'urban_rural_classification_nan', 'Median_Household_Income', 'High_School_Diploma_25plus',
         'Unemployment', 'Median_Housing_Value', 'Median_Gross_Rent', 'Renter_Occupied_Housing_Units', 'Total_Population', 'Median_Age', 'Per_Capita_Income', 'Families_Below_Poverty',
         'year_centered'
    ]
    
    features = payload.get('features', payload)
    df = pd.DataFrame([features])
    df.rename(columns=COLUMN_MAP, inplace=True)
    df = df.reindex(columns=FEATURE_ORDER)

    lst = float(model.predict(df)[0])
    return {"lst": lst}
