"""
FastAPI Prediction Service

"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# App initialization 

app = FastAPI(
    title="House Price Prediction API",
    description="""
Predict house sale prices using an ML model trained on the Ames, Iowa dataset.

**How to use:**
1. POST to `/predict` with house features as JSON
2. Receive predicted price in USD

**Model:** XGBoost (best CV RMSE on Kaggle House Prices benchmark)
    """,
    version="1.0.0",
)

# Allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model loading

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl"),
)

_model = None  # Lazy-loaded on first request

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run `python src/train.py` first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


# Request / Response schemas 

class HouseFeatures(BaseModel):
    """
    Input features for house price prediction.

    """
    # Location & zoning
    MSZoning: str = Field(default="RL", description="General zoning classification")
    Neighborhood: str = Field(default="CollgCr", description="Physical location in Ames")
    Condition1: str = Field(default="Norm")
    Condition2: str = Field(default="Norm")

    # Lot
    LotFrontage: Optional[float] = Field(default=69.0, description="Linear feet of street connected to property")
    LotArea: float = Field(default=10500, description="Lot size in square feet")
    Street: str = Field(default="Pave")
    LotShape: str = Field(default="Reg")
    LandContour: str = Field(default="Lvl")
    LotConfig: str = Field(default="Inside")
    LandSlope: str = Field(default="Gtl")

    # Building basics
    BldgType: str = Field(default="1Fam")
    HouseStyle: str = Field(default="2Story")
    OverallQual: int = Field(default=6, ge=1, le=10, description="Overall material and finish quality (1-10)")
    OverallCond: int = Field(default=5, ge=1, le=10, description="Overall condition (1-10)")
    YearBuilt: int = Field(default=2000)
    YearRemodAdd: int = Field(default=2000)
    RoofStyle: str = Field(default="Gable")
    RoofMatl: str = Field(default="CompShg")

    # Exterior
    Exterior1st: str = Field(default="VinylSd")
    Exterior2nd: str = Field(default="VinylSd")
    MasVnrType: str = Field(default="None")
    MasVnrArea: float = Field(default=0.0)
    ExterQual: str = Field(default="TA")
    ExterCond: str = Field(default="TA")
    Foundation: str = Field(default="PConc")

    # Basement
    BsmtQual: str = Field(default="TA")
    BsmtCond: str = Field(default="TA")
    BsmtExposure: str = Field(default="No")
    BsmtFinType1: str = Field(default="Unf")
    BsmtFinSF1: float = Field(default=0.0)
    BsmtFinType2: str = Field(default="Unf")
    BsmtFinSF2: float = Field(default=0.0)
    BsmtUnfSF: float = Field(default=500.0)
    TotalBsmtSF: float = Field(default=1000.0)

    # HVAC
    Heating: str = Field(default="GasA")
    HeatingQC: str = Field(default="Ex")
    CentralAir: str = Field(default="Y")
    Electrical: str = Field(default="SBrkr")

    # Square footage
    FirstFlrSF: float = Field(default=1000.0, alias="1stFlrSF")
    SecondFlrSF: float = Field(default=700.0, alias="2ndFlrSF")
    LowQualFinSF: float = Field(default=0.0)
    GrLivArea: float = Field(default=1700.0, description="Above grade living area sq ft")

    # Bathrooms & bedrooms
    BsmtFullBath: float = Field(default=0.0)
    BsmtHalfBath: float = Field(default=0.0)
    FullBath: int = Field(default=2)
    HalfBath: int = Field(default=1)
    BedroomAbvGr: int = Field(default=3)
    KitchenAbvGr: int = Field(default=1)
    KitchenQual: str = Field(default="TA")
    TotRmsAbvGrd: int = Field(default=7)
    Functional: str = Field(default="Typ")

    # Fireplace
    Fireplaces: int = Field(default=0)
    FireplaceQu: str = Field(default="NA")

    # Garage
    GarageType: str = Field(default="Attchd")
    GarageYrBlt: float = Field(default=2000.0)
    GarageFinish: str = Field(default="Unf")
    GarageCars: float = Field(default=2.0)
    GarageArea: float = Field(default=480.0)
    GarageQual: str = Field(default="TA")
    GarageCond: str = Field(default="TA")
    PavedDrive: str = Field(default="Y")

    # Outdoor
    WoodDeckSF: float = Field(default=0.0)
    OpenPorchSF: float = Field(default=0.0)
    EnclosedPorch: float = Field(default=0.0)
    ThreeSsnPorch: float = Field(default=0.0, alias="3SsnPorch")
    ScreenPorch: float = Field(default=0.0)
    PoolArea: float = Field(default=0.0)
    Fence: str = Field(default="NA")
    MiscFeature: str = Field(default="NA")

    # Sale info
    MoSold: int = Field(default=6, ge=1, le=12)
    YrSold: int = Field(default=2010)
    SaleType: str = Field(default="WD")
    SaleCondition: str = Field(default="Normal")

    class Config:
        populate_by_name = True  # allow both alias and field name


class PredictionResponse(BaseModel):
    predicted_price: float = Field(description="Predicted sale price in USD")
    log_prediction: float = Field(description="Raw log-scale prediction (for debugging)")
    model_version: str = Field(description="Model file used")


# Endpoints

@app.get("/")
def root():
    return {
        "message": "House Price Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }


@app.get("/health")
def health_check():
    """
    A /health endpoint is essential for:
      - Docker HEALTHCHECK instructions
      - Kubernetes liveness probes
      - Load balancer checks (is this instance alive?)
    It should return fast and confirm the model is loaded.
    
    """
    try:
        model = get_model()
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "model_path": MODEL_PATH,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    """
    Predict the sale price for a house.

    Returns the predicted price in USD along with the raw log-scale prediction.
    """
    model = get_model()

    # Convert Pydantic model -> dict -> DataFrame
    feature_dict = features.model_dump(by_alias=True)

    # Build a single-row DataFrame
    df = pd.DataFrame([feature_dict])

    # Predict (model returns log1p-transformed price)
    log_prediction = model.predict(df)[0]

    # Inverse transform: expm1 converts log(price+1) -> price
    predicted_price = float(np.expm1(log_prediction))

    return PredictionResponse(
        predicted_price=round(predicted_price, 2),
        log_prediction=round(float(log_prediction), 6),
        model_version=os.path.basename(MODEL_PATH),
    )


@app.get("/model-info")
def model_info():
    """Return information about the loaded model pipeline."""
    model = get_model()
    steps = [step[0] for step in model.steps]
    return {
        "pipeline_steps": steps,
        "model_type": type(model.named_steps["model"]).__name__,
    }
