"""
Feature Engineering Pipeline

"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# Column definitions
# I explicitly list columns rather than auto-detecting them.
# This prevents surprises when test data has slightly different nulls/dtypes.

NUMERIC_FEATURES = [
    "LotFrontage", "LotArea", "MasVnrArea",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea",
    "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath",
    "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd",
    "Fireplaces", "GarageYrBlt", "GarageCars", "GarageArea",
    "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
    "3SsnPorch", "ScreenPorch", "PoolArea",
    "OverallQual", "OverallCond",
    "YearBuilt", "YearRemodAdd",
    "MoSold", "YrSold",
]

CATEGORICAL_FEATURES = [
    "MSZoning", "Street", "LotShape", "LandContour", "LotConfig",
    "LandSlope", "Neighborhood", "Condition1", "Condition2",
    "BldgType", "HouseStyle", "RoofStyle", "RoofMatl",
    "Exterior1st", "Exterior2nd", "MasVnrType",
    "ExterQual", "ExterCond", "Foundation",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "Heating", "HeatingQC", "CentralAir", "Electrical",
    "KitchenQual", "Functional",
    "FireplaceQu", "GarageType", "GarageFinish",
    "GarageQual", "GarageCond", "PavedDrive",
    "Fence", "MiscFeature", "SaleType", "SaleCondition",
]

# Sub-pipelines for each column type 

def make_numeric_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def make_categorical_pipeline() -> Pipeline:
   return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])


# Master ColumnTransformer

def build_preprocessor() -> ColumnTransformer:
    """
    ColumnTransformer applies different pipelines to different column subsets,
    then horizontally stacks the results into one feature matrix.

    """
    return ColumnTransformer(
        transformers=[
            ("num", make_numeric_pipeline(), NUMERIC_FEATURES),
            ("cat", make_categorical_pipeline(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",   
        verbose_feature_names_out=False,
    )


# Data loading & target engineering

def load_data(train_path: str, test_path: str):
    """
    Load train/test CSVs and apply target transformation.
    
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X_train = train.drop(columns=["Id", "SalePrice"])
    y_train = np.log1p(train["SalePrice"])   # log-transform target
    X_test = test.drop(columns=["Id"])
    test_ids = test["Id"]

    return X_train, y_train, X_test, test_ids
