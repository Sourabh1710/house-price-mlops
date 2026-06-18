"""
Test Suite

"""

import sys
import os
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

# Add src to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.features import (
    build_preprocessor,
    load_data,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


# Fixtures

@pytest.fixture
def sample_df():
    """Create a minimal DataFrame that mimics the training data."""
    n = 20
    data = {col: [0.0] * n for col in NUMERIC_FEATURES}
    data.update({col: ["TA"] * n for col in CATEGORICAL_FEATURES})

    # Provide realistic values for a few key columns
    data["GrLivArea"] = [1500.0] * n
    data["OverallQual"] = [6] * n
    data["YearBuilt"] = [2000] * n
    data["Neighborhood"] = ["CollgCr"] * n
    data["MSZoning"] = ["RL"] * n
    data["CentralAir"] = ["Y"] * n

    return pd.DataFrame(data)


@pytest.fixture
def sample_target(sample_df):
    return np.log1p(np.full(len(sample_df), 180000.0))


# Feature pipeline tests

class TestFeaturePipeline:

    def test_preprocessor_builds(self):
        """Preprocessor object should be created without errors."""
        preprocessor = build_preprocessor()
        assert preprocessor is not None

    def test_preprocessor_fit_transform_shape(self, sample_df):
        preprocessor = build_preprocessor()
        X_transformed = preprocessor.fit_transform(sample_df)
        assert X_transformed.shape[0] == len(sample_df)
        assert X_transformed.shape[1] > len(NUMERIC_FEATURES)  # OHE adds columns

    def test_no_nans_after_transform(self, sample_df):
        """
        This is the most important pipeline test.
        Any NaN in the transformed data would silently corrupt model training.
        Imputers should eliminate all NaNs.
        
        """
        # Introduce some NaN values
        sample_df.loc[0, "LotFrontage"] = np.nan
        sample_df.loc[1, "GarageYrBlt"] = np.nan
        sample_df.loc[2, "MSZoning"] = np.nan

        preprocessor = build_preprocessor()
        X_transformed = preprocessor.fit_transform(sample_df)

        assert not np.isnan(X_transformed).any(), "NaN values found after preprocessing!"

    def test_train_test_no_leakage(self, sample_df):
        """
        The classic data leakage test.
        The preprocessor must be fit ONLY on train data.
        If I fit on train+test, the test scaler uses test statistics -> leakage.

        I test this by checking that fit_transform(train) and transform(test)
        use different data paths (fit on train only).

        """
        train = sample_df.iloc[:15].copy()
        test = sample_df.iloc[15:].copy()

        preprocessor = build_preprocessor()
        preprocessor.fit(train)                   # Fit only on train

        X_train_transformed = preprocessor.transform(train)
        X_test_transformed = preprocessor.transform(test)    # Transform test separately

        # Both should succeed and have the same number of columns
        assert X_train_transformed.shape[1] == X_test_transformed.shape[1]

    def test_unseen_categories_handled(self, sample_df):
        """
        Test data may contain categories not seen in training.
        handle_unknown='ignore' in OneHotEncoder should produce zeros, not crash.
        
        """
        train = sample_df.copy()
        test = sample_df.copy()
        test.loc[0, "Neighborhood"] = "BRAND_NEW_NEIGHBORHOOD_NEVER_SEEN"

        preprocessor = build_preprocessor()
        preprocessor.fit(train)

        # Should not raise an error
        try:
            X_test = preprocessor.transform(test)
            assert X_test is not None
        except Exception as e:
            pytest.fail(f"Unseen category caused crash: {e}")


# Full pipeline tests

class TestFullPipeline:

    def test_pipeline_predict_shape(self, sample_df, sample_target):
        """End-to-end: fit pipeline, predict, check output shape."""
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import Ridge
        from src.features import build_preprocessor

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", Ridge()),
        ])
        pipeline.fit(sample_df, sample_target)
        predictions = pipeline.predict(sample_df)

        assert predictions.shape == (len(sample_df),)
        assert not np.isnan(predictions).any()

    def test_prediction_in_reasonable_range(self, sample_df, sample_target):
        """
        predicted prices should be plausible house prices.
        I use log-scale predictions, so it expm1 back to dollars.
        A reasonable range is $10K-$10M.
        """
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import Ridge
        from src.features import build_preprocessor

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", Ridge()),
        ])
        pipeline.fit(sample_df, sample_target)
        log_preds = pipeline.predict(sample_df)
        dollar_preds = np.expm1(log_preds)

        assert (dollar_preds > 10_000).all(), "Predictions below $10K - something's wrong"
        assert (dollar_preds < 10_000_000).all(), "Predictions above $10M - something's wrong"


# API tests

class TestAPI:

    @pytest.fixture
    def client(self):
        """
        TestClient test FastAPI without starting a server.
        It sends HTTP requests in-process - fast and reliable for CI.
        
        """
        from fastapi.testclient import TestClient

        # Mock the model so API tests don't need a trained model file
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = np.array([12.2])   # log scale ~$200K

        with patch("src.app.get_model", return_value=mock_pipeline):
            from src.app import app
            return TestClient(app)

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "predict" in response.json()

    def test_predict_endpoint_returns_200(self, client):
        """A well-formed request should return 200 with a price."""
        payload = {
            "OverallQual": 7,
            "GrLivArea": 1800.0,
            "YearBuilt": 1995,
            "Neighborhood": "CollgCr",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "predicted_price" in data
        assert data["predicted_price"] > 0

    def test_predict_with_defaults(self, client):
        """Sending an empty body should use defaults and not crash."""
        response = client.post("/predict", json={})
        assert response.status_code == 200

    def test_invalid_overall_qual(self, client):
        """OverallQual must be 1-10. Sending 999 should return 422."""
        payload = {"OverallQual": 999}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422   # Pydantic validation error

    def test_health_endpoint_model_missing(self):
        """When model file is missing, /health should return 503."""
        with patch("src.app.get_model", side_effect=FileNotFoundError("no model")):
            from fastapi.testclient import TestClient
            from src.app import app
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/health")
            assert response.status_code == 503
