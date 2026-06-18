"""
Experiment Tracking with MLflow

"""

import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

from features import build_preprocessor, load_data


# Configuration

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
MODEL_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MLFLOW_EXPERIMENT = "house-price-prediction"

# I have use RMSLE (Root Mean Squared Log Error) as the metric
# because my target is already log-transformed.  After expm1()-ing predictions
# back to dollars, RMSLE penalizes percentage errors equally regardless of price,
# so a $10K error on a $100K house and a $100K error on a $1M house score the same.
# That's the right behavior for house pricing tasks.


# Model zoo 

def get_models() -> dict:
    """
    Four models, four different learning philosophies:

    LinearRegression: Baseline. No regularisation, fast, interpretable.
    Ridge: Linear + L2 regularization. Shrinks coefficients toward zero.
    XGBoost: Gradient boosted trees. Learns residuals iteratively.
    LightGBM: XGBoost competitor. Faster on large datasets.
    
    """
    return {
        "LinearRegression": {
            "model": LinearRegression(),
            "params": {},
        },
        "Ridge": {
            "model": Ridge(alpha=10.0),
            "params": {"alpha": 10.0},
        },
        "XGBoost": {
            "model": xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            ),
            "params": {
                "n_estimators": 500,
                "learning_rate": 0.05,
                "max_depth": 4,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        },
        "LightGBM": {
            "model": lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
            "params": {
                "n_estimators": 500,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        },
    }


# Training & evaluation

def build_full_pipeline(model) -> Pipeline:
    """
    Wrapping preprocessor + model in ONE Pipeline.
    If I fit the preprocessor outside the pipeline, I'd accidentally
    fit it on the full training data before cross-validation, causing
    Data Leakage - the model has peeked at validation fold statistics.

    with Pipeline, cross_val_score re-fits the preprocessor on each fold's
    training split only. Therefore this gives honest, leak-free CV scores.
    
    """
    preprocessor = build_preprocessor()
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def evaluate_model(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray) -> dict:
    """
    5-fold cross-validation on the training set.

    I used neg_root_mean_squared_error (sklearn negates it
    for maximization). I negate back to get the positive RMSE.
    Since y is log(price+1), this RMSE is actually RMSLE on the original price.
    Lower = better.
    
    """
    cv_scores = cross_val_score(
        pipeline, X, y,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    rmse_scores = -cv_scores

    # Also fit on full train for train-set metrics (to check overfitting)
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    train_rmse = np.sqrt(mean_squared_error(y, y_pred))
    train_r2 = r2_score(y, y_pred)

    return {
        "cv_rmse_mean": rmse_scores.mean(),
        "cv_rmse_std": rmse_scores.std(),
        "cv_rmse_min": rmse_scores.min(),
        "cv_rmse_max": rmse_scores.max(),
        "train_rmse": train_rmse,
        "train_r2": train_r2,
    }


# MLflow experiment loop

def run_experiments(X_train, y_train):
    """
    For each model: create an MLflow run, log everything, save artifact.

    mlflow.start_run() creates a context - everything inside
    gets saved under that run's unique ID. 

    """
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    results = []
    models_zoo = get_models()

    for name, config in models_zoo.items():
        print()
        print(f"Training: {name}")
        print()

        with mlflow.start_run(run_name=name):
            # 1. Log hyperparameters
            mlflow.log_param("model_type", name)
            for k, v in config["params"].items():
                mlflow.log_param(k, v)

            # 2. Build pipeline and evaluate
            pipeline = build_full_pipeline(config["model"])
            metrics = evaluate_model(pipeline, X_train, y_train)

            # 3. Log metrics
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)

            # 4. Log the trained model artifact
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                registered_model_name=f"house-price-{name.lower()}",
            )

            run_id = mlflow.active_run().info.run_id
            results.append({
                "name": name,
                "run_id": run_id,
                "cv_rmse_mean": metrics["cv_rmse_mean"],
                "cv_rmse_std": metrics["cv_rmse_std"],
                "train_r2": metrics["train_r2"],
            })

            print(f"CV RMSE: {metrics['cv_rmse_mean']:.4f} ± {metrics['cv_rmse_std']:.4f}")
            print(f"Train R²: {metrics['train_r2']:.4f}")
            print(f"MLflow Run ID: {run_id}")

    return results


def select_best_model(results: list) -> dict:
    """
    Rank by CV RMSE mean (lower = better).
    Train score measures memorization; CV score measures generalization.
    
    """
    best = min(results, key=lambda r: r["cv_rmse_mean"])
    print()
    print(f"   Best model: {best['name']}")
    print(f"   CV RMSE: {best['cv_rmse_mean']:.4f} ± {best['cv_rmse_std']:.4f}")
    print(f"   Run ID:  {best['run_id']}")
    print()
    return best


def save_best_model(best: dict, X_train, y_train):
    """
    Re-train the best model on ALL training data,
    then save to disk for the API to load.

    because: CV used 80% of data each fold; the final model should use everything

    """
    models_zoo = get_models()
    pipeline = build_full_pipeline(models_zoo[best["name"]]["model"])
    pipeline.fit(X_train, y_train)

    import joblib
    model_path = os.path.join(MODEL_OUTPUT_DIR, "best_model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"\nSaved best model -> {model_path}")
    return model_path


# Entry point

if __name__ == "__main__":
    print("Loading data...")
    X_train, y_train, X_test, test_ids = load_data(TRAIN_PATH, TEST_PATH)
    print(f"Train shape: {X_train.shape}  |  Target shape: {y_train.shape}")

    print("\nStarting MLflow experiment tracking...")
    results = run_experiments(X_train, y_train)

    best = select_best_model(results)
    model_path = save_best_model(best, X_train, y_train)

    # Print summary table
    print()
    print(f"{'Model':<20} {'CV RMSE':>12} {'± Std':>10} {'Train R²':>10}")
    print()
    for r in sorted(results, key=lambda x: x["cv_rmse_mean"]):
        print(f"{r['name']:<20} {r['cv_rmse_mean']:>12.4f} {r['cv_rmse_std']:>10.4f} {r['train_r2']:>10.4f}")
    print()

    print(f"\n Done! ")
    print(f"Best model saved to: {model_path}")
