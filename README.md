<div align="center">

#  House Price Prediction API

**Production ML pipeline - from raw CSV to live REST API in one repo.**

[![CI/CD](https://github.com/YOUR_USERNAME/house-price-mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/YOUR_USERNAME/house-price-mlops/actions)
[![Coverage](https://img.shields.io/badge/coverage-87%25-brightgreen)](https://github.com/YOUR_USERNAME/house-price-mlops)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white)](https://hub.docker.com/r/YOUR_USERNAME/house-price-api)
[![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-live-009688?logo=fastapi&logoColor=white)](https://YOUR_APP.onrender.com/docs)

[**Live API →**](https://house-price-api-latest-kmxu.onrender.com/docs) · [**MLflow Experiments →**](#experiment-results) · [**Quick Start →**](#quick-start)

</div>

---

## What this is

Most ML tutorials end at a `.pkl` file. This one doesn't.

This project takes the [Kaggle House Prices dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) (1,460 houses, 79 features) and builds the full production stack around it: a leak-proof preprocessing pipeline, MLflow experiment tracking across 4 models, a FastAPI service with auto-generated docs, a multi-stage Docker image, and a GitHub Actions pipeline that tests → builds → deploys on every push.

**The live API is one `curl` away:**

```bash
curl -s -X POST https://YOUR_APP.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "OverallQual": 8,
    "GrLivArea": 2000,
    "YearBuilt": 2005,
    "Neighborhood": "NridgHt",
    "GarageArea": 550
  }' | python -m json.tool
```

```json
{
  "predicted_price": 287543.12,
  "log_prediction": 12.568432,
  "model_version": "best_model.pkl"
}
```

---

## System Architecture

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                        TRAINING                                 │
  │                                                                 │
  │  train.csv ──► ColumnTransformer ──► 4 Models ──► MLflow UI    │
  │                │                                                │
  │                ├─ Numeric:  Median Impute → StandardScaler      │
  │                └─ Categoric: Mode Impute → OneHotEncoder        │
  │                                                                 │
  │  ⚠️  Pipeline fitted on train-only → zero data leakage         │
  └──────────────────────────┬──────────────────────────────────────┘
                             │  best_model.pkl
  ┌──────────────────────────▼──────────────────────────────────────┐
  │                        SERVING                                  │
  │                                                                 │
  │  POST /predict  →  FastAPI  →  pipeline.predict()  →  expm1()  │
  │  GET  /health   →  liveness probe (Docker + Render)            │
  │  GET  /docs     →  Swagger UI (auto-generated)                 │
  └──────────────────────────┬──────────────────────────────────────┘
                             │
  ┌──────────────────────────▼──────────────────────────────────────┐
  │                        CI / CD                                  │
  │                                                                 │
  │  git push main                                                  │
  │       │                                                         │
  │       ├─► [1] pytest (87% coverage) ──── FAIL? → stop here    │
  │       ├─► [2] docker build + push to Hub                       │
  │       └─► [3] render deploy hook → live in ~60s               │
  └─────────────────────────────────────────────────────────────────┘
```

---

## Experiment Results

Tracked with **MLflow**. Four models, 5-fold CV, scored by RMSLE (log-scale RMSE — the official Kaggle metric for this competition).

| Rank | Model | CV RMSLE ↓ | Std | Train R² | Notes |
|------|-------|-----------|-----|----------|-------|
| 🥇 | **GradientBoosting** | **0.1223** | 0.0105 | 0.9854 | Winner — deployed |
| 🥈 | Ridge | 0.1392 | 0.0257 | 0.9211 | Best linear model |
| 🥉 | RandomForest | 0.1446 | 0.0112 | 0.9825 | High variance |
| 4 | LinearRegression | 0.1550 | 0.0300 | 0.9432 | Baseline |

> **Context:** Kaggle top 10% cutoff ≈ 0.115 RMSLE. This model sits in the **top 25%** of 5,000+ submissions — without any feature engineering beyond standard preprocessing. XGBoost (in `train.py`) pushes this further.

The low Std on GradientBoosting (0.0105) matters as much as the mean — it means the model is **consistent** across folds, not just lucky on one split.

---

## Engineering Decisions

Three choices that separate this from a notebook dump:

**1. Pipeline wraps preprocessor + model together**
Fitting the scaler before cross-validation exposes val-fold statistics to training — data leakage. With sklearn `Pipeline`, the preprocessor re-fits fresh on each CV fold's training split. CV scores are honest. This is the #1 mistake junior candidates make.

**2. Log-transform the target (`np.log1p`)**
Raw `SalePrice` is right-skewed (a handful of $700K mansions drag the tail). Log-transforming makes residuals nearly normal, which cuts RMSLE by ~15% on linear models. The inverse (`np.expm1`) is applied at prediction time — the API always returns dollar amounts.

**3. `handle_unknown='ignore'` on OneHotEncoder**
Test data contains neighborhood codes and sale types not seen in training. Instead of crashing, unseen categories silently become all-zero rows. Verified in the test suite with a synthetic "ATLANTIS_NEIGHBORHOOD" category.

---

## Stack

![Python](https://img.shields.io/badge/-Python_3.11-3776AB?logo=python&logoColor=white&style=flat-square)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?logo=scikit-learn&logoColor=white&style=flat-square)
![XGBoost](https://img.shields.io/badge/-XGBoost-189AB4?style=flat-square)
![MLflow](https://img.shields.io/badge/-MLflow-0194E2?logo=mlflow&logoColor=white&style=flat-square)
![FastAPI](https://img.shields.io/badge/-FastAPI-009688?logo=fastapi&logoColor=white&style=flat-square)
![Docker](https://img.shields.io/badge/-Docker-2496ED?logo=docker&logoColor=white&style=flat-square)
![GitHub Actions](https://img.shields.io/badge/-GitHub_Actions-2088FF?logo=github-actions&logoColor=white&style=flat-square)
![Render](https://img.shields.io/badge/-Render-46E3B7?logo=render&logoColor=white&style=flat-square)

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/house-price-mlops
cd house-price-mlops
pip install -r requirements.txt

# 2. Train (with MLflow tracking)
python src/train.py
mlflow ui                          # → localhost:5000 to compare runs

# 3. Serve
uvicorn src.app:app --reload       # → localhost:8000/docs

# 4. Test
pytest tests/ --cov=src -v

# 5. Docker
docker build -t house-price-api .
docker run -p 8000:8000 house-price-api
```

---

## Project Layout

```
house-price-mlops/
├── src/
│   ├── features.py          # ColumnTransformer — the leak-proof preprocessor
│   ├── train.py             # MLflow tracking: LinearReg, Ridge, XGBoost, LightGBM
│   └── app.py               # FastAPI: /predict  /health  /docs
├── tests/
│   └── test_pipeline.py     # 7 tests: leakage, NaN handling, API responses
├── .github/workflows/
│   └── ci-cd.yml            # test → docker build → render deploy
├── Dockerfile               # Multi-stage build (builder + slim runtime)
├── render.yaml              # Infrastructure-as-code
└── requirements.txt         # Version-pinned
```

---

## CI/CD Setup

Three GitHub Secrets needed:

| Secret | Where |
|--------|-------|
| `DOCKERHUB_USERNAME` | hub.docker.com → username |
| `DOCKERHUB_TOKEN` | Docker Hub → Account Settings → Security → New Token |
| `RENDER_DEPLOY_HOOK_URL` | Render → Service → Settings → Deploy Hook |

Push to `main`. Everything else is automatic.

---

<div align="center">
<sub>Built with the Kaggle House Prices dataset · Ames, Iowa · 1,460 training samples · 79 features</sub>
</div>
