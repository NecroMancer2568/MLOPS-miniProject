# MP076 - ATM Cash Demand Prediction (Nagpur)



## Overview

An end-to-end MLOps project that predicts ATM cash withdrawal demand in Nagpur. It covers data cleaning, feature engineering, model comparison, experiment tracking, holdout prediction, monitoring, API deployment, Docker, CI, and Kubernetes.



**Target:** `withdrawal_thousand_rs`

**Problem type:** Regression

**Primary metric:** Mean Absolute Error (MAE)



## Pipeline Architecture

1\. Data quality inspection and cleaning

2\. Data versioning with DVC

3\. Feature engineering

4\. Time-series model evaluation

5\. Model comparison and later-period adaptation

6\. MLflow experiment tracking

7\. Holdout prediction generation

8\. Drift monitoring

9\. FastAPI inference service

10\. Docker containerization

11\. GitHub Actions CI

12\. Kubernetes deployment



## Data Quality Handling

The preprocessing pipeline handles:

\- Data leakage columns

\- Inconsistent `avg_ticket_rs` units

\- Sentinel and invalid values

\- Duplicate records

\- Stuck sensor values

\- Target decimal/scale anomalies

\- Categorical spelling inconsistencies



Cleaning and validation are implemented in `scripts/dataset_cleaning/`.



## Features



**Numerical**

\- nearby_shops

\- days_to_salary

\- footfall_index

\- distance_to_market_km

\- avg_ticket_rs

\- other_atms_1km

\- day_of_week

\- day_of_month

\- month

\- is_weekend



**Categorical**

\- atm_type

\- area



Date-derived features are generated during preprocessing and inference.



## Models

Four model baselines were evaluated:

\- Dummy Median

\- Ridge Regression

\- Random Forest

\- HistGradientBoosting



Time-series cross-validation and later-period evaluation were used. Later-period adaptation was also evaluated to address changing data patterns.



### Selected Model

HistGradientBoosting with later-period adaptation.



Recent later-period evaluation MAE: **49.92 thousand Rs**. This is a model selection score, not an independent untouched test score.



## Experiment Tracking

MLflow is used to log model experiments and evaluation metrics.



Model registry registration was attempted but remains unresolved due to model serialization/trust restrictions.



## Holdout Predictions

Generated predictions are saved in:



`predictions/holdout_predictions.csv`



Columns:

\- record_id

\- date

\- predicted_withdrawal_thousand_rs



## Drift Monitoring

The monitoring pipeline compares distributions across datasets and calculates drift metrics including PSI and statistical checks.



Report:



`monitoring/reports/drift_report.json`



## API



| Endpoint | Method | Description |

|---|---|---|

| `/health` | GET | Health and model-loaded status |

| `/predict` | POST | Single ATM withdrawal prediction |

| `/predict/batch` | POST | Batch predictions |

| `/metrics` | GET | Service metrics |



Start the local API:



```powershell

python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

```



Swagger documentation:



`http://127.0.0.1:8000/docs`



## Docker



Build:



```powershell

docker build -t mp076-api:1.0 .

```



Run:



```powershell

docker run --name mp076-test -p 8001:8000 mp076-api:1.0

```



Health check:



```powershell

Invoke-RestMethod http://127.0.0.1:8001/health

```



Prediction endpoints:

\- `http://127.0.0.1:8001/predict`

\- `http://127.0.0.1:8001/predict/batch`



## Tests



Run all tests:



```powershell

python -m pytest tests -v

```



Latest verified result: **15 passed**.



## DVC



The pipeline stages are defined in `dvc.yaml`:

\- clean

\- train

\- predict

\- monitor



Reproduce the pipeline:



```powershell

dvc repro

```



## Kubernetes

Deployment manifests are in the `k8s/` directory.



Kubernetes deployment is pending final environment verification.



## CI

GitHub Actions workflow:



`.github/workflows/ci.yml`



Workflow execution is pending final verification.



## Team

Group G078 — Batch 77

SCSE3040 MLOps Mini Project
