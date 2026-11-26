# MLOps Mini Project — ATM Cash Demand

**Course:** SCSE3040 — MLOps  
**Group:** G078  
**Problem ID:** MP076  
**Dataset:** ATM Cash Demand — Nagpur

## Project Overview

This project focuses on building an end-to-end MLOps pipeline for ATM cash demand forecasting. The work includes raw data versioning, data-quality investigation, cleaning and preprocessing, model training, experiment tracking, model registration, and testing.

## Dataset

| Dataset | Rows | Purpose |
|---|---:|---|
| `train.csv` | 1,000 | Training data and data-quality checks |
| `later.csv` | 300 | Later-period data |
| `holdout_features.csv` | 250 | Holdout features for future predictions |

## Work Completed

### 1. Data Versioning

- Added `train.csv` and `later.csv` to DVC tracking.
- Added `holdout_features.csv` to the repository.
- Pushed the raw-data tracking changes to the `rishika` branch.

### 2. Data-Quality Investigation

Initial checks are implemented in `src/data_checks.py`.

Findings recorded so far:

| Check | Finding |
|---|---|
| Unit mix-up | `avg_ticket_rs` values in `Market` require scaling; recorded factor is `0.001` |
| Sentinel values | `nearby_shops = -999`, 45 occurrences |
| Duplicate rows | 0 extra full-row copies found |
| Target typos | 9 target values above 1,000 identified |
| Category levels | Checked `atm_type` and `area`; train and later levels match |
| Leakage and stuck sensor | Initial checks performed; results need confirmation against the problem’s exact criteria |

### 3. Cleaning and Preprocessing

Cleaning code is in:

`scripts/dataset_cleaning/clean_and_preprocess.py`

The pipeline currently:

- Corrects the `avg_ticket_rs` unit inconsistency.
- Corrects target values above 1,000.
- Imputes negative `nearby_shops` values using training-data group medians, with a global-median fallback.
- Cleans categorical whitespace.
- Adds calendar features from `date`.
- Validates and saves the processed datasets.

The pipeline ran successfully on train, later, and holdout data. All three validation checks passed.

### 4. Model Training and MLflow Experiment Tracking

Training code is implemented in:

`src/train.py`

Model configuration is maintained separately in:

`params.yaml`

The training pipeline:

- Loads configuration using `yaml.safe_load`.
- Reads the cleaned training dataset.
- Performs an 80:20 training-validation split.
- Trains and evaluates three Random Forest configurations.
- Tracks model parameters, evaluation metrics, and model artifacts using MLflow.
- Registers the best-performing model in the MLflow Model Registry.
- Assigns the `champion` alias to the registered model.

#### Model Evaluation Results

| Model Configuration | MAE | RMSE | R² |
|---|---:|---:|---:|
| 50 trees, max depth 5 | 55.725 | 69.573 | 0.834 |
| 100 trees, max depth 10 | 45.261 | 56.550 | 0.891 |
| 150 trees, no depth limit | 45.721 | 57.454 | 0.887 |

The Random Forest model with 100 estimators and a maximum depth of 10 achieved the lowest MAE among the three evaluated configurations and was registered as the champion model.

**Registered model name:** `mp076-atm-cash-demand`  
**Champion alias:** `champion`  
**Registered version:** `1`

### 5. Testing

Tests are implemented in:

`tests/test_processed_data.py`

The current test suite checks:

- Processed dataset files exist.
- Expected row counts are maintained.
- The target column is present in training and later datasets.
- The target column is absent from holdout features.
- Record IDs are unique within each processed dataset.

**Latest test result:** 4 passed.

Run tests using:

```bash
python -m pytest -q
```

Run model training using:

```bash
python src/train.py
```

### 6. Data Lead Contribution

- Raw dataset versioning with DVC.
- Initial data-quality checks.
- Data cleaning and preprocessing.
- Cleaning summary correction to calculate the imputation count from the data.
- Data findings recorded in `reports/findings.yaml`.
- Model training and configuration management.
- MLflow experiment tracking and model registration.
- Champion model alias assignment.
- Processed-data validation tests.

## Current Repository Structure

```text
MLOPS-miniProject/
├── data/
│   ├── raw/
│   └── processed/
├── reports/
│   └── findings.yaml
├── scripts/
│   └── dataset_cleaning/
│       └── clean_and_preprocess.py
├── src/
│   ├── data_checks.py
│   └── train.py
├── tests/
│   └── test_processed_data.py
├── params.yaml
├── requirements.txt
├── Readme.md
└── .gitignore
```

## Current Status

**Completed:**
- Raw data versioning with DVC.
- Initial data-quality investigation.
- Data cleaning and preprocessing.
- Model training and evaluation.
- MLflow experiment tracking.
- Model registration and champion alias assignment.
- Processed-data validation tests.

**Upcoming work:**

- FastAPI prediction service.
- Docker containerization.
- GitHub Actions CI workflow.
- Kubernetes deployment.
- Model and data monitoring.
- Model retraining and champion promotion evaluation.

## Repository

[GitHub Repository](https://github.com/NecroMancer2568/MLOPS-miniProject)

**Working branch:** `rishika`