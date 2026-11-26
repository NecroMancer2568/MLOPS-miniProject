# MLOps Mini Project — ATM Cash Demand

**Course:** SCSE3040 — MLOps\
**Group:** G078\
**Problem ID:** MP076\
**Dataset:** ATM Cash Demand — Nagpur

## Project Overview

This project focuses on preparing ATM cash withdrawal data for an MLOps pipeline. The current work covers raw data versioning, data-quality investigation, and cleaning/preprocessing.

## Dataset

| Dataset                | Rows  | Purpose                                 |
| ---------------------- | ----- | --------------------------------------- |
| `train.csv`            | 1,000 | Training data and data-quality checks   |
| `later.csv`            | 300   | Later-period data                       |
| `holdout_features.csv` | 250   | Holdout features for future predictions |

## Work Completed

### 1. Data Versioning

- Added `train.csv` and `later.csv` to DVC tracking.
- Added `holdout_features.csv` to the repository.
- Pushed the raw-data tracking changes to the `rishika` branch.

### 2. Data-Quality Investigation

Initial checks are implemented in `src/data_checks.py`.

Findings recorded so far:

| Check                    | Finding                                                                                  |
| ------------------------ | ---------------------------------------------------------------------------------------- |
| Unit mix-up              | `avg_ticket_rs` values in `Market` require scaling; recorded factor is `0.001`           |
| Sentinel values          | `nearby_shops = -999`, 45 occurrences                                                    |
| Duplicate rows           | 0 extra full-row copies found                                                            |
| Target typos             | 9 target values above 1,000 identified                                                   |
| Category levels          | Checked `atm_type` and `area`; train and later levels match                              |
| Leakage and stuck sensor | Initial checks performed; results need confirmation against the problem’s exact criteria |

### 3. Cleaning and Preprocessing

Cleaning code is in:

`scripts/dataset_cleaning/clean_and_preprocess.py`

The pipeline currently:

- Corrects the `avg_ticket_rs` unit inconsistency
- Corrects target values above 1,000
- Imputes negative `nearby_shops` values using training-data group medians, with a global-median fallback
- Cleans categorical whitespace
- Adds calendar features from `date`
- Validates and saves the processed datasets

The pipeline ran successfully on train, later, and holdout data. All three validation checks passed.

### 4. Data Lead Contribution

- Raw dataset versioning with DVC
- Initial data-quality checks
- Data cleaning and preprocessing
- Cleaning summary correction to calculate the imputation count from the data
- Data findings recorded in `reports/findings.yaml`

## Current Repository Structure

```
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
│   └── data_checks.py
└── Readme.md
```

## Current Status

**Completed:** Raw data versioning, initial data checks, cleaning/preprocessing, and recording initial findings.

Model training, MLflow tracking, API, Docker, CI, Kubernetes, and monitoring will be documented here as they are completed.

## Repository

[https://github.com/NecroMancer2568/MLOPS-miniProject](https://github.com/NecroMancer2568/MLOPS-miniProject)

**Working branch:** `rishika`
