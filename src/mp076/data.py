"""The ATM input contract and preparation for v1 training."""

from pathlib import Path

import pandas as pd
import yaml

TARGET = "withdrawal_thousand_rs"
ID = "record_id"
DATE = "date"
NUMERIC_FEATURES = [
    "nearby_shops", "days_to_salary", "footfall_index",
    "distance_to_market_km", "avg_ticket_rs", "other_atms_1km",
]
CATEGORICAL_FEATURES = ["atm_type", "area"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_config():
    return yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))


def input_csv(configured_path):
    """Support the supplied project folder and the later DVC layout."""
    path = Path(configured_path)
    if path.exists() or path.parent != Path("data/raw"):
        return path
    return path.parent.parent / path.name


def clean_training(raw):
    """Correct the known source-specific issues before fitting on train.csv."""
    frame = raw[[ID, DATE, *FEATURES, TARGET]].drop_duplicates().copy()
    frame[DATE] = pd.to_datetime(frame[DATE], format="%Y-%m-%d")
    frame["nearby_shops"] = frame["nearby_shops"].replace(-999, float("nan"))
    market = frame["area"].eq("Market")
    # The affected tickets are 0.50–4.99 in thousands of rupees.
    in_thousands = frame["avg_ticket_rs"].between(0.50, 4.99)
    frame.loc[market & in_thousands, "avg_ticket_rs"] *= 1000
    # The original training targets contain a gap from 950 to 3233.
    extra_zero = frame[TARGET].gt(3000)
    frame.loc[extra_zero, TARGET] /= 10
    return frame.sort_values(DATE, kind="stable").reset_index(drop=True)
