"""Data drift & performance degradation monitoring for MP076."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN, extract_features_from_cleaned_df

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("MP076Monitor")


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Compute Population Stability Index safely with zero-bin smoothing and small-sample guards."""
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_props = np.clip(ref_counts / len(ref), eps, None)
    cur_props = np.clip(cur_counts / len(cur), eps, None)
    ref_props = ref_props / ref_props.sum()
    cur_props = cur_props / cur_props.sum()

    psi = float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))
    return round(max(0.0, psi), 4)


def compute_categorical_drift(
    reference: pd.Series, current: pd.Series, eps: float = 1e-6
) -> Dict[str, Any]:
    """Compare categorical proportions and compute categorical PSI."""
    ref_clean = reference.dropna().astype(str)
    cur_clean = current.dropna().astype(str)
    categories = sorted(set(ref_clean.unique()) | set(cur_clean.unique()))
    if not categories or len(ref_clean) == 0 or len(cur_clean) == 0:
        return {"psi": 0.0, "max_abs_diff": 0.0, "reference_dist": {}, "current_dist": {}}

    ref_dist = ref_clean.value_counts(normalize=True).to_dict()
    cur_dist = cur_clean.value_counts(normalize=True).to_dict()

    psi = 0.0
    max_diff = 0.0
    for cat in categories:
        r = max(float(ref_dist.get(cat, 0.0)), eps)
        c = max(float(cur_dist.get(cat, 0.0)), eps)
        psi += (c - r) * np.log(c / r)
        max_diff = max(max_diff, abs(float(cur_dist.get(cat, 0.0)) - float(ref_dist.get(cat, 0.0))))

    return {
        "psi": round(max(0.0, float(psi)), 4),
        "max_abs_diff": round(float(max_diff), 4),
        "reference_dist": {k: round(float(v), 4) for k, v in sorted(ref_dist.items())},
        "current_dist": {k: round(float(v), 4) for k, v in sorted(cur_dist.items())},
    }


def compare_distributions(
    df_ref: pd.DataFrame,
    df_cur: pd.DataFrame,
    bins: int = 10,
    warn_thresh: float = 0.1,
    alert_thresh: float = 0.25,
    ks_alpha: float = 0.05,
) -> Dict[str, Any]:
    """Run numeric (PSI, KS, mean/median shift) and categorical drift comparisons."""
    numeric_report: Dict[str, Any] = {}
    drifted_features: List[str] = []

    for col in NUMERIC_FEATURES:
        ref_vals = df_ref[col].dropna().to_numpy(dtype=np.float64)
        cur_vals = df_cur[col].dropna().to_numpy(dtype=np.float64)
        psi = compute_psi(ref_vals, cur_vals, bins=bins)
        ks_stat, ks_p = ks_2samp(ref_vals, cur_vals) if len(ref_vals) > 0 and len(cur_vals) > 0 else (0.0, 1.0)
        status = "ALERT" if psi >= alert_thresh else ("WARN" if psi >= warn_thresh else "OK")
        if status in ("WARN", "ALERT") or ks_p < ks_alpha:
            drifted_features.append(col)

        numeric_report[col] = {
            "psi": psi,
            "psi_status": status,
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_p), 6),
            "ref_mean": round(float(np.mean(ref_vals)), 4),
            "cur_mean": round(float(np.mean(cur_vals)), 4),
            "mean_shift": round(float(np.mean(cur_vals) - np.mean(ref_vals)), 4),
            "ref_median": round(float(np.median(ref_vals)), 4),
            "cur_median": round(float(np.median(cur_vals)), 4),
        }

    categorical_report: Dict[str, Any] = {}
    for col in CATEGORICAL_FEATURES:
        cat_drift = compute_categorical_drift(df_ref[col], df_cur[col])
        cat_status = "ALERT" if cat_drift["psi"] >= alert_thresh else ("WARN" if cat_drift["psi"] >= warn_thresh else "OK")
        categorical_report[col] = {**cat_drift, "psi_status": cat_status}

    return {
        "reference_rows": len(df_ref),
        "current_rows": len(df_cur),
        "numeric_features": numeric_report,
        "categorical_features": categorical_report,
        "flagged_features": drifted_features,
    }


def evaluate_later_performance(df_later: pd.DataFrame, model_path: Path) -> Dict[str, Any]:
    """Evaluate saved model on later_clean overall and per area."""
    if not model_path.exists():
        return {"available": False, "reason": f"Model not found at {model_path.as_posix()}"}

    model = joblib.load(model_path)
    X_later = extract_features_from_cleaned_df(df_later)
    y_later = df_later[TARGET_COLUMN].to_numpy(dtype=np.float64)
    preds = np.clip(np.asarray(model.predict(X_later), dtype=np.float64), 0.0, None)

    area_metrics: Dict[str, Any] = {}
    for area_name, grp in df_later.groupby("area"):
        idx = grp.index.to_numpy()
        area_metrics[str(area_name)] = {
            "rows": int(len(idx)),
            "mae": round(float(mean_absolute_error(y_later[idx], preds[idx])), 4),
            "rmse": round(float(root_mean_squared_error(y_later[idx], preds[idx])), 4),
        }

    return {
        "available": True,
        "evaluation_scope": "saved_artifact_on_later_clean",
        "overall": {
            "rows": int(len(y_later)),
            "mae": round(float(mean_absolute_error(y_later, preds)), 4),
            "rmse": round(float(root_mean_squared_error(y_later, preds)), 4),
            "r2": round(float(r2_score(y_later, preds)), 4),
        },
        "area_wise": area_metrics,
    }


def run_monitoring(params_path: str = "params.yaml") -> Dict[str, Any]:
    with open(params_path, "r", encoding="utf-8") as fp:
        params = yaml.safe_load(fp)

    paths = params["paths"]
    mon = params["monitoring"]

    processed_dir = Path(paths["processed_dir"])
    df_train = pd.read_csv(processed_dir / "train_clean.csv")
    df_later = pd.read_csv(processed_dir / "later_clean.csv")
    df_holdout = pd.read_csv(processed_dir / "holdout_clean.csv")

    train_vs_later = compare_distributions(
        df_train,
        df_later,
        bins=int(mon["psi_bins"]),
        warn_thresh=float(mon["psi_warn_threshold"]),
        alert_thresh=float(mon["psi_alert_threshold"]),
        ks_alpha=float(mon["ks_alpha"]),
    )
    train_vs_holdout = compare_distributions(
        df_train,
        df_holdout,
        bins=int(mon["psi_bins"]),
        warn_thresh=float(mon["psi_warn_threshold"]),
        alert_thresh=float(mon["psi_alert_threshold"]),
        ks_alpha=float(mon["ks_alpha"]),
    )

    target_shift = {
        "train_mean": round(float(df_train[TARGET_COLUMN].mean()), 4),
        "later_mean": round(float(df_later[TARGET_COLUMN].mean()), 4),
        "train_zero_rate": round(float((df_train[TARGET_COLUMN] == 0).mean()), 4),
        "later_zero_rate": round(float((df_later[TARGET_COLUMN] == 0).mean()), 4),
    }

    later_perf = evaluate_later_performance(df_later, Path(paths["model_path"]))

    alert_features = [
        f
        for f, stats in train_vs_holdout["numeric_features"].items()
        if stats["psi_status"] == "ALERT"
    ] + [
        f
        for f, stats in train_vs_later["numeric_features"].items()
        if stats["psi_status"] == "ALERT"
    ]
    alert_features = sorted(set(alert_features))

    recommendation = (
        "RETRAIN_RECOMMENDED"
        if alert_features
        else ("MONITOR" if (train_vs_later["flagged_features"] or train_vs_holdout["flagged_features"]) else "NO_ACTION")
    )

    report = {
        "train_vs_later": train_vs_later,
        "train_vs_holdout": train_vs_holdout,
        "target_shift_train_vs_later": target_shift,
        "later_model_performance": later_perf,
        "retraining_trigger": {
            "status": recommendation,
            "auto_deploy": False,
            "alert_features": alert_features,
            "reason": (
                f"Severe covariate shift detected (PSI >= {mon['psi_alert_threshold']}) on {alert_features}."
                if alert_features
                else "No severe PSI drift detected."
            ),
        },
    }

    out_path = Path(paths["drift_report_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fp:
        json.dump(report, fp, indent=2)
        fp.write("\n")

    logger.info(f"Saved drift report to '{out_path.as_posix()}'; trigger={recommendation}.")
    return report


def main():
    parser = argparse.ArgumentParser(description="Run MP076 data drift & performance monitoring.")
    parser.add_argument("--params", type=str, default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()
    run_monitoring(params_path=args.params)


if __name__ == "__main__":
    main()
