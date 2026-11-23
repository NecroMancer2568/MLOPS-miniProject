"""
src/atm_demand/audit.py

Programmatic data audit for MP076 (ATM Cash Demand - Nagpur).
Detects each of the seven kinds of data problems outlined in the Student Handbook:
1. Leaking column
2. Unit mixup
3. Sentinel value
4. Duplicate rows
5. Stuck sensor
6. Target typos
7. Spelling variants
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import yaml


def check_leaking_column(
    df_train: pd.DataFrame,
    df_holdout: pd.DataFrame,
    target_col: str = "withdrawal_thousand_rs",
) -> Dict[str, Any]:
    """
    Check for leaking columns:
    Columns present in train that secretly contain the answer (missing from holdout or correlation ~ 1.0).
    """
    train_features = set(df_train.columns) - {target_col, "record_id", "date"}
    holdout_features = set(df_holdout.columns) - {"record_id", "date"}
    missing_in_holdout = train_features - holdout_features

    leaking_col = None
    if missing_in_holdout:
        leaking_col = list(missing_in_holdout)[0]

    if not leaking_col and target_col in df_train.columns:
        for c in train_features:
            if pd.api.types.is_numeric_dtype(df_train[c]):
                corr = abs(df_train[c].corr(df_train[target_col]))
                if corr > 0.98:
                    leaking_col = c
                    break

    return {
        "present": bool(leaking_col is not None),
        "column": leaking_col,
    }


def check_unit_mixup(
    df_train: pd.DataFrame, group_col: str = "area"
) -> Dict[str, Any]:
    """
    Check for unit mixup:
    One office/area recorded a column in a different unit (e.g. thousand rupees vs rupees).
    """
    numeric_cols = [
        "nearby_shops",
        "days_to_salary",
        "footfall_index",
        "distance_to_market_km",
        "avg_ticket_rs",
        "other_atms_1km",
    ]

    for col in numeric_cols:
        if col not in df_train.columns:
            continue
        group_medians = df_train.groupby(group_col)[col].median()
        for group_val, median_val in group_medians.items():
            other_median = group_medians[group_medians.index != group_val].median()
            if other_median > 0:
                ratio = median_val / other_median
                # Check for standard factor offsets (~10, 60, 100, 1000 or inverse)
                if ratio < 0.05:
                    factor = round(ratio, 3)
                    return {
                        "present": True,
                        "column": col,
                        "group_value": group_val,
                        "factor": float(factor),
                    }
                elif ratio > 20:
                    factor = round(ratio, 1)
                    return {
                        "present": True,
                        "column": col,
                        "group_value": group_val,
                        "factor": float(factor),
                    }

    return {
        "present": False,
        "column": None,
        "group_value": None,
        "factor": None,
    }


def check_sentinel_value(
    df_train: pd.DataFrame,
    sentinel_candidates: Tuple[int, ...] = (-999, -1, 999, 9999),
) -> Dict[str, Any]:
    """
    Check for sentinel values:
    Impossible numbers typed instead of a blank (e.g. -999, -1, 9999).
    """
    for col in df_train.select_dtypes(include=["number"]).columns:
        if col == "withdrawal_thousand_rs":
            continue
        val_counts = df_train[col].value_counts()
        for sentinel in sentinel_candidates:
            if sentinel in val_counts:
                count = int(val_counts[sentinel])
                return {
                    "present": True,
                    "column": col,
                    "value": int(sentinel),
                    "count": count,
                }

    return {
        "present": False,
        "column": None,
        "value": None,
        "count": None,
    }


def check_duplicate_rows(df_train: pd.DataFrame) -> Dict[str, Any]:
    """
    Check for duplicate rows:
    Count number of extra copies.
    """
    dup_count = int(df_train.duplicated().sum())
    return {
        "present": bool(dup_count > 0),
        "count": dup_count if dup_count > 0 else None,
    }


def check_stuck_sensor(
    df_train: pd.DataFrame, min_consecutive_days: int = 7
) -> Dict[str, Any]:
    """
    Check for stuck sensors:
    One column shows the exact same value on every row for a long run of days.
    """
    df_sorted = df_train.sort_values("date").copy()
    numeric_cols = [
        "nearby_shops",
        "days_to_salary",
        "footfall_index",
        "distance_to_market_km",
        "avg_ticket_rs",
        "other_atms_1km",
    ]

    for col in numeric_cols:
        if col not in df_sorted.columns:
            continue
        # Find dates where all records on that date share the exact same value
        grouped = df_sorted.groupby("date")[col].agg(["nunique", "first"])
        single_val_dates = grouped[grouped["nunique"] == 1]
        if len(single_val_dates) >= min_consecutive_days:
            # Check if consecutive dates have the exact same constant value
            s = single_val_dates["first"]
            consec_groups = (s != s.shift()).cumsum()
            for _, grp in single_val_dates.groupby(consec_groups):
                if len(grp) >= min_consecutive_days:
                    return {
                        "present": True,
                        "column": col,
                        "start_date": str(grp.index[0]),
                        "end_date": str(grp.index[-1]),
                    }

    return {
        "present": False,
        "column": None,
        "start_date": None,
        "end_date": None,
    }


def check_target_typos(
    df_train: pd.DataFrame,
    target_col: str = "withdrawal_thousand_rs",
    threshold: float = 1000.0,
) -> Dict[str, Any]:
    """
    Check for target typos:
    A small group of target values sitting far above all others (10 times too big).
    """
    if target_col not in df_train.columns:
        return {"present": False, "count": None}

    typo_mask = df_train[target_col] > threshold
    count = int(typo_mask.sum())
    return {
        "present": bool(count > 0),
        "count": count if count > 0 else None,
    }


def check_spelling_variants(
    df_train: pd.DataFrame, df_later: pd.DataFrame
) -> Dict[str, Any]:
    """
    Check for spelling variants in categorical columns:
    Differences in capitalization, whitespace, or aliases.
    """
    cat_cols = df_train.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if col in ["record_id", "date"]:
            continue
        train_vals = set(df_train[col].dropna())
        norm_train = {v.strip().lower() for v in train_vals}
        if len(train_vals) != len(norm_train):
            return {
                "present": True,
                "column": col,
                "true_levels": len(norm_train),
            }
        if col in df_later.columns:
            later_vals = set(df_later[col].dropna())
            if not train_vals.issubset(later_vals) and len(train_vals) > len(later_vals):
                return {
                    "present": True,
                    "column": col,
                    "true_levels": len(later_vals),
                }

    return {
        "present": False,
        "column": None,
        "true_levels": None,
    }


def run_full_audit(
    train_path: str = "data/train.csv",
    later_path: str = "data/later.csv",
    holdout_path: str = "data/holdout_features.csv",
) -> Dict[str, Any]:
    """Run all seven checks and return formatted findings dictionary."""
    df_train = pd.read_csv(train_path)
    df_later = pd.read_csv(later_path)
    df_holdout = pd.read_csv(holdout_path)

    issues = {
        "leaking_column": check_leaking_column(df_train, df_holdout),
        "unit_mixup": check_unit_mixup(df_train, group_col="area"),
        "sentinel_value": check_sentinel_value(df_train),
        "duplicate_rows": check_duplicate_rows(df_train),
        "stuck_sensor": check_stuck_sensor(df_train),
        "target_typos": check_target_typos(df_train),
        "spelling_variants": check_spelling_variants(df_train, df_later),
    }
    return issues


def print_audit_report(issues: Dict[str, Any]) -> None:
    """Print readable audit summary table."""
    print("=" * 70)
    print("MP076 (ATM Cash Demand - Nagpur) - Data Audit Report")
    print("=" * 70)
    for name, details in issues.items():
        pres = details.get("present")
        status = "PRESENT [!]" if pres else "ABSENT  [OK]"
        extra = {k: v for k, v in details.items() if k != "present" and v is not None}
        extra_str = f" -> {extra}" if extra else ""
        print(f" - {name:<18}: {status}{extra_str}")
    print("=" * 70)


def update_findings_yaml(
    findings_file: Path, audit_results: Dict[str, Any]
) -> None:
    """Update only data_issues section of findings.yaml while preserving everything else."""
    with open(findings_file, "r") as fp:
        data = yaml.safe_load(fp)

    data["data_issues"] = audit_results

    with open(findings_file, "w") as fp:
        yaml.dump(data, fp, sort_keys=False, default_flow_style=False)


def main():
    parser = argparse.ArgumentParser(description="Audit data for MP076 mini project.")
    parser.add_argument("--train", default="data/train.csv", help="Path to train.csv")
    parser.add_argument("--later", default="data/later.csv", help="Path to later.csv")
    parser.add_argument(
        "--holdout", default="data/holdout_features.csv", help="Path to holdout_features.csv"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write detected data_issues to findings.yaml and reports/findings.yaml",
    )
    args = parser.parse_args()

    results = run_full_audit(args.train, args.later, args.holdout)
    print_audit_report(results)

    if args.apply:
        for f in [Path("findings.yaml"), Path("reports/findings.yaml")]:
            f.parent.mkdir(parents=True, exist_ok=True)
            if not f.exists() and Path("findings.yaml").exists():
                import shutil
                shutil.copy("findings.yaml", f)
            if f.exists():
                update_findings_yaml(f, results)
                print(f"Updated {f}")


if __name__ == "__main__":
    main()
