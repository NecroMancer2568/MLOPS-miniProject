from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")
TARGET = "withdrawal_thousand_rs"
DATE = "date"

train = pd.read_csv(DATA_DIR / "train.csv")
later = pd.read_csv(DATA_DIR / "later.csv")
holdout = pd.read_csv(DATA_DIR / "holdout_features.csv")

print("Shapes:", train.shape, later.shape, holdout.shape)

# 1. Possible leaking columns: feature absent from holdout, or almost identical to target
features = [c for c in train.columns if c != TARGET]
missing_from_holdout = [c for c in features if c not in holdout.columns]
print("\nPossible leakage — train features missing from holdout:", missing_from_holdout)

for col in features:
    if pd.api.types.is_numeric_dtype(train[col]):
        corr = train[col].corr(train[TARGET])
        if corr is not None and abs(corr) > 0.999:
            print(f"Possible target leakage: {col}, correlation={corr:.6f}")

# 2. Unit mix-up: small avg_ticket_rs values by area
if "avg_ticket_rs" in train and "area" in train:
    print("\nUnit mix-up — avg_ticket_rs below 100 by area:")
    print(train.loc[train["avg_ticket_rs"] < 100]
          .groupby("area")["avg_ticket_rs"].agg(["count", "min", "max"]))

# 3. Sentinel values
print("\nSentinel — nearby_shops == -999:",
      int((train["nearby_shops"] == -999).sum()))

# 4. Duplicate rows (extra copies, not unique record_id)
print("\nDuplicate full rows — extra copies:",
      int(train.duplicated().sum()))

# 5. Stuck sensor — longest identical-value run in date-sorted rows
print("\nStuck-sensor candidates (longest consecutive row run):")
ordered = train.sort_values(DATE)
for col in train.select_dtypes(include="number").columns:
    values = ordered[col].reset_index(drop=True)
    same_as_previous = values.eq(values.shift())
    groups = (~same_as_previous).cumsum()
    run_lengths = values.groupby(groups).size()
    if len(run_lengths):
        longest = int(run_lengths.max())
        if longest >= 3:
            run_group = run_lengths.idxmax()
            run_rows = ordered.iloc[(groups == run_group).to_numpy()]
            print(col, "run_length=", longest,
                  "start=", run_rows[DATE].iloc[0],
                  "end=", run_rows[DATE].iloc[-1],
                  "value=", run_rows[col].iloc[0])

# 6. Target typos — unusually high target values
print("\nTarget values > 1000:")
print(train.loc[train[TARGET] > 1000, ["record_id", DATE, TARGET]])

# 7. Category spelling variants — compare train/later categories
print("\nCategory levels by dataset:")
for col in ["atm_type", "area"]:
    print(col, "train=", sorted(train[col].dropna().astype(str).unique()))
    print(col, "later=", sorted(later[col].dropna().astype(str).unique()))