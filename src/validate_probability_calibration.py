
from pathlib import Path

import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "test_order_risk_predictions.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "reports"
    / "probability_calibration_summary.csv"
)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Missing predictions file: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

required = [
    "late_delivery_risk",
    "risk_probability",
    "order_value_net",
]

missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=required).copy()

y_true = df["late_delivery_risk"].astype(int)
prob = df["risk_probability"].astype(float)

if not prob.between(0, 1).all():
    raise ValueError("Predicted probabilities must be between 0 and 1.")

print("\n=== PROBABILITY QUALITY ===")
print(f"Test orders: {len(df):,}")
print(f"Observed positive-label rate: {y_true.mean():.2%}")
print(f"Mean predicted probability: {prob.mean():.2%}")
print(f"Brier score: {brier_score_loss(y_true, prob):.4f}")
print(f"Log loss: {log_loss(y_true, prob):.4f}")

# Group predictions into probability bands.
df["probability_band"] = pd.qcut(
    prob,
    q=10,
    duplicates="drop",
)

calibration = (
    df.groupby("probability_band", observed=True)
    .agg(
        orders=("late_delivery_risk", "size"),
        mean_predicted_probability=("risk_probability", "mean"),
        observed_positive_rate=("late_delivery_risk", "mean"),
        average_order_value=("order_value_net", "mean"),
    )
    .reset_index()
)

calibration["mean_predicted_pct"] = (
    calibration["mean_predicted_probability"] * 100
)
calibration["observed_rate_pct"] = (
    calibration["observed_positive_rate"] * 100
)
calibration["difference_percentage_points"] = (
    calibration["observed_rate_pct"]
    - calibration["mean_predicted_pct"]
)

print("\n=== CALIBRATION BY PROBABILITY BAND ===")
print(
    calibration[
        [
            "probability_band",
            "orders",
            "mean_predicted_pct",
            "observed_rate_pct",
            "difference_percentage_points",
        ]
    ].round(2).to_string(index=False)
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
calibration.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved calibration summary: {OUTPUT_FILE}")
