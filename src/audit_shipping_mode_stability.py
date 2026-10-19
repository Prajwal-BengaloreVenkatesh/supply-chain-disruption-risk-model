
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "order_level_modeling_dataset.csv"
)
OUTPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "monthly_shipping_mode_audit.csv"
)

df = pd.read_csv(INPUT_FILE, parse_dates=["order_date"])

# 1. Verify whether shipping mode and scheduled days
# carry the same information.
print("\n=== 1. SHIPPING MODE / SCHEDULE MAPPING ===")

mode_mapping = (
    df.groupby("shipping_mode")["scheduled_shipping_days"]
    .agg(["nunique", "min", "max", "size"])
)
print(mode_mapping)

reverse_mapping = (
    df.groupby("scheduled_shipping_days")["shipping_mode"]
    .nunique()
)
print(
    "\nScheduled durations linked to multiple shipping modes:",
    int((reverse_mapping > 1).sum()),
)

# 2. Calculate monthly late-delivery rates.
df["year_month"] = df["order_date"].dt.to_period("M").astype(str)

monthly = (
    df.groupby(["year_month", "shipping_mode"])
    .agg(
        orders=("late_delivery_risk", "size"),
        late_rate=("late_delivery_risk", "mean"),
    )
    .reset_index()
)

monthly["late_rate_pct"] = monthly["late_rate"] * 100
monthly.to_csv(OUTPUT_FILE, index=False)

print("\n=== 2. MONTHLY STABILITY BY SHIPPING MODE ===")
stability = (
    monthly.groupby("shipping_mode")
    .agg(
        months_observed=("year_month", "nunique"),
        minimum_monthly_orders=("orders", "min"),
        maximum_monthly_orders=("orders", "max"),
        minimum_monthly_rate=("late_rate_pct", "min"),
        maximum_monthly_rate=("late_rate_pct", "max"),
    )
    .round(2)
)
print(stability)

# 3. Compare earlier orders with a later time period.
# We can use this boundary for an initial chronological audit.
cutoff = pd.Timestamp("2017-07-01")

df["period"] = "Later period: Jul 2017 onward"
df.loc[df["order_date"] < cutoff, "period"] = (
    "Earlier period: before Jul 2017"
)

print("\n=== 3. PERIOD COMPARISON ===")
period_summary = (
    df.groupby("period")
    .agg(
        orders=("order_id", "size"),
        late_rate=("late_delivery_risk", "mean"),
    )
)
period_summary["late_rate_pct"] = (
    period_summary["late_rate"] * 100
)
print(period_summary.round(2))

print("\n=== 4. SHIPPING MODE BY PERIOD ===")
period_mode = (
    df.groupby(["period", "shipping_mode"])
    .agg(
        orders=("order_id", "size"),
        late_rate=("late_delivery_risk", "mean"),
    )
)
period_mode["late_rate_pct"] = period_mode["late_rate"] * 100
print(period_mode.round(2))

print(f"\nMonthly audit saved to: {OUTPUT_FILE}")
print("Audit complete.")
