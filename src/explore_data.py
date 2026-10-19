
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "order_level_modeling_dataset.csv"
)

OUTPUT_DIR = PROJECT_DIR / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)
df["order_date"] = pd.to_datetime(df["order_date"], errors="raise")

target = "late_delivery_risk"

print("\n=== 1. DATA QUALITY ===")
print(f"Rows: {len(df):,}")
print(f"Unique orders: {df['order_id'].nunique():,}")
print(f"Duplicate order IDs: {df['order_id'].duplicated().sum():,}")
print(f"Total missing cells: {int(df.isna().sum().sum()):,}")

if df["order_id"].duplicated().any():
    raise ValueError("Order IDs are not unique.")

if df[target].isna().any() or not df[target].isin([0, 1]).all():
    raise ValueError("The target contains invalid values.")

# Rate = average of a binary target, expressed as a percentage.
late_rate = df[target].mean() * 100

print("\n=== 2. BASELINE ===")
print(f"Overall late-delivery rate: {late_rate:.2f}%")
print(
    "Accuracy from always predicting late: "
    f"{late_rate:.2f}%"
)

# --------------------------------------------------
# Generate risk-rate summaries
# --------------------------------------------------
summary_tables = []

def analyse_category(column):
    result = (
        df.groupby(column, dropna=False)[target]
        .agg(orders="size", late_rate="mean")
        .reset_index()
        .rename(columns={column: "category"})
    )

    result["dimension"] = column
    result["late_rate_pct"] = result["late_rate"] * 100

    # Require at least 50 orders before comparing groups.
    result = result.loc[result["orders"] >= 50].copy()

    result = result.sort_values(
        "late_rate_pct", ascending=False
    )

    summary_tables.append(
        result[["dimension", "category", "orders", "late_rate_pct"]]
    )

    print(f"\n=== LATE-DELIVERY RATE BY {column.upper()} ===")
    print(
        result[
            ["category", "orders", "late_rate_pct"]
        ].round(2).to_string(index=False)
    )

    return result

by_shipping = analyse_category("shipping_mode")
by_schedule = analyse_category("scheduled_shipping_days")
by_market = analyse_category("market")
by_region = analyse_category("destination_region")

if summary_tables:
    combined = pd.concat(summary_tables, ignore_index=True)
    combined.to_csv(
        PROJECT_DIR / "data" / "processed" / "eda_risk_summary.csv",
        index=False,
    )

# --------------------------------------------------
# Chart 1: Target distribution
# --------------------------------------------------
counts = df[target].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(
    ["Not late (0)", "Late (1)"],
    [counts.get(0, 0), counts.get(1, 0)],
)

ax.set_title("Order-Level Late-Delivery Risk Distribution")
ax.set_ylabel("Number of orders")

for bar in bars:
    value = int(bar.get_height())
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value,
        f"{value:,}\n({value / len(df):.1%})",
        ha="center",
        va="bottom",
    )

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "01_target_distribution.png", dpi=150)
plt.close(fig)

# --------------------------------------------------
# Chart 2–4: Late rate by selected factors
# --------------------------------------------------
def plot_rate(data, category_column, title, filename):
    if data.empty:
        return

    plot_data = data.sort_values("late_rate_pct")

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        plot_data["category"].astype(str),
        plot_data["late_rate_pct"],
    )

    ax.axvline(
        late_rate,
        linestyle="--",
        label=f"Overall rate: {late_rate:.1f}%",
    )
    ax.set_title(title)
    ax.set_xlabel("Late-delivery rate (%)")
    ax.legend()

    for bar in bars:
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.1f}%",
            va="center",
            fontsize=8,
        )

    ax.set_xlim(
        0,
        max(100, plot_data["late_rate_pct"].max() + 8),
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(fig)

plot_rate(
    by_shipping,
    "shipping_mode",
    "Late-Delivery Rate by Shipping Mode",
    "02_risk_by_shipping_mode.png",
)

plot_rate(
    by_schedule,
    "scheduled_shipping_days",
    "Late-Delivery Rate by Scheduled Shipping Days",
    "03_risk_by_scheduled_days.png",
)

plot_rate(
    by_market,
    "market",
    "Late-Delivery Rate by Market",
    "04_risk_by_market.png",
)

# --------------------------------------------------
# Chart 5: Monthly trend
# --------------------------------------------------
monthly = (
    df.set_index("order_date")[target]
    .resample("MS")
    .agg(["count", "mean"])
    .rename(columns={"count": "orders", "mean": "late_rate"})
)

monthly["late_rate_pct"] = monthly["late_rate"] * 100
monthly.to_csv(
    PROJECT_DIR / "data" / "processed" / "monthly_risk_summary.csv"
)

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(
    monthly.index,
    monthly["late_rate_pct"],
    marker="o",
    markersize=3,
)

ax.axhline(
    late_rate,
    linestyle="--",
    label=f"Overall rate: {late_rate:.1f}%",
)

ax.set_title("Monthly Late-Delivery Rate by Order Date")
ax.set_ylabel("Late-delivery rate (%)")
ax.set_xlabel("Order date")
ax.legend()

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "05_monthly_risk_trend.png", dpi=150)
plt.close(fig)

print("\n=== EXPLORATION COMPLETE ===")
print(f"Charts saved to: {OUTPUT_DIR}")
print("Risk summaries saved to data/processed/")
