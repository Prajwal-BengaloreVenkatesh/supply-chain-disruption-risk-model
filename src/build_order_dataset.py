
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. Locate and load the original dataset
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "DataCoSupplyChainDataset.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "order_level_modeling_dataset.csv"
)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

df = pd.read_csv(
    INPUT_FILE,
    encoding="latin-1",
    low_memory=False,
)

print(f"Original rows: {len(df):,}")

# --------------------------------------------------
# 2. Parse the order date
# --------------------------------------------------
df["order_datetime"] = pd.to_datetime(
    df["order date (DateOrders)"],
    format="%m/%d/%Y %H:%M",
    errors="coerce",
)

if df["order_datetime"].isna().any():
    raise ValueError("Some order dates could not be parsed.")

# --------------------------------------------------
# 3. Exclude cancelled orders from this model
# --------------------------------------------------
cancelled_order_ids = df.loc[
    df["Delivery Status"].eq("Shipping canceled"),
    "Order Id",
].unique()

eligible = df.loc[
    ~df["Order Id"].isin(cancelled_order_ids)
].copy()

print(f"Cancelled orders excluded: {len(cancelled_order_ids):,}")
print(f"Eligible item rows: {len(eligible):,}")
print(f"Eligible unique orders: {eligible['Order Id'].nunique():,}")

# --------------------------------------------------
# 4. Check fields that should be consistent per order
# --------------------------------------------------
consistency_columns = [
    "Late_delivery_risk",
    "Days for shipment (scheduled)",
    "Shipping Mode",
    "order_datetime",
    "Market",
    "Order Country",
    "Order Region",
    "Customer Segment",
    "Type",
]

inconsistent = {}

for col in consistency_columns:
    unique_counts = eligible.groupby("Order Id")[col].nunique(
        dropna=False
    )
    number_inconsistent = int((unique_counts > 1).sum())

    if number_inconsistent > 0:
        inconsistent[col] = number_inconsistent

if inconsistent:
    print("\nFields with multiple values within an order:")
    for col, count in inconsistent.items():
        print(f"{col}: {count:,} orders")

    raise ValueError(
        "Resolve the order-level inconsistencies before aggregation."
    )

print("Order-level consistency check passed.")

# --------------------------------------------------
# 5. Aggregate item rows into one order-level record
# --------------------------------------------------
order_level = (
    eligible.groupby("Order Id", as_index=False)
    .agg(
        order_date=("order_datetime", "first"),
        late_delivery_risk=("Late_delivery_risk", "first"),
        scheduled_shipping_days=(
            "Days for shipment (scheduled)", "first"
        ),
        shipping_mode=("Shipping Mode", "first"),
        market=("Market", "first"),
        destination_country=("Order Country", "first"),
        destination_region=("Order Region", "first"),
        customer_segment=("Customer Segment", "first"),
        payment_type=("Type", "first"),

        order_value_net=("Order Item Total", "sum"),
        total_discount=("Order Item Discount", "sum"),
        item_lines=("Order Item Id", "count"),
        total_quantity=("Order Item Quantity", "sum"),
        unique_product_categories=("Category Name", "nunique"),
        average_product_price=("Order Item Product Price", "mean"),
    )
    .rename(columns={"Order Id": "order_id"})
)

# --------------------------------------------------
# 6. Add calendar features known at order placement
# --------------------------------------------------
order_level["order_month"] = order_level["order_date"].dt.month
order_level["order_day_of_week"] = (
    order_level["order_date"].dt.dayofweek
)
order_level["order_quarter"] = order_level["order_date"].dt.quarter

# --------------------------------------------------
# 7. Final validation and save
# --------------------------------------------------
assert order_level["order_id"].is_unique
assert order_level["late_delivery_risk"].notna().all()
assert order_level["late_delivery_risk"].isin([0, 1]).all()

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
order_level.to_csv(OUTPUT_FILE, index=False)

print("\n=== ORDER-LEVEL DATASET CREATED ===")
print(f"Rows: {len(order_level):,}")
print(f"Columns: {len(order_level.columns):,}")
print(f"Unique order IDs: {order_level['order_id'].nunique():,}")

print("\n=== TARGET DISTRIBUTION ===")
print(order_level["late_delivery_risk"].value_counts().sort_index())

print("\n=== MISSING VALUES ===")
print(
    order_level.isna().sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\n=== SAMPLE ROWS ===")
print(order_level.head().to_string(index=False))

print(f"\nSaved dataset to: {OUTPUT_FILE}")
