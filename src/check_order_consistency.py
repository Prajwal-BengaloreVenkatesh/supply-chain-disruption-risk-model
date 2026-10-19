
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "data" / "raw" / "DataCoSupplyChainDataset.csv"

df = pd.read_csv(DATA_FILE, encoding="latin-1", low_memory=False)

print("\n=== 1. DELIVERY STATUS COUNTS ===")
print(df["Delivery Status"].value_counts(dropna=False))

cancelled = df["Delivery Status"].eq("Shipping canceled")
completed = df.loc[~cancelled].copy()

print("\n=== 2. MODELLING POPULATION ===")
print(f"All item rows: {len(df):,}")
print(f"Cancelled item rows: {cancelled.sum():,}")
print(f"Non-cancelled item rows: {len(completed):,}")
print(f"Unique orders in non-cancelled data: {completed['Order Id'].nunique():,}")

print("\nTarget distribution among non-cancelled rows:")
print(completed["Late_delivery_risk"].value_counts().sort_index())

print("\nTarget proportions among non-cancelled rows:")
print(
    completed["Late_delivery_risk"]
    .value_counts(normalize=True)
    .sort_index()
    .round(4)
)

print("\n=== 3. TARGET CONSISTENCY WITHIN EACH ORDER ===")
target_counts = completed.groupby("Order Id")["Late_delivery_risk"].nunique()
print(f"Orders with mixed target labels: {(target_counts > 1).sum():,}")

print("\n=== 4. DELIVERY FIELD CONSISTENCY WITHIN EACH ORDER ===")
for col in [
    "Delivery Status",
    "Days for shipment (scheduled)",
    "Shipping Mode",
    "order date (DateOrders)",
    "Market",
    "Order Country",
]:
    counts = completed.groupby("Order Id")[col].nunique(dropna=False)
    print(f"{col}: {int((counts > 1).sum()):,} orders have multiple values")

print("\n=== 5. ORDER-ITEM COUNTS ===")
items_per_order = completed.groupby("Order Id").size()
print(items_per_order.describe().round(2))

print("\n=== 6. COMPLETE ===")
