
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "data" / "raw" / "DataCoSupplyChainDataset.csv"

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE, encoding="latin-1", low_memory=False)

print("\n=== 1. DATASET GRAIN ===")
print(f"Rows (order items): {len(df):,}")
print(f"Unique orders: {df['Order Id'].nunique():,}")
print(f"Unique order-item IDs: {df['Order Item Id'].nunique():,}")
print(f"Duplicate order-item IDs: {df['Order Item Id'].duplicated().sum():,}")

print("\n=== 2. TARGET DISTRIBUTION ===")
print(df["Late_delivery_risk"].value_counts(dropna=False).sort_index())
print("\nTarget proportions:")
print(
    df["Late_delivery_risk"]
    .value_counts(normalize=True, dropna=False)
    .sort_index()
    .round(4)
)

print("\n=== 3. TARGET VS DELIVERY STATUS ===")
print(pd.crosstab(df["Delivery Status"], df["Late_delivery_risk"]))

status_label = (
    df["Delivery Status"].eq("Late delivery").astype(int)
)
status_match = df["Late_delivery_risk"].eq(status_label).mean() * 100
print(f"\nTarget/status agreement: {status_match:.2f}%")

print("\n=== 4. ACTUAL VS SCHEDULED SHIPPING DAYS ===")
duration_label = (
    df["Days for shipping (real)"]
    > df["Days for shipment (scheduled)"]
).astype(int)

duration_match = (
    df["Late_delivery_risk"].eq(duration_label).mean() * 100
)
print(f"Target/duration agreement: {duration_match:.2f}%")

print("\nMismatches by delivery status:")
mismatch = df.loc[
    df["Late_delivery_risk"].ne(duration_label),
    "Delivery Status"
]
print(mismatch.value_counts(dropna=False))

print("\n=== 5. DATE RANGE ===")
for col in [
    "order date (DateOrders)",
    "shipping date (DateOrders)",
]:
    parsed = pd.to_datetime(
        df[col],
        format="%m/%d/%Y %H:%M",
        errors="coerce",
    )
    print(f"\n{col}")
    print(f"Unparsed dates: {parsed.isna().sum():,}")
    print(f"Earliest date: {parsed.min()}")
    print(f"Latest date: {parsed.max()}")

print("\n=== 6. SENSITIVE COLUMNS TO EXCLUDE ===")
sensitive = [
    "Customer Email",
    "Customer Fname",
    "Customer Lname",
    "Customer Password",
    "Customer Street",
    "Customer Id",
    "Order Customer Id",
]
print([col for col in sensitive if col in df.columns])

print("\n=== 7. VALIDATION COMPLETE ===")
