
from pathlib import Path
import pandas as pd

# Locate the project folder and raw dataset
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "data" / "raw" / "DataCoSupplyChainDataset.csv"

# Load the dataset
df = pd.read_csv(DATA_FILE, encoding="latin-1", low_memory=False)

print("\n--- DATASET SIZE ---")
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]:,}")

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- FIRST FIVE ROWS ---")
print(df.head().to_string())

print("\n--- MISSING VALUES ---")
print(df.isna().sum().sort_values(ascending=False).head(20))

print("\n--- DUPLICATE ROWS ---")
print(f"Exact duplicate rows: {df.duplicated().sum():,}")

print("\n--- DATA TYPES ---")
print(df.dtypes.to_string())

print("\n--- TARGET-RELATED COLUMNS ---")
for col in df.columns:
    if any(word in col.lower() for word in
           ["delivery", "shipping", "order date", "customer", "sales", "profit"]):
        print(col)
