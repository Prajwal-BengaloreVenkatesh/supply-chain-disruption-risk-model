
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "order_level_modeling_dataset.csv"
)

df = pd.read_csv(INPUT_FILE, parse_dates=["order_date"])

TARGET = "late_delivery_risk"

# Chronological split: no random shuffling.
validation_start = pd.Timestamp("2017-01-01")
validation_end = pd.Timestamp("2017-07-01")

train = df.loc[df["order_date"] < validation_start].copy()

validation = df.loc[
    (df["order_date"] >= validation_start)
    & (df["order_date"] < validation_end)
].copy()

if train.empty or validation.empty:
    raise ValueError("Training or validation data is empty.")

print("\n=== TRAINING AND VALIDATION PERIODS ===")
print(f"Training orders: {len(train):,}")
print(f"Training dates: {train['order_date'].min()} to "
      f"{train['order_date'].max()}")
print(f"Validation orders: {len(validation):,}")
print(f"Validation dates: {validation['order_date'].min()} to "
      f"{validation['order_date'].max()}")
print(f"Training late rate: {train[TARGET].mean():.2%}")
print(f"Validation late rate: {validation[TARGET].mean():.2%}")

categorical_features = [
    "shipping_mode",
    "market",
    "destination_region",
    "customer_segment",
    "payment_type",
]

numeric_features = [
    "order_value_net",
    "total_discount",
    "item_lines",
    "total_quantity",
    "unique_product_categories",
    "average_product_price",
    "order_month",
    "order_day_of_week",
    "order_quarter",
]

features = categorical_features + numeric_features

X_train = train[features]
y_train = train[TARGET]

X_val = validation[features]
y_val = validation[TARGET]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=2000)),
])

model.fit(X_train, y_train)

probabilities = model.predict_proba(X_val)[:, 1]

print("\n=== VALIDATION MODEL PERFORMANCE AT 0.50 ===")
predictions = (probabilities >= 0.50).astype(int)

print(f"Balanced accuracy: "
      f"{balanced_accuracy_score(y_val, predictions):.4f}")
print(f"Precision: "
      f"{precision_score(y_val, predictions, zero_division=0):.4f}")
print(f"Recall: "
      f"{recall_score(y_val, predictions, zero_division=0):.4f}")
print(f"F1-score: "
      f"{f1_score(y_val, predictions, zero_division=0):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_val, probabilities):.4f}")

print("\n=== VALIDATION THRESHOLD COMPARISON ===")

results = []

for threshold in [
    0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70,
]:
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).ravel()

    results.append({
        "threshold": threshold,
        "precision_pct": round(
            precision_score(y_val, predictions, zero_division=0) * 100, 2
        ),
        "recall_pct": round(
            recall_score(y_val, predictions, zero_division=0) * 100, 2
        ),
        "f1_pct": round(
            f1_score(y_val, predictions, zero_division=0) * 100, 2
        ),
        "balanced_accuracy_pct": round(
            balanced_accuracy_score(y_val, predictions) * 100, 2
        ),
        "orders_flagged": int(predictions.sum()),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

output_file = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "validation_threshold_analysis.csv"
)

results_df.to_csv(output_file, index=False)
print(f"\nSaved validation results to: {output_file}")