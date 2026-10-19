
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. Load the order-level dataset
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "order_level_modeling_dataset.csv"
)

df = pd.read_csv(INPUT_FILE, parse_dates=["order_date"])

TARGET = "late_delivery_risk"
CUTOFF = pd.Timestamp("2017-07-01")

# --------------------------------------------------
# 2. Define the chronological train/test split
# --------------------------------------------------
train = df.loc[df["order_date"] < CUTOFF].copy()
test = df.loc[df["order_date"] >= CUTOFF].copy()

if train.empty or test.empty:
    raise ValueError("Training or testing period is empty.")

print("\n=== TIME-BASED SPLIT ===")
print(f"Training orders: {len(train):,}")
print(
    f"Training dates: {train['order_date'].min()} "
    f"to {train['order_date'].max()}"
)
print(f"Testing orders: {len(test):,}")
print(
    f"Testing dates: {test['order_date'].min()} "
    f"to {test['order_date'].max()}"
)
print(
    f"Training late rate: {train[TARGET].mean():.2%}"
)
print(
    f"Testing late rate: {test[TARGET].mean():.2%}"
)

# --------------------------------------------------
# 3. Select features known at order placement
# --------------------------------------------------
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

feature_columns = categorical_features + numeric_features

X_train = train[feature_columns]
y_train = train[TARGET]

X_test = test[feature_columns]
y_test = test[TARGET]

print("\n=== FEATURE CHECK ===")
print(f"Input features: {len(feature_columns)}")
print("Excluded: order ID, raw order date, target,")
print("actual shipping duration, delivery status,")
print("and scheduled duration (redundant with shipping mode).")

# --------------------------------------------------
# 4. Prepare categorical and numeric features
# --------------------------------------------------
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)

# --------------------------------------------------
# 5. Train two models
# --------------------------------------------------
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=2000)),
    ]
)

logistic_model.fit(X_train, y_train)

# --------------------------------------------------
# 6. Evaluate on the later period only
# --------------------------------------------------
def evaluate_model(name, model, X, y):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    print(f"\n=== {name} ===")
    print(f"Accuracy:          {accuracy_score(y, predictions):.4f}")
    print(
        "Balanced accuracy: "
        f"{balanced_accuracy_score(y, predictions):.4f}"
    )
    print(
        f"Precision:         "
        f"{precision_score(y, predictions, zero_division=0):.4f}"
    )
    print(
        f"Recall:            "
        f"{recall_score(y, predictions, zero_division=0):.4f}"
    )
    print(
        f"F1-score:          "
        f"{f1_score(y, predictions, zero_division=0):.4f}"
    )
    print(f"ROC-AUC:           {roc_auc_score(y, probabilities):.4f}")
    print(
        "Average precision: "
        f"{average_precision_score(y, probabilities):.4f}"
    )

    print("Confusion matrix (rows=actual 0/1, columns=predicted 0/1):")
    print(confusion_matrix(y, predictions, labels=[0, 1]))


evaluate_model(
    "MAJORITY-CLASS BASELINE",
    baseline,
    X_test,
    y_test,
)

evaluate_model(
    "LOGISTIC REGRESSION",
    logistic_model,
    X_test,
    y_test,
)

print("\nBaseline modelling complete.")

# --------------------------------------------------
# 7. Analyse classification thresholds
# --------------------------------------------------
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

import pandas as pd

probabilities = logistic_model.predict_proba(X_test)[:, 1]

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

threshold_results = []

for threshold in thresholds:
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    threshold_results.append({
        "threshold": threshold,
        "precision_pct": round(
            precision_score(y_test, predictions, zero_division=0) * 100, 2
        ),
        "recall_pct": round(
            recall_score(y_test, predictions, zero_division=0) * 100, 2
        ),
        "f1_pct": round(
            f1_score(y_test, predictions, zero_division=0) * 100, 2
        ),
        "balanced_accuracy_pct": round(
            balanced_accuracy_score(y_test, predictions) * 100, 2
        ),
        "orders_flagged": int(predictions.sum()),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    })

threshold_df = pd.DataFrame(threshold_results)

print("\n=== THRESHOLD COMPARISON ===")
print(threshold_df.to_string(index=False))

output_file = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "threshold_analysis.csv"
)

threshold_df.to_csv(output_file, index=False)

print(f"\nThreshold analysis saved to: {output_file}")
