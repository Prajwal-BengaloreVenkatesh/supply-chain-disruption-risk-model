
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
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

PREDICTIONS_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "test_order_risk_predictions.csv"
)

SCENARIO_FILE = (
    PROJECT_DIR
    / "reports"
    / "financial_scenario_summary.csv"
)

df = pd.read_csv(INPUT_FILE, parse_dates=["order_date"])

TARGET = "late_delivery_risk"
TEST_START = pd.Timestamp("2017-07-01")
VALIDATION_START = pd.Timestamp("2017-01-01")

# Keep the model and its selected thresholds independent
# of the final test period.
train = df.loc[
    df["order_date"] < VALIDATION_START
].copy()

validation = df.loc[
    (df["order_date"] >= VALIDATION_START)
    & (df["order_date"] < TEST_START)
].copy()

test = df.loc[
    df["order_date"] >= TEST_START
].copy()

if train.empty or validation.empty or test.empty:
    raise ValueError("Training, validation, or test split is empty.")

print("\n=== CHRONOLOGICAL SPLIT ===")
print(f"Training orders: {len(train):,}")
print(f"Validation orders: {len(validation):,}")
print(f"Test orders: {len(test):,}")


# --------------------------------------------------
# 2. Prepare the model inputs
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

features = categorical_features + numeric_features

X_train = train[features]
y_train = train[TARGET]

X_val = validation[features]
y_val = validation[TARGET]

X_test = test[features]
y_test = test[TARGET]

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

# Two candidate thresholds carried forward from validation.
thresholds = {
    "focused_055": 0.55,
    "broad_040": 0.40,
}

validation_probabilities = model.predict_proba(X_val)[:, 1]

print("\n=== VALIDATION CHECK ===")
for name, threshold in thresholds.items():
    predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    print(
        f"{name}: "
        f"precision={precision_score(y_val, predictions, zero_division=0):.3f}, "
        f"recall={recall_score(y_val, predictions, zero_division=0):.3f}, "
        f"balanced_accuracy={balanced_accuracy_score(y_val, predictions):.3f}"
    )

# --------------------------------------------------
# 3. Predict risk on the later test period
# --------------------------------------------------
test_probabilities = model.predict_proba(X_test)[:, 1]

predictions = test[
    ["order_id", "order_date", TARGET, "order_value_net"]
].copy()

predictions["risk_probability"] = test_probabilities

for name, threshold in thresholds.items():
    predictions[f"flag_{name}"] = (
        predictions["risk_probability"] >= threshold
    ).astype(int)

PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
predictions.to_csv(PREDICTIONS_FILE, index=False)

print("\n=== TEST PERIOD MODEL PERFORMANCE ===")

for name, threshold in thresholds.items():
    flags = (
        predictions[f"flag_{name}"].to_numpy()
    )

    print(f"\nPolicy: {name} (threshold={threshold:.2f})")
    print(f"Orders flagged: {int(flags.sum()):,}")
    print(f"Flag rate: {flags.mean():.2%}")
    print(
        f"Precision: "
        f"{precision_score(y_test, flags, zero_division=0):.3f}"
    )
    print(
        f"Recall: "
        f"{recall_score(y_test, flags, zero_division=0):.3f}"
    )
    print(
        f"F1-score: "
        f"{f1_score(y_test, flags, zero_division=0):.3f}"
    )
    print(
        f"Balanced accuracy: "
        f"{balanced_accuracy_score(y_test, flags):.3f}"
    )

print(
    f"\nTest ROC-AUC: "
    f"{roc_auc_score(y_test, test_probabilities):.3f}"
)


# --------------------------------------------------
# 4. Calculate historical order-value exposure
# --------------------------------------------------
all_late = predictions[TARGET].eq(1)

total_order_value = predictions["order_value_net"].sum()
late_order_value = predictions.loc[
    all_late, "order_value_net"
].sum()

print("\n=== HISTORICAL ORDER-VALUE EXPOSURE ===")
print(f"Total test-period order value: {total_order_value:,.2f}")
print(f"Value associated with late labels: {late_order_value:,.2f}")
print(
    "Share of order value associated with late labels: "
    f"{late_order_value / total_order_value:.2%}"
    if total_order_value else "Not available"
)


# --------------------------------------------------
# 5. Model-based financial scenarios
# --------------------------------------------------
# These rates are hypothetical assumptions, NOT observed
# losses in the dataset. Adjust them for different scenarios.
delay_cost_rates = [0.02, 0.05, 0.10]
mitigation_effectiveness_rates = [0.10, 0.25, 0.40]

scenario_rows = []

for name, threshold in thresholds.items():
    flag_column = f"flag_{name}"

    flagged = predictions[flag_column].eq(1)
    true_positive = flagged & all_late
    false_positive = flagged & ~all_late
    missed_late = ~flagged & all_late

    flagged_value = predictions.loc[
        flagged, "order_value_net"
    ].sum()

    correctly_flagged_late_value = predictions.loc[
        true_positive, "order_value_net"
    ].sum()

    false_alert_value = predictions.loc[
        false_positive, "order_value_net"
    ].sum()

    missed_late_value = predictions.loc[
        missed_late, "order_value_net"
    ].sum()

    # Sum probabilities weighted by order value.
    # This is a model-derived exposure proxy, not actual loss.
    risk_weighted_flagged_value = (
        predictions.loc[flagged, "risk_probability"]
        * predictions.loc[flagged, "order_value_net"]
    ).sum()

    for cost_rate in delay_cost_rates:
        for mitigation_rate in mitigation_effectiveness_rates:

            estimated_exposure_cost = (
                risk_weighted_flagged_value * cost_rate
            )

            potential_avoided_cost = (
                estimated_exposure_cost * mitigation_rate
            )

            historical_missed_value_cost_proxy = (
                missed_late_value * cost_rate
            )

            scenario_rows.append({
                "policy": name,
                "threshold": threshold,
                "test_orders": len(predictions),
                "orders_flagged": int(flagged.sum()),
                "flagged_order_value": flagged_value,
                "risk_weighted_flagged_value":
                    risk_weighted_flagged_value,
                "historical_late_order_value_flagged":
                    correctly_flagged_late_value,
                "order_value_from_false_alerts": false_alert_value,
                "missed_late_order_value": missed_late_value,
                "hypothetical_delay_cost_rate_pct":
                    cost_rate * 100,
                "assumed_mitigation_effectiveness_pct":
                    mitigation_rate * 100,
                "estimated_cost_on_flagged_risk_exposure":
                    estimated_exposure_cost,
                "hypothetical_cost_avoided":
                    potential_avoided_cost,
                "estimated_cost_remaining_on_flagged_exposure":
                    estimated_exposure_cost - potential_avoided_cost,
                "missed_late_order_cost_proxy":
                    historical_missed_value_cost_proxy,
            })

scenario_df = pd.DataFrame(scenario_rows)

SCENARIO_FILE.parent.mkdir(parents=True, exist_ok=True)
scenario_df.to_csv(SCENARIO_FILE, index=False)

print("\n=== EXAMPLE SCENARIO ===")
print(
    scenario_df[
        [
            "policy",
            "hypothetical_delay_cost_rate_pct",
            "assumed_mitigation_effectiveness_pct",
            "estimated_cost_on_flagged_risk_exposure",
            "hypothetical_cost_avoided",
            "missed_late_order_cost_proxy",
        ]
    ].head(6).round(2).to_string(index=False)
)

print("\n=== FILES SAVED ===")
print(f"Test predictions: {PREDICTIONS_FILE}")
print(f"Financial scenarios: {SCENARIO_FILE}")
print("\nScenario analysis complete.")
print(
    "Reminder: scenario costs are assumptions, not measured losses. "
    "Order values use the dataset's recorded units."
)
