
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_DIR
    / "reports"
    / "financial_scenario_summary.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "reports"
    / "policy_comparison.csv"
)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Scenario file not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

required = [
    "policy",
    "threshold",
    "orders_flagged",
    "flagged_order_value",
    "risk_weighted_flagged_value",
    "historical_late_order_value_flagged",
    "order_value_from_false_alerts",
    "missed_late_order_value",
    "hypothetical_delay_cost_rate_pct",
    "assumed_mitigation_effectiveness_pct",
    "estimated_cost_on_flagged_risk_exposure",
    "hypothetical_cost_avoided",
    "estimated_cost_remaining_on_flagged_exposure",
    "missed_late_order_cost_proxy",
]

missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Compare both policies with identical assumptions.
comparison = df.loc[
    df["hypothetical_delay_cost_rate_pct"].eq(5.0)
    & df["assumed_mitigation_effectiveness_pct"].eq(25.0)
].copy()

if comparison.empty:
    raise ValueError("No rows found for the selected scenario assumptions.")

display_columns = [
    "policy",
    "threshold",
    "orders_flagged",
    "flagged_order_value",
    "historical_late_order_value_flagged",
    "order_value_from_false_alerts",
    "missed_late_order_value",
    "estimated_cost_on_flagged_risk_exposure",
    "hypothetical_cost_avoided",
    "estimated_cost_remaining_on_flagged_exposure",
    "missed_late_order_cost_proxy",
]

comparison = comparison[display_columns].sort_values("threshold")

print("\n=== POLICY COMPARISON ===")
print("Assumptions: 5% delay-cost rate, 25% mitigation effectiveness")
print(comparison.round(2).to_string(index=False))

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
comparison.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved comparison to: {OUTPUT_FILE}")
print(
    "\nAll cost figures are hypothetical scenario estimates, "
    "not verified financial losses or savings."
)
