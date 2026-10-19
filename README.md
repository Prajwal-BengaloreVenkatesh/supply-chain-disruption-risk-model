# Supply Chain Disruption Risk Model

A portfolio project combining predictive analytics and Power BI to explore recorded late-delivery risk, compare alerting policies, and illustrate how operational monitoring choices can affect workload and hypothetical financial exposure.

> **Important:** This is an exploratory portfolio project using historical DataCo supply-chain data from 2015–2018. The model predicts a recorded risk label; it is not a live logistics forecasting system. Financial figures are illustrative scenario estimates, not realized savings or verified losses.

## Business question

How can a supply-chain team use order-level risk scores to choose between:

- **Broad monitoring (40% cutoff):** flag more orders to capture a greater share of positive-label orders, with a larger alert workload.
- **Focused alerts (55% cutoff):** flag fewer orders with higher precision, while missing more positive-label orders.

The dashboard presents the trade-off rather than claiming that one policy is always best.

## Dashboard

The Power BI report contains four pages:

1. **Executive Risk Overview** — test-period KPIs, policy alert volumes, and monthly recorded positive-label rate.
2. **Delivery Risk Drivers** — descriptive rate comparisons by shipping mode, market, destination region, and month.
3. **Financial Scenario Analysis** — policy comparison and sensitivity analysis under explicit hypothetical delay-cost and mitigation assumptions.
4. **Model Performance & Alert Quality** — precision, recall, alert workload, and missed positive-label orders for the two monitoring cutoffs.

Suggested screenshot paths once screenshots are added to the repository:

- `reports/figures/page1_executive_overview.png`
- `reports/figures/page2_delivery_risk_drivers.png`
- `reports/figures/page3_financial_scenario_analysis.png`
- `reports/figures/page4_model_performance.png`

## Data and preparation

- Source: DataCo Supply Chain dataset, covering January 2015 through January 2018.
- Original raw data: 180,519 item-level rows.
- Eligible order-level data after excluding cancelled orders: 62,897 unique orders.
- The prepared order-level dataset contains aggregated order attributes, including order date, shipping mode, market, destination, order value and product-mix features.
- Direct identifiers and sensitive customer fields were excluded from the modelling features.

The recorded target is `late_delivery_risk`. In the source data, this label agrees completely with `Delivery Status`; therefore, it should be interpreted as a recorded dataset label rather than an independently verified future outcome. The analysis is descriptive and does not establish that any shipping mode, market, or region causes delays.

## Modeling approach

- Model: Logistic Regression.
- Split strategy: chronological rather than random, to better reflect evaluation on later observations.
- Training period: January 2015–December 2016 (39,950 orders).
- Validation period: January–June 2017 (9,876 orders).
- Test/evaluation period: July 2017–January 2018 (13,071 orders).
- Test-period positive-label rate: 57.56%.
- Test ROC-AUC: 0.746.

The test period has been inspected during project development, so it should not be described as a pristine, untouched holdout. Performance represents evaluation on this historical dataset and does not guarantee performance on current or future logistics data.

## Monitoring policy comparison

Reported results on the evaluation period:

| Metric | Broad monitoring (40%) | Focused alerts (55%) |
|---|---:|---:|
| Orders flagged | 10,714 (81.97%) | 4,570 (34.96%) |
| Precision | 61.6% | 88.8% |
| Recall | 87.7% | 54.0% |

**Interpretation:** Broad monitoring identifies a larger share of positive-label orders, at the cost of a much larger alert workload and lower precision. Focused alerts are more selective and more precise, but leave more positive-label orders unflagged. Policy choice depends on operational capacity and the relative cost of missed delays versus unnecessary investigations.

## Illustrative financial scenarios

The scenario analysis explores assumptions about delay cost and mitigation effectiveness. Under the reference assumptions of a 5% delay-cost rate and 25% mitigation effectiveness, the comparison reports hypothetical cost avoided of approximately:

- Broad monitoring: **31,834.06 dataset monetary units**.
- Focused alerts: **19,734.12 dataset monetary units**.

The missed-late-order cost proxy is approximately 19,888.29 for broad monitoring and 65,381.98 for focused alerts under the same comparison setup.

These are **scenario outputs only**. They are not observed financial savings, confirmed losses, or proof that interventions would work as assumed. The underlying dataset currency is not independently verified here; values are therefore described as dataset monetary units rather than euros or dollars. The scenario does not establish real intervention costs or causal impact.

## Repository contents

The working project is organized around:

```text
supply-chain-disruption-risk-model/
├── data/
│   ├── raw/                         # Source dataset (check license before sharing)
│   └── processed/
│       ├── order_level_modeling_dataset.csv
│       └── test_order_risk_predictions.csv
├── src/                             # Data preparation, modeling, and scenario scripts
├── reports/
│   ├── figures/                     # Dashboard screenshots / exported charts
│   ├── financial_scenario_summary.csv
│   ├── policy_comparison.csv
│   └── supply_chain_disruption_risk_model.pbix
└── README.md
```

Adjust the tree to match the actual files present before publishing. Do not commit credentials, private data, or a raw dataset unless its license permits redistribution. If files are too large, document how to obtain the source data and keep generated outputs appropriately sized.

## Reproducibility checklist

Before treating the project as reproducible, ensure the repository includes:

- The source scripts for preprocessing, chronological splitting, training/evaluation, and scenario analysis.
- A dependency file such as `requirements.txt` with the tested Python package versions.
- Clear instructions for placing the source dataset in `data/raw/` and running each pipeline step in order.
- The final `.pbix` report if its size is reasonable, plus readable page screenshots for visitors who do not have Power BI Desktop.
- Documentation of the threshold logic, metric definitions, financial assumptions, and known limitations.

## Limitations and responsible interpretation

- Historical data only; no live feeds or real-time monitoring.
- The positive target is a recorded label that aligns with another source status field, not an independently verified prediction of future delay.
- Descriptive associations do not establish causes.
- Shipping mode is tightly associated with scheduled shipping duration in this dataset; avoid causal interpretation of differences by mode.
- The scenario analysis uses assumptions and omits verified intervention costs and real-world intervention outcomes.
- Results may not generalize to a current carrier, market, geography, or operating environment.

## Skills demonstrated

Python data preparation and feature aggregation; chronological model evaluation; classification metrics and threshold trade-offs; scenario-based analysis; DAX measures; Power Query data-type handling; and multi-page Power BI dashboard design.
