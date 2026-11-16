
from pathlib import Path

import pandas as pd

REPORT_DIR = Path("reports")

df = pd.read_csv(
    REPORT_DIR / "threshold_business_analysis.csv"
)

# Hypothetical business scenarios.
# These are not JPMorganChase financial figures.
scenarios = [
    {
        "scenario": "Conservative",
        "contact_cost": 15,
        "success_rate": 0.10,
        "retained_value": 150,
    },
    {
        "scenario": "Base Case",
        "contact_cost": 10,
        "success_rate": 0.20,
        "retained_value": 200,
    },
    {
        "scenario": "Optimistic",
        "contact_cost": 5,
        "success_rate": 0.30,
        "retained_value": 300,
    },
]

results = []

for scenario in scenarios:
    for _, row in df.iterrows():
        expected_retained = (
            row["true_positives"] * scenario["success_rate"]
        )

        expected_benefit = (
            expected_retained * scenario["retained_value"]
        )

        campaign_cost = (
            row["contacts"] * scenario["contact_cost"]
        )

        net_value = expected_benefit - campaign_cost

        results.append({
            "scenario": scenario["scenario"],
            "model": row["model"],
            "threshold": row["threshold"],
            "contacts": int(row["contacts"]),
            "true_positives": int(row["true_positives"]),
            "expected_retained": expected_retained,
            "campaign_cost": campaign_cost,
            "expected_benefit": expected_benefit,
            "estimated_net_value": net_value,
        })

results_df = pd.DataFrame(results)

results_df.to_csv(
    REPORT_DIR / "business_sensitivity_analysis.csv",
    index=False,
)

best_scenarios = (
    results_df.sort_values(
        "estimated_net_value", ascending=False
    )
    .groupby("scenario", sort=False)
    .head(1)
    .sort_values("scenario")
)

print("\nBEST MODEL AND THRESHOLD BY SCENARIO")
print(best_scenarios.to_string(index=False))

print("\nBusiness sensitivity analysis: PASSED")
print("Results saved to reports/business_sensitivity_analysis.csv")
