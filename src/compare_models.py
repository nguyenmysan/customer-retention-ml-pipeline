
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

from preprocess import load_data, split_data, FEATURES

ARTIFACT_DIR = Path("artifacts")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

# Illustrative assumptions, not real bank financial data.
CONTACT_COST = 10
SUCCESS_RATE = 0.20
VALUE_PER_RETAINED_CUSTOMER = 200

df = load_data()
train, val, test = split_data(df)

X_val = val[FEATURES]
y_val = val["Exited"]

models = {
    "Logistic Regression": joblib.load(
        ARTIFACT_DIR / "logistic_regression_pipeline.joblib"
    ),
    "Random Forest": joblib.load(
        ARTIFACT_DIR / "random_forest_pipeline.joblib"
    ),
}

comparison = []
threshold_results = []

plt.figure(figsize=(8, 6))

for name, model in models.items():
    probabilities = model.predict_proba(X_val)[:, 1]

    fpr, tpr, _ = roc_curve(y_val, probabilities)
    auc = roc_auc_score(y_val, probabilities)

    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    for threshold in np.arange(0.20, 0.81, 0.05):
        predictions = (probabilities >= threshold).astype(int)

        tp = int(((predictions == 1) & (y_val == 1)).sum())
        fp = int(((predictions == 1) & (y_val == 0)).sum())
        contacts = tp + fp

        expected_retained = tp * SUCCESS_RATE
        expected_value = (
            expected_retained * VALUE_PER_RETAINED_CUSTOMER
        )
        campaign_cost = contacts * CONTACT_COST
        estimated_net_value = expected_value - campaign_cost

        threshold_results.append({
            "model": name,
            "threshold": round(float(threshold), 2),
            "precision": precision_score(
                y_val, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_val, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_val, predictions, zero_division=0
            ),
            "true_positives": tp,
            "false_positives": fp,
            "contacts": contacts,
            "estimated_net_value": estimated_net_value,
        })

    default_predictions = (probabilities >= 0.50).astype(int)

    comparison.append({
        "model": name,
        "roc_auc": auc,
        "average_precision": average_precision_score(
            y_val, probabilities
        ),
        "precision": precision_score(
            y_val, default_predictions, zero_division=0
        ),
        "recall": recall_score(
            y_val, default_predictions, zero_division=0
        ),
        "f1": f1_score(
            y_val, default_predictions, zero_division=0
        ),
    })

plt.plot([0, 1], [0, 1], "k--", label="Random baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Customer Churn Models")
plt.legend()
plt.tight_layout()
plt.savefig(REPORT_DIR / "model_roc_comparison.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 6))

for name, model in models.items():
    probabilities = model.predict_proba(X_val)[:, 1]
    precision, recall, _ = precision_recall_curve(
        y_val, probabilities
    )
    plt.plot(recall, precision, label=name)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(
    REPORT_DIR / "model_precision_recall_comparison.png",
    dpi=200,
)
plt.close()

comparison_df = pd.DataFrame(comparison)
threshold_df = pd.DataFrame(threshold_results)

comparison_df.to_csv(
    REPORT_DIR / "model_comparison.csv", index=False
)
threshold_df.to_csv(
    REPORT_DIR / "threshold_business_analysis.csv",
    index=False,
)

best = threshold_df.loc[
    threshold_df["estimated_net_value"].idxmax()
]

print("\nMODEL COMPARISON")
print(comparison_df.round(4).to_string(index=False))

print("\nBEST ILLUSTRATIVE BUSINESS SCENARIO")
print(best.to_string())

print("\nCharts and CSV reports saved successfully.")
print("Test set remains untouched.")
