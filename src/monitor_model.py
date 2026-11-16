"""Batch data-drift monitoring for the bank churn model."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocess import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    load_data,
    split_data,
)

BASELINE_PATH = Path("artifacts/monitoring_baseline.json")
REPORT_PATH = Path("reports/monitoring_report.json")
PSI_THRESHOLD = 0.20
EPSILON = 1e-6


def build_baseline(train):
    baseline = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "Training split only (GroupShuffleSplit)",
        "training_rows": len(train),
        "features": {},
    }

    for feature in NUMERIC_FEATURES:
        values = pd.to_numeric(train[feature], errors="coerce")
        valid = values.dropna().to_numpy(dtype=float)

        # Quantile bins with duplicate edges removed.
        edges = np.unique(
            np.quantile(valid, np.linspace(0, 1, 11))
        )

        if len(edges) < 2:
            edges = np.array([valid[0] - 1, valid[0] + 1])

        # Extend to infinity to handle future out-of-range values.
        edges[0] = -np.inf
        edges[-1] = np.inf

        counts, _ = np.histogram(valid, bins=edges)
        proportions = (counts / len(values)).tolist()

        baseline["features"][feature] = {
            "type": "numeric",
            "edges": edges.tolist(),
            "proportions": proportions,
            "missing_proportion": float(values.isna().mean()),
        }

    for feature in CATEGORICAL_FEATURES:
        proportions = (
            train[feature]
            .fillna("__MISSING__")
            .astype(str)
            .value_counts(normalize=True)
            .to_dict()
        )

        baseline["features"][feature] = {
            "type": "categorical",
            "proportions": proportions,
        }

    return baseline


def psi(expected, actual):
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    expected = np.clip(expected, EPSILON, None)
    actual = np.clip(actual, EPSILON, None)

    return float(
        np.sum((actual - expected) * np.log(actual / expected))
    )


def calculate_feature_psi(reference, current):
    if reference["type"] == "numeric":
        edges = np.asarray(reference["edges"], dtype=float)
        values = pd.to_numeric(current, errors="coerce")
        valid = values.dropna().to_numpy(dtype=float)

        counts, _ = np.histogram(valid, bins=edges)
        actual = np.append(counts / len(current), values.isna().mean())

        expected = np.append(
            reference["proportions"],
            reference["missing_proportion"],
        )
        return psi(expected, actual)

    expected_categories = reference["proportions"]
    actual_categories = (
        current.fillna("__MISSING__")
        .astype(str)
        .value_counts(normalize=True)
        .to_dict()
    )

    categories = sorted(
        set(expected_categories) | set(actual_categories)
    )

    expected = [expected_categories.get(c, 0) for c in categories]
    actual = [actual_categories.get(c, 0) for c in categories]

    return psi(expected, actual)


def monitor(baseline, current, threshold=PSI_THRESHOLD):
    missing = set(FEATURES) - set(current.columns)
    if missing:
        raise ValueError(f"Missing monitoring features: {sorted(missing)}")

    results = {}
    for feature in FEATURES:
        score = calculate_feature_psi(
            baseline["features"][feature],
            current[feature],
        )
        results[feature] = {
            "psi": round(score, 6),
            "drift_detected": bool(score >= threshold),
        }

    drifted = [
        feature for feature, result in results.items()
        if result["drift_detected"]
    ]

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reference_rows": baseline["training_rows"],
        "monitored_rows": len(current),
        "psi_threshold": threshold,
        "drift_detected": bool(drifted),
        "drifted_features": drifted,
        "feature_results": results,
    }


def main():
    df = load_data()
    train, validation, _ = split_data(df)

    baseline = build_baseline(train)
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with BASELINE_PATH.open("w") as f:
        json.dump(baseline, f, indent=2)

    report = monitor(baseline, validation)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w") as f:
        json.dump(report, f, indent=2)

    print(f"Baseline written to {BASELINE_PATH}")
    print(f"Report written to {REPORT_PATH}")
    print(f"Drift detected: {report['drift_detected']}")
    print(f"Drifted features: {report['drifted_features']}")

    for feature, result in report["feature_results"].items():
        print(f"{feature}: PSI={result['psi']:.6f}")


if __name__ == "__main__":
    main()
