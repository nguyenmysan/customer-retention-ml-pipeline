"""Tests for batch data-drift monitoring."""

import numpy as np
import pandas as pd
import pytest

from src.monitor_model import build_baseline, monitor


@pytest.fixture
def training_data():
    rng = np.random.default_rng(42)
    n = 1000

    return pd.DataFrame({
        "CreditScore": rng.normal(650, 70, n),
        "Age": rng.normal(40, 10, n),
        "Tenure": rng.integers(0, 11, n),
        "Balance": rng.uniform(0, 200000, n),
        "NumOfProducts": rng.choice([1, 2, 3], n),
        "HasCrCard": rng.integers(0, 2, n),
        "IsActiveMember": rng.integers(0, 2, n),
        "EstimatedSalary": rng.uniform(10000, 150000, n),
        "Geography": rng.choice(["France", "Germany", "Spain"], n),
        "Gender": rng.choice(["Male", "Female"], n),
    })


def test_no_drift_on_identical_data(training_data):
    baseline = build_baseline(training_data)
    report = monitor(baseline, training_data.copy())

    assert report["drift_detected"] is False
    assert report["drifted_features"] == []


def test_detect_age_drift(training_data):
    baseline = build_baseline(training_data)
    changed = training_data.copy()
    changed["Age"] += 20

    report = monitor(baseline, changed)

    assert report["drift_detected"] is True
    assert "Age" in report["drifted_features"]
    assert report["feature_results"]["Age"]["psi"] >= 0.20


def test_detect_category_drift(training_data):
    baseline = build_baseline(training_data)
    changed = training_data.copy()
    changed["Geography"] = "Germany"

    report = monitor(baseline, changed)

    assert report["drift_detected"] is True
    assert "Geography" in report["drifted_features"]


def test_missing_feature_raises_error(training_data):
    baseline = build_baseline(training_data)
    changed = training_data.drop(columns=["Age"])

    with pytest.raises(ValueError, match="Missing monitoring features"):
        monitor(baseline, changed)
