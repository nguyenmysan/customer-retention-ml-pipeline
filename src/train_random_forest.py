
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from preprocess import (
    load_data,
    split_data,
    build_preprocessor,
    FEATURES,
)

ARTIFACT_DIR = Path("artifacts")
REPORT_DIR = Path("reports")


def evaluate_model(model, X, y, dataset_name):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    metrics = {
        "dataset": dataset_name,
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(
            y, predictions, zero_division=0
        ),
        "recall": recall_score(
            y, predictions, zero_division=0
        ),
        "f1": f1_score(
            y, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y, probabilities),
    }

    print(f"\n{dataset_name} RESULTS")

    for name, value in metrics.items():
        if name != "dataset":
            print(f"{name}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, predictions))

    return metrics


def main():
    ARTIFACT_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    df = load_data()
    train, val, test = split_data(df)

    X_train = train[FEATURES]
    y_train = train["Exited"]

    X_val = val[FEATURES]
    y_val = val["Exited"]

    model = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    print("Training Random Forest...")
    model.fit(X_train, y_train)

    train_metrics = evaluate_model(
        model, X_train, y_train, "Training"
    )

    val_metrics = evaluate_model(
        model, X_val, y_val, "Validation"
    )

    pd.DataFrame([
        train_metrics,
        val_metrics,
    ]).to_csv(
        REPORT_DIR / "random_forest_metrics.csv",
        index=False,
    )

    joblib.dump(
        model,
        ARTIFACT_DIR / "random_forest_pipeline.joblib",
    )

    print("\nRandom Forest model saved successfully.")
    print("Test set reserved for final evaluation.")


if __name__ == "__main__":
    main()
