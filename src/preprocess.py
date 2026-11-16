
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

DATA_PATH = Path("data/train.csv")

NUMERIC_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

CATEGORICAL_FEATURES = [
    "Geography",
    "Gender",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data():
    df = pd.read_csv(DATA_PATH)

    required = FEATURES + ["Exited", "CustomerId"]
    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df


def split_data(df):
    # Keep repeated CustomerIds within the same split.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_val_idx, test_idx = next(
        splitter.split(
            df,
            df["Exited"],
            groups=df["CustomerId"],
        )
    )

    train_val = df.iloc[train_val_idx].copy()
    test = df.iloc[test_idx].copy()

    val_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42,
    )

    train_idx, val_idx = next(
        val_splitter.split(
            train_val,
            train_val["Exited"],
            groups=train_val["CustomerId"],
        )
    )

    train = train_val.iloc[train_idx].copy()
    val = train_val.iloc[val_idx].copy()

    return train, val, test


def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


if __name__ == "__main__":
    df = load_data()
    train, val, test = split_data(df)

    print("Train:", train.shape)
    print("Validation:", val.shape)
    print("Test:", test.shape)

    print("\nChurn rates:")
    for name, subset in [
        ("Train", train),
        ("Validation", val),
        ("Test", test),
    ]:
        print(f"{name}: {subset['Exited'].mean():.4f}")

    train_ids = set(train["CustomerId"])
    val_ids = set(val["CustomerId"])
    test_ids = set(test["CustomerId"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    print("\nCustomerId leakage check: PASSED")

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(train[FEATURES])

    print("Transformed training shape:", transformed.shape)
    print("Preprocessing check: PASSED")
