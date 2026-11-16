# Bank Customer Churn Prediction & Retention Analysis | JPMorganChase Case Study

## Project Overview

This project develops a machine learning pipeline to predict customer churn and support data-driven customer retention decisions.

The analysis uses a synthetic bank customer dataset from Kaggle Playground Series S4E1.

This is an independent academic project using JPMorganChase as an illustrative business context. It is not affiliated with, sponsored by, or based on proprietary data from JPMorganChase.

## Business Problem

Customer churn can increase acquisition costs and reduce long-term customer value.

The project aims to:
- Identify customers at risk of churn.
- Compare machine learning classification models.
- Evaluate precision, recall, F1-score, and ROC-AUC.
- Translate predictions into potential retention actions.
- Examine how campaign costs and assumptions affect decisions.

## Dataset

Source: Kaggle Playground Series S4E1

https://www.kaggle.com/competitions/playground-series-s4e1/data

The training dataset contains 165,034 records and 14 columns.

Target variable:
- Exited = 1: Customer churn
- Exited = 0: Customer does not churn

The dataset is synthetic and should not be interpreted as actual customer records.

## Installation

Clone the repository:

```bash
git clone https://github.com/nguyenmysan/customer-retention-ml-pipeline.git
cd customer-retention-ml-pipeline
```

Create a virtual environment.

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Dataset Setup

Download train.csv from the Kaggle competition dataset page.

Place it in:

data/train.csv

The dataset is excluded from GitHub and must be downloaded separately.

## Run the Project

Run all commands from the repository root.

### 1. Data Preprocessing

```bash
python src/preprocess.py
```

This checks the data split, feature transformations, and CustomerId overlap.

### 2. Logistic Regression

```bash
python src/train_model.py
```

### 3. Random Forest

```bash
python src/train_random_forest.py
```

### 4. Model Comparison

```bash
python src/compare_models.py
```

### 5. Business Sensitivity Analysis

```bash
python src/business_sensitivity.py
```

## Model Performance

Validation results at a classification threshold of 0.50:

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.7581 | 0.8658 |
| Precision | 0.4525 | 0.7630 |
| Recall | 0.7391 | 0.5206 |
| F1-score | 0.5613 | 0.6189 |
| ROC-AUC | 0.8209 | 0.8870 |

Random Forest achieved the higher ROC-AUC and F1-score.

Logistic Regression achieved higher recall at the default threshold.

## Business Analysis

The project evaluates hypothetical retention campaigns using model predictions.

Illustrative base-case assumptions:
- Contact cost: $10 per customer
- Retention success rate: 20%
- Value per retained customer: $200

Under these assumptions, Random Forest at a threshold of 0.30 produced the highest estimated net value among the evaluated configurations.

This result is based on validation data and hypothetical assumptions. It is not observed revenue, demonstrated campaign effectiveness, or an actual JPMorganChase financial result.

The model and threshold may change when business assumptions or campaign capacity change.

## Reproducibility

The project uses:
- Python
- pandas and NumPy
- scikit-learn
- Matplotlib and Seaborn
- joblib

Random seed: 42

The data is split using GroupShuffleSplit to keep records sharing a CustomerId within the same split.

The preprocessing pipeline is fitted on training data.

Trained models are saved as joblib files in the artifacts directory.

Because model artifacts are not committed, users should run the training scripts before model comparison.

## Outputs

The reports directory contains:
- Exploratory analysis outputs
- Model evaluation metrics
- ROC and Precision-Recall curves
- Threshold business analysis
- Business sensitivity analysis

The artifacts directory stores locally generated model pipelines.

## Limitations

- The dataset is synthetic.
- Repeated CustomerIds and conflicting churn labels require caution.
- Model predictions do not establish causal effects.
- Retention success and customer value are hypothetical assumptions.
- Additional validation, fairness analysis, monitoring, and real-world testing would be necessary before deployment.
- Final test-set evaluation is reserved until model selection is finalized.

## Academic Purpose

Developed as an independent academic machine learning project for the MSBA program at the University of the Pacific.
