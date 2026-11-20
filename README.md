
# Bank Customer Churn Prediction & Retention Analytics

### JPMorganChase-Inspired Machine Learning Case Study

**Technologies:** Python | Scikit-learn | FastAPI | Docker | MLOps

---

## Project Overview

This project develops an end-to-end machine learning pipeline to predict customer churn and support data-driven retention decisions in the banking industry.

Using synthetic banking data from Kaggle Playground Series S4E1, the project compares Logistic Regression and Random Forest models to identify customers at higher risk of leaving.

The solution includes exploratory data analysis, preprocessing, model training, evaluation, business sensitivity analysis, model deployment through FastAPI, Docker containerization, and Population Stability Index (PSI) monitoring.

The primary objective is to demonstrate how predictive analytics can help prioritize retention outreach when business resources are limited.

> **Disclaimer:** JPMorganChase is used only as a hypothetical business scenario. This independent academic project is not affiliated with, sponsored by, or endorsed by JPMorganChase. No actual JPMorganChase customer records or campaign outcomes were used.

---

## 1. Business Problem and Objectives

Customer churn presents a challenge for financial institutions because losing customers can reduce long-term revenue and increase customer acquisition costs.

Retention teams often have limited staff capacity and cannot contact every customer.

### Business Question

**How can a bank identify customers at higher risk of leaving and prioritize retention outreach when staff capacity is limited?**

### Project Objectives

1. Explore customer characteristics associated with churn.
2. Develop and compare predictive machine learning models.
3. Evaluate model performance using appropriate classification metrics.
4. Select an outreach threshold based on validation results.
5. Translate predictions into a practical retention review queue.
6. Evaluate potential financial outcomes under hypothetical retention scenarios.
7. Demonstrate a reproducible machine learning workflow.

The project does not claim that contacting a flagged customer will necessarily prevent churn. A controlled retention pilot would be required to measure the effectiveness of outreach.

---

## 2. Project Highlights

| Category | Final Results |
|---|---|
| Dataset | Kaggle Playground Series S4E1 |
| Total Observations | 165,034 |
| Recorded Churn Rate | 21.2% |
| Machine Learning Models | Logistic Regression and Random Forest |
| Selected Model | Random Forest |
| Validation ROC-AUC | 0.8870 |
| Held-Out Test ROC-AUC | 0.8883 |
| Decision Threshold | 0.30 |
| Held-Out Test Set | 33,363 observations |
| Observations Flagged | 8,378 |
| Test Recall | 71.9% |
| Test Precision | 61.3% |
| Model Serving | FastAPI |
| Containerization | Docker |
| Monitoring | Population Stability Index (PSI) |
| Automated Testing | 20 passing tests reported in the final presentation |

The final test results were reported after model selection and threshold tuning using validation data.

The latest automated test count should be independently verified by running the test suite.

---

## 3. Dataset Description

### Data Source

The dataset was obtained from Kaggle Playground Series S4E1:

[Binary Classification with a Bank Churn Dataset](https://www.kaggle.com/competitions/playground-series-s4e1/data)

The dataset contains synthetic banking customer observations designed for binary classification.

### Dataset Characteristics

| Attribute | Description |
|---|---|
| Observations | 165,034 |
| Columns | 14 |
| Target Variable | Exited |
| Recorded Churn Rate | Approximately 21.2% |
| Missing Values | No missing values reported |

### Important Features

| Feature | Description |
|---|---|
| CreditScore | Customer credit score |
| Geography | Customer geographic location |
| Gender | Customer gender |
| Age | Customer age |
| Tenure | Number of years with the bank |
| Balance | Customer account balance |
| NumOfProducts | Number of banking products |
| HasCrCard | Credit card ownership indicator |
| IsActiveMember | Customer activity indicator |
| EstimatedSalary | Estimated annual salary |
| Exited | Customer churn indicator |

### Target Variable

- `Exited = 1`: Customer recorded as leaving.
- `Exited = 0`: Customer recorded as staying.

Customer identifiers were excluded from model predictors.

`CustomerId` was used for group-aware splitting to reduce leakage from repeated customer records.

### Dataset Limitations

The dataset is synthetic and does not represent actual JPMorganChase customers.

Predictive performance on this dataset cannot establish real-world model performance or business impact.

---

## 4. Project Structure

```text
customer-retention-ml-pipeline/
│
├── api/
│   └── main.py
│
├── src/
│   ├── bank_churn_analysis.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── train_random_forest.py
│   ├── compare_models.py
│   ├── business_sensitivity.py
│   └── monitor_model.py
│
├── tests/
│   ├── test_api.py
│   └── test_monitor_model.py
│
├── docs/
├── reports/
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

Additional generated files may include:

```text
data/train.csv
artifacts/random_forest_pipeline.joblib
artifacts/monitoring_baseline.json
reports/monitoring_report.json
```

Generated model artifacts and dataset files may not be included directly in the public repository.

They can be recreated by following the documented workflow.

---

## 5. Machine Learning Workflow

The project follows a structured machine learning process.

### Step 1: Data Collection

Download the synthetic banking dataset from Kaggle.

### Step 2: Exploratory Data Analysis

Examine:

- Dataset structure
- Churn distribution
- Customer demographics
- Customer activity
- Banking product ownership
- Relationships between features and churn

### Step 3: Data Preparation

The preprocessing workflow includes:

- Numerical feature imputation
- Categorical feature imputation
- Numerical scaling
- One-hot encoding
- Exclusion of identifiers from model predictors

### Step 4: Data Splitting

The dataset is divided into training, validation, and held-out test sets.

| Dataset | Observations |
|---|---:|
| Training | 99,011 |
| Validation | 32,660 |
| Test | 33,363 |
| **Total** | **165,034** |

Customer-disjoint splitting is used to reduce data leakage.

### Step 5: Model Training

Two supervised classification models are compared:

1. Logistic Regression
2. Random Forest

### Step 6: Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### Step 7: Threshold Selection

The validation dataset is used to select the model and a decision threshold of `0.30`.

### Step 8: Final Evaluation

The selected model and threshold are evaluated on the held-out test dataset.

### Step 9: Model Deployment

The fitted pipeline is saved and made available through FastAPI.

### Step 10: Monitoring

PSI-based monitoring is used to identify changes in feature distributions.

---

## 6. Exploratory Data Analysis

Exploratory analysis examines the relationship between customer characteristics and recorded churn.

### Customer Activity

| Customer Status | Recorded Churn Rate |
|---|---:|
| Inactive | 29.7% |
| Active | 12.5% |

Inactive observations showed a higher recorded churn rate than active observations.

However, this relationship is observational and does not establish that customer activity directly causes churn.

### Business Interpretation

Customer activity may help retention teams identify customers who deserve further review.

A real-world retention program would need additional validation before using these patterns to guide operational decisions.

---

## 7. Model Development and Evaluation

### Validation Results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.7581 | 0.8658 |
| Precision | 0.4525 | 0.7630 |
| Recall | 0.7391 | 0.5206 |
| F1-score | 0.5613 | 0.6189 |
| ROC-AUC | 0.8209 | 0.8870 |

These classification metrics were reported at the default threshold of `0.50`.

Random Forest achieved the higher validation ROC-AUC.

Logistic Regression achieved higher recall at the default threshold.

### Selected Model

**Random Forest**

The final project selected Random Forest based on its stronger overall ranking performance.

### Selected Decision Threshold

**0.30**

The lower threshold was selected to identify more potentially high-risk observations for review.

---

## 8. Final Held-Out Test Results

The selected Random Forest model was evaluated using 33,363 held-out synthetic observations.

### Model Comparison

| Model | Test ROC-AUC |
|---|---:|
| Logistic Regression | 0.8197 |
| Random Forest | **0.8883** |

Random Forest demonstrated stronger risk-ranking performance on the held-out test set.

### Classification Results

| Metric | Result |
|---|---:|
| Test Observations | 33,363 |
| Decision Threshold | 0.30 |
| Observations Flagged | 8,378 |
| Recorded Exits Correctly Flagged | 5,138 |
| Non-Exits Flagged | 3,240 |
| Recorded Exits Missed | 2,004 |
| Recall | 71.9% |
| Precision | 61.3% |

### Business Interpretation

At the selected threshold:

- Approximately 25.1% of observations were flagged for review.
- Approximately 71.9% of recorded exits were identified.
- Approximately 61.3% of flagged observations corresponded to recorded exits.
- The model missed 2,004 recorded exits.

These results demonstrate predictive classification performance, not the effectiveness of retention outreach.

A bank would still need to verify eligibility, customer contact preferences, and available staffing before taking action.

---

## 9. Business Sensitivity Analysis

The project evaluates potential retention outreach outcomes under three hypothetical financial scenarios.

The analysis uses the 8,378 flagged observations from the held-out test set.

### Retention Scenarios

| Scenario | Cost per Contact | Assumed Save Rate | Illustrative Net Value |
|---|---:|---:|---:|
| Conservative | $15 | 10% | -$48,600 |
| Base Case | $10 | 20% | $121,740 |
| Optimistic | $5 | 30% | $420,530 |

### Break-Even Analysis

Under the base-case assumptions, the approximate break-even save rate is **8.2%**.

### Interpretation

The conservative scenario produces a negative estimated net value.

The base-case and optimistic scenarios produce positive illustrative net values.

However, these estimates depend on assumptions about contact costs, customer value, retention effectiveness, and the number of contacted customers.

They are not actual bank financial results or observed campaign outcomes.

### Business Recommendation

A randomized retention pilot should be conducted before implementing outreach at scale.

The pilot should compare customers assigned to outreach with a control group to estimate incremental retention.

---

## 10. How to Run the Project (macOS & Windows)

Follow the steps below to reproduce the project from the GitHub repository.

### Requirements

Recommended tools:

- Python 3.12
- Git
- Visual Studio Code
- Docker Desktop (optional)

Install the project dependencies using `requirements.txt`.

### Step 1: Clone the Repository

**macOS – Terminal**

```bash
git clone https://github.com/nguyenmysan/jpmorganchase-customer-churn-prediction.git
cd jpmorganchase-customer-churn-prediction
```

**Windows – PowerShell**

```powershell
git clone https://github.com/nguyenmysan/jpmorganchase-customer-churn-prediction.git
cd jpmorganchase-customer-churn-prediction
```

### Step 2: Create a Virtual Environment

**macOS – Terminal**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows – PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

**macOS – Terminal**

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Windows – PowerShell**

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 4: Download the Dataset

Download `train.csv` from:

https://www.kaggle.com/competitions/playground-series-s4e1/data

**macOS – Terminal**

```bash
mkdir -p data
```

**Windows – PowerShell**

```powershell
New-Item -ItemType Directory -Force data
```

Place the downloaded file at:

```text
data/train.csv
```

### Step 5: Run Data Preprocessing

**macOS – Terminal**

```bash
python src/preprocess.py
```

**Windows – PowerShell**

```powershell
python src/preprocess.py
```

### Step 6: Train Logistic Regression

**macOS – Terminal**

```bash
python src/train_model.py
```

**Windows – PowerShell**

```powershell
python src/train_model.py
```

### Step 7: Train Random Forest

**macOS – Terminal**

```bash
python src/train_random_forest.py
```

**Windows – PowerShell**

```powershell
python src/train_random_forest.py
```

### Step 8: Compare Models

**macOS – Terminal**

```bash
python src/compare_models.py
```

**Windows – PowerShell**

```powershell
python src/compare_models.py
```

### Step 9: Run Business Sensitivity Analysis

**macOS – Terminal**

```bash
python src/business_sensitivity.py
```

**Windows – PowerShell**

```powershell
python src/business_sensitivity.py
```

### Step 10: Run Exploratory Data Analysis

**macOS – Terminal**

```bash
python src/bank_churn_analysis.py
```

**Windows – PowerShell**

```powershell
python src/bank_churn_analysis.py
```

### Step 11: Verify the Saved Model

The trained Random Forest pipeline should be available at:

```text
artifacts/random_forest_pipeline.joblib
```

**macOS – Terminal**

```bash
python -c "from joblib import load; model = load('artifacts/random_forest_pipeline.joblib'); print(type(model).__name__)"
```

**Windows – PowerShell**

```powershell
python -c "from joblib import load; model = load('artifacts/random_forest_pipeline.joblib'); print(type(model).__name__)"
```

A successful run confirms that the saved model can be loaded.

### Step 12: Run Automated Tests

**macOS – Terminal**

```bash
python -m pytest tests/ -v
```

**Windows – PowerShell**

```powershell
python -m pytest tests/ -v
```

The final presentation reports 20 passing tests. Confirm the actual count using the latest repository version.

### Step 13: Start the FastAPI Application

**macOS – Terminal**

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

**Windows – PowerShell**

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

Available endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | Service health |
| POST | `/predict` | Customer churn prediction |

Use `/predict` to submit a valid synthetic customer record and obtain a churn-risk prediction.

Stop the API with `Ctrl + C` before starting Docker on the same port.

### Step 14: Build the Docker Image

Make sure Docker Desktop is installed and running.

**macOS – Terminal**

```bash
docker build -t bank-churn-api .
```

**Windows – PowerShell**

```powershell
docker build -t bank-churn-api .
```

### Step 15: Run the Docker Container

The saved model must already exist in the `artifacts` directory.

**macOS – Terminal**

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/artifacts:/app/artifacts:ro" \
  bank-churn-api
```

**Windows – PowerShell**

```powershell
docker run --rm -p 8000:8000 --mount "type=bind,source=$((Get-Location).Path)\artifacts,target=/app/artifacts,readonly" bank-churn-api
```

Open:

http://127.0.0.1:8000/docs

Verify that the API responds and the model loads successfully.

### Step 16: Run MLOps Monitoring

Stop the Docker container with `Ctrl + C` if it is still running.

**macOS – Terminal**

```bash
python -m src.monitor_model
```

**Windows – PowerShell**

```powershell
python -m src.monitor_model
```

Expected monitoring files:

```text
artifacts/monitoring_baseline.json
reports/monitoring_report.json
```

The monitoring process uses Population Stability Index (PSI) to detect feature-distribution changes.

### Step 17: Review the Final Outputs

**macOS – Terminal**

```bash
ls -la artifacts
ls -la reports
```

**Windows – PowerShell**

```powershell
Get-ChildItem artifacts
Get-ChildItem reports
```

Confirm that the expected artifacts and reports have been generated.

### Reproduction Notes

- Run all commands from the repository root.
- Keep the virtual environment activated when running Python commands.
- Download the Kaggle dataset before preprocessing or training.
- Train the model before starting FastAPI or Docker.
- Use the same dataset and dependency versions to reproduce reported results.
- macOS execution is documented in the project.
- Windows commands are provided for independent reproduction and should be tested on Windows.
- Compare the generated results with the final report before submission.

---

## 11. FastAPI Model Deployment

FastAPI provides an interface for accessing the trained machine learning model.

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | Service and model status |
| POST | `/predict` | Customer churn prediction |

### Example Prediction Input

```json
{
  "CreditScore": 650,
  "Geography": "France",
  "Gender": "Female",
  "Age": 42,
  "Tenure": 5,
  "Balance": 125000,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 85000
}
```

### Documented Prediction Example

The final presentation reports the following prediction for a synthetic customer:

| Output | Result |
|---|---:|
| Predicted Churn Risk | 0.1384 |
| Review Threshold | 0.30 |
| Review Flag | 0 |

The example customer would not be prioritized for immediate retention review under the selected threshold.

The exact prediction should be verified against the saved model artifact used for the final presentation.

---

## 12. Docker Containerization

Docker provides a reproducible environment for running the FastAPI application.

The Docker workflow packages the API and dependencies into a consistent environment.

The fitted model is supplied through a read-only bind mount.

The model artifact must be generated before starting the container.

Docker build and run instructions for both macOS and Windows are included in Section 10.

---

## 13. MLOps and Model Monitoring

The project implements batch-oriented monitoring using the Population Stability Index (PSI).

### Monitoring Objective

Monitoring helps identify changes in input feature distributions compared with the training-data baseline.

### Population Stability Index

PSI measures differences between reference and current feature distributions.

The documented monitoring threshold is:

`PSI >= 0.20`

Values at or above this threshold are flagged for investigation.

### Monitoring Workflow

1. Establish a baseline from training data.
2. Compare current feature distributions against the baseline.
3. Calculate PSI values.
4. Generate a monitoring report.
5. Identify features requiring investigation.

### Monitoring Interpretation

A PSI alert does not automatically mean that the model is inaccurate.

It indicates that the input distribution has changed enough to justify further investigation.

The monitoring implementation is a batch demonstration rather than a live production monitoring system.

Automatic model retraining is not implemented.

---

## 14. Automated Testing

Automated testing is included to help verify API behavior and monitoring functionality.

### Test Coverage Areas

The documented tests include:

- API health checks
- Prediction endpoint behavior
- Input validation
- Monitoring calculations
- Monitoring alert behavior

### Test Results

The earlier README documented:

**10 passing tests and 1 warning**

The final project presentation reports:

**20 passing tests**

The latest test count should be confirmed by executing the test suite on the final repository version.

This README does not claim that the 20-test result has been independently reproduced.

---

## 15. Model Reproducibility

Reproducibility is an important component of this project.

The machine learning workflow saves the fitted preprocessing and prediction pipeline.

### Saved Model

The documented Random Forest artifact is:

```text
artifacts/random_forest_pipeline.joblib
```

### Reproducibility Objectives

A reviewer should be able to:

1. Clone the repository.
2. Install the required dependencies.
3. Download the permitted dataset.
4. Execute preprocessing and training.
5. Reproduce the evaluation workflow.
6. Save and reload the fitted model.
7. Generate predictions using valid input.
8. Run automated tests.
9. Start the FastAPI service.
10. Run Docker and monitoring checks.

The final presentation reports that a fresh-process model reload reproduced the reference prediction score.

Exact results depend on the submitted dataset, model artifact, software versions, and execution configuration.

---

## 16. Proposed Retention Strategy

The project recommends a controlled retention pilot rather than immediate large-scale deployment.

### Phase 1: Define Eligibility

Identify customers eligible for retention outreach.

Confirm:

- Contact preferences
- Customer eligibility
- Duplicate customer identifiers
- Available staffing
- Outreach capacity

### Phase 2: Prioritize Customers

Use model scores and the selected threshold to construct a review queue.

### Phase 3: Conduct a Randomized Pilot

Randomly assign eligible customers to:

- Outreach group
- Control group

### Phase 4: Measure Outcomes

Evaluate:

- Incremental retention
- Contact costs
- Net financial contribution
- Customer complaints
- Opt-out rates

### Phase 5: Evaluate Expansion

Expand the program only if pilot results demonstrate credible business value.

---

## 17. Project Limitations

Several limitations should be considered before applying this approach in a real banking environment.

### Synthetic Data

The dataset does not contain real JPMorganChase customer records.

### Predictive Performance

Model metrics measure the ability to identify recorded churn outcomes.

They do not measure the causal effect of retention outreach.

### Financial Assumptions

Business sensitivity results depend on hypothetical assumptions.

### Customer Privacy

Real-world implementation would require appropriate privacy protections and data governance.

### Fairness

Model errors and outcomes should be evaluated across relevant customer groups.

### Monitoring

The current monitoring implementation demonstrates batch drift detection.

It does not include continuous production monitoring or automatic retraining.

### Operational Readiness

Before deployment, the organization would need to validate feature availability, observation timing, contact permissions, and operational requirements.

---

## 18. Final Project Deliverables

The final project is designed to provide the following materials:

| Deliverable | Description |
|---|---|
| Final Business Report | PDF explaining the project and results |
| Rendered Analysis | HTML report or equivalent readable output |
| Source Code | Python analysis and implementation |
| Dataset Documentation | Data source and access instructions |
| Saved Model | Fitted preprocessing and prediction pipeline |
| FastAPI | Model-serving prototype |
| Docker | Containerized API environment |
| MLOps | PSI-based monitoring |
| Automated Tests | Reproducibility and functionality checks |
| PowerPoint | Stakeholder-focused final presentation |
| Assignment Validation | Peer reproduction and verification |
| Feedback Revision Log | Documentation of changes following feedback |

The presence and completeness of each deliverable should be checked in the final repository and submission package.

---

## 19. Final Business Recommendation

The project demonstrates that machine learning can help prioritize customer churn risk in a synthetic banking scenario.

Random Forest achieved a held-out ROC-AUC of **0.8883**.

Using the selected threshold of **0.30**, the model identified approximately **71.9% of recorded exits** while flagging approximately one quarter of test observations.

These results support the use of predictive scores as a potential input to a retention review process.

However, prediction accuracy alone does not establish business value.

The recommended next step is a randomized retention pilot to measure whether outreach produces incremental retention and positive net financial contribution.

---

