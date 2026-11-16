

# JPMorganChase Bank Customer Churn Prediction & Retention Analysis

### An Independent Machine Learning & Business Analytics Case Study


## Project Overview

This project develops an end-to-end machine learning pipeline inspired by JPMorganChase to predict bank customer churn and support data-driven customer retention decisions.

The project uses the **Binary Classification with a Bank Churn Dataset** from Kaggle Playground Series – Season 4, Episode 1 (S4E1).

The project combines predictive analytics, business decision-making, API development, containerization, and MLOps monitoring.

### Key Project Components

- Exploratory Data Analysis (EDA)
- Data preprocessing and feature transformation
- Logistic Regression classification
- Random Forest classification
- Model evaluation and comparison
- Customer retention business analysis
- Business sensitivity analysis
- FastAPI REST API
- Docker containerization
- MLOps data drift monitoring
- Automated testing with pytest

**Disclaimer:** This is an independent academic case study inspired by JPMorganChase. It is not affiliated with, endorsed by, or sponsored by JPMorganChase. The dataset is synthetic and does not contain actual JPMorganChase customer information.

---

## 1. Business Problem and Objectives

Customer churn occurs when customers stop using a company's products or services.

In banking, customer churn can increase customer acquisition costs, reduce long-term customer value, and negatively affect profitability.

### Project Objectives

1. Identify customers with a higher probability of churn.
2. Compare machine learning classification models.
3. Evaluate accuracy, precision, recall, F1-score, and ROC-AUC.
4. Examine how classification thresholds influence retention decisions.
5. Estimate hypothetical financial outcomes of customer retention campaigns.
6. Develop a reproducible machine learning workflow.
7. Demonstrate model deployment, containerization, and data drift monitoring.

---

## 2. Dataset

**Official Dataset Name:** Binary Classification with a Bank Churn Dataset

**Competition:** Kaggle Playground Series – Season 4, Episode 1 (S4E1)

**Source:** https://www.kaggle.com/competitions/playground-series-s4e1/data

### Dataset Information

| Property | Value |
|---|---|
| Dataset Name | Binary Classification with a Bank Churn Dataset |
| Competition | Kaggle Playground Series S4E1 |
| File | train.csv |
| Total Records | 165,034 |
| Total Columns | 14 |
| Missing Values | 0 |
| Target Variable | Exited |
| Dataset Type | Synthetic |

### Target Variable

- `Exited = 1`: Customer churn
- `Exited = 0`: Customer does not churn

### Numerical Features

- CreditScore
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary

### Categorical Features

- Geography
- Gender

The columns `id`, `CustomerId`, and `Surname` are excluded from predictive features.

`CustomerId` is used for group-aware data splitting to reduce customer-level data leakage.

The dataset is synthetic and does not represent actual JPMorganChase customers.

---

## 3. Project Structure

```text
customer-retention-ml-pipeline/
│
├── api/
│   └── main.py
│
├── data/
│   └── train.csv
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
├── artifacts/
│   ├── random_forest_pipeline.joblib
│   └── monitoring_baseline.json
│
├── reports/
│   └── monitoring_report.json
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

**Note:** Dataset files, trained model artifacts, and monitoring outputs are generated or stored locally and may not be included in the GitHub repository.

---

## 4. Technologies Used

### Programming and Data Analysis

- Python
- pandas
- NumPy
- Matplotlib
- Seaborn

### Machine Learning

- scikit-learn
- Logistic Regression
- Random Forest
- StandardScaler
- OneHotEncoder
- SimpleImputer
- GroupShuffleSplit
- joblib

### API and Deployment

- FastAPI
- Uvicorn
- Docker
- Docker Desktop

### MLOps and Testing

- Population Stability Index (PSI)
- Batch Data Drift Monitoring
- pytest
- FastAPI TestClient

### Development Tools

- Visual Studio Code
- Git
- GitHub

---

# 5. How to Run the Project (macOS & Windows)

This section provides step-by-step instructions for running the complete project on macOS and Windows.

Each step includes commands for both operating systems.

Run all commands from the repository root directory unless otherwise specified.

## Prerequisites

Install the following tools:

- Python (Python 3.12 recommended)
- Git
- Visual Studio Code (recommended)
- Docker Desktop (required for Docker deployment)

Official download links:

- Python: https://www.python.org/downloads/
- Git: https://git-scm.com/downloads
- Visual Studio Code: https://code.visualstudio.com/
- Docker Desktop: https://www.docker.com/products/docker-desktop/

The project was tested locally on macOS using Python 3.14.10. The Docker image uses Python 3.12.

Windows instructions are provided for reproducibility but have not been independently tested.

### Step 1: Clone the GitHub Repository

**macOS — Terminal**

```bash
git clone https://github.com/nguyenmysan/customer-retention-ml-pipeline.git
cd customer-retention-ml-pipeline
```

**Windows — PowerShell**

```powershell
git clone https://github.com/nguyenmysan/customer-retention-ml-pipeline.git
cd customer-retention-ml-pipeline
```

### Step 2: Create a Virtual Environment

**macOS — Terminal**

```bash
python3 -m venv .venv
```

**Windows — PowerShell**

```powershell
py -m venv .venv
```

### Step 3: Activate the Virtual Environment

**macOS — Terminal**

```bash
source .venv/bin/activate
```

**Windows — PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, Windows users can run Python directly from the virtual environment by replacing `python` in subsequent commands with `.\.venv\Scripts\python.exe`.

### Step 4: Install Dependencies

**macOS — Terminal**

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Windows — PowerShell**

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 5: Download the Dataset

**Dataset:** Binary Classification with a Bank Churn Dataset

Download `train.csv` from:

https://www.kaggle.com/competitions/playground-series-s4e1/data

Create the data directory if necessary.

**macOS — Terminal**

```bash
mkdir -p data
```

**Windows — PowerShell**

```powershell
New-Item -ItemType Directory -Force data
```

Place the downloaded dataset at:

```text
customer-retention-ml-pipeline/
└── data/
    └── train.csv
```

The dataset is not included in the GitHub repository.

### Step 6: Run Data Preprocessing

This step loads the dataset, prepares the features, and checks customer-level data splitting.

**macOS — Terminal**

```bash
python src/preprocess.py
```

**Windows — PowerShell**

```powershell
python src/preprocess.py
```

Verified dataset split sizes:

| Dataset | Records |
|---|---:|
| Training | 99,011 |
| Validation | 32,660 |
| Testing | 33,363 |
| Total | 165,034 |

The project uses `GroupShuffleSplit` to prevent records associated with the same CustomerId from appearing in multiple datasets.

### Step 7: Train Logistic Regression

Train the Logistic Regression baseline model.

**macOS — Terminal**

```bash
python src/train_model.py
```

**Windows — PowerShell**

```powershell
python src/train_model.py
```

The trained pipeline is saved locally in the `artifacts/` directory.

### Step 8: Train Random Forest

Train the Random Forest classification model.

**macOS — Terminal**

```bash
python src/train_random_forest.py
```

**Windows — PowerShell**

```powershell
python src/train_random_forest.py
```

The trained model is saved at:

```text
artifacts/random_forest_pipeline.joblib
```

This model is required for FastAPI and Docker deployment.

### Step 9: Compare Machine Learning Models

Compare Logistic Regression and Random Forest using validation metrics.

**macOS — Terminal**

```bash
python src/compare_models.py
```

**Windows — PowerShell**

```powershell
python src/compare_models.py
```

The comparison evaluates accuracy, precision, recall, F1-score, and ROC-AUC.

### Step 10: Run Business Sensitivity Analysis

Evaluate hypothetical customer retention campaigns under different business assumptions.

**macOS — Terminal**

```bash
python src/business_sensitivity.py
```

**Windows — PowerShell**

```powershell
python src/business_sensitivity.py
```

The analysis examines campaign costs, retention success rates, customer value, and classification thresholds.

### Step 11: Run MLOps Data Drift Monitoring

Run batch data drift monitoring using Population Stability Index (PSI).

**macOS — Terminal**

```bash
python -m src.monitor_model
```

**Windows — PowerShell**

```powershell
python -m src.monitor_model
```

Generated files:

```text
artifacts/monitoring_baseline.json
reports/monitoring_report.json
```

The monitoring system uses a PSI alert threshold of 0.20.

The validation dataset produced no drift alerts in the verified local run.

A simulated 20-year increase in the Age feature successfully triggered a drift alert.

### Step 12: Run Automated Tests

Run the FastAPI and MLOps test suites.

**macOS — Terminal**

```bash
python -m pytest tests/ -v
```

**Windows — PowerShell**

```powershell
python -m pytest tests/ -v
```

The test suite contains:

- 6 FastAPI tests
- 4 MLOps monitoring tests

Verified local result on macOS:

```text
10 passed, 1 warning in 2.11s
```

The warning concerns a dependency deprecation and does not indicate a failed test.

### Step 13: Start the FastAPI Application

Ensure that the Random Forest model has been trained.

**macOS — Terminal**

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

**Windows — PowerShell**

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

Check API health:

http://127.0.0.1:8000/health

Expected response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "Random Forest"
}
```

### Step 14: Test the Prediction Endpoint

Open Swagger UI:

http://127.0.0.1:8000/docs

The following steps are the same on macOS and Windows:

1. Select `POST /predict`.
2. Click `Try it out`.
3. Enter the JSON request below.
4. Click `Execute`.

Example request:

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

Example response from the tested model:

```json
{
  "churn_probability": 0.1279,
  "predicted_churn": 0,
  "risk_level": "Low",
  "threshold": 0.3,
  "recommended_action": "Continue standard customer engagement"
}
```

Prediction values may change after model retraining.

Stop FastAPI using Ctrl+C before starting Docker on the same port.

### Step 15: Build the Docker Image

Make sure Docker Desktop is installed and running.

**macOS — Terminal**

```bash
docker build -t bank-churn-api .
```

**Windows — PowerShell**

```powershell
docker build -t bank-churn-api .
```

The Docker image uses Python 3.12.

### Step 16: Run the Docker Container

The trained model is mounted into the container from the local `artifacts/` directory.

**macOS — Terminal**

```bash
docker run --rm -p 8000:8000 -v "$(pwd)/artifacts:/app/artifacts:ro" bank-churn-api
```

**Windows — PowerShell**

```powershell
docker run --rm -p 8000:8000 --mount "type=bind,source=$((Get-Location).Path)\artifacts,target=/app/artifacts,readonly" bank-churn-api
```

On Windows, Docker Desktop should be configured to use Linux containers.

The trained model directory is mounted read-only and is not embedded in the Docker image.

### Step 17: Verify Docker Deployment

Open the following URLs on either operating system.

**Swagger UI:**

http://127.0.0.1:8000/docs

**Health Check:**

http://127.0.0.1:8000/health

Expected health response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "Random Forest"
}
```

Test the `POST /predict` endpoint using the example JSON from Step 14.

The Docker deployment was successfully tested on macOS.

Press Ctrl+C to stop the container.

### Step 18: Optional Exploratory Data Analysis

The exploratory analysis script can also be executed separately.

**macOS — Terminal**

```bash
python src/bank_churn_analysis.py
```

**Windows — PowerShell**

```powershell
python src/bank_churn_analysis.py
```

---

## 6. Bonus Implementations (15 Points)

This project includes three additional technical implementations developed to demonstrate practical machine learning deployment, containerization, and monitoring skills.

The following components target the three optional bonus categories.

### Bonus 1: FastAPI Model Deployment (+5 Points)

A REST API was developed using FastAPI to serve customer churn predictions.

**Implementation:**

- Developed the `/`, `/health`, and `/predict` endpoints.
- Integrated the trained Random Forest pipeline.
- Implemented customer input validation.
- Returned churn probability and predicted churn class.
- Included risk level and recommended retention action.
- Added interactive API documentation through Swagger UI.
- Implemented automated API testing.

**Verification:**

- API startup verified locally.
- Health endpoint successfully returned model status.
- Prediction endpoint returned a valid churn prediction.
- All 6 automated FastAPI tests passed.

**Bonus Target: +5 Points**

### Bonus 2: Docker Containerization (+5 Points)

The FastAPI application was containerized using Docker.

**Implementation:**

- Created a Dockerfile using Python 3.12.
- Built the `bank-churn-api` Docker image.
- Packaged the API and Python dependencies.
- Exposed port 8000 for API access.
- Mounted the trained Random Forest model using a read-only volume.
- Provided Docker execution instructions for macOS and Windows.

**Verification:**

- Docker image successfully built on macOS.
- Docker container successfully started.
- API health endpoint verified inside the container.
- Customer churn prediction endpoint verified inside the container.

**Bonus Target: +5 Points**

### Bonus 3: MLOps Data Drift Monitoring (+5 Points)

Batch data drift monitoring was implemented using Population Stability Index (PSI).

**Implementation:**

- Created a reference baseline from the training dataset.
- Monitored 10 predictive features.
- Calculated PSI for feature distribution changes.
- Configured a PSI drift threshold of 0.20.
- Generated JSON monitoring reports.
- Implemented simulated drift detection.
- Developed automated monitoring tests.

**Verification:**

- Validation data produced no drift alerts.
- Simulated Age feature drift was successfully detected.
- Monitoring baseline and report files were generated.
- All 4 automated MLOps tests passed.

**Bonus Target: +5 Points**

### Bonus Verification Summary

| Bonus Component | Target Points | Verification |
|---|---:|---|
| FastAPI | +5 | 6 API tests passed |
| Docker | +5 | Image built; health and prediction endpoints verified |
| MLOps | +5 | 4 tests passed; simulated drift detected |
| **Total** | **+15** | **All three components implemented** |

**Total Bonus Points Targeted: 15/15**

These implementations were successfully tested locally on macOS. Final bonus credit depends on the instructor's evaluation and grading rubric.

---

## 7. Data Preprocessing and Reproducibility

The project uses `GroupShuffleSplit` to reduce customer-level data leakage.

### Preprocessing Pipeline

The preprocessing workflow includes:

- Median imputation for numerical features
- StandardScaler for numerical features
- Most-frequent imputation for categorical features
- OneHotEncoder for categorical features

The preprocessing transformations are fitted using training data.

### Dataset Splitting

| Dataset | Records |
|---|---:|
| Training | 99,011 |
| Validation | 32,660 |
| Testing | 33,363 |
| Total | 165,034 |

**Random Seed:** 42

The trained machine learning pipelines are saved locally using joblib.

---

## 8. Machine Learning Model Performance

Two classification models were evaluated:

1. Logistic Regression
2. Random Forest

### Validation Results

The following results use a classification threshold of 0.50.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.7581 | 0.8658 |
| Precision | 0.4525 | 0.7630 |
| Recall | 0.7391 | 0.5206 |
| F1-score | 0.5613 | 0.6189 |
| ROC-AUC | 0.8209 | 0.8870 |

### Model Comparison

Random Forest achieved higher:

- Accuracy
- Precision
- F1-score
- ROC-AUC

Logistic Regression achieved higher recall at the default classification threshold.

Random Forest was selected for the FastAPI prediction service.

---

## 9. Business Retention Analysis

The project evaluates hypothetical customer retention campaigns using predicted churn probabilities.

### Base-Case Assumptions

| Parameter | Assumption |
|---|---:|
| Contact Cost | $10 |
| Retention Success Rate | 20% |
| Customer Value | $200 |
| Selected Threshold | 0.30 |

Under the evaluated assumptions, Random Forest at a threshold of 0.30 produced the highest estimated net value among the tested configurations.

**Illustrative Estimated Net Value: $115,950**

This result is based on validation data and hypothetical assumptions.

It is not observed revenue, proven campaign effectiveness, or an actual JPMorganChase financial result.

---

## 10. FastAPI REST API

The project provides a REST API for customer churn prediction.

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | / | API information |
| GET | /health | API and model health |
| POST | /predict | Customer churn prediction |

### Example Prediction Request

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

### Example Prediction Response

```json
{
  "churn_probability": 0.1279,
  "predicted_churn": 0,
  "risk_level": "Low",
  "threshold": 0.3,
  "recommended_action": "Continue standard customer engagement"
}
```

### Prediction Interpretation

For this example:

- Churn Probability: 12.79%
- Predicted Churn: 0
- Risk Level: Low
- Classification Threshold: 30%
- Recommended Action: Continue standard customer engagement

Prediction values may vary after retraining.

---

## 11. Docker Deployment

Docker packages the FastAPI service and its dependencies into a reproducible environment.

The Docker image uses Python 3.12.

The trained Random Forest model is loaded through a local read-only volume mount rather than being embedded in the image.

### Verified Deployment

The Docker implementation was successfully built and tested on macOS.

Verified operations:

- Docker image build
- Docker container startup
- FastAPI health endpoint
- Random Forest model loading
- Customer churn prediction endpoint

---

## 12. MLOps Data Drift Monitoring

The project implements batch data drift monitoring using Population Stability Index (PSI).

### Monitoring Process

1. Build a baseline from the training split.
2. Compare validation data with the baseline.
3. Calculate PSI for each predictive feature.
4. Identify features exceeding the drift threshold.
5. Generate a JSON monitoring report.

### Drift Threshold

```text
PSI >= 0.20
```

This is a configurable monitoring heuristic, not a formal statistical significance test.

### Normal Monitoring Results

The validation dataset produced no drift alerts.

```text
Drift detected: False
Drifted features: []
```

### Simulated Data Drift

A simulated scenario increased the Age feature by 20 years.

The monitoring system detected the change:

```text
Drift detected: True
Drifted features: ['Age']
Age PSI: 8.002136
```

### Monitoring Outputs

```text
artifacts/monitoring_baseline.json
reports/monitoring_report.json
```

The monitoring implementation is batch-based.

It does not include continuous production monitoring, automatic retraining, or real-time external alert delivery.

---

## 13. Automated Testing

The project uses pytest to validate the FastAPI and MLOps implementations.

### FastAPI Tests

- Root endpoint
- Health endpoint
- Successful prediction
- Missing input fields
- Invalid age
- Invalid credit card value

### MLOps Tests

- No drift on identical data
- Numerical feature drift detection
- Categorical feature drift detection
- Missing feature error handling

### Test Results

| Component | Passed |
|---|---:|
| FastAPI | 6/6 |
| MLOps | 4/4 |
| **Total** | **10/10** |

Verified local execution on macOS:

```text
10 passed, 1 warning in 2.11s
```

The warning concerns a Starlette/httpx dependency deprecation and did not cause a test failure.

---

## 14. Troubleshooting

### Dataset Not Found

Ensure that `train.csv` is located at:

```text
data/train.csv
```

### Model Not Found

Train the Random Forest model before starting FastAPI or Docker.

**macOS — Terminal**

```bash
python src/train_random_forest.py
```

**Windows — PowerShell**

```powershell
python src/train_random_forest.py
```

### Missing Python Dependencies

**macOS — Terminal**

```bash
python -m pip install -r requirements.txt
```

**Windows — PowerShell**

```powershell
python -m pip install -r requirements.txt
```

### Python Import Errors

Ensure that commands are executed from the repository root and that the virtual environment is activated.

### Port 8000 Already in Use

Stop the existing FastAPI or Docker server using Ctrl+C.

Alternatively, use a different host port when running Docker.

### Docker Cannot Load the Model

Confirm that:

- Docker Desktop is running.
- The Random Forest model has been trained.
- The `artifacts` directory exists.
- The model directory is mounted correctly into the container.

### Windows PowerShell Activation Error

If PowerShell blocks virtual environment activation, use the Python executable directly.

For example:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

---

## 15. Project Limitations

- The dataset is synthetic.
- Repeated CustomerIds and potentially conflicting churn labels require caution.
- Validation performance does not guarantee real-world performance.
- Retention campaign assumptions are hypothetical.
- Model predictions do not establish causal effects.
- PSI monitoring does not directly measure prediction accuracy or fairness.
- Monitoring is batch-based rather than continuous.
- Additional security, fairness evaluation, governance, and production validation would be required before real-world deployment.
- Final test-set evaluation should be performed after model selection is finalized.

---

## 16. Future Improvements

Potential future improvements include:

- Hyperparameter tuning
- Additional machine learning models
- SHAP-based model explainability
- Customer segmentation
- Probability calibration
- Fairness analysis
- Real-time monitoring
- Automated retraining
- CI/CD integration
- Cloud deployment
- Business intelligence dashboard integration


