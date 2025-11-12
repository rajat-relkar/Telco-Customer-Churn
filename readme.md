# 📊 Telco Customer Churn Prediction

Predict customer churn for a telecom company with a fully operational MLOps pipeline — from data prep and model training to deployment on AWS ECS with REST API and Gradio UI.

---

## 🧭 Purpose

This project builds and ships a **complete machine learning solution** for predicting telecom customer churn.  
It covers everything — data validation, feature engineering, model training, experiment tracking, containerization, CI/CD, and cloud deployment.

---

## 💡 Problem Solved & Benefits

| Challenge | Solution | Benefit |
|------------|-----------|----------|
| Customer retention uncertainty | Predict churn probability using ML | Enables proactive retention campaigns |
| Difficult access to models | REST API + Gradio UI | Business users can test without coding |
| Inconsistent deployments | CI/CD with Docker + GitHub Actions | Reliable, repeatable deployments |
| Lack of experiment tracking | MLflow integration | Full traceability and reproducibility |

---

## 🏗️ What I Built

- **Data & Modeling:** Feature engineering + XGBoost classifier  
- **Model Tracking:** MLflow for runs, metrics, and artifacts  
- **Inference Service:** FastAPI app with `/predict` (POST) and `/` (health check)  
- **Web UI:** Gradio interface at `/ui` for easy manual testing  
- **Containerization:** Docker image using `uvicorn` entrypoint  
- **CI/CD:** GitHub Actions → Docker Hub → AWS ECS deployment  
- **Observability:** CloudWatch logs for containers and ECS events  

---

## ⚙️ Architecture Overview

### 🧩 ML Pipeline Flow

#### **1. Training Pipeline** (`scripts/run_pipeline.py`)
```
Data → Validation (Great Expectations) → Preprocessing → Feature Engineering → XGBoost Training → MLflow Logging
```
Artifacts:
- `model/`, `feature_columns.txt`, `preprocessing.pkl`  
- Logged to MLflow under experiment **"Telco Churn"**

#### **2. Serving Pipeline** (`src/app/main.py`, `src/serving/inference.py`)
```
FastAPI REST API + Gradio UI → Load Model from MLflow → Transform Features → Predict
```
Feature transformations mirror training logic to ensure consistency.

---

## 🧪 MLflow Integration

| Type | Example |
|------|----------|
| **Experiment Name** | `Telco Churn` |
| **Tracking URI** | `file:./mlruns` |
| **Logged Metrics** | precision, recall, f1, roc_auc, data_quality_pass |
| **Parameters** | model type, threshold=0.35, test_size=0.2 |

Access MLflow UI:
```bash
mlflow ui --backend-store-uri file:./mlruns
```

---

## 🧠 Feature Engineering Consistency

| Type | Logic |
|------|-------|
| Binary | `Yes/No`, `Male/Female` → 0/1 mapping |
| Multi-category | One-hot encoding (`drop_first=True`) |
| Boolean | Converted to integers |
| Alignment | `FEATURE_COLS` enforced during inference |

---

## 🚀 Model Serving

| Aspect | Details |
|--------|----------|
| **Model Path** | `/app/model` (MLflow pyfunc format) |
| **Feature Order** | From `feature_columns.txt` |
| **Prediction Output** | `"Likely to churn"` / `"Not likely to churn"` |
| **API Endpoints** | `/` → health check <br> `/predict` → JSON payload <br> `/ui` → Gradio web app |

Example request:
```bash
curl -X POST "http://<alb-dns>/predict" -H "Content-Type: application/json" -d '{"gender": "Male", "SeniorCitizen": 0, ...}'
```

---

## 🔍 Data Validation

- **Tool:** Great Expectations  
- **Script:** `src/utils/validate_data.py`  
- **Checks:**  
  - `CustomerID` presence  
  - Valid `gender` values  
  - Numeric ranges for `tenure` / `charges`  
- **Integration:** Logged to MLflow as `data_quality_pass`

---

## 🐳 Docker Containerization

| Setting | Value |
|----------|--------|
| **Base Image** | `python:3.11-slim` |
| **Workdir** | `/app` |
| **Entrypoint** | `uvicorn src.app.main:app --host 0.0.0.0 --port 8000` |
| **Env Var** | `PYTHONPATH=/app/src` |
| **Port** | `8000` |

---

## ⚡ CI/CD Pipeline

- Trigger: Push to `main`
- Workflow:  
  1. Build Docker image  
  2. Push to Docker Hub (`anasriad8/telco-fastapi:latest`)  
  3. Optional ECS update trigger  
- Secrets required:  
  - `DOCKERHUB_USERNAME`  
  - `DOCKERHUB_TOKEN`  

---

## ☁️ AWS Deployment Overview

| Component | Purpose |
|------------|----------|
| **ECS Fargate** | Run container (serverless) |
| **ALB (HTTP:80)** | Routes to Target Group on port `8000` |
| **Security Groups** | ALB → inbound 80; ECS → inbound 8000 |
| **CloudWatch** | Container logs + ECS service events |

**Deployment Flow:**
```
Push to main → GitHub Actions → Docker Hub → ECS Update → ALB Health Check → Live Traffic
```

---

## 📂 Project Structure

```
Telco-Customer-Churn/
│
├── data/
│   ├── raw/               # Original datasets
│   └── processed/         # Cleaned data
│
├── mlruns/                # MLflow tracking
├── artifacts/             # Feature columns + preprocessing
│
├── src/
│   ├── app/               # FastAPI + Gradio
│   ├── features/          # Feature engineering
│   ├── serving/           # Inference logic
│   └── utils/             # Data validation, helpers
│
├── scripts/
│   ├── run_pipeline.py    # Training pipeline
│   └── test_*.py          # Manual tests
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 📈 Results (Extracted from `EDA.ipynb`)

### ROC AUC
- (source) - If business wants a ranking of churn risk → use ROC-AUC or PR-AUC to evaluate the model.- (source) from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score
    auc = roc_auc_score(y_test, proba)
    mlflow.log_metric("roc_auc", auc)### PRECISION
- (source) - If retention campaigns are expensive → balance precision and recall using F1 score or a precision-recall trade-off.- (output)               precision    recall  f1-score   support- (output)               precision    recall  f1-score   support- (source) - The cost is a small drop in precision — meaning more loyal customers will be flagged as churn risks — but if retention offers are low-cost, this is fine.- (output)               precision    recall  f1-score   support### RECALL
- (source) In churn prediction, recall (and metrics derived from it, like F1) is usually the most important — here’s why:- (source) - If retention campaigns are cheap → prioritize recall (catch every possible churner).
- If retention campaigns are expensive → balance precision and recall using F1 score or a precision-recall trade-off.
- High Recall: A high recall score indicates the model is good at finding most of the positive cases and has a low number of false negatives.- (source) THRESHOLD = 0.25  # lower than 0.5 to boost recall (see next to choose the right value)- (output)               precision    recall  f1-score   support- (output)               precision    recall  f1-score   support### F1
- (source) In churn prediction, recall (and metrics derived from it, like F1) is usually the most important — here’s why:- (source) - If retention campaigns are expensive → balance precision and recall using F1 score or a precision-recall trade-off.- (output)               precision    recall  f1-score   support- (source) from sklearn.metrics import precision_score, recall_score, f1_score
    f1 = f1_score(y_test, preds, pos_label=1)
    print(f"{thresh:<8}{prec:<8.3f}{rec:<8.3f}{f1:<8.3f}")- (output)               precision    recall  f1-score   support### ACCURACY
- (output)     accuracy                          0.739      1409- (output)     accuracy                          0.715      1409- (output)     accuracy                          0.704      1409- (output)     accuracy                          0.633      1409- (output)     accuracy                          0.633      1409### CLASSIFICATION REPORT
- (source) from sklearn.metrics import classification_report- (source) from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, digits=3))- (source) from sklearn.metrics import classification_report
# Classification report
print(classification_report(y_test, y_pred, digits=3))- (source) from sklearn.metrics import classification_report
# Classification report
print(classification_report(y_test, y_pred, digits=3))- (source) from sklearn.metrics import classification_report
# Classification report
print(classification_report(y_test, y_pred, digits=3))

---

## 🧰 Tech Stack

**ML & Data:** XGBoost, pandas, Great Expectations, MLflow  
**Backend:** FastAPI, Gradio, Pydantic  
**MLOps:** Docker, GitHub Actions, AWS ECS Fargate, CloudWatch  
**Language:** Python 3.11  

---

## 🧑‍💻 Development Notes

- No formal test suite; use scripts in `scripts/test_*.py`
- Model training artifacts must match inference columns
- MLflow tracking is **file-based** (no server required)
- Use `EDA.ipynb` for exploratory analysis and insights

---

## 📈 Future Improvements

- Add unit tests and data drift checks (Evidently AI)
- Automate ECS service update via CI/CD
- Enable remote MLflow tracking server
- Add Streamlit dashboard for analytics

---

## 🏁 Getting Started

```bash
# Clone repo
git clone https://github.com/<your-username>/Telco-Customer-Churn.git
cd Telco-Customer-Churn

# Install dependencies
pip install -r requirements.txt

# Run training pipeline
python scripts/run_pipeline.py

# Start API
uvicorn src.app.main:app --reload

```

---

**Author:** Rajat Relkar  
**License:** MIT  
**Keywords:** `mlops`, `fastapi`, `mlflow`, `aws`, `docker`, `xgboost`, `great-expectations`
