
# Arpan-ML-CICD

This repository implements a complete, modular machine learning pipeline with end-to-end lifecycle automation. It leverages **MLflow**, **GitHub Actions**, **Docker**, and supports deployment across **AWS** and **Azure** environments. The goal is to enable **reproducible**, **scalable**, and **automated** training, evaluation, and deployment of ML models.

---

## 🚀 Features

- **Modular ML Pipeline**: Covers all key stages:
  - Data ingestion & preprocessing
  - Feature engineering
  - Model training & evaluation
  - Model deployment

- **Model Optimization**:
  - Hyperparameter tuning using `GridSearchCV`
  - Supports ensemble regressors: `Random Forest`, `XGBoost`, `CatBoost`, `AdaBoost`, etc.

- **MLOps Automation**:
  - MLflow for experiment tracking, model versioning, and artifact logging
  - GitHub Actions for CI/CD
  - Dockerized workflows for portability and reproducibility

- **Cross-Cloud Deployment**:
  - **AWS**:
    - SageMaker
    - EC2
    - Elastic Beanstalk
    - Docker + ECR
  - **Azure**:
    - Azure Container Apps
    - GitHub Actions with self-hosted runners

---

## 📁 Project Structure

```bash
├── data/                      # Sample datasets
├── src/                       # Core ML pipeline code
│   ├── ingest.py              # Data ingestion
│   ├── preprocess.py          # Data cleaning & feature engineering
│   ├── train.py               # Model training & evaluation
│   └── deploy.py              # Deployment logic
├── docker/                    # Docker configurations
├── .github/workflows/        # GitHub Actions CI/CD pipelines
├── mlruns/                   # MLflow experiment tracking artifacts
├── requirements.txt
├── Dockerfile
└── README.md
