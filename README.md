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

```
⚙️ Setup & Usage

## Prerequisites
Python 3.8+

Docker

MLflow

AWS CLI / Azure CLI

GitHub CLI

##Installation
# Clone the repository

```bash

git clone https://github.com/Arpan-Roy-1993/Arpan-ML-CICD.git
cd Arpan-ML-CICD
```
# Install dependencies
```bash

pip install -r requirements.txt
```
Run Training Locally
```bash
python src/train.py
```
Track Experiments
```bash

mlflow ui
```
#🐳 Containerization
Build and run the project inside a Docker container:

```bash
docker build -t ml-cicd-pipeline .
docker run -p 5000:5000 ml-cicd-pipeline
```
# Deployment
AWS
ECR for container registry

SageMaker/Elastic Beanstalk for model hosting

EC2 for flexible deployment

Azure
Deployed to Azure Container Apps via GitHub Actions

Includes setup for self-hosted runners

# CI/CD with GitHub Actions
Automatically triggers:

Model training and evaluation

MLflow artifact logging

Container build and push

Cross-cloud deployment

See .github/workflows/ for CI/CD pipeline definitions.

# TODOs
Add monitoring & alerting (EvidentlyAI, Prometheus, etc.)

Integrate Feature Store

Add real-time inference API
