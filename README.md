# CICD-for-Machine-Learning: Credit Card Fraud Detection
[![CI](https://github.com/MohammedAmine0012/credit-card-fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/MohammedAmine0012/credit-card-fraud-detection/actions/workflows/ci.yml)
[![Continuous Deployment](https://github.com/MohammedAmine0012/credit-card-fraud-detection/actions/workflows/cd.yml/badge.svg)](https://github.com/MohammedAmine0012/credit-card-fraud-detection/actions/workflows/cd.yml)

[![Open the Space](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md-dark.svg)](https://huggingface.co/spaces/Moamineelhilali/Credit-Card-Fraud-Detection)

Automated MLOps pipeline for credit card fraud detection using scikit-learn, Streamlit, and GitHub Actions.

## Project Description
This project demonstrates an end-to-end machine learning workflow for detecting credit card fraud. It includes:
- Data preprocessing and feature engineering
- Model training with class imbalance handling (SMOTE)
- Automated checks and evaluation
- CI/CD pipeline with GitHub Actions
- Web application deployment with Streamlit to Hugging Face Spaces

Deployed app: https://huggingface.co/spaces/Moamineelhilali/Credit-Card-Fraud-Detection  
Model repository: https://huggingface.co/Moamineelhilali/credit-card-fraud-model

The process is automated using GitHub Actions. On push to main, CI verifies that the published model can be downloaded from the Hugging Face Hub. On CI success, CD deploys the Streamlit app to the Space; the app downloads the model from the model repo at runtime (no binaries are pushed to the Space).

## Pipeline

![CICD](./asset/CICD-pipeline.png)

## Dataset
- **Source**: [Credit Card Fraud Detection on Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Features**: 30 numerical features (anonymized as V1-V28) + Time + Amount
- **Target**: Binary classification (0: Genuine, 1: Fraud)
- **Class Distribution**: Highly imbalanced (0.17% fraud cases)

## Results
Example performance from a typical RandomForestClassifier pipeline on this task (for illustration):

| Model                  | Accuracy | F1 Score | Precision | Recall | ROC AUC |
|------------------------|----------|----------|-----------|--------|---------|
| RandomForestClassifier | 99.9%+   | 0.84+    | 0.93+     | 0.76+  | 0.94+   |

Notes:
- Metrics are sensitive to sampling, preprocessing, and class imbalance handling (SMOTE).
- In CI, we avoid pushing large artifacts and do not train on every commit. The Space app loads the model from the HF model repo.

## Local Setup
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run App/app.py
   ```

## CI/CD Workflow
- **CI (Continuous Integration)**
  - Lightweight check: installs minimal deps and verifies the model can be downloaded and loaded from the Hugging Face model repo.
  - Keeps runs fast and avoids re-training on every commit.

- **CD (Continuous Deployment)**
  - Deploys only the Streamlit app files to the Hugging Face Space.
  - Ensures the Space has `app.py` at the root and a `requirements.txt`.
  - The app downloads the trained model from the model repository at runtime.
