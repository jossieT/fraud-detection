# Fraud Detection for E-commerce and Bank Transactions

This project aims to build accurate and explainable fraud detection models for Adey Innovations Inc. It focuses on handling extreme class imbalance while maintaining a balance between security and user experience.

## Project Structure

- `data/`
  - `raw/`: Raw datasets (`Fraud_Data.csv`, `creditcard.csv`, `IpAddress_to_Country.csv`).
  - `processed/`: Cleaned and engineered datasets.
- `notebooks/`
  - `eda-fraud-data.ipynb`: EDA for e-commerce fraud data.
  - `eda-creditcard.ipynb`: EDA for bank transaction data.
  - `feature-engineering.ipynb`: Preprocessing and feature engineering pipeline.
  - `modeling.ipynb`: Model training, evaluation, and comparison.
- `src/`
  - `data_loader.py`: Utility for loading CSV data.
  - `preprocessing.py`: Data cleaning and timestamp normalization.
  - `feature_engineering.py`: Time-based features, IP-to-Country mapping, and encoding.
  - `modeling.py`: Baseline (Logistic Regression) and Ensemble (Random Forest) models.
  - `evaluation.py`: Performance metrics (AUC-PR, F1) and visualizations.
- `models/`: Location for saved production models (`.joblib`).
- `tests/`: Automated verification scripts.

## Key Features Implemented

### 1. Data Processing & EDA

- **Class Imbalance Visualization**: Detailed analysis of the extreme skew in fraud cases.
- **Feature Engineering**:
  - `time_since_signup`: Calculated as the difference between purchase and signup time.
  - **IP-to-Country Mapping**: Vectorized mapping using IP ranges for localized fraud patterns.
  - **Categorical Encoding**: Label encoding for browser, source, and gender.

### 2. Modeling & Evaluation

- **Handling Imbalance**: Models use `class_weight='balanced'` to prevent bias toward the majority class.
- **Models**:
  - **Baseline**: Logistic Regression for interpretability.
  - **Ensemble**: Random Forest for capturing non-linear fraud patterns (Production Model).
- **Metrics**: Prioritized **AUC-PR** (Area Under Precision-Recall Curve) and **F1-Score** over accuracy due to class imbalance.
- **Cross-Validation**: 5-fold Stratified K-Fold to ensure model stability.

## Getting Started

### Installation

1. Clone the repository.
2. Create a virtual environment: `python -m venv .venv`
3. Install dependencies: `pip install -r requirements.txt`

### Running the Pipeline

1. Run `notebooks/feature-engineering.ipynb` to process the raw data.
2. Run `notebooks/modeling.ipynb` to train models and evaluate performance.
3. Verification: Run `python tests/verify_modeling.py` to ensure the end-to-end flow works with dummy data.

## Business Justification

The Random Forest model was selected as the final candidate because it provides a superior balance between Precision and Recall. In fraud detection, missing a fraud case (False Negative) is costly, but so is blocking a legitimate user (False Positive). Our model focuses on maximizing the F1-Score and AUC-PR to ensure high reliability in a production environment.
