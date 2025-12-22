
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_loader import load_data
from preprocessing import clean_fraud_data, clean_ip_data
from feature_engineering import feature_engineer_fraud_data, encode_categorical
from modeling import train_baseline_model, train_ensemble_model, stratified_cross_validation, save_model
from evaluation import evaluate_model

def create_dummy_data():
    """Creates dummy data for testing."""
    os.makedirs('../data/raw', exist_ok=True)
    
    # Fraud Data
    fraud_data = {
        'user_id': range(1, 101), # More data for training
        'signup_time': pd.to_datetime(['2024-01-01']*100),
        'purchase_time': pd.to_datetime(['2024-01-02']*100),
        'purchase_value': np.random.randint(10, 100, 100),
        'device_id': [f'dev_{i}' for i in range(100)],
        'source': ['SEO', 'Ads'] * 50,
        'browser': ['Chrome', 'Firefox'] * 50,
        'sex': ['M', 'F'] * 50,
        'age': np.random.randint(18, 60, 100),
        'ip_address': [1.2e9 + i for i in range(100)],
        'class': [0]*90 + [1]*10 # Imbalanced
    }
    pd.DataFrame(fraud_data).to_csv('../data/raw/Fraud_Data.csv', index=False)
    
    # IP Data
    ip_data = {
        'lower_bound_ip_address': [1.2e9],
        'upper_bound_ip_address': [1.3e9],
        'country': ['Utopia']
    }
    pd.DataFrame(ip_data).to_csv('../data/raw/IpAddress_to_Country.csv', index=False)

def run_test():
    print("Creating dummy data...")
    create_dummy_data()
    
    print("Processing data...")
    fraud_df = load_data('../data/raw/Fraud_Data.csv')
    ip_df = load_data('../data/raw/IpAddress_to_Country.csv')
    
    fraud_df = clean_fraud_data(fraud_df)
    ip_df = clean_ip_data(ip_df)
    fraud_df = feature_engineer_fraud_data(fraud_df, ip_df)
    cat_cols = ['source', 'browser', 'sex', 'country']
    fraud_df = encode_categorical(fraud_df, cat_cols)
    
    # Save processed for notebook simulation
    os.makedirs('../data/processed', exist_ok=True)
    fraud_df.to_csv('../data/processed/Fraud_Data_Processed.csv', index=False)
    
    print("Training models...")
    X = fraud_df.drop(columns=['class', 'user_id', 'signup_time', 'purchase_time', 'device_id'])
    y = fraud_df['class']
    
    model_lr = train_baseline_model(X, y)
    model_rf = train_ensemble_model(X, y)
    
    print("Evaluating models...")
    evaluate_model(model_lr, X, y, "Baseline LR")
    evaluate_model(model_rf, X, y, "Ensemble RF")
    
    print("Running CV...")
    cv_res = stratified_cross_validation(model_rf, X, y, k=2) # Small k for dummy
    print(f"CV Results: {cv_res}")
    
    print("Saving model...")
    os.makedirs('../models', exist_ok=True)
    save_model(model_rf, '../models/best_model.joblib')
    
    assert os.path.exists('../models/best_model.joblib')
    print("Verification successful!")

if __name__ == "__main__":
    run_test()
