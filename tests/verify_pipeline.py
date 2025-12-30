import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_loader import load_data
from preprocessing import clean_fraud_data, clean_ip_data, DataTransformer
from feature_engineering import feature_engineer_fraud_data

def create_dummy_data():
    """Creates dummy data for testing."""
    os.makedirs('../data/raw', exist_ok=True)
    
    # Fraud Data
    fraud_data = {
        'user_id': [1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'signup_time': pd.date_range(start='2024-01-01', periods=10, freq='D'),
        'purchase_time': pd.date_range(start='2024-01-02', periods=10, freq='D'),
        'purchase_value': np.random.randint(10, 100, 10),
        'device_id': ['dev_1', 'dev_1', 'dev_2', 'dev_3', 'dev_4', 'dev_5', 'dev_6', 'dev_7', 'dev_8', 'dev_9'],
        'source': ['SEO', 'Ads'] * 5,
        'browser': ['Chrome', 'Firefox'] * 5,
        'sex': ['M', 'F'] * 5,
        'age': np.random.randint(18, 60, 10),
        'ip_address': [1.2e9 + i for i in range(10)],
        'class': [0, 1] * 5
    }
    pd.DataFrame(fraud_data).to_csv('../data/raw/Fraud_Data.csv', index=False)
    
    # IP Data
    ip_data = {
        'lower_bound_ip_address': [1.2e9, 1.3e9],
        'upper_bound_ip_address': [1.20000001e9, 1.30000001e9],
        'country': ['Utopia', 'Atlantis']
    }
    pd.DataFrame(ip_data).to_csv('../data/raw/IpAddress_to_Country.csv', index=False)

def test_pipeline():
    print("Testing pipeline with Task 1 enhancements...")
    create_dummy_data()
    
    fraud_df = load_data('../data/raw/Fraud_Data.csv')
    ip_df = load_data('../data/raw/IpAddress_to_Country.csv')
    
    # Clean
    fraud_df = clean_fraud_data(fraud_df)
    ip_df = clean_ip_data(ip_df)
    
    # Feature Engineer
    fraud_df = feature_engineer_fraud_data(fraud_df, ip_df)
    
    print(f"Features added: {fraud_df.columns.tolist()}")
    assert 'country' in fraud_df.columns
    assert 'user_freq' in fraud_df.columns
    assert 'device_freq' in fraud_df.columns
    
    # Transform
    X = fraud_df.drop(columns=['class', 'user_id', 'signup_time', 'purchase_time', 'device_id'])
    y = fraud_df['class']
    
    transformer = DataTransformer(use_smote=False) # Disable SMOTE for small dummy set
    X_transformed = transformer.fit_transform(X)
    
    print(f"Transformed X shape: {X_transformed.shape}")
    print("Pipeline test successful!")

if __name__ == "__main__":
    test_pipeline()
