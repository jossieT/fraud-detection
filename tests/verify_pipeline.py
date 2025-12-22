
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_loader import load_data
from preprocessing import clean_fraud_data, clean_ip_data
from feature_engineering import feature_engineer_fraud_data, encode_categorical

def create_dummy_data():
    """Creates dummy data for testing."""
    os.makedirs('../data/raw', exist_ok=True)
    
    # Fraud Data
    fraud_data = {
        'user_id': range(1, 11),
        'signup_time': pd.date_range(start='2024-01-01', periods=10, freq='D'),
        'purchase_time': pd.date_range(start='2024-01-02', periods=10, freq='D'),
        'purchase_value': np.random.randint(10, 100, 10),
        'device_id': [f'dev_{i}' for i in range(10)],
        'source': ['SEO', 'Ads'] * 5,
        'browser': ['Chrome', 'Firefox'] * 5,
        'sex': ['M', 'F'] * 5,
        'age': np.random.randint(18, 60, 10),
        'ip_address': [1.2e9 + i for i in range(10)], # Float/Int style
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
    
    # Credit Card Data
    cc_data = {
        'Time': range(10),
        'V1': np.random.randn(10),
        'Class': [0, 1] * 5
    }
    pd.DataFrame(cc_data).to_csv('../data/raw/creditcard.csv', index=False)

def test_pipeline():
    print("Creating dummy data...")
    create_dummy_data()
    
    print("Testing pipeline...")
    fraud_df = load_data('../data/raw/Fraud_Data.csv')
    ip_df = load_data('../data/raw/IpAddress_to_Country.csv')
    
    assert fraud_df is not None
    assert ip_df is not None
    
    fraud_df = clean_fraud_data(fraud_df)
    ip_df = clean_ip_data(ip_df)
    
    fraud_df = feature_engineer_fraud_data(fraud_df, ip_df)
    
    cols_to_check = ['time_since_signup', 'hour_of_day', 'day_of_week', 'country']
    for col in cols_to_check:
        assert col in fraud_df.columns, f"Missing {col}"
        
    print("Pipeline test successful!")
    print(fraud_df[['user_id', 'country', 'time_since_signup']].head())

if __name__ == "__main__":
    test_pipeline()
