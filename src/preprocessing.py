import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_fraud_data(df):
    """
    Cleans the fraud data.
    - Handles missing values (removes rows with null critical fields).
    - Removes duplicates.
    - Converts timestamps to datetime objects.
    - Ensures IP addresses are numeric.
    """
    try:
        if df is None:
            return None
            
        df = df.copy()
        initial_count = len(df)
        
        # Remove duplicates
        df.drop_duplicates(inplace=True)
        
        # Handle missing values: document strategy is dropping since they are few usually
        critical_cols = ['user_id', 'signup_time', 'purchase_time', 'ip_address']
        df.dropna(subset=[col for col in critical_cols if col in df.columns], inplace=True)
        
        # Convert timestamps
        if 'signup_time' in df.columns:
            df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce')
        if 'purchase_time' in df.columns:
            df['purchase_time'] = pd.to_datetime(df['purchase_time'], errors='coerce')
        
        # Ensure IP addresses are numeric (float/int)
        if 'ip_address' in df.columns:
            df['ip_address'] = pd.to_numeric(df['ip_address'], errors='coerce')
            
        final_count = len(df)
        logging.info(f"Cleaned Fraud Data: Removed {initial_count - final_count} rows.")
        return df
    except Exception as e:
        logging.error(f"Error cleaning fraud data: {e}")
        return None

def clean_creditcard_data(df):
    """
    Cleans the credit card data.
    """
    try:
        if df is None:
            return None
            
        df = df.copy()
        initial_count = len(df)
        
        df.drop_duplicates(inplace=True)
        # Note: creditcard.csv is usually clean, but we ensure no nulls in Class
        if 'Class' in df.columns:
            df.dropna(subset=['Class'], inplace=True)
            
        final_count = len(df)
        logging.info(f"Cleaned Credit Card Data: Removed {initial_count - final_count} rows.")
        return df
    except Exception as e:
        logging.error(f"Error cleaning credit card data: {e}")
        return None

def clean_ip_data(df):
    """
    Cleans the IP address to country mapping data.
    """
    try:
        if df is None:
            return None
        df = df.copy()
        
        # Ensure IP ranges are numeric
        df['lower_bound_ip_address'] = pd.to_numeric(df['lower_bound_ip_address'], errors='coerce')
        df['upper_bound_ip_address'] = pd.to_numeric(df['upper_bound_ip_address'], errors='coerce')
        df.dropna(subset=['lower_bound_ip_address', 'upper_bound_ip_address'], inplace=True)
        
        return df
    except Exception as e:
        logging.error(f"Error cleaning IP data: {e}")
        return None

class DataTransformer:
    """
    Handles scaling, encoding, and resampling.
    """
    def __init__(self, use_smote=True):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.use_smote = use_smote
        self.cat_cols = []
        self.num_cols = []

    def fit_transform(self, X, y=None):
        """Fits transformers on training data and returns transformed data."""
        X = X.copy()
        
        self.cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.num_cols = X.select_dtypes(include=['number']).columns.tolist()
        
        # Scale numerical
        if self.num_cols:
            X[self.num_cols] = self.scaler.fit_transform(X[self.num_cols])
            
        # Encode categorical
        if self.cat_cols:
            encoded = self.encoder.fit_transform(X[self.cat_cols])
            encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(self.cat_cols), index=X.index)
            X = pd.concat([X.drop(columns=self.cat_cols), encoded_df], axis=1)
            
        # Resample ONLY if y is provided (training set)
        if self.use_smote and y is not None:
            logging.info(f"Class distribution BEFORE SMOTE: {y.value_counts().to_dict()}")
            smote = SMOTE(random_state=42)
            X, y = smote.fit_resample(X, y)
            logging.info(f"Class distribution AFTER SMOTE: {y.value_counts().to_dict()}")
            return X, y
            
        return X

    def transform(self, X):
        """Applies fitted transformers to test data."""
        X = X.copy()
        
        if self.num_cols:
            X[self.num_cols] = self.scaler.transform(X[self.num_cols])
            
        if self.cat_cols:
            encoded = self.encoder.transform(X[self.cat_cols])
            encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(self.cat_cols), index=X.index)
            X = pd.concat([X.drop(columns=self.cat_cols), encoded_df], axis=1)
            
        return X
