
import pandas as pd

def clean_fraud_data(df):
    """
    Cleans the fraud data.
    - Handles missing values (if any).
    - Converts timestamps to datetime objects.
    
    Args:
        df (pd.DataFrame): Raw fraud data.
        
    Returns:
        pd.DataFrame: Cleaned fraud data.
    """
    if df is None:
        return None
        
    df = df.copy()
    
    # Handle missing values (simple strategy: drop rows with critical missing info or fill)
    # Checking for missing values first is usually good practice in EDA, here we implement a basic cleaning
    # Assuming 'user_id', 'signup_time', 'purchase_time' are critical.
    # We will just drop duplicates if any
    df.drop_duplicates(inplace=True)
    
    # Convert timestamps
    if 'signup_time' in df.columns:
        df['signup_time'] = pd.to_datetime(df['signup_time'])
    if 'purchase_time' in df.columns:
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        
    return df

def clean_creditcard_data(df):
    """
    Cleans the credit card data.
    
    Args:
        df (pd.DataFrame): Raw credit card data.
        
    Returns:
        pd.DataFrame: Cleaned credit card data.
    """
    if df is None:
        return None
        
    df = df.copy()
    df.drop_duplicates(inplace=True)
    # Credit card data is usually already numerical and clean (PCA components), 
    # but we can add more checks here if needed.
    
    return df

def clean_ip_data(df):
    """
    Cleans the IP address to country mapping data.
    
    Args:
        df (pd.DataFrame): Raw IP address data.
        
    Returns:
        pd.DataFrame: Cleaned IP address data.
    """
    if df is None:
        return None
    df = df.copy()
    
    # Ensure IP ranges are integers
    if 'lower_bound_ip_address' in df.columns:
        df['lower_bound_ip_address'] = pd.to_numeric(df['lower_bound_ip_address'], errors='coerce')
    if 'upper_bound_ip_address' in df.columns:
        df['upper_bound_ip_address'] = pd.to_numeric(df['upper_bound_ip_address'], errors='coerce')
        
    return df
