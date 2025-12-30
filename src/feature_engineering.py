import pandas as pd
import numpy as np
import logging

def add_time_since_signup(df):
    """
    Adds a 'time_since_signup' feature (seconds).
    Relevance: Immediate purchases after signup are often high-risk (bot behavior).
    """
    df = df.copy()
    if 'purchase_time' in df.columns and 'signup_time' in df.columns:
        df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
    return df

def add_hour_of_day(df):
    """
    Adds 'hour_of_day' feature.
    Relevance: Fraud patterns often vary by time of day (e.g., late-night spikes).
    """
    df = df.copy()
    if 'purchase_time' in df.columns:
        df['hour_of_day'] = df['purchase_time'].dt.hour
    return df

def add_day_of_week(df):
    """Adds 'day_of_week' feature."""
    df = df.copy()
    if 'purchase_time' in df.columns:
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
    return df

def add_transaction_frequency(df):
    """
    Adds transaction frequency per user and device.
    Relevance: High frequency of transactions from the same user or device in a short time is a strong fraud indicator.
    """
    df = df.copy()
    if 'user_id' in df.columns:
        df['user_freq'] = df.groupby('user_id')['user_id'].transform('count')
    if 'device_id' in df.columns:
        df['device_freq'] = df.groupby('device_id')['device_id'].transform('count')
    return df

def ip_to_int(ip):
    """Convert string IP to integer if not already numeric."""
    if isinstance(ip, (int, float)):
        return ip
    try:
        parts = ip.split('.')
        return int(parts[0]) * 16777216 + int(parts[1]) * 65536 + int(parts[2]) * 256 + int(parts[3])
    except:
        return np.nan

def merge_country_info(fraud_df, ip_country_df):
    """
    Optimized IP-to-Country mapping using vectorized operations.
    """
    fraud_df = fraud_df.copy()
    ip_country_df = ip_country_df.sort_values('lower_bound_ip_address')
    
    # Vectorized approach: Use searchsorted to find the index where the ip would be inserted
    # This finds the row where ip >= lower_bound. We then check if ip <= upper_bound.
    indices = np.searchsorted(ip_country_df['lower_bound_ip_address'].values, fraud_df['ip_address'].values, side='right') - 1
    
    # Filter valid indices
    valid_mask = (indices >= 0) & (indices < len(ip_country_df))
    
    # Initialize with Unknown
    fraud_df['country'] = 'Unknown'
    
    # Map countries for valid indices where ip is within the upper bound
    potential_matches = ip_country_df.iloc[indices[valid_mask]]
    final_mask = fraud_df.loc[valid_mask, 'ip_address'].values <= potential_matches['upper_bound_ip_address'].values
    
    fraud_df.loc[np.where(valid_mask)[0][final_mask], 'country'] = potential_matches.iloc[final_mask]['country'].values
    
    return fraud_df

def feature_engineer_fraud_data(df, ip_country_df=None):
    """
    Applies all feature engineering steps.
    """
    df = add_time_since_signup(df)
    df = add_hour_of_day(df)
    df = add_day_of_week(df)
    df = add_transaction_frequency(df)
    
    if ip_country_df is not None and 'ip_address' in df.columns:
        df = merge_country_info(df, ip_country_df)
        
    return df
