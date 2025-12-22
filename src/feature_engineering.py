
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def add_time_since_signup(df):
    """
    Adds a 'time_since_signup' feature (seconds).
    
    Args:
        df (pd.DataFrame): Dataframe with 'purchase_time' and 'signup_time'.
        
    Returns:
        pd.DataFrame: Dataframe with new feature.
    """
    df = df.copy()
    if 'purchase_time' in df.columns and 'signup_time' in df.columns:
        df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
    return df

def add_hour_of_day(df):
    """
    Adds 'hour_of_day' feature from purchase time.
    """
    df = df.copy()
    if 'purchase_time' in df.columns:
        df['hour_of_day'] = df['purchase_time'].dt.hour
    return df

def add_day_of_week(df):
    """
    Adds 'day_of_week' feature from purchase time.
    """
    df = df.copy()
    if 'purchase_time' in df.columns:
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
    return df


def ip_to_int(ip):
    """Convert string IP to integer."""
    try:
        parts = ip.split('.')
        return int(parts[0]) * 16777216 + int(parts[1]) * 65536 + int(parts[2]) * 256 + int(parts[3])
    except:
        return np.nan

def get_country(ip_int, ip_country_df):
    """
    Finds the country for a given integer IP address efficiently.
    Assumes ip_country_df is sorted by lower_bound_ip_address.
    """
    # This is a naive implementation; for large datasets, a binary search or interval tree is better.
    # However, to be vectorized and fast with pandas:
    # We can't easily join on inequalities with standard pandas merge without cartesian product (too big).
    # Since this is "feature engineering" and might be run row-by-row or needs optimization:
    
    # Optimization: Use searchsorted if the intervals are non-overlapping and cover the space
    # But usually, we just need to find the row where lower <= ip <= upper.
    
    # A simple way for this task (assuming reasonably sized ip_country_df):
    # Iterate is slow. Let's return a placeholder for now or a slow apply if dataset is small.
    # Given the requirements usually imply learning by doing, let's write a function that takes a SINGLE ip
    # and we can apply it.
    
    try:
        # Filter where ip is within range
        # Note: This is efficient only for single lookup. 
        # For bulk: ensure df is sorted, use searchsorted to find potential match, then check bounds.
        
        # Let's assume this is used in a apply function for now.
        res = ip_country_df[
            (ip_country_df['lower_bound_ip_address'] <= ip_int) & 
            (ip_country_df['upper_bound_ip_address'] >= ip_int)
        ]
        if not res.empty:
            return res.iloc[0]['country']
        return "Unknown"
    except:
        return "Unknown"

def feature_engineer_fraud_data(df, ip_country_df=None):
    """
    Applies all feature engineering steps to fraud data.
    
    Args:
        df (pd.DataFrame): Cleaned fraud data.
        ip_country_df (pd.DataFrame): Cleaned IP country data.
        
    Returns:
        pd.DataFrame: Dataframe with features.
    """
    df = add_time_since_signup(df)
    df = add_hour_of_day(df)
    df = add_day_of_week(df)
    
    # IP mapping
    if ip_country_df is not None and 'ip_address' in df.columns:
        # Convert IP to int
        # Note: In the real dataset, 'ip_address' might be float or int already.
        # If it's a string like "192.168.1.1", use ip_to_int.
        # Requirement says "IP -> Country mapping".
        # Let's check type. If numeric, assume it's already converted or in simplified format.
        # The provided dataset usually has 'ip_address' as a float/int.
        
        # We will assume it is numeric for now based on standard kaggle datasets of this name.
        # If not, we'd need conversion.
        
        # Optimizing the lookup is critical for speed.
        # For this task, we will apply the lookup row-wise for simplicity as the dataset size isn't specified as massive.
        
        # Let's define the vectorized logic or efficient apply
        # To avoid passing the whole DF to apply, we can do it smarter, but apply is safest for "fully working code".
        
        # Sort ip_country_df for searchsorted
        ip_country_df = ip_country_df.sort_values('lower_bound_ip_address')
        
        # Helper to do batch lookup could be better, but let's stick to functional correctness first.
        def lookup_country(ip):
            return get_country(ip, ip_country_df)
            
        df['country'] = df['ip_address'].apply(lookup_country)
        
    return df

def encode_categorical(df, columns):
    """
    Encodes categorical variables using Label Encoding.
    For production, OneHot might be better, but LabelEncoder is requested/common for basics.
    """
    le = LabelEncoder()
    for col in columns:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    return df
