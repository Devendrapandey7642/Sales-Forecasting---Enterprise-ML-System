"""
Utility functions for data processing and analysis
"""
import os
import pandas as pd
import numpy as np

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"✓ Created directory: {directory}")

def load_csv(filepath):
    """Load CSV file"""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"✗ File not found: {filepath}")
        return None

def save_csv(df, filepath):
    """Save dataframe to CSV"""
    df.to_csv(filepath, index=False)
    print(f"✓ Saved: {filepath}")

def remove_duplicates(df):
    """Remove duplicate rows"""
    initial_len = len(df)
    df = df.drop_duplicates()
    removed = initial_len - len(df)
    print(f"  Removed {removed} duplicates")
    return df

def handle_missing_values(df, method='mean'):
    """Handle missing values"""
    if method == 'mean':
        df = df.fillna(df.mean())
    elif method == 'forward_fill':
        df = df.fillna(method='ffill')
    elif method == 'drop':
        df = df.dropna()
    return df
