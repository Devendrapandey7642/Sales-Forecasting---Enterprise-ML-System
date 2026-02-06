"""
Feature Engineering: Create lag, rolling, and other advanced features
"""
import pandas as pd
import numpy as np

def create_lag_features(df, target_col, lags=[7, 14, 30]):
    """Create lag features"""
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df

def create_rolling_features(df, target_col, windows=[7, 14, 30]):
    """Create rolling window features"""
    for window in windows:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window).std()
    return df

def create_temporal_features(df, date_col):
    """Create temporal features (day, month, quarter, etc.)"""
    df[date_col] = pd.to_datetime(df[date_col])
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    df['year'] = df[date_col].dt.year
    df['week_of_year'] = df[date_col].dt.isocalendar().week
    return df

def engineer_features(df, target_col='sales', date_col='date'):
    """Apply all feature engineering"""
    df = create_temporal_features(df, date_col)
    df = create_lag_features(df, target_col)
    df = create_rolling_features(df, target_col)
    return df
