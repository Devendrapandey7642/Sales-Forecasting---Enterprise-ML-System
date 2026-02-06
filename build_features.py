"""
Build engineered features from raw dataset
"""
import pandas as pd
import numpy as np
from tqdm import tqdm

# Load dataset
print("Loading final dataset...")
df = pd.read_csv('data/processed/final_dataset.csv')
df['date'] = pd.to_datetime(df['date'])
print(f"✓ Loaded: {df.shape}")

# Temporal features
print("\n📅 Creating temporal features...")
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter
df['day_of_week'] = df['date'].dt.dayofweek
df['day_of_month'] = df['date'].dt.day
df['week_of_year'] = df['date'].dt.isocalendar().week
print(f"✓ Temporal features: 6 new columns")

# Sort for time series
print("\n⏱️  Sorting by time...")
df = df.sort_values(['item_id', 'store_id', 'date']).reset_index(drop=True)

# Lag features
print("\n📊 Creating lag features...")
for lag_days in [7, 30]:
    df[f'quantity_lag_{lag_days}'] = df.groupby(['item_id', 'store_id'])['quantity'].shift(lag_days)
    df[f'sum_total_lag_{lag_days}'] = df.groupby(['item_id', 'store_id'])['sum_total'].shift(lag_days)
print(f"✓ Lag features: 4 new columns")

# Rolling mean
print("\n📈 Creating rolling mean features...")
for window in [7, 14, 30]:
    df[f'quantity_rolling_mean_{window}'] = df.groupby(['item_id', 'store_id'])['quantity'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    df[f'sum_total_rolling_mean_{window}'] = df.groupby(['item_id', 'store_id'])['sum_total'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
print(f"✓ Rolling mean features: 6 new columns")

# Clean NaN
print("\n🧹 Cleaning NaN values...")
print(f"  Before: {df.shape[0]:,} rows")
df_clean = df.dropna()
print(f"  After: {df_clean.shape[0]:,} rows")
print(f"  Removed: {df.shape[0] - df_clean.shape[0]:,} rows")

# Save
print("\n💾 Saving featured dataset...")
output_path = 'data/processed/featured_dataset.csv'
df_clean.to_csv(output_path, index=False)
print(f"✓ Saved: {output_path}")

# Summary
print("\n" + "="*60)
print("📊 FEATURE ENGINEERING SUMMARY")
print("="*60)
print(f"  Total rows: {df_clean.shape[0]:,}")
print(f"  Total features: {df_clean.shape[1]}")
print(f"\n  Features created:")
print(f"    ✓ Temporal (6): year, month, quarter, day_of_week, day_of_month, week_of_year")
print(f"    ✓ Lag (4): quantity_lag_7/30, sum_total_lag_7/30")
print(f"    ✓ Rolling Mean (6): quantity_rolling_mean_7/14/30, sum_total_rolling_mean_7/14/30")
print("="*60)
