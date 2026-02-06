"""
Data Pipeline: Merge all raw datasets into a single master dataset
Handles all 8 CSV files and creates a comprehensive master dataset for forecasting
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path

def load_raw_data(data_dir='./data/raw'):
    """Load all raw CSV files"""
    print("📂 Loading raw CSV files...\n")
    
    raw_files = {
        'sales': os.path.join(data_dir, 'sales.csv'),
        'price_history': os.path.join(data_dir, 'price_history.csv'),
        'discounts_history': os.path.join(data_dir, 'discounts_history.csv'),
        'catalog': os.path.join(data_dir, 'catalog.csv'),
        'stores': os.path.join(data_dir, 'stores.csv'),
        'online': os.path.join(data_dir, 'online.csv'),
        'markdowns': os.path.join(data_dir, 'markdowns.csv'),
        'actual_matrix': os.path.join(data_dir, 'actual_matrix.csv')
    }
    
    dfs = {}
    for key, filepath in raw_files.items():
        if os.path.exists(filepath):
            dfs[key] = pd.read_csv(filepath)
            print(f"  ✓ {key:20s} {str(dfs[key].shape):15s} {list(dfs[key].columns)}")
        else:
            print(f"  ✗ {key}: File not found")
    
    return dfs

def clean_dataframe(df, name=''):
    """Remove unnamed index column and handle nulls"""
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    return df

def merge_datasets(dfs):
    """
    Merge all 8 datasets into one master dataset
    Strategy:
    1. sales → main table (7.4M rows, actual transactions)
    2. + stores → store info
    3. + catalog → product info
    4. + price_history → historical prices
    5. + discounts_history → promotional discounts
    6. + online → online channel sales
    7. + markdowns → clearance/markdowns
    8. actual_matrix → validation reference
    """
    
    print("\n🔄 Merging datasets...\n")
    
    # Start with sales as main table
    print("  1. Start with SALES as main table")
    final_df = clean_dataframe(dfs['sales'].copy(), 'sales')
    print(f"     Shape: {final_df.shape}")
    
    # Merge 2: stores
    print("  2. Merge with STORES (on store_id)")
    stores = clean_dataframe(dfs['stores'], 'stores')
    final_df = final_df.merge(stores, on='store_id', how='left')
    print(f"     Shape: {final_df.shape}")
    
    # Merge 3: catalog
    print("  3. Merge with CATALOG (on item_id)")
    catalog = clean_dataframe(dfs['catalog'], 'catalog')
    catalog = catalog.drop_duplicates(subset=['item_id'], keep='first')
    final_df = final_df.merge(catalog, on='item_id', how='left')
    print(f"     Shape: {final_df.shape}")
    
    # Merge 4: price_history
    print("  4. Merge with PRICE_HISTORY (on item_id, store_id, date)")
    price_history = clean_dataframe(dfs['price_history'], 'price_history')
    price_history = price_history.sort_values(['item_id', 'store_id', 'date'])
    price_history = price_history.drop_duplicates(
        subset=['item_id', 'store_id', 'date'], keep='last'
    )
    final_df = final_df.merge(
        price_history[['item_id', 'store_id', 'date', 'price', 'code']],
        on=['item_id', 'store_id', 'date'],
        how='left',
        suffixes=('', '_history')
    )
    print(f"     Shape: {final_df.shape}")
    
    # Merge 5: discounts_history (aggregate)
    print("  5. Merge with DISCOUNTS (aggregate on item_id, store_id, date)")
    discounts = clean_dataframe(dfs['discounts_history'], 'discounts')
    discounts_agg = discounts.groupby(['item_id', 'store_id', 'date']).agg({
        'sale_price_before_promo': 'mean',
        'sale_price_time_promo': 'mean',
        'number_disc_day': 'first'
    }).reset_index().rename(columns={
        'sale_price_before_promo': 'promo_price_before',
        'sale_price_time_promo': 'promo_price_after',
        'number_disc_day': 'promo_days'
    })
    final_df = final_df.merge(discounts_agg, on=['item_id', 'store_id', 'date'], how='left')
    print(f"     Shape: {final_df.shape}")
    
    # Merge 6: online (aggregate)
    print("  6. Merge with ONLINE SALES (aggregate on item_id, store_id, date)")
    online = clean_dataframe(dfs['online'], 'online')
    online_agg = online.groupby(['item_id', 'store_id', 'date']).agg({
        'quantity': 'sum',
        'price_base': 'mean',
        'sum_total': 'sum'
    }).reset_index().rename(columns={
        'quantity': 'online_qty',
        'price_base': 'online_price',
        'sum_total': 'online_revenue'
    })
    final_df = final_df.merge(online_agg, on=['item_id', 'store_id', 'date'], how='left')
    print(f"     Shape: {final_df.shape}")
    
    # Merge 7: markdowns (aggregate)
    print("  7. Merge with MARKDOWNS (aggregate on item_id, store_id, date)")
    markdowns = clean_dataframe(dfs['markdowns'], 'markdowns')
    markdowns_agg = markdowns.groupby(['item_id', 'store_id', 'date']).agg({
        'normal_price': 'mean',
        'price': 'mean',
        'quantity': 'sum'
    }).reset_index().rename(columns={
        'normal_price': 'markdown_normal_price',
        'price': 'markdown_price',
        'quantity': 'markdown_qty'
    })
    final_df = final_df.merge(markdowns_agg, on=['item_id', 'store_id', 'date'], how='left')
    print(f"     Shape: {final_df.shape}")
    
    print(f"\n✓ All merges complete!")
    return final_df

def handle_missing_values(df):
    """Handle missing values smartly"""
    print("\n📊 Handling missing values...\n")
    
    missing_before = df.isnull().sum().sum()
    
    # Fill numeric columns with 0 (no online, no markdown, etc.)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(0)
    
    # Fill categorical columns with 'Unknown'
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna('Unknown')
    
    missing_after = df.isnull().sum().sum()
    print(f"  Missing values before: {missing_before}")
    print(f"  Missing values after:  {missing_after}")
    
    return df

def build_final_dataset(raw_dir='./data/raw', output_dir='./data/processed'):
    """Build the final master dataset - main pipeline"""
    
    print("="*70)
    print("🚀 BUILDING FINAL MASTER DATASET")
    print("="*70)
    
    # Step 1: Load
    dfs = load_raw_data(raw_dir)
    
    # Step 2: Merge
    final_df = merge_datasets(dfs)
    
    # Step 3: Handle missing values
    final_df = handle_missing_values(final_df)
    
    # Step 4: Data quality
    print("\n📋 Final Dataset Quality Check:\n")
    print(f"  Shape: {final_df.shape}")
    print(f"  Columns: {len(final_df.columns)}")
    print(f"  Date range: {final_df['date'].min()} to {final_df['date'].max()}")
    print(f"  Stores: {final_df['store_id'].nunique()}")
    print(f"  Items: {final_df['item_id'].nunique()}")
    print(f"  Missing values: {final_df.isnull().sum().sum()}")
    
    # Step 5: Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'final_dataset.csv')
    final_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Final dataset saved!")
    print(f"   Path: {output_path}")
    print("="*70)
    
    return final_df

if __name__ == '__main__':
    final_df = build_final_dataset()
    print(f"\n📈 Final shape: {final_df.shape}")
    print(f"📁 Saved to: ./data/processed/final_dataset.csv")
