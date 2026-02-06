"""
Build final dataset from RAW CSV files with merge logic
"""
import pandas as pd
import numpy as np

print("="*70)
print("BUILDING FINAL DATASET FROM RAW FILES")
print("="*70)

# 1. LOAD ALL RAW FILES
print("\n📂 Loading raw CSV files...")
sales = pd.read_csv('data/raw/sales.csv')
stores = pd.read_csv('data/raw/stores.csv')
catalog = pd.read_csv('data/raw/catalog.csv')
price_history = pd.read_csv('data/raw/price_history.csv')
discounts = pd.read_csv('data/raw/discounts_history.csv')
online = pd.read_csv('data/raw/online.csv')
markdowns = pd.read_csv('data/raw/markdowns.csv')

print(f"  ✓ sales: {sales.shape}")
print(f"  ✓ stores: {stores.shape}")
print(f"  ✓ catalog: {catalog.shape}")
print(f"  ✓ price_history: {price_history.shape}")
print(f"  ✓ discounts: {discounts.shape}")
print(f"  ✓ online: {online.shape}")
print(f"  ✓ markdowns: {markdowns.shape}")

# 2. START WITH SALES (main table)
print("\n🔄 Merging datasets...")
print(f"  → Starting with sales: {sales.shape}")

# Drop unnamed index column if exists
if 'Unnamed: 0' in sales.columns:
    sales = sales.drop('Unnamed: 0', axis=1)

final_df = sales.copy()

# 3. MERGE WITH STORES
if 'Unnamed: 0' in stores.columns:
    stores = stores.drop('Unnamed: 0', axis=1)
final_df = final_df.merge(stores, on='store_id', how='left')
print(f"  ✓ Merged with stores: {final_df.shape}")

# 4. MERGE WITH CATALOG
if 'Unnamed: 0' in catalog.columns:
    catalog = catalog.drop('Unnamed: 0', axis=1)
final_df = final_df.merge(catalog, on='item_id', how='left')
print(f"  ✓ Merged with catalog: {final_df.shape}")

# 5. MERGE WITH PRICE_HISTORY (latest price per day/store/item)
if 'Unnamed: 0' in price_history.columns:
    price_history = price_history.drop('Unnamed: 0', axis=1)
price_latest = price_history.sort_values(['item_id', 'store_id', 'date']).drop_duplicates(
    subset=['item_id', 'store_id', 'date'], keep='last'
)
final_df = final_df.merge(
    price_latest[['item_id', 'store_id', 'date', 'price']],
    on=['item_id', 'store_id', 'date'],
    how='left',
    suffixes=('', '_hist')
)
print(f"  ✓ Merged with price_history: {final_df.shape}")

# 6. MERGE WITH DISCOUNTS
if 'Unnamed: 0' in discounts.columns:
    discounts = discounts.drop('Unnamed: 0', axis=1)
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
print(f"  ✓ Merged with discounts: {final_df.shape}")

# 7. MERGE WITH ONLINE SALES
if 'Unnamed: 0' in online.columns:
    online = online.drop('Unnamed: 0', axis=1)
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
print(f"  ✓ Merged with online: {final_df.shape}")

# 8. MERGE WITH MARKDOWNS
if 'Unnamed: 0' in markdowns.columns:
    markdowns = markdowns.drop('Unnamed: 0', axis=1)
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
print(f"  ✓ Merged with markdowns: {final_df.shape}")

# 9. CLEAN UP
print("\n🧹 Cleaning up...")
print(f"  Missing values: {final_df.isnull().sum().sum()}")
final_df = final_df.fillna(0)  # Fill NaN with 0
print(f"  ✓ Filled missing values")

# 10. SAVE
output_path = 'data/processed/final_dataset.csv'
final_df.to_csv(output_path, index=False)

print(f"\n✅ FINAL DATASET CREATED")
print("="*70)
print(f"  Path: {output_path}")
print(f"  Shape: {final_df.shape}")
print(f"  Rows: {final_df.shape[0]:,}")
print(f"  Columns: {final_df.shape[1]}")
print("="*70)
