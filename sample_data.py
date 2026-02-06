"""
Sample data to reduce size for faster development
Sample 10% of rows from each CSV file
"""
import pandas as pd
import os

data_dir = 'data/raw'

# Files to sample
files = [
    'sales.csv',
    'price_history.csv',
    'discounts_history.csv',
    'catalog.csv',
    'stores.csv',
    'online.csv',
    'markdowns.csv',
    'actual_matrix.csv'
]

print("="*60)
print("SAMPLING DATA (10% of each file)")
print("="*60)

for fname in files:
    filepath = os.path.join(data_dir, fname)
    
    # Load
    df = pd.read_csv(filepath)
    original_rows = len(df)
    
    # Sample 10%
    df_sampled = df.sample(frac=0.1, random_state=42)
    
    # Save back
    df_sampled.to_csv(filepath, index=False)
    
    print(f"\n{fname}")
    print(f"  Before: {original_rows:,} rows")
    print(f"  After:  {len(df_sampled):,} rows")
    print(f"  Saved:  {filepath}")

print("\n" + "="*60)
print("✅ Data sampling complete!")
print("="*60)
