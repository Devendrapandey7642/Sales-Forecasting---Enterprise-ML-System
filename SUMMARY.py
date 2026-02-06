"""
Final Summary Report
"""
import pandas as pd
import os

print("\n" + "="*80)
print(" "*20 + "🎉 SALES FORECASTING PROJECT SUMMARY 🎉")
print("="*80)

# Check files
print("\n📊 DATASETS CREATED:")
print("-" * 80)

files_to_check = [
    ('data/raw/sales.csv', 'Raw Sales Data'),
    ('data/processed/final_dataset.csv', 'Master Dataset (Merged)'),
    ('data/processed/featured_dataset.csv', 'Featured Dataset'),
]

for path, desc in files_to_check:
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024 * 1024)  # MB
        rows = len(pd.read_csv(path))
        print(f"  ✓ {desc}")
        print(f"    Path: {path}")
        print(f"    Size: {size:.2f} MB | Rows: {rows:,}\n")

# Model results
print("🏆 MODEL RESULTS:")
print("-" * 80)

if os.path.exists('reports/model_comparison.csv'):
    results = pd.read_csv('reports/model_comparison.csv', index_col=0)
    print(results.to_string())
    
    best = results['R²'].idxmax()
    print(f"\n  🥇 Best Model: {best}")
    print(f"     R² Score: {results.loc[best, 'R²']:.4f}")
    print(f"     RMSE: {results.loc[best, 'RMSE']:.4f}")
    print(f"     MAE: {results.loc[best, 'MAE']:.4f}\n")

# Feature importance
print("🔝 TOP 10 IMPORTANT FEATURES:")
print("-" * 80)

if os.path.exists('reports/feature_importance_detailed.csv'):
    features = pd.read_csv('reports/feature_importance_detailed.csv')
    for i, row in features.head(10).iterrows():
        pct = row['importance_pct']
        print(f"  {i+1:2d}. {row['feature']:30s} {pct:6.2f}%")

# Reports generated
print("\n📁 REPORTS & FILES GENERATED:")
print("-" * 80)

reports = [
    'reports/model_comparison.csv',
    'reports/predictions.csv',
    'reports/evaluation_metrics.csv',
    'reports/detailed_predictions.csv',
    'reports/feature_importance_detailed.csv',
    'reports/feature_importance_plot.png',
    'reports/feature_correlation.csv',
    'models/best_model.pkl',
    'models/scaler.pkl',
]

for report in reports:
    if os.path.exists(report):
        print(f"  ✓ {report}")

# Python scripts
print("\n🐍 PYTHON SCRIPTS:")
print("-" * 80)

scripts = [
    ('build_dataset.py', 'Build master dataset from raw CSV files'),
    ('build_features.py', 'Create engineered features'),
    ('train_simple.py', 'Train ML models'),
    ('evaluate_model.py', 'Evaluate model performance'),
    ('explain_simple.py', 'Feature importance analysis'),
    ('sample_data.py', 'Sample data for faster development'),
]

for script, desc in scripts:
    if os.path.exists(script):
        print(f"  ✓ {script:20s} - {desc}")

# Jupyter notebooks
print("\n📓 JUPYTER NOTEBOOKS:")
print("-" * 80)

notebooks = [
    'notebooks/01_build_final_dataset.ipynb',
    'notebooks/02_eda.ipynb',
    'notebooks/03_feature_engineering.ipynb',
    'notebooks/04_model_training.ipynb',
    'notebooks/05_model_evaluation.ipynb',
    'notebooks/06_xai.ipynb',
]

for nb in notebooks:
    if os.path.exists(nb):
        print(f"  ✓ {nb}")

print("\n" + "="*80)
print("📈 PIPELINE SUMMARY:")
print("="*80)
print("""
1️⃣  DATA PIPELINE
    ✓ Loaded 8 raw CSV files (sampled to 10%)
    ✓ Merged into master dataset (743,268 rows)
    ✓ Created 27 merged features

2️⃣  FEATURE ENGINEERING
    ✓ Temporal features (6): year, month, quarter, day_of_week, etc.
    ✓ Lag features (4): quantity/sum_total lag 7 & 30 days
    ✓ Rolling features (6): rolling mean windows 7/14/30
    ✓ Final dataset: 146,608 rows, 43 features

3️⃣  MODEL TRAINING
    ✓ Linear Regression (Baseline): R² = 0.9019
    ✓ Random Forest: R² = 0.9279 ⭐ BEST
    ✓ Gradient Boosting: R² = 0.8196
    
4️⃣  MODEL EVALUATION
    ✓ RMSE: 5.1488
    ✓ MAE: 1.5320
    ✓ R² Score: 0.9279
    
5️⃣  EXPLAINABILITY
    ✓ Feature importance analysis
    ✓ Top features: rolling means (62% importance)
    ✓ Generated visualizations
""")

print("="*80)
print("🚀 NEXT STEPS:")
print("="*80)
print("""
1. Use the trained model for predictions on new data
2. Monitor model performance over time
3. Retrain model with new data periodically
4. Adjust features based on business insights
5. Deploy model to production for real-time forecasting
""")

print("="*80)
print("✅ PROJECT COMPLETE!")
print("="*80 + "\n")
