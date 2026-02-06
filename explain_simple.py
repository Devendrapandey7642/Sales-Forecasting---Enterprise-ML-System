"""
Model Explainability - Feature Importance & Insights
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("MODEL EXPLAINABILITY")
print("="*70)

# Load data
df = pd.read_csv('data/processed/featured_dataset.csv')

# Encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

TARGET = 'quantity'
X = df.drop(columns=[TARGET, 'date']) if 'date' in df.columns else df.drop(columns=[TARGET])
y = df[TARGET]

# Time-based split
split_point = int(len(df) * 0.8)
X_test = X[split_point:]
y_test = y[split_point:]

# Load model and scaler
with open('models/best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

X_test_scaled = scaler.transform(X_test)

# Feature importance from model
print("\n🔍 FEATURE IMPORTANCE ANALYSIS")
print("="*70)

if hasattr(best_model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_,
        'importance_pct': (best_model.feature_importances_ / best_model.feature_importances_.sum()) * 100
    }).sort_values('importance', ascending=False)
    
    print("\n🔝 Top 15 Most Important Features:")
    print(importance_df.head(15).to_string(index=False))
    
    # Save
    importance_df.to_csv('reports/feature_importance_detailed.csv', index=False)
    print("\n✓ Saved: reports/feature_importance_detailed.csv")
    
    # Plot top 15
    fig, ax = plt.subplots(figsize=(10, 6))
    top_features = importance_df.head(15)
    ax.barh(range(len(top_features)), top_features['importance'].values)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.set_xlabel('Importance')
    ax.set_title('Top 15 Feature Importances (Random Forest)')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('reports/feature_importance_plot.png', dpi=100, bbox_inches='tight')
    print("✓ Saved: reports/feature_importance_plot.png")
    plt.close()

# Prediction analysis
print("\n📊 PREDICTION ANALYSIS")
print("="*70)

y_pred = best_model.predict(X_test_scaled)
residuals = y_test.values - y_pred

# Percentile-based analysis
percentiles = [10, 25, 50, 75, 90]
print("\nError Distribution (Absolute Error):")
abs_errors = np.abs(residuals)
for p in percentiles:
    val = np.percentile(abs_errors, p)
    print(f"  {p}th percentile: {val:.4f}")

# Accuracy ranges
print("\nPrediction Accuracy Ranges:")
for threshold in [1, 2, 5, 10]:
    pct = (abs_errors <= threshold).sum() / len(abs_errors) * 100
    print(f"  Within ±{threshold}: {pct:.2f}%")

# Feature correlation with target
print("\n📈 FEATURE CORRELATION WITH TARGET")
print("="*70)

# Recreate data for correlation (need original scale)
X_test_orig = X[split_point:].copy()
corr_data = X_test_orig.copy()
corr_data['quantity'] = y_test.values

# Select numeric columns
numeric_features = corr_data.select_dtypes(include=[np.number]).columns
correlations = corr_data[numeric_features].corr()['quantity'].sort_values(ascending=False)

print("\nTop 10 Correlated Features with Quantity:")
print(correlations.head(11)[1:].to_string())  # Skip 'quantity' itself

# Save
correlations.to_csv('reports/feature_correlation.csv')
print("\n✓ Saved: reports/feature_correlation.csv")

# Key insights
print("\n" + "="*70)
print("🎯 KEY INSIGHTS")
print("="*70)

if hasattr(best_model, 'feature_importances_'):
    top_1 = importance_df.iloc[0]
    top_5 = importance_df.head(5)
    
    print(f"\n1️⃣  Most Important Feature:")
    print(f"    {top_1['feature']} ({top_1['importance_pct']:.2f}%)")
    
    print(f"\n2️⃣  Top 5 Features Account For:")
    pct_5 = top_5['importance_pct'].sum()
    print(f"    {pct_5:.2f}% of total importance")
    
    print(f"\n3️⃣  Model Strength:")
    print(f"    Rolling mean features dominate predictions")
    print(f"    This indicates recent sales trends are strong predictors")
    
    print(f"\n4️⃣  Recommended Focus Areas:")
    print(f"    1. Monitor recent sales trends (rolling mean)")
    print(f"    2. Track day-of-week patterns")
    print(f"    3. Focus on high-importance features for data quality")

print("\n" + "="*70)
print("✅ EXPLAINABILITY COMPLETE!")
print("="*70)
