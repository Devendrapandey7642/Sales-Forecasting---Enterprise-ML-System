"""
SHAP Explainability Analysis
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("SHAP EXPLAINABILITY ANALYSIS")
print("="*70)

# Load data
print("\n📂 Loading data...")
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
X_train = X[:split_point]
X_test = X[split_point:]
y_test = y[split_point:]

# Load model and scaler
with open('models/best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature importance from model
print("\n🔍 Feature Importance Analysis...")
if hasattr(best_model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔝 Top 15 Most Important Features:")
    print(importance_df.head(15).to_string(index=False))
    
    # Save
    importance_df.to_csv('reports/feature_importance_detailed.csv', index=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['feature'][:10], importance_df['importance'][:10])
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    plt.savefig('reports/feature_importance_plot.png', dpi=100)
    print("\n✓ Saved: reports/feature_importance_plot.png")

# SHAP Analysis
print("\n📊 Computing SHAP values...")
try:
    # Use TreeExplainer for tree-based models
    explainer = shap.TreeExplainer(best_model)
    
    # Use sample of test set for speed
    sample_size = min(1000, len(X_test_scaled))
    X_sample = X_test_scaled[:sample_size]
    
    print(f"  Computing SHAP for {sample_size} samples...")
    shap_values = explainer.shap_values(X_sample)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    
    print("✓ SHAP values computed")
    
    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=X.columns, 
                      plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('reports/shap_summary_bar.png', dpi=100)
    print("✓ Saved: reports/shap_summary_bar.png")
    
    # Summary scatter plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=X.columns, show=False)
    plt.tight_layout()
    plt.savefig('reports/shap_summary_scatter.png', dpi=100)
    print("✓ Saved: reports/shap_summary_scatter.png")
    
except Exception as e:
    print(f"⚠️  SHAP computation skipped: {str(e)}")

print("\n" + "="*70)
print("✅ EXPLAINABILITY ANALYSIS COMPLETE!")
print("="*70)

# Summary
print("\n📋 FILES GENERATED:")
print("  ✓ reports/feature_importance_detailed.csv")
print("  ✓ reports/feature_importance_plot.png")
print("  ✓ reports/shap_summary_bar.png")
print("  ✓ reports/shap_summary_scatter.png")
