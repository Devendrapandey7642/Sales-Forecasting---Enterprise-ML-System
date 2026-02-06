"""
Train multiple models for sales forecasting
Uses time-based split (not random)
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load featured dataset
print("Loading featured dataset...")
df = pd.read_csv('data/processed/featured_dataset.csv')
print(f"✓ Loaded: {df.shape}")

# Define target and features
TARGET = 'quantity'  # Predicting quantity sold
print(f"\n🎯 Target variable: {TARGET}")

# Encode categorical columns
print(f"\n🔤 Encoding categorical features...")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = pd.factorize(df[col])[0]
print(f"✓ Encoded {len(categorical_cols)} categorical columns")

# Separate features and target
X = df.drop(columns=[TARGET, 'date', 'item_id', 'store_id', 'price_base', 'sum_total'])
y = df[TARGET]

print(f"✓ Features: {X.shape[1]}")
print(f"✓ Samples: {X.shape[0]:,}")
print(f"\nFeatures:\n  {list(X.columns)}")

# TIME-BASED SPLIT (80-20)
split_point = int(len(df) * 0.8)
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

print(f"\n📊 Train-Test Split (Time-Based):")
print(f"  Train size: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"  Test size: {X_test.shape[0]:,} samples ({X_test.shape[0]/len(df)*100:.1f}%)")

# Scale features
print(f"\n🔧 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"✓ Scaler saved")

# Train models
print(f"\n🚀 Training models...\n")
models = {}
results = {}

# 1. Linear Regression (Baseline)
print("1️⃣  Linear Regression (Baseline)...")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
models['Linear Regression'] = lr
results['Linear Regression'] = {'RMSE': rmse_lr, 'MAE': mae_lr, 'R²': r2_lr}
print(f"   RMSE: {rmse_lr:.4f} | MAE: {mae_lr:.4f} | R²: {r2_lr:.4f}\n")

# 2. Random Forest
print("2️⃣  Random Forest...")
rf = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
models['Random Forest'] = rf
results['Random Forest'] = {'RMSE': rmse_rf, 'MAE': mae_rf, 'R²': r2_rf}
print(f"   RMSE: {rmse_rf:.4f} | MAE: {mae_rf:.4f} | R²: {r2_rf:.4f}\n")

# 3. Gradient Boosting (simplified)
print("3️⃣  Gradient Boosting...")
gb = GradientBoostingRegressor(n_estimators=10, max_depth=3, learning_rate=0.1, random_state=42)
gb.fit(X_train_scaled, y_train)
y_pred_gb = gb.predict(X_test_scaled)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
mae_gb = mean_absolute_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)
models['Gradient Boosting'] = gb
results['Gradient Boosting'] = {'RMSE': rmse_gb, 'MAE': mae_gb, 'R²': r2_gb}
print(f"   RMSE: {rmse_gb:.4f} | MAE: {mae_gb:.4f} | R²: {r2_gb:.4f}\n")

# Find best model
print("="*70)
print("📊 MODEL COMPARISON")
print("="*70)
results_df = pd.DataFrame(results).T
print(results_df)
best_model_name = results_df['R²'].idxmax()
best_model = models[best_model_name]
print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   R² Score: {results_df.loc[best_model_name, 'R²']:.4f}")
print("="*70)

# Save best model
print(f"\n💾 Saving best model...")
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"✓ Saved: models/best_model.pkl")

# Save results
results_df.to_csv('reports/model_comparison.csv')
print(f"✓ Saved: reports/model_comparison.csv")

# Save predictions
pred_results = pd.DataFrame({
    'actual': y_test.values,
    'lr_pred': y_pred_lr,
    'rf_pred': y_pred_rf,
    'gb_pred': y_pred_gb,
    'best_pred': best_model.predict(X_test_scaled)
})
pred_results.to_csv('reports/predictions.csv', index=False)
print(f"✓ Saved: reports/predictions.csv")

print(f"\n✅ Training complete!")
