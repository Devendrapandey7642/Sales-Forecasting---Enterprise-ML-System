"""
Model Evaluation - Detailed metrics and analysis
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("MODEL EVALUATION")
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

# Get predictions
y_pred = best_model.predict(X_test_scaled)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

# Residuals analysis
residuals = y_test.values - y_pred

print(f"\n📊 TEST SET EVALUATION")
print("="*70)
print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"MAE (Mean Absolute Error):      {mae:.4f}")
print(f"R² Score:                       {r2:.4f}")
print(f"MAPE:                           {mape:.4f}")
print("="*70)

print(f"\n📈 RESIDUALS ANALYSIS")
print("="*70)
print(f"Mean Residual:                  {residuals.mean():.4f}")
print(f"Std Residual:                   {residuals.std():.4f}")
print(f"Min Residual:                   {residuals.min():.4f}")
print(f"Max Residual:                   {residuals.max():.4f}")
print("="*70)

print(f"\n📉 PREDICTION ACCURACY")
print("="*70)
print(f"Actual Mean:                    {y_test.mean():.4f}")
print(f"Predicted Mean:                 {y_pred.mean():.4f}")
print(f"Actual Std:                     {y_test.std():.4f}")
print(f"Predicted Std:                  {y_pred.std():.4f}")
print("="*70)

# Save evaluation report
report = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'R²', 'MAPE', 'Mean_Residual', 'Std_Residual'],
    'Value': [rmse, mae, r2, mape, residuals.mean(), residuals.std()]
})
report.to_csv('reports/evaluation_metrics.csv', index=False)
print(f"\n✓ Saved: reports/evaluation_metrics.csv")

# Prediction analysis
pred_analysis = pd.DataFrame({
    'actual': y_test.values,
    'predicted': y_pred,
    'residual': residuals,
    'abs_error': np.abs(residuals),
    'pct_error': np.abs(residuals) / (y_test.values + 1) * 100
})
pred_analysis.to_csv('reports/detailed_predictions.csv', index=False)
print(f"✓ Saved: reports/detailed_predictions.csv")

# Error distribution
print(f"\n✅ Evaluation complete!")
