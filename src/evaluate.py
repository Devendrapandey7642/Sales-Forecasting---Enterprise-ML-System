"""
Model Evaluation: Calculate metrics and compare models
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }
    return metrics

def print_metrics(metrics):
    """Print metrics in a nice format"""
    print("\n" + "="*40)
    print("MODEL EVALUATION METRICS")
    print("="*40)
    for metric_name, value in metrics.items():
        print(f"{metric_name:6s}: {value:10.4f}")
    print("="*40 + "\n")

def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set"""
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    print_metrics(metrics)
    return metrics, y_pred
