"""
Utility functions for the advanced sales forecasting dashboard
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class DataManager:
    """Manages all data loading and caching"""
    
    @staticmethod
    def load_predictions():
        try:
            path = os.path.join(BASE_DIR, "reports", "predictions.csv")
            return pd.read_csv(path)
        except:
            return None
    
    @staticmethod
    def load_evaluation_metrics():
        try:
            path = os.path.join(BASE_DIR, "reports", "evaluation_metrics.csv")
            return pd.read_csv(path)
        except:
            return None
    
    @staticmethod
    def load_feature_importance():
        try:
            path = os.path.join(BASE_DIR, "reports", "feature_importance.csv")
            return pd.read_csv(path)
        except:
            return None
    
    @staticmethod
    def load_model_comparison():
        try:
            path = os.path.join(BASE_DIR, "reports", "model_comparison.csv")
            return pd.read_csv(path)
        except:
            return None
    
    @staticmethod
    def load_dataset():
        try:
            path = os.path.join(BASE_DIR, "data", "processed", "featured_dataset.csv")
            return pd.read_csv(path)
        except:
            return None
    
    @staticmethod
    def load_model():
        try:
            path = os.path.join(BASE_DIR, "models", "best_model.pkl")
            with open(path, 'rb') as f:
                return pickle.load(f)
        except:
            return None
    
    @staticmethod
    def load_scaler():
        try:
            path = os.path.join(BASE_DIR, "models", "scaler.pkl")
            with open(path, 'rb') as f:
                return pickle.load(f)
        except:
            return None


class PredictionEngine:
    """Advanced prediction capabilities"""
    
    @staticmethod
    def generate_confidence_intervals(predictions, std_dev=None, confidence=0.95):
        """Generate upper and lower bounds for predictions"""
        z_score = 1.96 if confidence == 0.95 else 1.645
        if std_dev is None:
            std_dev = np.std(predictions) * 0.1
        
        upper_bound = predictions + (z_score * std_dev)
        lower_bound = predictions - (z_score * std_dev)
        
        return lower_bound, upper_bound
    
    @staticmethod
    def scenario_prediction(base_prediction, scenario_type, factor=0.1):
        """Generate scenario-based predictions"""
        scenarios = {
            'with_discount': base_prediction * (1 - factor),
            'without_discount': base_prediction * 1.05,
            'price_increase': base_prediction * (1 - factor * 0.5),
            'price_decrease': base_prediction * (1 + factor * 0.3),
        }
        return scenarios.get(scenario_type, base_prediction)
    
    @staticmethod
    def ensemble_prediction(predictions_dict, method='average', weights=None):
        """Combine predictions from multiple models"""
        if method == 'average':
            return np.mean(list(predictions_dict.values()))
        elif method == 'weighted' and weights:
            weighted_sum = sum(pred * weights.get(name, 1) 
                             for name, pred in predictions_dict.items())
            return weighted_sum / sum(weights.values())
        return np.mean(list(predictions_dict.values()))


class InsightGenerator:
    """Generate business insights from data"""
    
    @staticmethod
    def detect_seasonality(df, date_col='date', value_col='quantity'):
        """Detect seasonal patterns"""
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            monthly_avg = df.groupby(df[date_col].dt.month)[value_col].mean()
            
            max_month = monthly_avg.idxmax()
            min_month = monthly_avg.idxmin()
            
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            max_increase = ((monthly_avg[max_month] - monthly_avg.mean()) / 
                           monthly_avg.mean() * 100)
            
            return f"Sales peak in {months[max_month-1]} ({max_increase:.1f}% above average)"
        except:
            return "Seasonality analysis unavailable"
    
    @staticmethod
    def detect_anomalies(series, threshold=2):
        """Detect outliers using z-score"""
        z_scores = np.abs((series - series.mean()) / series.std())
        anomalies = series[z_scores > threshold]
        return len(anomalies)
    
    @staticmethod
    def generate_insights(df, predictions, actuals):
        """Generate auto-insights"""
        insights = []
        
        try:
            mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
            insights.append(f"Average prediction accuracy: {100-mape:.1f}%")
        except:
            pass
        
        try:
            growth = ((df.iloc[-1] - df.iloc[0]) / df.iloc[0] * 100)
            if growth > 0:
                insights.append(f"Sales trending up by {growth:.1f}%")
            else:
                insights.append(f"Sales declining by {abs(growth):.1f}%")
        except:
            pass
        
        return insights


class DataQualityChecker:
    """Monitor data quality"""
    
    @staticmethod
    def check_missing_values(df):
        """Check for missing values"""
        missing = df.isnull().sum()
        missing_percent = (missing / len(df)) * 100
        
        return pd.DataFrame({
            'Column': missing.index,
            'Missing_Count': missing.values,
            'Percentage': missing_percent.values
        }).sort_values('Missing_Count', ascending=False)
    
    @staticmethod
    def detect_drift(df_old, df_new, columns=None):
        """Detect data distribution shift"""
        if columns is None:
            columns = df_old.select_dtypes(include=[np.number]).columns
        
        drift_detected = []
        for col in columns:
            old_mean = df_old[col].mean()
            new_mean = df_new[col].mean()
            change_percent = abs((new_mean - old_mean) / old_mean * 100)
            
            if change_percent > 10:
                drift_detected.append((col, change_percent))
        
        return drift_detected
    
    @staticmethod
    def data_health_score(df):
        """Calculate overall data health score"""
        missing_percent = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        health_score = 100 - missing_percent
        return max(0, min(100, health_score))
