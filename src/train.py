"""
Model Training: Train ML/DL models for sales forecasting
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pickle

def load_processed_data(filepath='./data/processed/final_dataset.csv'):
    """Load the processed dataset"""
    return pd.read_csv(filepath)

def train_model(X_train, y_train, model_type='rf'):
    """Train a model"""
    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'gb':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    return model

def save_model(model, filepath='./models/best_model.pkl'):
    """Save trained model"""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved: {filepath}")

def train_pipeline(data_path='./data/processed/final_dataset.csv', target_col='sales'):
    """Full training pipeline"""
    df = load_processed_data(data_path)
    
    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = train_model(X_train_scaled, y_train, model_type='gb')
    
    # Save model
    save_model(model)
    
    return model, X_test_scaled, y_test

if __name__ == '__main__':
    model, X_test, y_test = train_pipeline()
    print(f"✓ Training complete!")
