"""
Auto-Retraining Scheduler - Autonomous ML System
Features:
- Scheduled retraining (weekly/monthly)
- Best model auto-selection
- Hyperparameter auto-tuning
- Performance decay detection
- Automatic transition to production
"""

import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import sys
sys.path.append(str(Path(__file__).parent))
from mlops import (
    ModelRegistry, ExperimentTracker, AuditLog, PerformanceMonitor, 
    TrainProdComparison, calculate_data_hash
)


class AutoRetrainingScheduler:
    """Manages automatic model retraining on schedule"""
    
    SCHEDULE_PATH = Path(__file__).parent.parent / "models" / "schedule.json"
    
    @staticmethod
    def should_retrain(schedule_type: str = "weekly") -> bool:
        """Check if retraining should run based on schedule"""
        
        import json
        
        if not AutoRetrainingScheduler.SCHEDULE_PATH.exists():
            AutoRetrainingScheduler.SCHEDULE_PATH.write_text(json.dumps({
                "last_retrain": None,
                "schedule_type": schedule_type
            }, indent=2))
        
        schedule = json.loads(AutoRetrainingScheduler.SCHEDULE_PATH.read_text())
        last_retrain = schedule.get("last_retrain")
        
        if not last_retrain:
            return True
        
        last_time = datetime.fromisoformat(last_retrain)
        now = datetime.now()
        
        if schedule_type == "weekly":
            return (now - last_time).days >= 7
        elif schedule_type == "daily":
            return (now - last_time).days >= 1
        elif schedule_type == "monthly":
            return (now - last_time).days >= 30
        
        return False
    
    @staticmethod
    def run_retrain_cycle(
        dataset_path: str,
        schedule_type: str = "weekly",
        auto_promote: bool = True
    ) -> Dict[str, Any]:
        """Run complete retraining cycle"""
        
        import json
        
        AuditLog.log_action(
            action="retrain_started",
            details=f"Auto-retraining cycle started ({schedule_type})"
        )
        
        start_time = datetime.now()
        
        # Load data
        df = pd.read_csv(dataset_path)
        data_hash = calculate_data_hash(df)
        
        # Prepare data
        target_col = 'quantity'
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Retrain models
        results = {
            "timestamp": datetime.now().isoformat(),
            "data_hash": data_hash,
            "models": {},
            "best_model": None,
            "promoted": False
        }
        
        # 1. Linear Regression (baseline)
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        lr_pred = lr.predict(X_test_scaled)
        lr_metrics = BestModelSelector._calculate_metrics(y_test, lr_pred)
        results["models"]["LinearRegression"] = {
            "model": lr,
            "metrics": lr_metrics,
            "hyperparams": {}
        }
        
        # 2. Random Forest with tuning
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, 20]
        }
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        rf_search = GridSearchCV(rf, rf_params, cv=3, n_jobs=-1)
        rf_search.fit(X_train_scaled, y_train)
        rf_pred = rf_search.predict(X_test_scaled)
        rf_metrics = BestModelSelector._calculate_metrics(y_test, rf_pred)
        results["models"]["RandomForest"] = {
            "model": rf_search.best_estimator_,
            "metrics": rf_metrics,
            "hyperparams": rf_search.best_params_
        }
        
        # 3. Gradient Boosting with tuning
        gb_params = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05],
            'max_depth': [5, 7, 10]
        }
        gb = GradientBoostingRegressor(random_state=42)
        gb_search = GridSearchCV(gb, gb_params, cv=3, n_jobs=-1)
        gb_search.fit(X_train_scaled, y_train)
        gb_pred = gb_search.predict(X_test_scaled)
        gb_metrics = BestModelSelector._calculate_metrics(y_test, gb_pred)
        results["models"]["GradientBoosting"] = {
            "model": gb_search.best_estimator_,
            "metrics": gb_metrics,
            "hyperparams": gb_search.best_params_
        }
        
        # Find best model
        best_name, best_result = BestModelSelector.select_best_model(results["models"])
        results["best_model"] = {
            "name": best_name,
            "metrics": best_result["metrics"],
            "hyperparams": best_result["hyperparams"]
        }
        
        # Save best model
        model_path = str(Path(__file__).parent.parent / "models" / "best_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(best_result["model"], f)
        
        # Save scaler
        scaler_path = str(Path(__file__).parent.parent / "models" / "scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Register model
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        registry_info = ModelRegistry.register_model(
            model_path=model_path,
            model_name="AutoTrained_" + best_name,
            version=version,
            metrics=best_result["metrics"],
            hyperparams=best_result["hyperparams"],
            training_data_hash=data_hash,
            stage="staging"
        )
        
        # Log experiment
        ExperimentTracker.log_experiment(
            name=f"Auto-Retrain_{schedule_type}_{version}",
            hyperparams=best_result["hyperparams"],
            metrics=best_result["metrics"],
            model_type=best_name,
            status="completed"
        )
        
        # Performance monitoring
        PerformanceMonitor.record_metrics(
            metrics=best_result["metrics"],
            dataset_hash=data_hash,
            environment="staging"
        )
        
        # Check if should promote
        if auto_promote:
            prod_model = ModelRegistry.get_production_model()
            if prod_model:
                improved = all(
                    best_result["metrics"].get(k, 0) >= prod_model["metrics"].get(k, 0)
                    for k in best_result["metrics"].keys()
                )
                if improved:
                    ModelRegistry.promote_to_production(registry_info["id"])
                    results["promoted"] = True
                    AuditLog.log_action(
                        action="auto_promotion",
                        details=f"Model {registry_info['id']} auto-promoted to production"
                    )
            else:
                # No production model, auto-promote
                ModelRegistry.promote_to_production(registry_info["id"])
                results["promoted"] = True
        
        # Update schedule
        schedule = json.loads(AutoRetrainingScheduler.SCHEDULE_PATH.read_text())
        schedule["last_retrain"] = datetime.now().isoformat()
        schedule["schedule_type"] = schedule_type
        AutoRetrainingScheduler.SCHEDULE_PATH.write_text(json.dumps(schedule, indent=2))
        
        duration = (datetime.now() - start_time).total_seconds()
        
        AuditLog.log_action(
            action="retrain_completed",
            details=f"Retraining completed in {duration:.1f}s. Best: {best_name}"
        )
        
        return results


class BestModelSelector:
    """Select best model based on metrics"""
    
    @staticmethod
    def _calculate_metrics(y_true, y_pred) -> Dict[str, float]:
        """Calculate standard metrics"""
        return {
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred)),
            "MAPE": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
        }
    
    @staticmethod
    def select_best_model(models: Dict[str, Dict]) -> Tuple[str, Dict]:
        """Select best model using weighted metrics"""
        
        # Weights: R2 (50%), RMSE (30%), MAE (20%)
        weights = {"R2": 0.5, "RMSE": -0.3, "MAE": -0.2}
        
        best_name = None
        best_score = float('-inf')
        best_result = None
        
        for model_name, result in models.items():
            metrics = result["metrics"]
            
            # Calculate weighted score
            score = (
                metrics.get("R2", 0) * weights["R2"] -
                (metrics.get("RMSE", 0) / 100) * weights["RMSE"] -
                (metrics.get("MAE", 0) / 100) * weights["MAE"]
            )
            
            if score > best_score:
                best_score = score
                best_name = model_name
                best_result = result
        
        return best_name, best_result


class HyperparameterTuner:
    """Auto-tune hyperparameters"""
    
    TUNING_HISTORY_PATH = Path(__file__).parent.parent / "models" / "tuning_history.json"
    
    @staticmethod
    def tune_model(
        X_train,
        y_train,
        X_test,
        y_test,
        model_type: str = "RandomForest"
    ) -> Dict[str, Any]:
        """Tune model hyperparameters and find best config"""
        
        import json
        
        if model_type == "RandomForest":
            params = {
                'n_estimators': [50, 100, 150, 200],
                'max_depth': [5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        elif model_type == "GradientBoosting":
            params = {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
            model = GradientBoostingRegressor(random_state=42)
        
        else:
            return None
        
        # GridSearch
        search = GridSearchCV(model, params, cv=5, n_jobs=-1, verbose=1)
        search.fit(X_train, y_train)
        
        # Evaluate
        y_pred = search.predict(X_test)
        metrics = BestModelSelector._calculate_metrics(y_test, y_pred)
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "model_type": model_type,
            "best_params": search.best_params_,
            "best_score": search.best_score_,
            "metrics": metrics,
            "cv_results": len(search.cv_results_["params"])
        }
        
        # Log tuning result
        if not HyperparameterTuner.TUNING_HISTORY_PATH.exists():
            HyperparameterTuner.TUNING_HISTORY_PATH.write_text(json.dumps({
                "tuning_runs": []
            }, indent=2))
        
        history = json.loads(HyperparameterTuner.TUNING_HISTORY_PATH.read_text())
        history["tuning_runs"].append(result)
        HyperparameterTuner.TUNING_HISTORY_PATH.write_text(json.dumps(history, indent=2))
        
        return result
