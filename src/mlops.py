"""
MLOps Layer - Enterprise-grade model lifecycle management
Features:
- Model Registry with versioning
- Experiment Tracking
- Train vs Production Comparison
- Audit Logs
- Performance Monitoring
- Automated Rollback
"""

import json
import pickle
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import hashlib


class ModelRegistry:
    """Manages model versioning, metadata, and lifecycle"""
    
    REGISTRY_PATH = Path(__file__).parent.parent / "models" / "registry.json"
    MODELS_DIR = Path(__file__).parent.parent / "models"
    
    @staticmethod
    def _ensure_registry():
        """Ensure registry file exists"""
        if not ModelRegistry.REGISTRY_PATH.exists():
            ModelRegistry.REGISTRY_PATH.write_text(json.dumps({
                "models": [],
                "production": None,
                "staging": None
            }, indent=2))
    
    @staticmethod
    def register_model(
        model_path: str,
        model_name: str,
        version: str,
        metrics: Dict[str, float],
        hyperparams: Dict[str, Any],
        training_data_hash: str,
        stage: str = "staging"  # staging, production, archived
    ) -> Dict[str, Any]:
        """Register a new model version"""
        
        ModelRegistry._ensure_registry()
        
        model_info = {
            "id": f"{model_name}_v{version}",
            "name": model_name,
            "version": version,
            "path": model_path,
            "registered_at": datetime.now().isoformat(),
            "metrics": metrics,
            "hyperparams": hyperparams,
            "training_data_hash": training_data_hash,
            "stage": stage,
            "description": f"{model_name} version {version}",
            "tags": ["auto-generated"],
            "performance_trend": []
        }
        
        registry = json.loads(ModelRegistry.REGISTRY_PATH.read_text())
        registry["models"].append(model_info)
        
        if stage == "production":
            if registry.get("production"):
                old_prod = registry["production"]
                old_prod["stage"] = "archived"
            registry["production"] = model_info
        elif stage == "staging":
            registry["staging"] = model_info
        
        ModelRegistry.REGISTRY_PATH.write_text(json.dumps(registry, indent=2))
        
        return model_info
    
    @staticmethod
    def promote_to_production(model_id: str) -> bool:
        """Promote a staged model to production"""
        
        registry = json.loads(ModelRegistry.REGISTRY_PATH.read_text())
        
        model = next((m for m in registry["models"] if m["id"] == model_id), None)
        if not model:
            return False
        
        # Archive old production
        if registry.get("production"):
            old_prod = next((m for m in registry["models"] 
                           if m["id"] == registry["production"]["id"]), None)
            if old_prod:
                old_prod["stage"] = "archived"
        
        # Promote model
        model["stage"] = "production"
        model["promoted_at"] = datetime.now().isoformat()
        registry["production"] = model
        
        ModelRegistry.REGISTRY_PATH.write_text(json.dumps(registry, indent=2))
        
        AuditLog.log_action(
            action="model_promotion",
            details=f"Model {model_id} promoted to production"
        )
        
        return True
    
    @staticmethod
    def rollback_production(steps: int = 1) -> bool:
        """Rollback production to previous model version"""
        
        registry = json.loads(ModelRegistry.REGISTRY_PATH.read_text())
        
        # Get production history (archived + production)
        history = sorted(
            [m for m in registry["models"] if m["stage"] in ["production", "archived"]],
            key=lambda x: x["registered_at"],
            reverse=True
        )
        
        if len(history) < steps + 1:
            return False
        
        # Get the target model to restore
        target = history[steps]
        
        # Archive current production
        if registry.get("production"):
            old_prod = next((m for m in registry["models"] 
                           if m["id"] == registry["production"]["id"]), None)
            if old_prod:
                old_prod["stage"] = "archived"
        
        # Restore previous model
        target["stage"] = "production"
        target["restored_at"] = datetime.now().isoformat()
        registry["production"] = target
        
        ModelRegistry.REGISTRY_PATH.write_text(json.dumps(registry, indent=2))
        
        AuditLog.log_action(
            action="model_rollback",
            details=f"Production rolled back to {target['id']}"
        )
        
        return True
    
    @staticmethod
    def get_production_model() -> Dict[str, Any]:
        """Get current production model info"""
        registry = json.loads(ModelRegistry.REGISTRY_PATH.read_text())
        return registry.get("production")
    
    @staticmethod
    def get_staging_model() -> Dict[str, Any]:
        """Get current staging model info"""
        registry = json.loads(ModelRegistry.REGISTRY_PATH.read_text())
        return registry.get("staging")
    
    @staticmethod
    def get_model_history(model_name: str = None, limit: int = 10) -> List[Dict]:
        """Get model version history"""
        registry = json.loads(ModelRegistry.REGISTRY_PATH.read_text())
        
        models = registry["models"]
        if model_name:
            models = [m for m in models if m["name"] == model_name]
        
        return sorted(models, key=lambda x: x["registered_at"], reverse=True)[:limit]
    
    @staticmethod
    def compare_models(model_id_1: str, model_id_2: str) -> Dict[str, Any]:
        """Compare two model versions"""
        registry = json.loads(ModelRegistry.REGISTRY_PATH.read_text())
        
        m1 = next((m for m in registry["models"] if m["id"] == model_id_1), None)
        m2 = next((m for m in registry["models"] if m["id"] == model_id_2), None)
        
        if not m1 or not m2:
            return None
        
        return {
            "model_1": m1,
            "model_2": m2,
            "metrics_diff": {
                key: m2["metrics"].get(key, 0) - m1["metrics"].get(key, 0)
                for key in m1["metrics"].keys()
            }
        }


class ExperimentTracker:
    """Track ML experiments and hyperparameter tuning"""
    
    EXPERIMENTS_PATH = Path(__file__).parent.parent / "models" / "experiments.json"
    
    @staticmethod
    def _ensure_experiments():
        """Ensure experiments file exists"""
        if not ExperimentTracker.EXPERIMENTS_PATH.exists():
            ExperimentTracker.EXPERIMENTS_PATH.write_text(json.dumps({
                "experiments": []
            }, indent=2))
    
    @staticmethod
    def log_experiment(
        name: str,
        hyperparams: Dict[str, Any],
        metrics: Dict[str, float],
        model_type: str,
        status: str = "completed"  # completed, running, failed
    ) -> Dict[str, Any]:
        """Log a new experiment"""
        
        ExperimentTracker._ensure_experiments()
        
        experiment = {
            "id": f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "hyperparams": hyperparams,
            "metrics": metrics,
            "model_type": model_type,
            "status": status,
            "duration_seconds": None
        }
        
        experiments = json.loads(ExperimentTracker.EXPERIMENTS_PATH.read_text())
        experiments["experiments"].append(experiment)
        
        ExperimentTracker.EXPERIMENTS_PATH.write_text(json.dumps(experiments, indent=2))
        
        return experiment
    
    @staticmethod
    def get_best_experiment(metric_name: str = "R2") -> Dict[str, Any]:
        """Get best experiment by metric"""
        experiments = json.loads(ExperimentTracker.EXPERIMENTS_PATH.read_text())
        
        completed = [e for e in experiments["experiments"] if e["status"] == "completed"]
        
        if not completed:
            return None
        
        return max(completed, key=lambda x: x["metrics"].get(metric_name, 0))
    
    @staticmethod
    def get_experiment_history(limit: int = 20) -> List[Dict]:
        """Get recent experiments"""
        experiments = json.loads(ExperimentTracker.EXPERIMENTS_PATH.read_text())
        
        return sorted(
            experiments["experiments"],
            key=lambda x: x["timestamp"],
            reverse=True
        )[:limit]


class AuditLog:
    """Enterprise audit logging for compliance and debugging"""
    
    LOG_PATH = Path(__file__).parent.parent / "models" / "audit.log"
    
    @staticmethod
    def log_action(
        action: str,
        details: str,
        user: str = "system",
        severity: str = "info"  # info, warning, critical
    ):
        """Log an action to audit trail"""
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details,
            "user": user,
            "severity": severity
        }
        
        with open(AuditLog.LOG_PATH, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        return log_entry
    
    @staticmethod
    def get_audit_trail(action_filter: str = None, limit: int = 100) -> List[Dict]:
        """Get audit trail"""
        
        if not AuditLog.LOG_PATH.exists():
            return []
        
        logs = []
        with open(AuditLog.LOG_PATH, "r") as f:
            for line in f:
                try:
                    logs.append(json.loads(line))
                except:
                    pass
        
        if action_filter:
            logs = [l for l in logs if l["action"] == action_filter]
        
        return sorted(logs, key=lambda x: x["timestamp"], reverse=True)[:limit]


class PerformanceMonitor:
    """Monitor model performance over time and detect decay"""
    
    METRICS_HISTORY_PATH = Path(__file__).parent.parent / "models" / "metrics_history.json"
    
    @staticmethod
    def record_metrics(
        metrics: Dict[str, float],
        dataset_hash: str,
        environment: str = "production"  # production, staging, test
    ):
        """Record metrics snapshot"""
        
        if not PerformanceMonitor.METRICS_HISTORY_PATH.exists():
            PerformanceMonitor.METRICS_HISTORY_PATH.write_text(json.dumps({
                "history": []
            }, indent=2))
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "dataset_hash": dataset_hash,
            "environment": environment
        }
        
        history = json.loads(PerformanceMonitor.METRICS_HISTORY_PATH.read_text())
        history["history"].append(record)
        
        PerformanceMonitor.METRICS_HISTORY_PATH.write_text(json.dumps(history, indent=2))
        
        return record
    
    @staticmethod
    def detect_performance_decay(
        metric_name: str = "R2",
        threshold_pct: float = 5.0,
        lookback_days: int = 7
    ) -> Tuple[bool, Dict[str, Any]]:
        """Detect if model performance is decaying"""
        
        if not PerformanceMonitor.METRICS_HISTORY_PATH.exists():
            return False, {}
        
        history = json.loads(PerformanceMonitor.METRICS_HISTORY_PATH.read_text())
        
        # Filter for production environment
        prod_metrics = [
            h for h in history["history"] 
            if h["environment"] == "production"
        ]
        
        if len(prod_metrics) < 2:
            return False, {}
        
        # Sort by timestamp
        prod_metrics = sorted(prod_metrics, key=lambda x: x["timestamp"])
        
        # Get recent baseline (first in range)
        baseline = prod_metrics[0]["metrics"].get(metric_name, 0)
        current = prod_metrics[-1]["metrics"].get(metric_name, 0)
        
        decay_pct = ((baseline - current) / baseline) * 100 if baseline != 0 else 0
        
        is_decaying = decay_pct > threshold_pct
        
        return is_decaying, {
            "metric_name": metric_name,
            "baseline": baseline,
            "current": current,
            "decay_percent": decay_pct,
            "threshold_percent": threshold_pct,
            "records": len(prod_metrics)
        }


class TrainProdComparison:
    """Compare training and production environments"""
    
    @staticmethod
    def compare(
        train_metrics: Dict[str, float],
        prod_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Compare train vs production metrics"""
        
        comparison = {
            "train": train_metrics,
            "production": prod_metrics,
            "drift": {},
            "warnings": []
        }
        
        for key in train_metrics.keys():
            if key in prod_metrics:
                diff_pct = (
                    (train_metrics[key] - prod_metrics[key]) / train_metrics[key] * 100
                ) if train_metrics[key] != 0 else 0
                
                comparison["drift"][key] = {
                    "diff_percent": diff_pct,
                    "train_value": train_metrics[key],
                    "prod_value": prod_metrics[key]
                }
                
                # Warn if significant difference
                if abs(diff_pct) > 10:
                    comparison["warnings"].append(
                        f"Significant drift in {key}: {abs(diff_pct):.1f}% difference"
                    )
        
        return comparison


def calculate_data_hash(df: pd.DataFrame) -> str:
    """Calculate hash of dataset for tracking"""
    data_str = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.md5(data_str).hexdigest()
