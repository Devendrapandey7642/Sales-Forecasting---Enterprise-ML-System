"""
Real-Time Forecasting System
Features:
- Streaming data simulation
- Live forecast refresh
- Real-time alerts on anomalies
- Rolling prediction window
- Stream processing pipeline
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Generator
from pathlib import Path
import json
import pickle
from collections import deque

import sys
sys.path.append(str(Path(__file__).parent))
from mlops import AuditLog


class StreamDataSimulator:
    """Simulates real-time streaming data"""
    
    def __init__(self, dataset_path: str, batch_size: int = 10):
        self.dataset = pd.read_csv(dataset_path)
        self.batch_size = batch_size
        self.current_idx = 0
        self.stream_stats = {
            "total_batches": 0,
            "total_records": 0,
            "start_time": datetime.now().isoformat()
        }
    
    def get_next_batch(self) -> Tuple[pd.DataFrame, bool]:
        """Get next batch of streaming data"""
        
        if self.current_idx >= len(self.dataset):
            # Reset stream
            self.current_idx = 0
            return None, True  # End of stream indicator
        
        end_idx = min(self.current_idx + self.batch_size, len(self.dataset))
        batch = self.dataset.iloc[self.current_idx:end_idx]
        
        self.current_idx = end_idx
        self.stream_stats["total_batches"] += 1
        self.stream_stats["total_records"] += len(batch)
        
        return batch, False
    
    def stream_generator(self, infinite: bool = False) -> Generator:
        """Generate stream of batches"""
        
        while True:
            batch, end_of_stream = self.get_next_batch()
            
            if batch is not None:
                yield batch
            
            if end_of_stream and not infinite:
                break
            elif end_of_stream and infinite:
                # Reset and continue
                self.current_idx = 0


class RealtimePredictionEngine:
    """Process real-time predictions on streaming data"""
    
    def __init__(self, model_path: str, scaler_path: str):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.prediction_window = deque(maxlen=100)  # Last 100 predictions
        self.batch_cache = deque(maxlen=10)  # Last 10 batches
    
    def predict_batch(self, batch: pd.DataFrame) -> Dict[str, any]:
        """Make predictions for a batch"""
        
        target_col = 'quantity'
        
        # Extract features (same as training)
        X = batch.drop(columns=[target_col], errors='ignore')
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        
        # Calculate metrics for this batch
        batch_stats = {
            "timestamp": datetime.now().isoformat(),
            "batch_size": len(batch),
            "predictions": predictions.tolist(),
            "pred_mean": float(np.mean(predictions)),
            "pred_std": float(np.std(predictions)),
            "pred_min": float(np.min(predictions)),
            "pred_max": float(np.max(predictions))
        }
        
        # Add to window
        self.prediction_window.extend(predictions)
        self.batch_cache.append(batch_stats)
        
        return batch_stats
    
    def get_rolling_stats(self, window_size: int = 50) -> Dict[str, float]:
        """Get rolling statistics over prediction window"""
        
        if len(self.prediction_window) == 0:
            return {}
        
        recent = list(self.prediction_window)[-window_size:]
        
        return {
            "window_mean": float(np.mean(recent)),
            "window_std": float(np.std(recent)),
            "window_min": float(np.min(recent)),
            "window_max": float(np.max(recent)),
            "records_in_window": len(recent)
        }


class AnomalyDetector:
    """Detect anomalies in real-time data"""
    
    def __init__(self, threshold_std: float = 2.5):
        self.threshold_std = threshold_std
        self.baseline_mean = None
        self.baseline_std = None
        self.anomalies = []
    
    def build_baseline(self, baseline_data: np.ndarray):
        """Build baseline statistics"""
        self.baseline_mean = np.mean(baseline_data)
        self.baseline_std = np.std(baseline_data)
    
    def detect_anomaly(self, value: float) -> Tuple[bool, Dict[str, any]]:
        """Detect if single value is anomalous"""
        
        if self.baseline_mean is None:
            return False, {}
        
        z_score = abs((value - self.baseline_mean) / self.baseline_std)
        is_anomaly = z_score > self.threshold_std
        
        result = {
            "value": value,
            "z_score": float(z_score),
            "baseline_mean": float(self.baseline_mean),
            "baseline_std": float(self.baseline_std),
            "is_anomaly": is_anomaly,
            "severity": "high" if z_score > 4 else "medium" if is_anomaly else "low"
        }
        
        if is_anomaly:
            result["timestamp"] = datetime.now().isoformat()
            self.anomalies.append(result)
        
        return is_anomaly, result
    
    def detect_batch_anomalies(self, predictions: np.ndarray) -> List[Dict]:
        """Detect anomalies in batch"""
        
        anomalies = []
        for i, pred in enumerate(predictions):
            is_anomaly, details = self.detect_anomaly(pred)
            if is_anomaly:
                details["index"] = i
                anomalies.append(details)
        
        return anomalies
    
    def get_anomaly_report(self, limit: int = 20) -> Dict:
        """Get anomaly report"""
        
        recent = self.anomalies[-limit:]
        
        return {
            "total_anomalies": len(self.anomalies),
            "recent_anomalies": recent,
            "high_severity": [a for a in recent if a["severity"] == "high"],
            "anomaly_rate": len(self.anomalies) / max(1, sum(1 for _ in self.anomalies)) if self.anomalies else 0
        }


class RealTimeAlertSystem:
    """Generate real-time alerts"""
    
    ALERTS_PATH = Path(__file__).parent.parent / "models" / "realtime_alerts.json"
    
    @staticmethod
    def _ensure_alerts():
        """Ensure alerts file exists"""
        if not RealTimeAlertSystem.ALERTS_PATH.exists():
            RealTimeAlertSystem.ALERTS_PATH.write_text(json.dumps({
                "alerts": []
            }, indent=2))
    
    @staticmethod
    def create_alert(
        alert_type: str,
        severity: str,
        message: str,
        related_value: float = None,
        threshold: float = None
    ) -> Dict:
        """Create and log alert"""
        
        RealTimeAlertSystem._ensure_alerts()
        
        alert = {
            "id": f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}",
            "timestamp": datetime.now().isoformat(),
            "type": alert_type,
            "severity": severity,  # low, medium, high, critical
            "message": message,
            "related_value": related_value,
            "threshold": threshold,
            "acknowledged": False
        }
        
        alerts = json.loads(RealTimeAlertSystem.ALERTS_PATH.read_text())
        alerts["alerts"].append(alert)
        RealTimeAlertSystem.ALERTS_PATH.write_text(json.dumps(alerts, indent=2))
        
        AuditLog.log_action(
            action="realtime_alert",
            details=f"Alert: {alert_type} - {message}",
            severity=severity
        )
        
        return alert
    
    @staticmethod
    def get_active_alerts(limit: int = 50) -> List[Dict]:
        """Get active (unacknowledged) alerts"""
        
        RealTimeAlertSystem._ensure_alerts()
        alerts = json.loads(RealTimeAlertSystem.ALERTS_PATH.read_text())
        
        unack = [a for a in alerts["alerts"] if not a["acknowledged"]]
        return sorted(unack, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    @staticmethod
    def acknowledge_alert(alert_id: str) -> bool:
        """Acknowledge an alert"""
        
        alerts = json.loads(RealTimeAlertSystem.ALERTS_PATH.read_text())
        
        for alert in alerts["alerts"]:
            if alert["id"] == alert_id:
                alert["acknowledged"] = True
                alert["acknowledged_at"] = datetime.now().isoformat()
                RealTimeAlertSystem.ALERTS_PATH.write_text(json.dumps(alerts, indent=2))
                return True
        
        return False


class RealtimeMonitoringPipeline:
    """Complete real-time monitoring pipeline"""
    
    def __init__(self, model_path: str, scaler_path: str, dataset_path: str):
        self.simulator = StreamDataSimulator(dataset_path)
        self.predictor = RealtimePredictionEngine(model_path, scaler_path)
        self.anomaly_detector = AnomalyDetector(threshold_std=2.5)
        self.alert_system = RealTimeAlertSystem()
        self.pipeline_stats = {
            "batches_processed": 0,
            "predictions_made": 0,
            "anomalies_detected": 0,
            "alerts_created": 0,
            "start_time": datetime.now().isoformat()
        }
    
    def process_batch(self, batch: pd.DataFrame) -> Dict:
        """Process single batch through pipeline"""
        
        # 1. Make predictions
        pred_batch = self.predictor.predict_batch(batch)
        
        # 2. Detect anomalies
        anomalies = self.anomaly_detector.detect_batch_anomalies(
            np.array(pred_batch["predictions"])
        )
        
        # 3. Create alerts if needed
        alerts = []
        for anomaly in anomalies:
            alert = self.alert_system.create_alert(
                alert_type="prediction_anomaly",
                severity=anomaly["severity"],
                message=f"Anomalous prediction: {anomaly['value']:.2f}",
                related_value=anomaly["value"],
                threshold=anomaly["baseline_mean"] + anomaly["baseline_std"] * self.anomaly_detector.threshold_std
            )
            alerts.append(alert)
        
        # Check for sudden changes
        rolling = self.predictor.get_rolling_stats()
        if rolling and abs(pred_batch["pred_mean"] - rolling["window_mean"]) > rolling["window_std"] * 2:
            alert = self.alert_system.create_alert(
                alert_type="sudden_change",
                severity="high",
                message=f"Sudden change detected in predictions",
                related_value=pred_batch["pred_mean"]
            )
            alerts.append(alert)
        
        # Update stats
        self.pipeline_stats["batches_processed"] += 1
        self.pipeline_stats["predictions_made"] += len(pred_batch["predictions"])
        self.pipeline_stats["anomalies_detected"] += len(anomalies)
        self.pipeline_stats["alerts_created"] += len(alerts)
        
        return {
            "batch_stats": pred_batch,
            "anomalies": anomalies,
            "alerts": alerts,
            "rolling_stats": rolling,
            "pipeline_stats": self.pipeline_stats
        }
    
    def run_realtime_monitoring(self, num_batches: int = None) -> List[Dict]:
        """Run monitoring for specified number of batches"""
        
        results = []
        batch_count = 0
        
        for batch in self.simulator.stream_generator():
            if num_batches and batch_count >= num_batches:
                break
            
            result = self.process_batch(batch)
            results.append(result)
            batch_count += 1
        
        return results
