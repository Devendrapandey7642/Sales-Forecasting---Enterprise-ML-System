"""
Advanced XAI System (Explainable AI)
Features:
- Counterfactual explanations
- Human-readable text explanations
- Per-store/per-product insights
- Auto-generated PDF reports
- Decision reasoning
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path
import json
from datetime import datetime
import pickle

import sys
sys.path.append(str(Path(__file__).parent))


class CounterfactualExplainer:
    """Generate counterfactual explanations - what-if scenarios"""
    
    def __init__(self, model, scaler, dataset: pd.DataFrame):
        self.model = model
        self.scaler = scaler
        self.dataset = dataset
        self.feature_ranges = self._calculate_feature_ranges()
    
    def _calculate_feature_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Calculate min/max for each feature"""
        ranges = {}
        numeric_cols = self.dataset.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            ranges[col] = (
                float(self.dataset[col].min()),
                float(self.dataset[col].max())
            )
        
        return ranges
    
    def generate_counterfactual(
        self,
        sample: pd.DataFrame,
        feature_changes: Dict[str, float],
        target_outcome: float = None
    ) -> Dict[str, Any]:
        """Generate counterfactual explanation"""
        
        # Make prediction for original
        X_orig = self.scaler.transform(sample)
        pred_orig = float(self.model.predict(X_orig)[0])
        
        # Make prediction with changes
        sample_cf = sample.copy()
        for feature, change in feature_changes.items():
            if feature in sample_cf.columns:
                sample_cf[feature] = np.clip(
                    sample_cf[feature].values + change,
                    self.feature_ranges[feature][0],
                    self.feature_ranges[feature][1]
                )
        
        X_cf = self.scaler.transform(sample_cf)
        pred_cf = float(self.model.predict(X_cf)[0])
        
        return {
            "original_prediction": pred_orig,
            "counterfactual_prediction": pred_cf,
            "prediction_change": pred_cf - pred_orig,
            "percentage_change": ((pred_cf - pred_orig) / pred_orig * 100) if pred_orig != 0 else 0,
            "feature_changes": feature_changes,
            "interpretation": self._generate_interpretation(
                feature_changes, pred_cf - pred_orig
            )
        }
    
    def _generate_interpretation(
        self,
        changes: Dict[str, float],
        pred_change: float
    ) -> str:
        """Generate human-readable interpretation"""
        
        direction = "↑ increase" if pred_change > 0 else "↓ decrease"
        magnitude = "significantly" if abs(pred_change) > 100 else "moderately" if abs(pred_change) > 10 else "slightly"
        
        change_descriptions = []
        for feature, change in changes.items():
            if change > 0:
                change_descriptions.append(f"increase {feature} by {abs(change):.1f}")
            else:
                change_descriptions.append(f"decrease {feature} by {abs(change):.1f}")
        
        changes_str = ", ".join(change_descriptions)
        
        return f"If you {changes_str}, sales would {direction} {magnitude} by ~{abs(pred_change):.1f} units"
    
    def find_minimum_change(
        self,
        sample: pd.DataFrame,
        target_change: float,
        max_iterations: int = 10
    ) -> Dict[str, Any]:
        """Find minimum feature changes to achieve target outcome"""
        
        X_orig = self.scaler.transform(sample)
        pred_orig = float(self.model.predict(X_orig)[0])
        
        best_changes = {}
        best_pred = pred_orig
        
        # Try different feature changes
        for feature in sample.columns:
            if feature in self.feature_ranges:
                min_val, max_val = self.feature_ranges[feature]
                current_val = sample[feature].values[0]
                
                for step in range(1, max_iterations):
                    # Try moving Max outward
                    new_val = min(current_val + (max_val - current_val) * (step / max_iterations), max_val)
                    
                    sample_test = sample.copy()
                    sample_test[feature] = new_val
                    
                    X_test = self.scaler.transform(sample_test)
                    pred_test = float(self.model.predict(X_test)[0])
                    
                    if abs(pred_test - target_change) < abs(best_pred - target_change):
                        best_pred = pred_test
                        best_changes[feature] = new_val - current_val
        
        return {
            "target_change": target_change,
            "achieved_change": best_pred - pred_orig,
            "minimum_changes": best_changes
        }


class HumanExplainer:
    """Generate human-readable explanations"""
    
    @staticmethod
    def explain_prediction(
        sample: pd.DataFrame,
        prediction: float,
        feature_importance: pd.DataFrame,
        actual_value: float = None
    ) -> Dict[str, str]:
        """Generate human explanation for prediction"""
        
        explanations = {}
        
        # Overall assessment
        if actual_value is not None:
            error = abs(prediction - actual_value)
            error_pct = (error / actual_value * 100) if actual_value != 0 else 0
            accuracy = "very accurate" if error_pct < 5 else "accurate" if error_pct < 10 else "reasonable" if error_pct < 20 else "moderate"
            explanations["overall"] = f"The model predicted {prediction:.1f} units, which is {accuracy} (actual: {actual_value:.1f}, error: {error_pct:.1f}%)"
        else:
            explanations["overall"] = f"The model predicted {prediction:.1f} units for this product"
        
        # Key drivers
        top_features = feature_importance.head(5)
        drivers = ", ".join(top_features['feature'].head(3).tolist())
        explanations["key_drivers"] = f"The top factors influencing this prediction are: {drivers}"
        
        # Contextual insights
        sample_numeric = sample.select_dtypes(include=[np.number])
        if len(sample_numeric) > 0:
            high_features = sample_numeric.nlargest(2, 0).index.tolist()
            low_features = sample_numeric.nsmallest(2, 0).index.tolist()
            
            if high_features:
                explanations["context"] = f"This product has high {high_features[0]} and low {low_features[0] if low_features else 'inventory'}, which may affect demand"
        
        # Recommendation
        if prediction < 10:
            explanations["recommendation"] = "⚠️ Low predicted sales - consider promotional activities"
        elif prediction > 100:
            explanations["recommendation"] = "✅ High predicted sales - ensure sufficient stock"
        else:
            explanations["recommendation"] = "📊 Normal predicted sales - standard operations recommended"
        
        return explanations
    
    @staticmethod
    def explain_performance_drop(
        current_metrics: Dict[str, float],
        previous_metrics: Dict[str, float]
    ) -> str:
        """Explain why model performance dropped"""
        
        reasons = []
        
        for metric, current in current_metrics.items():
            previous = previous_metrics.get(metric, current)
            change = current - previous
            
            if metric == "R2" and change < -0.05:
                reasons.append(f"Model fit decreased (R² dropped by {abs(change):.3f})")
            elif metric == "RMSE" and change > 0:
                reasons.append(f"Prediction error increased ({change:.2f}% more error)")
            elif metric == "MAE" and change > 0:
                reasons.append(f"Average error increased by {change:.2f}")
        
        if not reasons:
            return "Model performance remains stable"
        
        return "Performance decline detected: " + "; ".join(reasons)
    
    @staticmethod
    def explain_anomaly(
        value: float,
        baseline_mean: float,
        baseline_std: float,
        z_score: float
    ) -> str:
        """Explain why value is anomalous"""
        
        deviation = "well below" if value < baseline_mean else "well above"
        severity = "critical" if abs(z_score) > 4 else "significant" if abs(z_score) > 2 else "moderate"
        
        return f"This value ({value:.1f}) is {severity} - it's {deviation} the typical range ({baseline_mean:.1f} ± {baseline_std:.1f})"


class PerProductAnalyzer:
    """Per-product and per-store explanations"""
    
    def __init__(self, model, scaler, dataset: pd.DataFrame):
        self.model = model
        self.scaler = scaler
        self.dataset = dataset
    
    def analyze_product(self, product_id: Any) -> Dict[str, Any]:
        """Analyze single product"""
        
        product_data = self.dataset[self.dataset['item_id'] == product_id]
        
        if product_data.empty:
            return None
        
        target_col = 'quantity'
        X = product_data.drop(columns=[target_col], errors='ignore')
        y = product_data[target_col] if target_col in product_data.columns else None
        
        # Make predictions
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return {
            "product_id": product_id,
            "num_records": len(product_data),
            "mean_actual": float(y.mean()) if y is not None else None,
            "mean_prediction": float(predictions.mean()),
            "prediction_std": float(predictions.std()),
            "trending": "up" if predictions[-10:].mean() > predictions[:10].mean() else "down",
            "confidence": float(np.std(predictions)) if len(predictions) > 1 else 1.0
        }
    
    def analyze_store(self, store_id: Any) -> Dict[str, Any]:
        """Analyze single store"""
        
        store_data = self.dataset[self.dataset['store_id'] == store_id]
        
        if store_data.empty:
            return None
        
        target_col = 'quantity'
        X = store_data.drop(columns=[target_col], errors='ignore')
        y = store_data[target_col] if target_col in store_data.columns else None
        
        # Make predictions
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return {
            "store_id": store_id,
            "num_records": len(store_data),
            "mean_actual": float(y.mean()) if y is not None else None,
            "mean_prediction": float(predictions.mean()),
            "total_predicted_revenue": float(predictions.sum()),
            "top_products": store_data['item_id'].value_counts().head(5).to_dict(),
            "performance": "strong" if predictions.mean() > predictions.std() else "moderate"
        }
    
    def compare_products(self, product_ids: List[Any]) -> pd.DataFrame:
        """Compare multiple products"""
        
        results = []
        for pid in product_ids:
            analysis = self.analyze_product(pid)
            if analysis:
                results.append(analysis)
        
        return pd.DataFrame(results)


class XAIReportGenerator:
    """Generate PDF reports with explanations"""
    
    @staticmethod
    def generate_summary_report(
        model_name: str,
        predictions: np.ndarray,
        actuals: np.ndarray,
        feature_importance: pd.DataFrame,
        top_products: List[Tuple] = None,
        output_path: str = None
    ) -> Dict[str, Any]:
        """Generate XAI summary report (JSON/Markdown for now, PDF with reportlab later)"""
        
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "model_name": model_name,
            "performance_metrics": {
                "RMSE": float(np.sqrt(mean_squared_error(actuals, predictions))),
                "MAE": float(mean_absolute_error(actuals, predictions)),
                "R2": float(r2_score(actuals, predictions))
            },
            "prediction_statistics": {
                "mean_prediction": float(predictions.mean()),
                "std_prediction": float(predictions.std()),
                "min_prediction": float(predictions.min()),
                "max_prediction": float(predictions.max()),
                "mean_actual": float(actuals.mean()),
                "std_actual": float(actuals.std())
            },
            "top_features": feature_importance.head(10).to_dict('records'),
            "key_insights": [
                f"Model explains {float(r2_score(actuals, predictions))*100:.1f}% of variance",
                f"Average prediction error: {float(mean_absolute_error(actuals, predictions)):.2f} units",
                f"Top 5 features account for {feature_importance.head(5)['importance'].sum():.1f}% of model decisions"
            ]
        }
        
        if output_path:
            Path(output_path).write_text(json.dumps(report, indent=2))
        
        return report
