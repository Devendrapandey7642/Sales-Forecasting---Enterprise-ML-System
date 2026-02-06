"""
Business Decision Engine
Features:
- Inventory reorder recommendations
- Revenue loss prediction
- Profit vs discount optimizer
- Store expansion suggestions
- Decision intelligence (ML → actions)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import json

import sys
sys.path.append(str(Path(__file__).parent))


class InventoryOptimizer:
    """Smart inventory reorder recommendations"""
    
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset
        self.lead_time_days = 7  # Typical lead time
        self.safety_stock_multiplier = 1.5
    
    def recommend_reorder(
        self,
        product_id: Any,
        current_stock: float,
        predictions: np.ndarray,
        reorder_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """Recommend when and how much to reorder"""
        
        # Calculate future demand
        forecast_horizon = 30  # Next 30 days
        avg_daily_demand = predictions[:forecast_horizon].mean()
        std_demand = predictions[:forecast_horizon].std()
        
        # Calculate safety stock
        safety_stock = std_demand * self.safety_stock_multiplier * np.sqrt(self.lead_time_days)
        
        # Lead time demand
        lead_time_demand = avg_daily_demand * self.lead_time_days
        
        # Reorder point
        reorder_point = lead_time_demand + safety_stock
        
        # Economic order quantity (EOQ) approximation
        eoq = avg_daily_demand * 30
        
        recommendation = {
            "product_id": product_id,
            "current_stock": current_stock,
            "forecast_daily_demand": float(avg_daily_demand),
            "reorder_point": float(reorder_point),
            "order_quantity": float(eoq),
            "total_order_point": float(reorder_point + eoq),
            "recommended_action": "REORDER_NOW" if current_stock < reorder_point else "MONITOR",
            "urgency": "high" if current_stock < reorder_point * 0.5 else "medium" if current_stock < reorder_point else "low",
            "days_until_stockout": float((current_stock - safety_stock) / avg_daily_demand) if avg_daily_demand > 0 else float('inf'),
            "estimated_cost": float(eoq * 10),  # Assuming $10/unit cost
            "risk_level": "critical" if current_stock < safety_stock else "medium" if current_stock < reorder_point else "low"
        }
        
        return recommendation
    
    def batch_recommendations(
        self,
        products_data: List[Dict]
    ) -> List[Dict]:
        """Get recommendations for multiple products"""
        
        return [self.recommend_reorder(**prod) for prod in products_data]
    
    def get_reorder_schedule(self, products: List[Dict]) -> pd.DataFrame:
        """Get prioritized reorder schedule"""
        
        recs = self.batch_recommendations(products)
        df = pd.DataFrame(recs)
        
        # Sort by urgency and days until stockout
        df['priority_score'] = df.apply(
            lambda x: 100 if x['urgency'] == 'high' else 50 if x['urgency'] == 'medium' else 10,
            axis=1
        )
        df['priority_score'] += df['days_until_stockout'].fillna(float('inf')).apply(
            lambda x: 0 if x < 7 else 10 if x < 14 else 20
        )
        
        return df.sort_values('priority_score', ascending=False)


class RevenueLossPredictor:
    """Predict potential revenue loss from various scenarios"""
    
    def __init__(self, historical_data: pd.DataFrame):
        self.historical_data = historical_data
        self.avg_margin = 0.25  # 25% average margin
    
    def predict_loss_from_stockout(
        self,
        product_id: Any,
        forecast_sales: np.ndarray,
        stockout_days: int
    ) -> Dict[str, float]:
        """Predict revenue loss from stockout"""
        
        avg_daily_sales = forecast_sales.mean()
        lost_sales_units = avg_daily_sales * stockout_days
        avg_price = 15  # Average price per unit
        
        lost_revenue = lost_sales_units * avg_price
        lost_profit = lost_revenue * self.avg_margin
        
        return {
            "product_id": product_id,
            "stockout_days": stockout_days,
            "lost_sales_units": float(lost_sales_units),
            "lost_revenue": float(lost_revenue),
            "lost_profit": float(lost_profit),
            "expected_customer_churn": float(lost_sales_units * 0.05)  # 5% churn rate
        }
    
    def predict_loss_from_price_increase(
        self,
        current_price: float,
        price_increase_pct: float,
        sales_elasticity: float = 0.8,
        current_sales: float = 100
    ) -> Dict[str, float]:
        """Predict sales loss from price increase"""
        
        quantity_decrease_pct = price_increase_pct * sales_elasticity
        new_sales = current_sales * (1 - quantity_decrease_pct / 100)
        new_price = current_price * (1 + price_increase_pct / 100)
        
        old_revenue = current_sales * current_price
        new_revenue = new_sales * new_price
        revenue_change = new_revenue - old_revenue
        
        return {
            "current_price": current_price,
            "new_price": new_price,
            "price_increase_pct": price_increase_pct,
            "current_sales": int(current_sales),
            "new_sales": int(new_sales),
            "old_revenue": float(old_revenue),
            "new_revenue": float(new_revenue),
            "revenue_change": float(revenue_change),
            "revenue_change_pct": float(revenue_change / old_revenue * 100) if old_revenue > 0 else 0
        }
    
    def predict_total_loss(self, risks: List[Dict]) -> Dict[str, float]:
        """Calculate total revenue loss across all risks"""
        
        total_loss = sum(risk.get('lost_revenue', 0) for risk in risks)
        total_profit_loss = sum(risk.get('lost_profit', 0) for risk in risks)
        
        return {
            "total_revenue_loss": float(total_loss),
            "total_profit_loss": float(total_profit_loss),
            "risk_count": len(risks),
            "average_loss_per_risk": float(total_loss / len(risks)) if risks else 0
        }


class ProfitOptimizer:
    """Optimize pricing and discounts for maximum profit"""
    
    def __init__(self):
        self.cost_multiplier = 0.6  # 60% of price is cost
        self.target_margin = 0.30  # Target 30% margin
    
    def optimize_discount(
        self,
        base_price: float,
        current_sales: float,
        discount_elasticity: float = 1.2,
        current_margin: float = 0.25
    ) -> Dict[str, Any]:
        """Find optimal discount level"""
        
        scenarios = []
        
        for discount_pct in range(0, 31, 5):  # 0% to 30% discounts
            discount = base_price * discount_pct / 100
            new_price = base_price - discount
            
            # Sales increase from discount
            sales_increase = (current_sales * discount_pct / 100 * discount_elasticity)
            new_sales = current_sales + sales_increase
            
            # Profit calculation
            cost_per_unit = base_price * self.cost_multiplier
            profit_per_unit = new_price - cost_per_unit
            total_profit = profit_per_unit * new_sales
            
            scenarios.append({
                "discount_pct": discount_pct,
                "new_price": float(new_price),
                "old_sales": int(current_sales),
                "new_sales": int(new_sales),
                "sales_increase_pct": float(discount_pct * discount_elasticity),
                "profit_per_unit": float(profit_per_unit),
                "total_profit": float(total_profit),
                "margin": float(profit_per_unit / new_price * 100) if new_price > 0 else 0
            })
        
        # Find optimal
        best = max(scenarios, key=lambda x: x['total_profit'])
        
        return {
            "optimal_discount": best['discount_pct'],
            "optimal_price": best['new_price'],
            "expected_sales_increase": best['sales_increase_pct'],
            "expected_total_profit": best['total_profit'],
            "profit_gain_vs_current": best['total_profit'] - (current_sales * (base_price - base_price * self.cost_multiplier)),
            "scenarios": scenarios
        }
    
    def optimize_product_mix(
        self,
        products: List[Dict]
    ) -> List[Dict]:
        """Optimize product mix for maximum profit"""
        
        optimized = []
        
        for prod in products:
            optimization = self.optimize_discount(
                base_price=prod['price'],
                current_sales=prod['sales'],
                discount_elasticity=prod.get('elasticity', 1.2)
            )
            optimization['product_id'] = prod['product_id']
            optimized.append(optimization)
        
        # Sort by profit potential
        return sorted(optimized, key=lambda x: x['profit_gain_vs_current'], reverse=True)


class StoreExpansionAnalyzer:
    """Suggest store expansion opportunities"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def analyze_store_performance(self, store_id: Any) -> Dict[str, Any]:
        """Analyze store performance metrics"""
        
        store_data = self.data[self.data['store_id'] == store_id]
        
        if store_data.empty:
            return None
        
        target_col = 'quantity'
        
        metrics = {
            "store_id": store_id,
            "total_sales": int(store_data[target_col].sum()) if target_col in store_data.columns else 0,
            "avg_transaction": float(store_data[target_col].mean()) if target_col in store_data.columns else 0,
            "num_products": store_data['item_id'].nunique(),
            "num_categories": store_data['class_name'].nunique() if 'class_name' in store_data.columns else 0,
            "revenue": float(store_data['sum_total'].sum()) if 'sum_total' in store_data.columns else 0,
            "growth_potential": "high" if store_data.shape[0] > 1000 else "medium" if store_data.shape[0] > 500 else "low"
        }
        
        return metrics
    
    def suggest_expansion(self) -> List[Dict]:
        """Suggest top stores for expansion"""
        
        if 'store_id' not in self.data.columns:
            return []
        
        stores = []
        for store_id in self.data['store_id'].unique():
            analysis = self.analyze_store_performance(store_id)
            if analysis:
                stores.append(analysis)
        
        stores_df = pd.DataFrame(stores)
        
        # Score stores
        stores_df['expansion_score'] = (
            (stores_df['total_sales'] / stores_df['total_sales'].max() * 0.4) +
            (stores_df['num_products'] / stores_df['num_products'].max() * 0.3) +
            (stores_df['avg_transaction'] / stores_df['avg_transaction'].max() * 0.3)
        )
        
        top_stores = stores_df.nlargest(5, 'expansion_score')
        
        suggestions = []
        for _, store in top_stores.iterrows():
            suggestions.append({
                "store_id": store['store_id'],
                "current_revenue": store['revenue'],
                "expansion_opportunity": "High" if store['expansion_score'] > 0.8 else "Medium" if store['expansion_score'] > 0.5 else "Low",
                "recommended_action": f"Expand product range from {int(store['num_products'])} to {int(store['num_products'] * 1.3)}",
                "investment_required": float(store['revenue'] * 0.15),
                "expected_revenue_increase": float(store['revenue'] * 0.25)
            })
        
        return suggestions


class DecisionIntelligence:
    """Integrate all decision engines into actionable recommendations"""
    
    def __init__(self, model, scaler, dataset: pd.DataFrame):
        self.model = model
        self.scaler = scaler
        self.dataset = dataset
        self.inventory = InventoryOptimizer(dataset)
        self.revenue_loss = RevenueLossPredictor(dataset)
        self.profit = ProfitOptimizer()
        self.expansion = StoreExpansionAnalyzer(dataset)
    
    def generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive-level decision summary"""
        
        summary = {
            "generated_at": datetime.now().isoformat(),
            "inventory_alerts": self._get_inventory_alerts(),
            "revenue_risks": self._get_revenue_risks(),
            "profit_opportunities": self._get_profit_opportunities(),
            "expansion_recommendations": self.expansion.suggest_expansion(),
            "executive_actions": self._generate_actions()
        }
        
        return summary
    
    def _get_inventory_alerts(self) -> List[Dict]:
        """Get critical inventory alerts"""
        # Implementation would iterate through products
        return []
    
    def _get_revenue_risks(self) -> Dict[str, float]:
        """Get total revenue at risk"""
        return {
            "total_at_risk": 0.0,
            "critical_items": 0,
            "medium_risk_items": 0
        }
    
    def _get_profit_opportunities(self) -> List[Dict]:
        """Get top profit optimization opportunities"""
        # Implementation would analyze product mix
        return []
    
    def _generate_actions(self) -> List[str]:
        """Generate recommended business actions"""
        actions = [
            "📦 Implement recommended inventory reorders",
            "💰 Apply optimal pricing strategies to top products",
            "🏪 Initiate expansion process for high-performing stores",
            "⚠️ Monitor identified revenue risks"
        ]
        return actions
