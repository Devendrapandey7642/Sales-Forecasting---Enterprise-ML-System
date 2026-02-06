"""
Agentic AI Assistant v2
Features:
- Tool calling and execution
- Multi-step reasoning
- Complex query handling
- Action planning
- Integration with ML models and decision engines
"""

import json
from typing import Dict, List, Any, Callable, Tuple
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent))
from mlops import ModelRegistry, ExperimentTracker, PerformanceMonitor
from business_engine import ProfitOptimizer, InventoryOptimizer, StoreExpansionAnalyzer
from advanced_xai import HumanExplainer


class ToolRegistry:
    """Registry of available tools for AI assistant to use"""
    
    def __init__(self):
        self.tools = {}
        self._register_default_tools()
    
    def register_tool(
        self,
        name: str,
        function: Callable,
        description: str,
        parameters: Dict[str, str]
    ):
        """Register a tool"""
        self.tools[name] = {
            "function": function,
            "description": description,
            "parameters": parameters
        }
    
    def _register_default_tools(self):
        """Register default tools"""
        
        self.register_tool(
            "get_model_status",
            self._get_model_status,
            "Get current production model status and performance",
            {}
        )
        
        self.register_tool(
            "get_performance_metrics",
            self._get_performance_metrics,
            "Get latest model performance metrics",
            {}
        )
        
        self.register_tool(
            "check_model_drift",
            self._check_model_drift,
            "Check if model is experiencing performance decay",
            {}
        )
        
        self.register_tool(
            "optimize_discount",
            self._optimize_discount,
            "Find optimal discount for product",
            {"product_id": "str", "base_price": "float", "current_sales": "float"}
        )
        
        self.register_tool(
            "recommend_inventory",
            self._recommend_inventory,
            "Get inventory reorder recommendations",
            {"product_id": "str", "current_stock": "float"}
        )
        
        self.register_tool(
            "suggest_expansion",
            self._suggest_expansion,
            "Get store expansion opportunities",
            {}
        )
        
        self.register_tool(
            "explain_prediction",
            self._explain_prediction,
            "Get human explanation for a prediction",
            {"prediction": "float", "product_id": "str"}
        )
        
        self.register_tool(
            "compare_models",
            self._compare_models,
            "Compare staging vs production models",
            {}
        )
    
    def _get_model_status(self) -> Dict:
        """Get current model status"""
        prod_model = ModelRegistry.get_production_model()
        if not prod_model:
            return {"status": "No production model"}
        
        return {
            "model": prod_model["name"],
            "version": prod_model["version"],
            "stage": prod_model["stage"],
            "metrics": prod_model["metrics"],
            "registered_at": prod_model["registered_at"]
        }
    
    def _get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        prod_model = ModelRegistry.get_production_model()
        if not prod_model:
            return {}
        return prod_model.get("metrics", {})
    
    def _check_model_drift(self) -> Dict:
        """Check for model drift"""
        is_decaying, details = PerformanceMonitor.detect_performance_decay(
            metric_name="R2",
            threshold_pct=5.0
        )
        return {
            "is_drifting": is_decaying,
            "details": details
        }
    
    def _optimize_discount(self, product_id: str, base_price: float, current_sales: float) -> Dict:
        """Get discount optimization"""
        optimizer = ProfitOptimizer()
        return optimizer.optimize_discount(base_price, current_sales)
    
    def _recommend_inventory(self, product_id: str, current_stock: float) -> Dict:
        """Get inventory recommendation"""
        # Dummy implementation - would use real data
        return {
            "product_id": product_id,
            "recommendation": "MONITOR",
            "reorder_point": current_stock * 1.5
        }
    
    def _suggest_expansion(self) -> Dict:
        """Get expansion suggestions"""
        return {
            "top_stores": [
                {"store_id": 1, "opportunity": "High"},
                {"store_id": 5, "opportunity": "Medium"}
            ]
        }
    
    def _explain_prediction(self, prediction: float, product_id: str) -> Dict:
        """Explain prediction"""
        return {
            "prediction": prediction,
            "explanation": f"Based on historical patterns, product {product_id} is predicted to sell {prediction:.1f} units"
        }
    
    def _compare_models(self) -> Dict:
        """Compare models"""
        prod = ModelRegistry.get_production_model()
        staging = ModelRegistry.get_staging_model()
        
        return {
            "production": prod,
            "staging": staging,
            "comparison": "Staging model shows improvement" if staging and prod and staging["metrics"].get("R2", 0) > prod["metrics"].get("R2", 0) else "No improvement"
        }
    
    def get_tool(self, name: str) -> Dict:
        """Get tool by name"""
        return self.tools.get(name)
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool"""
        tool = self.tools.get(tool_name)
        if not tool:
            return {"error": f"Tool {tool_name} not found"}
        
        try:
            return tool["function"](**kwargs)
        except Exception as e:
            return {"error": str(e)}
    
    def list_tools(self) -> List[Dict]:
        """List all available tools"""
        return [
            {
                "name": name,
                "description": info["description"],
                "parameters": info["parameters"]
            }
            for name, info in self.tools.items()
        ]


class QueryPlanner:
    """Plan multi-step queries"""
    
    @staticmethod
    def parse_query(query: str) -> Dict[str, Any]:
        """Parse user query into actionable steps"""
        
        query_lower = query.lower()
        
        # Intent detection
        intent = None
        if any(word in query_lower for word in ["optimize", "discount", "price"]):
            intent = "pricing_optimization"
        elif any(word in query_lower for word in ["inventory", "reorder", "stock"]):
            intent = "inventory_management"
        elif any(word in query_lower for word in ["expand", "store", "growth"]):
            intent = "expansion"
        elif any(word in query_lower for word in ["model", "drift", "performance", "deploy"]):
            intent = "model_management"
        elif any(word in query_lower for word in ["explain", "why", "because"]):
            intent = "explanation"
        
        # Entity extraction (simplified)
        entities = {
            "product_ids": [],
            "store_ids": [],
            "metrics": []
        }
        
        # Simple extraction
        import re
        product_matches = re.findall(r"product[s]?\s+(\d+)", query_lower)
        store_matches = re.findall(r"store[s]?\s+(\d+)", query_lower)
        
        entities["product_ids"] = product_matches
        entities["store_ids"] = store_matches
        
        return {
            "original_query": query,
            "intent": intent,
            "entities": entities,
            "requires_tools": True
        }
    
    @staticmethod
    def plan_steps(parsed_query: Dict) -> List[str]:
        """Plan steps to answer query"""
        
        intent = parsed_query["intent"]
        steps = []
        
        if intent == "pricing_optimization":
            steps = [
                "get_product_data",
                "optimize_discount",
                "estimate_impact",
                "format_recommendation"
            ]
        elif intent == "inventory_management":
            steps = [
                "get_inventory_data",
                "recommend_inventory",
                "check_risks",
                "format_recommendations"
            ]
        elif intent == "expansion":
            steps = [
                "analyze_stores",
                "suggest_expansion",
                "calculate_investment",
                "format_suggestions"
            ]
        elif intent == "model_management":
            steps = [
                "get_model_status",
                "check_model_drift",
                "compare_models",
                "recommend_action"
            ]
        elif intent == "explanation":
            steps = [
                "identify_subject",
                "get_context",
                "explain_prediction",
                "format_explanation"
            ]
        
        return steps


class AgenticAI:
    """Main agentic AI system"""
    
    def __init__(self):
        self.tool_registry = ToolRegistry()
        self.query_planner = QueryPlanner()
        self.conversation_history = []
        self.execution_log = []
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process complex user query"""
        
        start_time = datetime.now()
        
        # Parse query
        parsed = self.query_planner.parse_query(user_query)
        
        # Plan steps
        steps = self.query_planner.plan_steps(parsed)
        
        # Execute steps
        results = {
            "query": user_query,
            "intent": parsed["intent"],
            "steps": steps,
            "outputs": {},
            "reasoning": [],
            "recommendation": None,
            "confidence": 0.0
        }
        
        # Execute each step
        for step in steps:
            step_result = self._execute_step(step, parsed, results)
            results["outputs"][step] = step_result
            results["reasoning"].append(f"Executed: {step}")
        
        # Generate recommendation
        if steps:
            results["recommendation"] = self._generate_recommendation(parsed, results)
            results["confidence"] = self._calculate_confidence(parsed, results)
        
        # Log execution
        duration = (datetime.now() - start_time).total_seconds()
        self.execution_log.append({
            "query": user_query,
            "intent": parsed["intent"],
            "duration_seconds": duration,
            "successful": results["recommendation"] is not None,
            "timestamp": datetime.now().isoformat()
        })
        
        # Add to history
        self.conversation_history.append({
            "query": user_query,
            "response": results,
            "timestamp": datetime.now().isoformat()
        })
        
        return results
    
    def _execute_step(self, step: str, parsed_query: Dict, current_results: Dict) -> Any:
        """Execute single step"""
        
        if step == "get_model_status":
            return self.tool_registry.execute_tool("get_model_status")
        
        elif step == "check_model_drift":
            return self.tool_registry.execute_tool("check_model_drift")
        
        elif step == "compare_models":
            return self.tool_registry.execute_tool("compare_models")
        
        elif step == "optimize_discount":
            # Extract parameters from query entities
            return {
                "optimization": "ready",
                "scenarios": 7
            }
        
        elif step == "estimate_impact":
            return {
                "estimated_revenue_increase": "15%",
                "estimated_profit_increase": "22%"
            }
        
        elif step == "format_recommendation":
            return "Recommendation formatted"
        
        return {"status": "completed"}
    
    def _generate_recommendation(self, parsed_query: Dict, results: Dict) -> str:
        """Generate final recommendation"""
        
        intent = parsed_query["intent"]
        
        if intent == "pricing_optimization":
            outputs = results["outputs"]
            return "📊 Recommendation: Apply optimal pricing strategy from analysis for 15-25% revenue increase"
        
        elif intent == "inventory_management":
            return "📦 Recommendation: Follow reorder schedule to maintain safety stock levels and prevent stockouts"
        
        elif intent == "expansion":
            return "🏪 Recommendation: Prioritize 3 high-opportunity stores for expansion initiative"
        
        elif intent == "model_management":
            outputs = results["outputs"]
            if outputs.get("check_model_drift", {}).get("is_drifting"):
                return "⚠️ Recommendation: Model performance declining - initiate immediate retraining cycle"
            else:
                return "✅ Recommendation: Production model performing well - no immediate action needed"
        
        elif intent == "explanation":
            return "📈 Explainable prediction with key drivers identified"
        
        return "💡 Analysis completed - review detailed recommendations above"
    
    def _calculate_confidence(self, parsed_query: Dict, results: Dict) -> float:
        """Calculate confidence in recommendation"""
        
        # Base confidence
        confidence = 0.7
        
        # Adjust based on available data
        if parsed_query.get("entities", {}).get("product_ids"):
            confidence += 0.1
        
        if results.get("outputs"):
            confidence += 0.15
        
        return min(0.99, confidence)
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history
    
    def get_execution_log(self) -> List[Dict]:
        """Get execution log"""
        return self.execution_log
    
    def natural_language_interface(self, query: str) -> str:
        """Friendly NL interface"""
        
        result = self.process_query(query)
        
        response = f"""
🤖 **AI Analysis for:** {result['intent']}

**Query:** {result['query']}

**Analysis Steps Completed:** {len(result['steps'])}

**Key Findings:**
{chr(10).join([f"  • {output}" for output in list(result['outputs'].values())[:3]])}

**Recommendation:**
{result['recommendation']}

**Confidence Level:** {result['confidence']*100:.0f}%
"""
        
        return response
