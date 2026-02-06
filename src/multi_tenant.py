"""
Multi-Tenant Support
Features:
- Multiple client support
- Data isolation per tenant
- White-label customization
- Tenant-level analytics
- Resource management
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd

import sys
sys.path.append(str(Path(__file__).parent))


class TenantManager:
    """Manage multiple tenants"""
    
    TENANTS_PATH = Path(__file__).parent.parent / "models" / "tenants.json"
    
    @staticmethod
    def _ensure_tenants():
        """Ensure tenants file exists"""
        if not TenantManager.TENANTS_PATH.exists():
            TenantManager.TENANTS_PATH.write_text(json.dumps({
                "tenants": [
                    {
                        "tenant_id": "default",
                        "tenant_name": "Default Organization",
                        "created_at": datetime.now().isoformat(),
                        "active": True,
                        "plan": "enterprise"
                    }
                ],
                "default_tenant": "default"
            }, indent=2))
    
    @staticmethod
    def create_tenant(
        tenant_name: str,
        admin_email: str,
        plan: str = "pro"  # starter, pro, enterprise
    ) -> Dict[str, Any]:
        """Create new tenant"""
        
        TenantManager._ensure_tenants()
        
        # Generate tenant ID
        import hashlib
        tenant_id = hashlib.sha256(tenant_name.encode()).hexdigest()[:8]
        
        tenants_data = json.loads(TenantManager.TENANTS_PATH.read_text())
        
        # Check if already exists
        if any(t["tenant_id"] == tenant_id for t in tenants_data["tenants"]):
            return {"error": "Tenant already exists"}
        
        new_tenant = {
            "tenant_id": tenant_id,
            "tenant_name": tenant_name,
            "admin_email": admin_email,
            "created_at": datetime.now().isoformat(),
            "active": True,
            "plan": plan,
            "features_enabled": TenantManager._get_plan_features(plan),
            "usage": {
                "monthly_api_calls": 0,
                "storage_gb": 0,
                "models_deployed": 0
            }
        }
        
        tenants_data["tenants"].append(new_tenant)
        TenantManager.TENANTS_PATH.write_text(json.dumps(tenants_data, indent=2))
        
        return new_tenant
    
    @staticmethod
    def _get_plan_features(plan: str) -> List[str]:
        """Get features for plan"""
        
        features = {
            "starter": [
                "basic_dashboard",
                "single_model",
                "monthly_retraining"
            ],
            "pro": [
                "advanced_dashboard",
                "multiple_models",
                "weekly_retraining",
                "basic_xai",
                "real_time_alerts"
            ],
            "enterprise": [
                "advanced_dashboard",
                "unlimited_models",
                "daily_retraining",
                "advanced_xai",
                "real_time_alerts",
                "api_access",
                "custom_integration",
                "dedicated_support",
                "multi_tenant_support",
                "white_label"
            ]
        }
        
        return features.get(plan, features["pro"])
    
    @staticmethod
    def get_tenant(tenant_id: str) -> Optional[Dict]:
        """Get tenant info"""
        
        TenantManager._ensure_tenants()
        tenants_data = json.loads(TenantManager.TENANTS_PATH.read_text())
        
        return next((t for t in tenants_data["tenants"] if t["tenant_id"] == tenant_id), None)
    
    @staticmethod
    def list_tenants() -> List[Dict]:
        """List all tenants"""
        
        TenantManager._ensure_tenants()
        tenants_data = json.loads(TenantManager.TENANTS_PATH.read_text())
        
        return tenants_data["tenants"]
    
    @staticmethod
    def update_tenant_usage(tenant_id: str, api_calls: int = 0, storage_gb: float = 0):
        """Update tenant usage metrics"""
        
        tenants_data = json.loads(TenantManager.TENANTS_PATH.read_text())
        
        for tenant in tenants_data["tenants"]:
            if tenant["tenant_id"] == tenant_id:
                tenant["usage"]["monthly_api_calls"] += api_calls
                tenant["usage"]["storage_gb"] += storage_gb
                break
        
        TenantManager.TENANTS_PATH.write_text(json.dumps(tenants_data, indent=2))


class DataIsolation:
    """Ensure data isolation between tenants"""
    
    DATA_ROOT = Path(__file__).parent.parent / "data" / "tenant_data"
    
    @staticmethod
    def get_tenant_data_path(tenant_id: str) -> Path:
        """Get data path for tenant"""
        
        path = DataIsolation.DATA_ROOT / tenant_id
        path.mkdir(parents=True, exist_ok=True)
        
        return path
    
    @staticmethod
    def get_tenant_dataset(tenant_id: str, dataset_type: str = "featured") -> Optional[pd.DataFrame]:
        """Get tenant-specific dataset"""
        
        data_path = DataIsolation.get_tenant_data_path(tenant_id)
        csv_path = data_path / f"{dataset_type}_dataset.csv"
        
        if csv_path.exists():
            return pd.read_csv(csv_path)
        
        return None
    
    @staticmethod
    def save_tenant_dataset(
        tenant_id: str,
        df: pd.DataFrame,
        dataset_type: str = "featured"
    ) -> bool:
        """Save tenant-specific dataset"""
        
        data_path = DataIsolation.get_tenant_data_path(tenant_id)
        csv_path = data_path / f"{dataset_type}_dataset.csv"
        
        df.to_csv(csv_path, index=False)
        
        return True
    
    @staticmethod
    def get_tenant_models_path(tenant_id: str) -> Path:
        """Get models path for tenant"""
        
        path = Path(__file__).parent.parent / "models" / "tenant_models" / tenant_id
        path.mkdir(parents=True, exist_ok=True)
        
        return path
    
    @staticmethod
    def ensure_data_isolation(tenant_id: str, user_tenant_id: str) -> bool:
        """Verify user can only access their tenant"""
        
        return tenant_id == user_tenant_id


class WhiteLabelManager:
    """White-label customization"""
    
    CONFIG_PATH = Path(__file__).parent.parent / "models" / "white_label.json"
    
    @staticmethod
    def _ensure_config():
        """Ensure config exists"""
        if not WhiteLabelManager.CONFIG_PATH.exists():
            WhiteLabelManager.CONFIG_PATH.write_text(json.dumps({
                "brands": {}
            }, indent=2))
    
    @staticmethod
    def create_brand(
        tenant_id: str,
        brand_name: str,
        logo_url: str = None,
        primary_color: str = "#1f77b4",
        secondary_color: str = "#ff7f0e"
    ) -> Dict[str, Any]:
        """Create white-label brand"""
        
        WhiteLabelManager._ensure_config()
        config = json.loads(WhiteLabelManager.CONFIG_PATH.read_text())
        
        brand = {
            "tenant_id": tenant_id,
            "brand_name": brand_name,
            "logo_url": logo_url,
            "primary_color": primary_color,
            "secondary_color": secondary_color,
            "created_at": datetime.now().isoformat()
        }
        
        config["brands"][tenant_id] = brand
        WhiteLabelManager.CONFIG_PATH.write_text(json.dumps(config, indent=2))
        
        return brand
    
    @staticmethod
    def get_brand(tenant_id: str) -> Optional[Dict]:
        """Get brand configuration"""
        
        WhiteLabelManager._ensure_config()
        config = json.loads(WhiteLabelManager.CONFIG_PATH.read_text())
        
        return config["brands"].get(tenant_id)
    
    @staticmethod
    def update_brand(
        tenant_id: str,
        **kwargs
    ) -> Dict:
        """Update brand configuration"""
        
        config = json.loads(WhiteLabelManager.CONFIG_PATH.read_text())
        
        if tenant_id in config["brands"]:
            config["brands"][tenant_id].update(kwargs)
            WhiteLabelManager.CONFIG_PATH.write_text(json.dumps(config, indent=2))
        
        return config["brands"].get(tenant_id)


class TenantAnalytics:
    """Analytics per tenant"""
    
    ANALYTICS_PATH = Path(__file__).parent.parent / "models" / "tenant_analytics.json"
    
    @staticmethod
    def _ensure_analytics():
        """Ensure analytics file exists"""
        if not TenantAnalytics.ANALYTICS_PATH.exists():
            TenantAnalytics.ANALYTICS_PATH.write_text(json.dumps({
                "analytics": []
            }, indent=2))
    
    @staticmethod
    def record_event(
        tenant_id: str,
        event_type: str,
        details: Dict = None
    ):
        """Record tenant event"""
        
        TenantAnalytics._ensure_analytics()
        
        event = {
            "tenant_id": tenant_id,
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        
        analytics = json.loads(TenantAnalytics.ANALYTICS_PATH.read_text())
        analytics["analytics"].append(event)
        
        TenantAnalytics.ANALYTICS_PATH.write_text(json.dumps(analytics, indent=2))
    
    @staticmethod
    def get_tenant_analytics(tenant_id: str, days: int = 30) -> Dict[str, Any]:
        """Get analytics for tenant"""
        
        TenantAnalytics._ensure_analytics()
        analytics = json.loads(TenantAnalytics.ANALYTICS_PATH.read_text())
        
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=days)
        
        tenant_events = [
            e for e in analytics["analytics"]
            if e["tenant_id"] == tenant_id and 
            datetime.fromisoformat(e["timestamp"]) >= cutoff_date
        ]
        
        # Aggregate stats
        stats = {
            "total_events": len(tenant_events),
            "events_by_type": {},
            "last_activity": None
        }
        
        for event in tenant_events:
            event_type = event["event_type"]
            stats["events_by_type"][event_type] = stats["events_by_type"].get(event_type, 0) + 1
        
        if tenant_events:
            stats["last_activity"] = max(e["timestamp"] for e in tenant_events)
        
        return stats


class ResourceQuota:
    """Manage resource quotas per tenant"""
    
    QUOTAS_PATH = Path(__file__).parent.parent / "models" / "quotas.json"
    
    @staticmethod
    def _ensure_quotas():
        """Ensure quotas file exists"""
        
        if not ResourceQuota.QUOTAS_PATH.exists():
            ResourceQuota.QUOTAS_PATH.write_text(json.dumps({
                "quotas": {}
            }, indent=2))
    
    @staticmethod
    def set_quota(
        tenant_id: str,
        quota_type: str,  # "api_calls", "storage_gb", "models"
        limit: int
    ):
        """Set quota for tenant"""
        
        ResourceQuota._ensure_quotas()
        quotas = json.loads(ResourceQuota.QUOTAS_PATH.read_text())
        
        if tenant_id not in quotas["quotas"]:
            quotas["quotas"][tenant_id] = {}
        
        quotas["quotas"][tenant_id][quota_type] = {
            "limit": limit,
            "current": 0,
            "reset_date": datetime.now().isoformat()
        }
        
        ResourceQuota.QUOTAS_PATH.write_text(json.dumps(quotas, indent=2))
    
    @staticmethod
    def check_quota(tenant_id: str, quota_type: str) -> bool:
        """Check if tenant has available quota"""
        
        ResourceQuota._ensure_quotas()
        quotas = json.loads(ResourceQuota.QUOTAS_PATH.read_text())
        
        if tenant_id not in quotas["quotas"]:
            return True  # No quota set = unlimited
        
        tenant_quota = quotas["quotas"].get(tenant_id, {})
        quota = tenant_quota.get(quota_type)
        
        if not quota:
            return True
        
        return quota["current"] < quota["limit"]
    
    @staticmethod
    def increment_quota(tenant_id: str, quota_type: str):
        """Increment quota usage"""
        
        quotas = json.loads(ResourceQuota.QUOTAS_PATH.read_text())
        
        if tenant_id in quotas["quotas"] and quota_type in quotas["quotas"][tenant_id]:
            quotas["quotas"][tenant_id][quota_type]["current"] += 1
            ResourceQuota.QUOTAS_PATH.write_text(json.dumps(quotas, indent=2))
    
    @staticmethod
    def get_usage(tenant_id: str) -> Dict[str, Dict]:
        """Get quota usage for tenant"""
        
        ResourceQuota._ensure_quotas()
        quotas = json.loads(ResourceQuota.QUOTAS_PATH.read_text())
        
        return quotas["quotas"].get(tenant_id, {})


class TenantBillingManager:
    """Handle billing and payments"""
    
    BILLING_PATH = Path(__file__).parent.parent / "models" / "billing.json"
    
    @staticmethod
    def _ensure_billing():
        """Ensure billing file exists"""
        if not TenantBillingManager.BILLING_PATH.exists():
            TenantBillingManager.BILLING_PATH.write_text(json.dumps({
                "invoices": []
            }, indent=2))
    
    @staticmethod
    def generate_invoice(
        tenant_id: str,
        plan: str,
        usage_charges: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate invoice for tenant"""
        
        TenantBillingManager._ensure_billing()
        
        # Base prices
        plan_prices = {
            "starter": 99,
            "pro": 299,
            "enterprise": 999
        }
        
        base_price = plan_prices.get(plan, 299)
        total_usage = sum(usage_charges.values())
        
        invoice = {
            "invoice_id": f"inv_{tenant_id}_{datetime.now().strftime('%Y%m%d')}",
            "tenant_id": tenant_id,
            "plan": plan,
            "base_price": base_price,
            "usage_charges": usage_charges,
            "total_usage": total_usage,
            "total_amount": base_price + total_usage,
            "issued_at": datetime.now().isoformat(),
            "due_date": (datetime.now() + timedelta(days=30)).isoformat(),
            "status": "unpaid"
        }
        
        billing = json.loads(TenantBillingManager.BILLING_PATH.read_text())
        billing["invoices"].append(invoice)
        TenantBillingManager.BILLING_PATH.write_text(json.dumps(billing, indent=2))
        
        return invoice
    
    @staticmethod
    def get_invoices(tenant_id: str) -> List[Dict]:
        """Get invoices for tenant"""
        
        TenantBillingManager._ensure_billing()
        billing = json.loads(TenantBillingManager.BILLING_PATH.read_text())
        
        return [inv for inv in billing["invoices"] if inv["tenant_id"] == tenant_id]


from datetime import timedelta
