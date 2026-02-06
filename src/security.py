"""
Security & Enterprise Layer
Features:
- Role-based access control (RBAC)
- Comprehensive audit trails
- Model rollback capabilities
- API rate limiting
- User session management
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
from enum import Enum

import sys
sys.path.append(str(Path(__file__).parent))
from mlops import AuditLog, ModelRegistry


class UserRole(Enum):
    """Available user roles"""
    ADMIN = "admin"              # Full access
    MANAGER = "manager"          # Can view, make decisions
    ANALYST = "analyst"          # Can view and analyze
    VIEWER = "viewer"            # Read-only access


class Permission(Enum):
    """Available permissions"""
    VIEW_DASHBOARD = "view_dashboard"
    VIEW_MODELS = "view_models"
    RETRAIN_MODEL = "retrain_model"
    DEPLOY_MODEL = "deploy_model"
    ROLLBACK_MODEL = "rollback_model"
    VIEW_AUDIT_LOG = "view_audit_log"
    MODIFY_SETTINGS = "modify_settings"
    MANAGE_USERS = "manage_users"


# Role-to-Permission mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        Permission.VIEW_DASHBOARD,
        Permission.VIEW_MODELS,
        Permission.RETRAIN_MODEL,
        Permission.DEPLOY_MODEL,
        Permission.ROLLBACK_MODEL,
        Permission.VIEW_AUDIT_LOG,
        Permission.MODIFY_SETTINGS,
        Permission.MANAGE_USERS
    ],
    UserRole.MANAGER: [
        Permission.VIEW_DASHBOARD,
        Permission.VIEW_MODELS,
        Permission.RETRAIN_MODEL,
        Permission.DEPLOY_MODEL,
        Permission.VIEW_AUDIT_LOG
    ],
    UserRole.ANALYST: [
        Permission.VIEW_DASHBOARD,
        Permission.VIEW_MODELS,
        Permission.VIEW_AUDIT_LOG
    ],
    UserRole.VIEWER: [
        Permission.VIEW_DASHBOARD,
        Permission.VIEW_MODELS
    ]
}


class UserManager:
    """Manage users and authentication"""
    
    USERS_PATH = Path(__file__).parent.parent / "models" / "users.json"
    SESSIONS_PATH = Path(__file__).parent.parent / "models" / "sessions.json"
    
    @staticmethod
    def _ensure_users():
        """Ensure users file exists"""
        if not UserManager.USERS_PATH.exists():
            UserManager.USERS_PATH.write_text(json.dumps({
                "users": [
                    {
                        "username": "admin",
                        "password": UserManager._hash_password("admin123"),
                        "email": "admin@company.com",
                        "role": "admin",
                        "created_at": datetime.now().isoformat(),
                        "active": True
                    }
                ]
            }, indent=2))
    
    @staticmethod
    def _hash_password(password: str) -> str:
        """Hash password"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def create_user(
        username: str,
        password: str,
        email: str,
        role: str = "analyst"
    ) -> Dict[str, Any]:
        """Create new user"""
        
        UserManager._ensure_users()
        users_data = json.loads(UserManager.USERS_PATH.read_text())
        
        # Check if user exists
        if any(u["username"] == username for u in users_data["users"]):
            return {"error": "User already exists"}
        
        new_user = {
            "username": username,
            "password": UserManager._hash_password(password),
            "email": email,
            "role": role,
            "created_at": datetime.now().isoformat(),
            "active": True
        }
        
        users_data["users"].append(new_user)
        UserManager.USERS_PATH.write_text(json.dumps(users_data, indent=2))
        
        AuditLog.log_action(
            action="user_created",
            details=f"User {username} created with role {role}",
            user="system"
        )
        
        return {
            "username": username,
            "role": role,
            "created_at": new_user["created_at"]
        }
    
    @staticmethod
    def authenticate(username: str, password: str) -> Optional[Dict]:
        """Authenticate user"""
        
        UserManager._ensure_users()
        users_data = json.loads(UserManager.USERS_PATH.read_text())
        
        user = next((u for u in users_data["users"] if u["username"] == username), None)
        
        if not user or user["password"] != UserManager._hash_password(password):
            AuditLog.log_action(
                action="auth_failed",
                details=f"Failed login attempt for {username}",
                severity="warning"
            )
            return None
        
        if not user["active"]:
            return None
        
        return {
            "username": user["username"],
            "email": user["email"],
            "role": user["role"]
        }
    
    @staticmethod
    def get_user_role(username: str) -> Optional[UserRole]:
        """Get user role"""
        
        UserManager._ensure_users()
        users_data = json.loads(UserManager.USERS_PATH.read_text())
        
        user = next((u for u in users_data["users"] if u["username"] == username), None)
        return UserRole(user["role"]) if user else None


class AccessControl:
    """Manage access control and permissions"""
    
    @staticmethod
    def has_permission(username: str, permission: Permission) -> bool:
        """Check if user has permission"""
        
        role = UserManager.get_user_role(username)
        if not role:
            return False
        
        allowed_permissions = ROLE_PERMISSIONS.get(role, [])
        return permission in allowed_permissions
    
    @staticmethod
    def require_permission(permission: Permission):
        """Decorator to require permission"""
        
        def decorator(func):
            def wrapper(*args, **kwargs):
                username = kwargs.get("username") or args[0] if args else None
                
                if not username or not AccessControl.has_permission(username, permission):
                    raise PermissionError(f"User lacks {permission} permission")
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    @staticmethod
    def get_user_permissions(username: str) -> List[str]:
        """Get all permissions for user"""
        
        role = UserManager.get_user_role(username)
        if not role:
            return []
        
        return [p.value for p in ROLE_PERMISSIONS.get(role, [])]


class SessionManager:
    """Manage user sessions"""
    
    SESSION_TIMEOUT_MINUTES = 30
    
    @staticmethod
    def create_session(username: str) -> str:
        """Create user session"""
        
        session_id = hashlib.sha256(
            f"{username}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        session_data = {
            "session_id": session_id,
            "username": username,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "active": True
        }
        
        if not SessionManager._get_sessions_path().exists():
            SessionManager._get_sessions_path().write_text(json.dumps({"sessions": []}))
        
        sessions = json.loads(SessionManager._get_sessions_path().read_text())
        sessions["sessions"].append(session_data)
        SessionManager._get_sessions_path().write_text(json.dumps(sessions, indent=2))
        
        return session_id
    
    @staticmethod
    def validate_session(session_id: str) -> bool:
        """Validate session is active and not expired"""
        
        if not SessionManager._get_sessions_path().exists():
            return False
        
        sessions = json.loads(SessionManager._get_sessions_path().read_text())
        session = next((s for s in sessions["sessions"] if s["session_id"] == session_id), None)
        
        if not session or not session["active"]:
            return False
        
        # Check timeout
        last_activity = datetime.fromisoformat(session["last_activity"])
        if (datetime.now() - last_activity).seconds > SessionManager.SESSION_TIMEOUT_MINUTES * 60:
            session["active"] = False
            SessionManager._update_sessions(sessions)
            return False
        
        # Update last activity
        session["last_activity"] = datetime.now().isoformat()
        SessionManager._update_sessions(sessions)
        
        return True
    
    @staticmethod
    def _get_sessions_path():
        return Path(__file__).parent.parent / "models" / "sessions.json"
    
    @staticmethod
    def _update_sessions(sessions):
        SessionManager._get_sessions_path().write_text(json.dumps(sessions, indent=2))


class ModelRollback:
    """Handle model rollback operations"""
    
    @staticmethod
    def list_rollback_options() -> List[Dict]:
        """List available models for rollback"""
        
        history = ModelRegistry.get_model_history()
        
        return [
            {
                "version": m["version"],
                "model": m["name"],
                "metrics": m["metrics"],
                "registered_at": m["registered_at"],
                "stage": m["stage"],
                "available_for_rollback": m["stage"] in ["archived", "staging"]
            }
            for m in history[:5]
        ]
    
    @staticmethod
    def perform_rollback(model_id: str, username: str) -> Dict[str, Any]:
        """Perform rollback to previous model"""
        
        if not AccessControl.has_permission(username, Permission.ROLLBACK_MODEL):
            raise PermissionError("User lacks rollback permission")
        
        success = ModelRegistry.rollback_production()
        
        if success:
            AuditLog.log_action(
                action="model_rollback",
                details=f"Rollback performed by {username}",
                user=username,
                severity="warning"
            )
        
        return {
            "success": success,
            "performed_by": username,
            "timestamp": datetime.now().isoformat()
        }


class AuditTrail:
    """Enhanced audit trail management"""
    
    @staticmethod
    def get_audit_trail(
        action_filter: str = None,
        user_filter: str = None,
        severity_filter: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get filtered audit trail"""
        
        logs = AuditLog.get_audit_trail(limit=limit*2)
        
        # Apply filters
        if action_filter:
            logs = [l for l in logs if l["action"] == action_filter]
        if user_filter:
            logs = [l for l in logs if l["user"] == user_filter]
        if severity_filter:
            logs = [l for l in logs if l["severity"] == severity_filter]
        
        return logs[:limit]
    
    @staticmethod
    def get_compliance_report(days: int = 30) -> Dict[str, Any]:
        """Generate compliance report"""
        
        logs = AuditLog.get_audit_trail(limit=10000)
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_logs = [
            l for l in logs 
            if datetime.fromisoformat(l["timestamp"]) >= cutoff_date
        ]
        
        report = {
            "period_days": days,
            "total_actions": len(recent_logs),
            "actions_by_type": {},
            "actions_by_user": {},
            "critical_actions": [],
            "failed_auths": 0
        }
        
        for log in recent_logs:
            # Count by type
            action = log["action"]
            report["actions_by_type"][action] = report["actions_by_type"].get(action, 0) + 1
            
            # Count by user
            user = log["user"]
            report["actions_by_user"][user] = report["actions_by_user"].get(user, 0) + 1
            
            # Track critical actions
            if log["severity"] == "critical":
                report["critical_actions"].append(log)
            
            if action == "auth_failed":
                report["failed_auths"] += 1
        
        return report


class APIRateLimiter:
    """Rate limiting for API access"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.request_history = {}
    
    def check_rate_limit(self, client_id: str) -> Tuple[bool, Dict]:
        """Check if client exceeded rate limit"""
        
        now = datetime.now()
        
        if client_id not in self.request_history:
            self.request_history[client_id] = []
        
        # Remove old requests (older than 1 minute)
        cutoff_time = now - timedelta(minutes=1)
        self.request_history[client_id] = [
            req_time for req_time in self.request_history[client_id]
            if req_time > cutoff_time
        ]
        
        # Check limit
        if len(self.request_history[client_id]) >= self.requests_per_minute:
            return False, {
                "limited": True,
                "current_requests": len(self.request_history[client_id]),
                "limit": self.requests_per_minute,
                "retry_after_seconds": 60
            }
        
        # Record this request
        self.request_history[client_id].append(now)
        
        return True, {
            "limited": False,
            "current_requests": len(self.request_history[client_id]),
            "remaining": self.requests_per_minute - len(self.request_history[client_id])
        }


from typing import Tuple

class EnterpriseSecurityManager:
    """Central security management"""
    
    def __init__(self):
        self.rate_limiter = APIRateLimiter(requests_per_minute=1000)
    
    def authenticate_and_authorize(
        self,
        username: str,
        password: str,
        required_permission: Permission = None,
        client_id: str = None
    ) -> Dict[str, Any]:
        """Complete auth flow"""
        
        # Check rate limit
        if client_id:
            allowed, limit_info = self.rate_limiter.check_rate_limit(client_id)
            if not allowed:
                return {"success": False, "error": "Rate limit exceeded", "limit_info": limit_info}
        
        # Authenticate
        user = UserManager.authenticate(username, password)
        if not user:
            return {"success": False, "error": "Authentication failed"}
        
        # Check permission if required
        if required_permission:
            if not AccessControl.has_permission(username, required_permission):
                return {"success": False, "error": "Insufficient permissions"}
        
        # Create session
        session_id = SessionManager.create_session(username)
        
        AuditLog.log_action(
            action="auth_success",
            details=f"Successful login for {username}",
            user=username
        )
        
        return {
            "success": True,
            "session_id": session_id,
            "username": username,
            "role": user["role"],
            "permissions": AccessControl.get_user_permissions(username)
        }
