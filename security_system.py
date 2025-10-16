"""
Production-Grade Security System
Provides input sanitization, authentication checks, and security validations.
"""

import re
import hashlib
import hmac
import secrets
import logging
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import base64
import urllib.parse
from pathlib import Path
import httpx

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityViolation:
    """Security violation record"""
    violation_type: str
    severity: SecurityLevel
    details: str
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    timestamp: datetime = None
    blocked: bool = False
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class SecurityContext:
    """Security context for requests"""
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class InputSanitizer:
    """
    Comprehensive input sanitization system.
    """
    
    def __init__(self):
        self.dangerous_patterns = [
            # SQL Injection patterns
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
            r"(--|\#|\/\*|\*\/)",
            r"(\b(SCRIPT|JAVASCRIPT|VBSCRIPT)\b)",
            
            # XSS patterns
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"vbscript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>",
            
            # Path traversal patterns
            r"\.\.\/",
            r"\.\.\\",
            r"\/etc\/passwd",
            r"\/etc\/shadow",
            r"C:\\Windows\\System32",
            
            # Command injection patterns
            r"[;&|`$]",
            r"\b(cat|ls|dir|type|more|less|head|tail|grep|find|awk|sed)\b",
            r"\b(ping|nslookup|tracert|netstat|ps|kill|killall)\b",
            
            # LDAP injection patterns
            r"[()=*!&|]",
            
            # NoSQL injection patterns
            r"\$where",
            r"\$ne",
            r"\$gt",
            r"\$lt",
            r"\$regex",
        ]
        
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.dangerous_patterns]
        
        # File extension blacklist
        self.dangerous_extensions = {
            '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js', '.jar',
            '.php', '.asp', '.aspx', '.jsp', '.py', '.pl', '.sh', '.ps1'
        }
        
        # MIME type blacklist
        self.dangerous_mime_types = {
            'application/x-executable',
            'application/x-msdownload',
            'application/x-msdos-program',
            'application/x-winexe',
            'application/x-javascript',
            'application/javascript',
            'text/javascript'
        }
    
    def sanitize_string(self, input_string: str, max_length: int = 1000) -> str:
        """Sanitize string input"""
        if not isinstance(input_string, str):
            return str(input_string)
        
        # Truncate if too long
        if len(input_string) > max_length:
            input_string = input_string[:max_length]
        
        # Remove null bytes
        input_string = input_string.replace('\x00', '')
        
        # Remove control characters except newlines and tabs
        input_string = ''.join(char for char in input_string 
                              if ord(char) >= 32 or char in '\n\t')
        
        # HTML encode dangerous characters
        dangerous_chars = {
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;',
            '&': '&amp;',
            '/': '&#x2F;'
        }
        
        for char, replacement in dangerous_chars.items():
            input_string = input_string.replace(char, replacement)
        
        return input_string.strip()
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename input"""
        if not filename:
            return "unnamed_file"
        
        # Remove path components
        filename = Path(filename).name
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove control characters
        filename = ''.join(char for char in filename if ord(char) >= 32)
        
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name[:250] + ('.' + ext if ext else '')
        
        # Check for dangerous extensions
        file_ext = Path(filename).suffix.lower()
        if file_ext in self.dangerous_extensions:
            filename = filename.replace(file_ext, '.txt')
        
        return filename or "unnamed_file"
    
    def sanitize_json(self, json_data: Union[str, dict]) -> dict:
        """Sanitize JSON input"""
        if isinstance(json_data, str):
            try:
                json_data = json.loads(json_data)
            except json.JSONDecodeError:
                return {}
        
        if not isinstance(json_data, dict):
            return {}
        
        sanitized = {}
        for key, value in json_data.items():
            # Sanitize key
            sanitized_key = self.sanitize_string(str(key), 100)
            
            # Sanitize value based on type
            if isinstance(value, str):
                sanitized[sanitized_key] = self.sanitize_string(value)
            elif isinstance(value, (int, float, bool)):
                sanitized[sanitized_key] = value
            elif isinstance(value, list):
                sanitized[sanitized_key] = [
                    self.sanitize_string(str(item)) if isinstance(item, str) else item
                    for item in value[:100]  # Limit list size
                ]
            elif isinstance(value, dict):
                sanitized[sanitized_key] = self.sanitize_json(value)
            else:
                sanitized[sanitized_key] = self.sanitize_string(str(value))
        
        return sanitized
    
    def detect_malicious_patterns(self, input_string: str) -> List[SecurityViolation]:
        """Detect malicious patterns in input"""
        violations = []
        
        if not isinstance(input_string, str):
            return violations
        
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.search(input_string):
                violation_type = self._get_violation_type(i)
                severity = self._get_violation_severity(violation_type)
                
                violations.append(SecurityViolation(
                    violation_type=violation_type,
                    severity=severity,
                    details=f"Detected {violation_type} pattern in input",
                    blocked=severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]
                ))
        
        return violations
    
    def _get_violation_type(self, pattern_index: int) -> str:
        """Get violation type based on pattern index"""
        if pattern_index < 4:
            return "sql_injection"
        elif pattern_index < 10:
            return "xss"
        elif pattern_index < 15:
            return "path_traversal"
        elif pattern_index < 18:
            return "command_injection"
        elif pattern_index < 20:
            return "ldap_injection"
        else:
            return "nosql_injection"
    
    def _get_violation_severity(self, violation_type: str) -> SecurityLevel:
        """Get severity level for violation type"""
        severity_map = {
            "sql_injection": SecurityLevel.CRITICAL,
            "xss": SecurityLevel.HIGH,
            "path_traversal": SecurityLevel.HIGH,
            "command_injection": SecurityLevel.CRITICAL,
            "ldap_injection": SecurityLevel.HIGH,
            "nosql_injection": SecurityLevel.HIGH
        }
        return severity_map.get(violation_type, SecurityLevel.MEDIUM)

class AuthenticationValidator:
    """
    Authentication and authorization validation system.
    """
    
    def __init__(self):
        self.session_timeout = 3600  # 1 hour
        self.max_login_attempts = 5
        self.lockout_duration = 900  # 15 minutes
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.failed_attempts: Dict[str, List[datetime]] = {}
    
    async def validate_user_session(self, user_id: str, session_token: str) -> Tuple[bool, str]:
        """Validate user session"""
        if not user_id or not session_token:
            return False, "Missing user ID or session token"

        # 1) Try Supabase JWT validation (preferred in production)
        try:
            supabase_url = os.environ.get("SUPABASE_URL")
            # apikey header required by Supabase Auth endpoints
            supabase_api_key = (
                os.environ.get("SUPABASE_ANON_KEY")
                or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
                or os.environ.get("SUPABASE_SERVICE_KEY")
                or os.environ.get("SUPABASE_KEY")
            )
            if supabase_url and supabase_api_key:
                headers = {
                    "Authorization": f"Bearer {session_token}",
                    "apikey": supabase_api_key,
                }
                # Async HTTP call to verify the user from JWT
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"{supabase_url}/auth/v1/user", headers=headers, timeout=5.0)
                    if resp.status_code == 200:
                        data = resp.json() or {}
                        token_user_id = data.get("id") or (data.get("user") or {}).get("id")
                        if token_user_id == user_id:
                            return True, "Session valid"
                        else:
                            return False, "Token user mismatch"
                    # For 401/403 keep falling back to in-memory session check
        except Exception:
            # Network or parse error; fall back to in-memory session logic
            pass

        # 2) Fallback to in-memory sessions (dev/testing compatibility)
        if user_id not in self.active_sessions:
            return False, "No active session found"

        session_data = self.active_sessions[user_id]

        # Check session token
        if session_data.get('token') != session_token:
            return False, "Invalid session token"

        # Check session expiry
        if datetime.utcnow() > session_data.get('expires_at', datetime.min):
            del self.active_sessions[user_id]
            return False, "Session expired"

        # Update last activity
        session_data['last_activity'] = datetime.utcnow()

        return True, "Session valid"
    
    def create_user_session(self, user_id: str, additional_data: Dict[str, Any] = None) -> str:
        """Create new user session"""
        session_token = self._generate_session_token()
        expires_at = datetime.utcnow() + timedelta(seconds=self.session_timeout)
        
        self.active_sessions[user_id] = {
            'token': session_token,
            'created_at': datetime.utcnow(),
            'expires_at': expires_at,
            'last_activity': datetime.utcnow(),
            'ip_address': additional_data.get('ip_address') if additional_data else None,
            'user_agent': additional_data.get('user_agent') if additional_data else None
        }
        
        return session_token
    
    def revoke_user_session(self, user_id: str) -> bool:
        """Revoke user session"""
        if user_id in self.active_sessions:
            del self.active_sessions[user_id]
            return True
        return False
    
    def check_login_attempts(self, user_id: str) -> Tuple[bool, str]:
        """Check if user is locked out due to failed login attempts"""
        if user_id not in self.failed_attempts:
            return True, "No failed attempts"
        
        attempts = self.failed_attempts[user_id]
        current_time = datetime.utcnow()
        
        # Remove old attempts
        attempts = [attempt for attempt in attempts 
                   if current_time - attempt < timedelta(seconds=self.lockout_duration)]
        self.failed_attempts[user_id] = attempts
        
        if len(attempts) >= self.max_login_attempts:
            return False, f"Account locked due to {len(attempts)} failed attempts"
        
        return True, f"{len(attempts)} failed attempts"
    
    def record_failed_login(self, user_id: str) -> None:
        """Record failed login attempt"""
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = []
        
        self.failed_attempts[user_id].append(datetime.utcnow())
    
    def clear_failed_logins(self, user_id: str) -> None:
        """Clear failed login attempts"""
        if user_id in self.failed_attempts:
            del self.failed_attempts[user_id]
    
    def _generate_session_token(self) -> str:
        """Generate secure session token"""
        return secrets.token_urlsafe(32)
    
    def validate_api_key(self, api_key: str) -> Tuple[bool, Optional[str]]:
        """Validate API key"""
        if not api_key:
            return False, None
        
        # In production, this would check against a database
        # For now, we'll use a simple validation
        if len(api_key) < 32:
            return False, None
        
        # Check if API key is properly formatted
        try:
            base64.b64decode(api_key)
            return True, "api_user"  # Return user ID
        except:
            return False, None

class SecurityValidator:
    """
    Main security validation system.
    """
    
    def __init__(self):
        self.input_sanitizer = InputSanitizer()
        self.auth_validator = AuthenticationValidator()
        self.security_violations: List[SecurityViolation] = []
        self.rate_limits: Dict[str, List[datetime]] = {}
    
    async def validate_request(self, request_data: Dict[str, Any], 
                        security_context: SecurityContext) -> Tuple[bool, List[SecurityViolation]]:
        """Validate entire request for security issues"""
        violations = []
        
        # Rate limiting check
        if not self._check_rate_limit(security_context):
            violations.append(SecurityViolation(
                violation_type="rate_limit_exceeded",
                severity=SecurityLevel.MEDIUM,
                details="Request rate limit exceeded",
                user_id=security_context.user_id,
                ip_address=security_context.ip_address
            ))
        
        # Input sanitization
        sanitized_data = self._sanitize_request_data(request_data, violations, security_context)
        
        # Authentication check - FIX: Added await
        if not await self._validate_authentication(request_data, security_context, violations):
            return False, violations
        
        # Store violations
        self.security_violations.extend(violations)
        
        return len(violations) == 0, violations
    
    def _check_rate_limit(self, security_context: SecurityContext) -> bool:
        """Check rate limiting"""
        if not security_context.ip_address:
            return True
        
        current_time = datetime.utcnow()
        window_start = current_time - timedelta(minutes=1)
        
        if security_context.ip_address not in self.rate_limits:
            self.rate_limits[security_context.ip_address] = []
        
        # Remove old requests
        self.rate_limits[security_context.ip_address] = [
            req_time for req_time in self.rate_limits[security_context.ip_address]
            if req_time > window_start
        ]
        
        # Check limit (100 requests per minute)
        if len(self.rate_limits[security_context.ip_address]) >= 100:
            return False
        
        # Add current request
        self.rate_limits[security_context.ip_address].append(current_time)
        return True
    
    def _sanitize_request_data(self, request_data: Dict[str, Any], 
                              violations: List[SecurityViolation],
                              security_context: SecurityContext) -> Dict[str, Any]:
        """Sanitize request data"""
        sanitized_data = {}
        
        for key, value in request_data.items():
            # Check for malicious patterns in key
            key_violations = self.input_sanitizer.detect_malicious_patterns(str(key))
            violations.extend(key_violations)
            
            # Sanitize key
            sanitized_key = self.input_sanitizer.sanitize_string(str(key))
            
            # Sanitize value
            if isinstance(value, str):
                # Check for malicious patterns
                value_violations = self.input_sanitizer.detect_malicious_patterns(value)
                violations.extend(value_violations)
                
                # Sanitize value
                sanitized_data[sanitized_key] = self.input_sanitizer.sanitize_string(value)
            
            elif isinstance(value, dict):
                # Recursively sanitize nested objects
                nested_violations = []
                sanitized_data[sanitized_key] = self._sanitize_request_data(
                    value, nested_violations, security_context
                )
                violations.extend(nested_violations)
            
            elif isinstance(value, list):
                # Sanitize list items
                sanitized_list = []
                for item in value:
                    if isinstance(item, str):
                        item_violations = self.input_sanitizer.detect_malicious_patterns(item)
                        violations.extend(item_violations)
                        sanitized_list.append(self.input_sanitizer.sanitize_string(item))
                    else:
                        sanitized_list.append(item)
                sanitized_data[sanitized_key] = sanitized_list
            
            else:
                sanitized_data[sanitized_key] = value
        
        return sanitized_data
    
    async def _validate_authentication(self, request_data: Dict[str, Any],
                                security_context: SecurityContext,
                                violations: List[SecurityViolation]) -> bool:
        """Validate authentication"""
        # Check if authentication is required
        if not self._requires_authentication(request_data):
            return True
        
        # Check session validation
        user_id = request_data.get('user_id')
        session_token = request_data.get('session_token')
        
        if not user_id or not session_token:
            violations.append(SecurityViolation(
                violation_type="missing_authentication",
                severity=SecurityLevel.HIGH,
                details="Missing user ID or session token",
                user_id=user_id,
                ip_address=security_context.ip_address
            ))
            return False
        
        # Validate session - FIX: Added await
        is_valid, message = await self.auth_validator.validate_user_session(user_id, session_token)
        if not is_valid:
            violations.append(SecurityViolation(
                violation_type="invalid_authentication",
                severity=SecurityLevel.HIGH,
                details=f"Authentication failed: {message}",
                user_id=user_id,
                ip_address=security_context.ip_address
            ))
            return False
        
        return True
    
    def _requires_authentication(self, request_data: Dict[str, Any]) -> bool:
        """Check if request requires authentication"""
        # Public endpoints that don't require authentication
        public_endpoints = [
            'health_check',
            'status',
            'login',
            'register',
            'public_data'
        ]
        
        endpoint = request_data.get('endpoint', '')
        return endpoint not in public_endpoints
    
    def validate_file_metadata(self, filename: str, file_size: int = 0, 
                              content_type: str = None) -> Tuple[bool, List[str]]:
        """
        Validate file metadata before processing.
        
        Args:
            filename: Name of the file
            file_size: Size of the file in bytes
            content_type: MIME type of the file
            
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        
        # Check file size (500MB limit)
        MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
        if file_size > MAX_FILE_SIZE:
            violations.append(f"File too large: {file_size / 1024 / 1024:.2f}MB (max: 500MB)")
        
        # Check file extension
        allowed_extensions = ['.xlsx', '.xls', '.csv', '.pdf', '.png', '.jpg', '.jpeg', 
                            '.gif', '.bmp', '.webp', '.svg', '.ods', '.zip', '.7z', '.rar']
        filename_lower = filename.lower()
        if not any(filename_lower.endswith(ext) for ext in allowed_extensions):
            violations.append(f"Invalid file type. Allowed extensions: {', '.join(allowed_extensions)}")
        
        # Check content type if provided
        if content_type:
            allowed_mimes = [
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',  # .xlsx
                'application/vnd.ms-excel',  # .xls
                'application/zip',  # .xlsx (also detected as zip)
                'text/csv',  # .csv
                'text/plain',  # .csv (sometimes detected as plain text)
                'application/pdf',  # .pdf
                'image/png',  # .png
                'image/jpeg',  # .jpg, .jpeg
                'image/gif',  # .gif
                'image/bmp',  # .bmp
                'image/webp',  # .webp
                'image/svg+xml',  # .svg
                'application/vnd.oasis.opendocument.spreadsheet',  # .ods
                'application/x-7z-compressed',  # .7z
                'application/x-rar-compressed',  # .rar
                'application/octet-stream'  # Generic binary (fallback)
            ]
            if content_type not in allowed_mimes:
                violations.append(f"Invalid content type: {content_type}")
        
        # Sanitize filename to prevent path traversal
        sanitized_filename = self.input_sanitizer.sanitize_string(filename)
        if '..' in filename or '/' in filename or '\\' in filename:
            violations.append("Filename contains invalid path characters")
        
        return len(violations) == 0, violations
    
    def get_security_statistics(self) -> Dict[str, Any]:
        """Get security statistics"""
        violation_counts = {}
        for violation in self.security_violations:
            violation_type = violation.violation_type
            if violation_type not in violation_counts:
                violation_counts[violation_type] = 0
            violation_counts[violation_type] += 1
        
        return {
            'total_violations': len(self.security_violations),
            'violation_types': violation_counts,
            'active_sessions': len(self.auth_validator.active_sessions),
            'rate_limited_ips': len(self.rate_limits),
            'failed_login_attempts': len(self.auth_validator.failed_attempts)
        }
    
    def cleanup_expired_data(self) -> int:
        """Clean up expired security data"""
        cleaned = 0
        
        # Clean up expired sessions
        current_time = datetime.utcnow()
        expired_sessions = [
            user_id for user_id, session_data in self.auth_validator.active_sessions.items()
            if current_time > session_data.get('expires_at', datetime.min)
        ]
        
        for user_id in expired_sessions:
            del self.auth_validator.active_sessions[user_id]
            cleaned += 1
        
        # Clean up old rate limit data
        window_start = current_time - timedelta(hours=1)
        for ip_address in list(self.rate_limits.keys()):
            self.rate_limits[ip_address] = [
                req_time for req_time in self.rate_limits[ip_address]
                if req_time > window_start
            ]
            if not self.rate_limits[ip_address]:
                del self.rate_limits[ip_address]
                cleaned += 1
        
        return cleaned

# Global security system instance
_global_security_system: Optional[SecurityValidator] = None

def get_global_security_system() -> SecurityValidator:
    """Get or create global security system"""
    global _global_security_system
    
    if _global_security_system is None:
        _global_security_system = SecurityValidator()
    
    return _global_security_system
