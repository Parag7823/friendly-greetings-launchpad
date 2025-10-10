"""
Unit Tests for Security System

Tests:
- Input sanitization (SQL injection, XSS, path traversal)
- Session validation
- Rate limiting
- Authentication checks
- Malicious pattern detection
"""

import pytest
import asyncio
import sys
import os
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

# Add parent directory to path to import security_system
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from security_system import (
    InputSanitizer,
    AuthenticationValidator,
    SecurityValidator,
    SecurityContext,
    SecurityLevel,
    SecurityViolation
)


class TestInputSanitizer:
    """Test input sanitization functionality"""
    
    def setup_method(self):
        self.sanitizer = InputSanitizer()
    
    def test_sanitize_normal_string(self):
        """Should not modify normal strings"""
        input_str = "Hello World 123"
        result = self.sanitizer.sanitize_string(input_str)
        assert result == "Hello World 123"
    
    def test_sanitize_html_characters(self):
        """Should escape HTML characters"""
        input_str = "<script>alert('xss')</script>"
        result = self.sanitizer.sanitize_string(input_str)
        assert "<script>" not in result
        assert "&lt;script&gt;" in result
    
    def test_sanitize_sql_injection(self):
        """Should escape SQL injection attempts"""
        input_str = "'; DROP TABLE users; --"
        result = self.sanitizer.sanitize_string(input_str)
        # Should escape dangerous characters
        assert "'" not in result or "&#x27;" in result
    
    def test_sanitize_null_bytes(self):
        """Should remove null bytes"""
        input_str = "test\x00data"
        result = self.sanitizer.sanitize_string(input_str)
        assert "\x00" not in result
        assert result == "testdata"
    
    def test_sanitize_control_characters(self):
        """Should remove control characters except newlines and tabs"""
        input_str = "test\x01\x02\x03data"
        result = self.sanitizer.sanitize_string(input_str)
        assert "\x01" not in result
        assert result == "testdata"
    
    def test_sanitize_preserves_newlines_and_tabs(self):
        """Should preserve newlines and tabs"""
        input_str = "line1\nline2\tcolumn"
        result = self.sanitizer.sanitize_string(input_str)
        assert "\n" in result
        assert "\t" in result
    
    def test_sanitize_truncates_long_strings(self):
        """Should truncate strings exceeding max length"""
        input_str = "x" * 2000
        result = self.sanitizer.sanitize_string(input_str, max_length=1000)
        assert len(result) == 1000
    
    def test_sanitize_filename(self):
        """Should sanitize filenames"""
        filename = "test<>file:name.xlsx"
        result = self.sanitizer.sanitize_filename(filename)
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
    
    def test_sanitize_filename_removes_path(self):
        """Should remove path components from filename"""
        filename = "/path/to/file.xlsx"
        result = self.sanitizer.sanitize_filename(filename)
        assert result == "file.xlsx"
    
    def test_sanitize_filename_dangerous_extension(self):
        """Should replace dangerous file extensions"""
        filename = "malicious.exe"
        result = self.sanitizer.sanitize_filename(filename)
        assert ".exe" not in result
        assert ".txt" in result
    
    def test_detect_sql_injection(self):
        """Should detect SQL injection patterns"""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "1 UNION SELECT * FROM users"
        ]
        
        for input_str in malicious_inputs:
            violations = self.sanitizer.detect_malicious_patterns(input_str)
            assert len(violations) > 0
            assert any(v.violation_type == "sql_injection" for v in violations)
    
    def test_detect_xss(self):
        """Should detect XSS patterns"""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert(1)",
            "<iframe src='evil.com'>",
            "onerror=alert(1)"
        ]
        
        for input_str in malicious_inputs:
            violations = self.sanitizer.detect_malicious_patterns(input_str)
            assert len(violations) > 0
            assert any(v.violation_type == "xss" for v in violations)
    
    def test_detect_path_traversal(self):
        """Should detect path traversal patterns"""
        malicious_inputs = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "/etc/shadow"
        ]
        
        for input_str in malicious_inputs:
            violations = self.sanitizer.detect_malicious_patterns(input_str)
            assert len(violations) > 0
            assert any(v.violation_type == "path_traversal" for v in violations)
    
    def test_detect_command_injection(self):
        """Should detect command injection patterns"""
        malicious_inputs = [
            "; rm -rf /",
            "| cat /etc/passwd",
            "& ping evil.com"
        ]
        
        for input_str in malicious_inputs:
            violations = self.sanitizer.detect_malicious_patterns(input_str)
            assert len(violations) > 0
    
    def test_no_false_positives_on_normal_data(self):
        """Should not flag normal data as malicious"""
        normal_inputs = [
            "John Doe",
            "test@example.com",
            "2024-01-15",
            "Product description with & symbol"
        ]
        
        for input_str in normal_inputs:
            violations = self.sanitizer.detect_malicious_patterns(input_str)
            # Should have minimal or no violations for normal data
            critical_violations = [v for v in violations if v.severity == SecurityLevel.CRITICAL]
            assert len(critical_violations) == 0


class TestAuthenticationValidator:
    """Test authentication and session validation"""
    
    def setup_method(self):
        self.auth_validator = AuthenticationValidator()
    
    @pytest.mark.asyncio
    async def test_create_session(self):
        """Should create valid session"""
        user_id = "user-123"
        session_token = self.auth_validator.create_user_session(user_id)
        
        assert session_token is not None
        assert len(session_token) > 0
        assert user_id in self.auth_validator.active_sessions
    
    @pytest.mark.asyncio
    async def test_validate_valid_session(self):
        """Should validate correct session"""
        user_id = "user-123"
        session_token = self.auth_validator.create_user_session(user_id)
        
        is_valid, message = await self.auth_validator.validate_user_session(user_id, session_token)
        
        assert is_valid is True
        assert "valid" in message.lower()
    
    @pytest.mark.asyncio
    async def test_validate_invalid_token(self):
        """Should reject invalid session token"""
        user_id = "user-123"
        self.auth_validator.create_user_session(user_id)
        
        is_valid, message = await self.auth_validator.validate_user_session(user_id, "invalid-token")
        
        assert is_valid is False
        assert "invalid" in message.lower()
    
    @pytest.mark.asyncio
    async def test_validate_missing_session(self):
        """Should reject non-existent session"""
        is_valid, message = await self.auth_validator.validate_user_session("nonexistent-user", "token")
        
        assert is_valid is False
        assert "no active session" in message.lower()
    
    @pytest.mark.asyncio
    async def test_validate_missing_credentials(self):
        """Should reject missing user_id or token"""
        is_valid, message = await self.auth_validator.validate_user_session("", "token")
        assert is_valid is False
        
        is_valid, message = await self.auth_validator.validate_user_session("user-123", "")
        assert is_valid is False
    
    def test_revoke_session(self):
        """Should revoke user session"""
        user_id = "user-123"
        self.auth_validator.create_user_session(user_id)
        
        result = self.auth_validator.revoke_user_session(user_id)
        
        assert result is True
        assert user_id not in self.auth_validator.active_sessions
    
    def test_check_login_attempts_no_failures(self):
        """Should allow login with no failed attempts"""
        user_id = "user-123"
        
        allowed, message = self.auth_validator.check_login_attempts(user_id)
        
        assert allowed is True
    
    def test_check_login_attempts_lockout(self):
        """Should lockout after max failed attempts"""
        user_id = "user-123"
        
        # Record max failed attempts
        for _ in range(5):
            self.auth_validator.record_failed_login(user_id)
        
        allowed, message = self.auth_validator.check_login_attempts(user_id)
        
        assert allowed is False
        assert "locked" in message.lower()
    
    def test_clear_failed_logins(self):
        """Should clear failed login attempts"""
        user_id = "user-123"
        
        # Record failed attempts
        for _ in range(3):
            self.auth_validator.record_failed_login(user_id)
        
        self.auth_validator.clear_failed_logins(user_id)
        
        allowed, message = self.auth_validator.check_login_attempts(user_id)
        assert allowed is True


class TestSecurityValidator:
    """Test complete security validation"""
    
    def setup_method(self):
        self.security_validator = SecurityValidator()
    
    @pytest.mark.asyncio
    async def test_validate_safe_request(self):
        """Should validate safe request"""
        request_data = {
            'endpoint': 'public_data',
            'user_id': 'user-123',
            'data': 'normal data'
        }
        context = SecurityContext(user_id='user-123', ip_address='192.168.1.1')
        
        is_valid, violations = self.security_validator.validate_request(request_data, context)
        
        assert is_valid is True
        assert len(violations) == 0
    
    @pytest.mark.asyncio
    async def test_validate_request_with_sql_injection(self):
        """Should detect SQL injection in request"""
        request_data = {
            'endpoint': 'public_data',
            'user_id': 'user-123',
            'query': "'; DROP TABLE users; --"
        }
        context = SecurityContext(user_id='user-123', ip_address='192.168.1.1')
        
        is_valid, violations = self.security_validator.validate_request(request_data, context)
        
        assert len(violations) > 0
        assert any(v.violation_type == "sql_injection" for v in violations)
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Should enforce rate limiting"""
        context = SecurityContext(user_id='user-123', ip_address='192.168.1.1')
        request_data = {'endpoint': 'public_data'}
        
        # Make 100 requests (at the limit)
        for _ in range(100):
            is_valid, violations = self.security_validator.validate_request(request_data, context)
        
        # 101st request should be rate limited
        is_valid, violations = self.security_validator.validate_request(request_data, context)
        
        assert len(violations) > 0
        assert any(v.violation_type == "rate_limit_exceeded" for v in violations)
    
    @pytest.mark.asyncio
    async def test_authentication_required(self):
        """Should require authentication for protected endpoints"""
        request_data = {
            'endpoint': 'process-excel',  # Protected endpoint
            'user_id': 'user-123'
            # Missing session_token
        }
        context = SecurityContext(user_id='user-123', ip_address='192.168.1.1')
        
        is_valid, violations = self.security_validator.validate_request(request_data, context)
        
        assert is_valid is False
        assert any(v.violation_type == "missing_authentication" for v in violations)
    
    def test_get_security_statistics(self):
        """Should return security statistics"""
        stats = self.security_validator.get_security_statistics()
        
        assert 'total_violations' in stats
        assert 'violation_types' in stats
        assert 'active_sessions' in stats
        assert 'rate_limited_ips' in stats
    
    def test_cleanup_expired_data(self):
        """Should cleanup expired sessions and rate limits"""
        # Create a session
        user_id = "user-123"
        self.security_validator.auth_validator.create_user_session(user_id)
        
        # Manually expire it
        session_data = self.security_validator.auth_validator.active_sessions[user_id]
        session_data['expires_at'] = datetime.utcnow() - timedelta(hours=1)
        
        cleaned = self.security_validator.cleanup_expired_data()
        
        assert cleaned > 0
        assert user_id not in self.security_validator.auth_validator.active_sessions


class TestSecurityIntegration:
    """Integration tests for security system"""
    
    def setup_method(self):
        self.security_validator = SecurityValidator()
    
    @pytest.mark.asyncio
    async def test_complete_request_validation_flow(self):
        """Should validate complete request flow"""
        # Create session
        user_id = "user-123"
        session_token = self.security_validator.auth_validator.create_user_session(user_id)
        
        # Make authenticated request
        request_data = {
            'endpoint': 'process-excel',
            'user_id': user_id,
            'session_token': session_token,
            'file_name': 'test.xlsx'
        }
        context = SecurityContext(user_id=user_id, ip_address='192.168.1.1')
        
        is_valid, violations = self.security_validator.validate_request(request_data, context)
        
        # Should be valid with proper authentication
        assert is_valid is True or len([v for v in violations if v.severity == SecurityLevel.CRITICAL]) == 0
    
    @pytest.mark.asyncio
    async def test_malicious_request_blocked(self):
        """Should block malicious requests"""
        request_data = {
            'endpoint': 'public_data',
            'user_id': 'user-123',
            'query': "<script>alert('xss')</script>",
            'file_path': '../../../etc/passwd'
        }
        context = SecurityContext(user_id='user-123', ip_address='192.168.1.1')
        
        is_valid, violations = self.security_validator.validate_request(request_data, context)
        
        # Should have multiple violations
        assert len(violations) > 0
        assert any(v.violation_type == "xss" for v in violations)
        assert any(v.violation_type == "path_traversal" for v in violations)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
