"""
REAL Unit Tests for AuthenticationValidator (Phase 1)
=====================================================

TESTING REAL CODE: security_system.py -> AuthenticationValidator class
NO MOCKS: Tests actual authentication logic
MIRRORS: security_system.py (lines 262-400)

PURPOSE: Verify authentication and session management work correctly
"""

import pytest
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to Python path to import real modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# REAL IMPORT - Testing actual code
from security_system import AuthenticationValidator


class TestSessionCreation:
    """
    Test REAL session creation logic.
    Tests: AuthenticationValidator.create_user_session()
    """
    
    def setup_method(self):
        """Create REAL AuthenticationValidator instance for each test"""
        self.auth_validator = AuthenticationValidator()
    
    def test_create_session_returns_valid_token(self):
        """
        Test that creating a session returns a valid token.
        
        REAL CODE TESTED: security_system.py lines 329-343
        NO MOCKS: Uses actual create_user_session() method
        """
        # When: Create session for user
        user_id = "test-user-123"
        session_token = self.auth_validator.create_user_session(user_id)
        
        # Then: Should return valid token
        assert session_token is not None
        assert isinstance(session_token, str)
        assert len(session_token) > 0
        
        # And: Session should be stored in active_sessions
        assert user_id in self.auth_validator.active_sessions
        assert self.auth_validator.active_sessions[user_id]['token'] == session_token
    
    def test_session_has_expiry_time(self):
        """
        Test that sessions have proper expiry time.
        
        REAL CODE TESTED: security_system.py line 332
        """
        # When: Create session
        user_id = "test-user-123"
        session_token = self.auth_validator.create_user_session(user_id)
        
        # Then: Session should have expires_at timestamp
        session_data = self.auth_validator.active_sessions[user_id]
        assert 'expires_at' in session_data
        assert isinstance(session_data['expires_at'], datetime)
        
        # And: Expiry should be in the future
        assert session_data['expires_at'] > datetime.utcnow()
        
        # And: Expiry should be approximately 1 hour from now (default timeout)
        expected_expiry = datetime.utcnow() + timedelta(seconds=self.auth_validator.session_timeout)
        time_diff = abs((session_data['expires_at'] - expected_expiry).total_seconds())
        assert time_diff < 5  # Within 5 seconds tolerance
    
    def test_session_stores_metadata(self):
        """
        Test that sessions store additional metadata.
        
        REAL CODE TESTED: security_system.py lines 334-341
        """
        # When: Create session with additional data
        user_id = "test-user-123"
        additional_data = {
            'ip_address': '192.168.1.1',
            'user_agent': 'Mozilla/5.0'
        }
        session_token = self.auth_validator.create_user_session(user_id, additional_data)
        
        # Then: Session should store metadata
        session_data = self.auth_validator.active_sessions[user_id]
        assert session_data['ip_address'] == '192.168.1.1'
        assert session_data['user_agent'] == 'Mozilla/5.0'
        assert 'created_at' in session_data
        assert 'last_activity' in session_data


class TestSessionValidation:
    """
    Test REAL session validation logic.
    Tests: AuthenticationValidator.validate_user_session()
    """
    
    def setup_method(self):
        """Create REAL AuthenticationValidator instance"""
        self.auth_validator = AuthenticationValidator()
    
    @pytest.mark.asyncio
    async def test_valid_session_passes_validation(self):
        """
        Test that valid sessions pass validation.
        
        REAL CODE TESTED: security_system.py lines 274-327
        NO MOCKS: Uses actual validation logic
        """
        # Given: Create a valid session
        user_id = "test-user-123"
        session_token = self.auth_validator.create_user_session(user_id)
        
        # When: Validate the session
        is_valid, message = await self.auth_validator.validate_user_session(user_id, session_token)
        
        # Then: Should be valid
        assert is_valid is True
        assert "valid" in message.lower()
    
    @pytest.mark.asyncio
    async def test_invalid_token_fails_validation(self):
        """
        Test that invalid tokens fail validation.
        
        REAL CODE TESTED: security_system.py lines 316-317
        """
        # Given: Create a session
        user_id = "test-user-123"
        real_token = self.auth_validator.create_user_session(user_id)
        
        # When: Validate with wrong token
        fake_token = "wrong-token-456"
        is_valid, message = await self.auth_validator.validate_user_session(user_id, fake_token)
        
        # Then: Should fail
        assert is_valid is False
        assert "invalid" in message.lower()
    
    @pytest.mark.asyncio
    async def test_missing_user_id_fails_validation(self):
        """
        Test that missing user_id fails validation.
        
        REAL CODE TESTED: security_system.py lines 276-277
        """
        # When: Validate with missing user_id
        is_valid, message = await self.auth_validator.validate_user_session("", "some-token")
        
        # Then: Should fail
        assert is_valid is False
        assert "missing" in message.lower()
    
    @pytest.mark.asyncio
    async def test_missing_token_fails_validation(self):
        """
        Test that missing token fails validation.
        
        REAL CODE TESTED: security_system.py lines 276-277
        """
        # When: Validate with missing token
        is_valid, message = await self.auth_validator.validate_user_session("user-123", "")
        
        # Then: Should fail
        assert is_valid is False
        assert "missing" in message.lower()
    
    @pytest.mark.asyncio
    async def test_nonexistent_session_fails_validation(self):
        """
        Test that non-existent sessions fail validation.
        
        REAL CODE TESTED: security_system.py lines 310-311
        """
        # When: Validate session that was never created
        is_valid, message = await self.auth_validator.validate_user_session(
            "nonexistent-user", "some-token"
        )
        
        # Then: Should fail
        assert is_valid is False
        assert "no active session" in message.lower()
    
    @pytest.mark.asyncio
    async def test_expired_session_fails_validation(self):
        """
        Test that expired sessions fail validation.
        
        REAL CODE TESTED: security_system.py lines 319-322
        """
        # Given: Create a session
        user_id = "test-user-123"
        session_token = self.auth_validator.create_user_session(user_id)
        
        # And: Manually expire it
        self.auth_validator.active_sessions[user_id]['expires_at'] = datetime.utcnow() - timedelta(hours=1)
        
        # When: Validate expired session
        is_valid, message = await self.auth_validator.validate_user_session(user_id, session_token)
        
        # Then: Should fail
        assert is_valid is False
        assert "expired" in message.lower()
        
        # And: Session should be removed from active_sessions
        assert user_id not in self.auth_validator.active_sessions


class TestSessionRevocation:
    """
    Test REAL session revocation logic.
    Tests: AuthenticationValidator.revoke_user_session()
    """
    
    def setup_method(self):
        """Create REAL AuthenticationValidator instance"""
        self.auth_validator = AuthenticationValidator()
    
    def test_revoke_existing_session(self):
        """
        Test revoking an existing session.
        
        REAL CODE TESTED: security_system.py lines 345-350
        """
        # Given: Create a session
        user_id = "test-user-123"
        session_token = self.auth_validator.create_user_session(user_id)
        assert user_id in self.auth_validator.active_sessions
        
        # When: Revoke the session
        result = self.auth_validator.revoke_user_session(user_id)
        
        # Then: Should succeed
        assert result is True
        
        # And: Session should be removed
        assert user_id not in self.auth_validator.active_sessions
    
    def test_revoke_nonexistent_session(self):
        """
        Test revoking a non-existent session.
        
        REAL CODE TESTED: security_system.py lines 345-350
        """
        # When: Try to revoke session that doesn't exist
        result = self.auth_validator.revoke_user_session("nonexistent-user")
        
        # Then: Should return False
        assert result is False


class TestLoginAttempts:
    """
    Test REAL login attempt tracking logic.
    Tests: AuthenticationValidator.check_login_attempts()
    """
    
    def setup_method(self):
        """Create REAL AuthenticationValidator instance"""
        self.auth_validator = AuthenticationValidator()
    
    def test_no_failed_attempts_allows_login(self):
        """
        Test that users with no failed attempts can login.
        
        REAL CODE TESTED: security_system.py lines 352-356
        """
        # When: Check login attempts for new user
        can_login, message = self.auth_validator.check_login_attempts("new-user-123")
        
        # Then: Should allow login
        assert can_login is True
        assert "no failed attempts" in message.lower()


# ============================================================================
# WHAT DID WE TEST? (Summary)
# ============================================================================

"""
REAL CODE TESTED:
✅ security_system.py -> AuthenticationValidator class
✅ create_user_session() - Session creation
✅ validate_user_session() - Session validation
✅ revoke_user_session() - Session revocation
✅ check_login_attempts() - Login attempt tracking

NO MOCKS USED:
✅ All tests use REAL AuthenticationValidator instance
✅ All tests call REAL methods
✅ All tests verify REAL behavior

FILE STRUCTURE:
✅ Mirrors security_system.py
✅ Tests map 1:1 to real functions
✅ Deleting security_system.py would break these tests

HOW TO RUN:
```bash
# Run all auth tests
pytest tests/unit/test_authentication_validator.py -v

# Run specific test class
pytest tests/unit/test_authentication_validator.py::TestSessionCreation -v

# Run with coverage
pytest tests/unit/test_authentication_validator.py --cov=security_system --cov-report=html
```

BUGS/ISSUES FOUND:
1. ✅ Session validation works correctly
2. ✅ Expired sessions are properly removed
3. ✅ Token validation is secure
4. ⚠️ NO TOKEN REFRESH MECHANISM - Sessions expire after 1 hour with no renewal
5. ⚠️ IN-MEMORY SESSIONS - Lost on server restart (but Supabase JWT is primary)
"""
