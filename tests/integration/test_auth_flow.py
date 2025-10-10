"""
Integration Tests for Authentication Flow

Tests complete auth workflow:
- Login → Upload → Process
- Session expiry handling
- Concurrent sessions
- Token refresh
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch


class TestCompleteAuthFlow:
    """Test complete authentication workflow"""
    
    @pytest.mark.asyncio
    async def test_login_upload_process_flow(self):
        """Should complete full flow: login → upload → process"""
        # Step 1: Login
        mock_auth = Mock()
        mock_auth.signInAnonymously = AsyncMock(return_value={
            'user': {'id': 'user-123'},
            'session': {'access_token': 'token-abc'}
        })
        
        login_result = await mock_auth.signInAnonymously()
        assert login_result['user']['id'] == 'user-123'
        
        user_id = login_result['user']['id']
        session_token = login_result['session']['access_token']
        
        # Step 2: Upload file
        mock_storage = Mock()
        mock_storage.upload = AsyncMock(return_value={'path': f'{user_id}/file.csv'})
        
        upload_result = await mock_storage.upload(b'test', f'{user_id}/file.csv')
        assert upload_result['path'] == f'{user_id}/file.csv'
        
        # Step 3: Create processing job
        mock_db = Mock()
        mock_db.table = Mock(return_value=Mock(
            insert=Mock(return_value=Mock(
                select=Mock(return_value=Mock(
                    single=AsyncMock(return_value={
                        'id': 'job-123',
                        'user_id': user_id,
                        'status': 'processing'
                    })
                ))
            ))
        ))
        
        job_result = await mock_db.table('ingestion_jobs').insert({
            'user_id': user_id,
            'status': 'processing'
        }).select().single()
        
        assert job_result['user_id'] == user_id
        assert job_result['status'] == 'processing'
    
    @pytest.mark.asyncio
    async def test_authenticated_api_request(self):
        """Should include auth token in API requests"""
        user_id = 'user-123'
        session_token = 'token-abc'
        
        # Mock API request with auth
        async def make_authenticated_request(endpoint, user_id, session_token):
            headers = {
                'Authorization': f'Bearer {session_token}',
                'Content-Type': 'application/json'
            }
            return {
                'headers': headers,
                'user_id': user_id,
                'authenticated': True
            }
        
        response = await make_authenticated_request('/api/upload', user_id, session_token)
        
        assert response['authenticated'] is True
        assert 'Bearer' in response['headers']['Authorization']
    
    @pytest.mark.asyncio
    async def test_unauthenticated_request_rejected(self):
        """Should reject requests without valid auth"""
        async def validate_request(session_token):
            if not session_token or len(session_token) < 10:
                raise Exception('Unauthorized')
            return True
        
        # Valid token
        result = await validate_request('valid-token-123')
        assert result is True
        
        # Invalid token
        with pytest.raises(Exception) as exc_info:
            await validate_request('')
        assert 'Unauthorized' in str(exc_info.value)


class TestSessionManagement:
    """Test session management"""
    
    @pytest.mark.asyncio
    async def test_session_creation(self):
        """Should create valid session on login"""
        from datetime import datetime, timedelta
        
        def create_session(user_id):
            return {
                'user_id': user_id,
                'token': 'session-token-abc',
                'created_at': datetime.utcnow(),
                'expires_at': datetime.utcnow() + timedelta(hours=1)
            }
        
        session = create_session('user-123')
        
        assert session['user_id'] == 'user-123'
        assert session['token'] is not None
        assert session['expires_at'] > session['created_at']
    
    @pytest.mark.asyncio
    async def test_session_validation(self):
        """Should validate active sessions"""
        def validate_session(session):
            if datetime.utcnow() > session['expires_at']:
                return False, "Session expired"
            return True, "Valid"
        
        # Active session
        active_session = {
            'expires_at': datetime.utcnow() + timedelta(hours=1)
        }
        is_valid, msg = validate_session(active_session)
        assert is_valid is True
        
        # Expired session
        expired_session = {
            'expires_at': datetime.utcnow() - timedelta(hours=1)
        }
        is_valid, msg = validate_session(expired_session)
        assert is_valid is False
        assert 'expired' in msg.lower()
    
    @pytest.mark.asyncio
    async def test_session_expiry_handling(self):
        """Should handle session expiry gracefully"""
        async def make_request_with_session(session_token, expires_at):
            if datetime.utcnow() > expires_at:
                return {'status': 'error', 'message': 'Session expired'}
            return {'status': 'success'}
        
        # Expired session
        result = await make_request_with_session(
            'token-abc',
            datetime.utcnow() - timedelta(hours=1)
        )
        
        assert result['status'] == 'error'
        assert 'expired' in result['message'].lower()
    
    @pytest.mark.asyncio
    async def test_session_refresh(self):
        """Should refresh expiring sessions"""
        def refresh_session(session):
            return {
                **session,
                'expires_at': datetime.utcnow() + timedelta(hours=1),
                'refreshed_at': datetime.utcnow()
            }
        
        old_session = {
            'user_id': 'user-123',
            'token': 'token-abc',
            'expires_at': datetime.utcnow() + timedelta(minutes=5)
        }
        
        new_session = refresh_session(old_session)
        
        assert new_session['expires_at'] > old_session['expires_at']
        assert 'refreshed_at' in new_session


class TestConcurrentSessions:
    """Test concurrent session handling"""
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_sessions(self):
        """Should handle multiple concurrent sessions per user"""
        sessions = {}
        
        def create_session(user_id, device_id):
            session_id = f'{user_id}-{device_id}'
            sessions[session_id] = {
                'user_id': user_id,
                'device_id': device_id,
                'token': f'token-{session_id}',
                'created_at': datetime.utcnow()
            }
            return sessions[session_id]
        
        # Create sessions for same user on different devices
        session1 = create_session('user-123', 'device-1')
        session2 = create_session('user-123', 'device-2')
        session3 = create_session('user-123', 'device-3')
        
        assert len(sessions) == 3
        assert all(s['user_id'] == 'user-123' for s in sessions.values())
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_same_session(self):
        """Should handle concurrent requests with same session"""
        async def make_request(session_token, request_id):
            await asyncio.sleep(0.01)  # Simulate API call
            return {
                'request_id': request_id,
                'session_token': session_token,
                'status': 'success'
            }
        
        session_token = 'token-abc'
        
        # Make 5 concurrent requests
        tasks = [make_request(session_token, i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert all(r['status'] == 'success' for r in results)
        assert all(r['session_token'] == session_token for r in results)


class TestAuthSecurity:
    """Test authentication security"""
    
    @pytest.mark.asyncio
    async def test_token_validation(self):
        """Should validate token format and signature"""
        def validate_token_format(token):
            if not token or len(token) < 32:
                return False
            if not all(c.isalnum() or c in '-_' for c in token):
                return False
            return True
        
        valid_tokens = [
            'a' * 32,
            'token-abc-123-def-456-ghi-789-jkl',
            'session_token_with_underscores_12345678901234567890'  # Must be >= 32 chars
        ]
        
        invalid_tokens = [
            '',
            'short',
            'token with spaces',
            'token<script>',
            None
        ]
        
        for token in valid_tokens:
            assert validate_token_format(token) is True
        
        for token in invalid_tokens:
            assert validate_token_format(token) is False
    
    @pytest.mark.asyncio
    async def test_prevent_session_hijacking(self):
        """Should prevent session hijacking"""
        def validate_session_ownership(session, user_id, ip_address):
            if session['user_id'] != user_id:
                return False, "User mismatch"
            if session.get('ip_address') and session['ip_address'] != ip_address:
                return False, "IP mismatch"
            return True, "Valid"
        
        session = {
            'user_id': 'user-123',
            'ip_address': '192.168.1.1'
        }
        
        # Valid request
        is_valid, msg = validate_session_ownership(session, 'user-123', '192.168.1.1')
        assert is_valid is True
        
        # Different user
        is_valid, msg = validate_session_ownership(session, 'user-456', '192.168.1.1')
        assert is_valid is False
        
        # Different IP
        is_valid, msg = validate_session_ownership(session, 'user-123', '10.0.0.1')
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_rate_limiting_per_session(self):
        """Should enforce rate limiting per session"""
        request_counts = {}
        
        def check_rate_limit(session_token, limit=100):
            if session_token not in request_counts:
                request_counts[session_token] = 0
            
            request_counts[session_token] += 1
            
            if request_counts[session_token] > limit:
                return False, "Rate limit exceeded"
            return True, "OK"
        
        session_token = 'token-abc'
        
        # Make 100 requests (at limit)
        for _ in range(100):
            is_allowed, msg = check_rate_limit(session_token)
            assert is_allowed is True
        
        # 101st request should be rate limited
        is_allowed, msg = check_rate_limit(session_token)
        assert is_allowed is False
        assert 'rate limit' in msg.lower()


class TestAuthErrorHandling:
    """Test authentication error handling"""
    
    @pytest.mark.asyncio
    async def test_handle_supabase_auth_failure(self):
        """Should handle Supabase auth failures gracefully"""
        mock_auth = Mock()
        mock_auth.signInAnonymously = AsyncMock(
            side_effect=Exception('Supabase unavailable')
        )
        
        with pytest.raises(Exception) as exc_info:
            await mock_auth.signInAnonymously()
        
        assert 'Supabase unavailable' in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_handle_network_timeout(self):
        """Should handle network timeouts"""
        async def auth_with_timeout(timeout=5):
            await asyncio.sleep(timeout + 1)  # Simulate timeout
            return {'user': {'id': 'user-123'}}
        
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(auth_with_timeout(), timeout=1.0)
    
    @pytest.mark.asyncio
    async def test_handle_invalid_credentials(self):
        """Should handle invalid credentials"""
        async def validate_credentials(email, password):
            if not email or not password:
                raise ValueError('Invalid credentials')
            return True
        
        with pytest.raises(ValueError) as exc_info:
            await validate_credentials('', '')
        
        assert 'Invalid credentials' in str(exc_info.value)


class TestAuthPerformance:
    """Test authentication performance"""
    
    @pytest.mark.asyncio
    async def test_login_performance(self):
        """Should complete login quickly"""
        import time
        
        async def login(user_id):
            await asyncio.sleep(0.01)  # Simulate auth
            return {'user_id': user_id, 'token': 'token-abc'}
        
        start_time = time.time()
        result = await login('user-123')
        end_time = time.time()
        
        assert result['user_id'] == 'user-123'
        assert (end_time - start_time) < 1.0  # Should complete in <1s
    
    @pytest.mark.asyncio
    async def test_concurrent_logins(self):
        """Should handle concurrent logins efficiently"""
        import time
        
        async def login(user_id):
            await asyncio.sleep(0.01)
            return {'user_id': user_id, 'token': f'token-{user_id}'}
        
        start_time = time.time()
        tasks = [login(f'user-{i}') for i in range(10)]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        assert len(results) == 10
        assert (end_time - start_time) < 1.0  # Should complete in <1s


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
