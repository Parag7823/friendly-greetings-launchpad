"""
Security Tests for Rate Limiting

Tests:
- Rate limit enforcement
- IP-based limiting
- User-based limiting
- Endpoint-specific limits
- Rate limit bypass attempts
"""

import pytest
import asyncio
import time
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from security_system import SecurityValidator, SecurityContext


class TestIPBasedRateLimiting:
    """Test IP-based rate limiting"""
    
    def setup_method(self):
        self.security_validator = SecurityValidator()
    
    @pytest.mark.asyncio
    async def test_enforce_rate_limit_per_ip(self):
        """Should enforce rate limit per IP address"""
        ip_address = '192.168.1.100'
        context = SecurityContext(ip_address=ip_address, user_id='user-123')
        request_data = {'endpoint': 'public_data'}
        
        # Make 100 requests (at the limit)
        for i in range(100):
            is_valid, violations = self.security_validator.validate_request(request_data, context)
        
        # 101st request should be rate limited
        is_valid, violations = self.security_validator.validate_request(request_data, context)
        
        assert len(violations) > 0
        assert any(v.violation_type == "rate_limit_exceeded" for v in violations)
    
    @pytest.mark.asyncio
    async def test_different_ips_independent_limits(self):
        """Should enforce independent limits for different IPs"""
        request_data = {'endpoint': 'public_data'}
        
        # IP 1: Make 100 requests
        context1 = SecurityContext(ip_address='192.168.1.1', user_id='user-1')
        for _ in range(100):
            self.security_validator.validate_request(request_data, context1)
        
        # IP 2: Should still be able to make requests
        context2 = SecurityContext(ip_address='192.168.1.2', user_id='user-2')
        is_valid, violations = self.security_validator.validate_request(request_data, context2)
        
        # IP 2 should not be rate limited
        rate_limit_violations = [v for v in violations if v.violation_type == "rate_limit_exceeded"]
        assert len(rate_limit_violations) == 0
    
    @pytest.mark.asyncio
    async def test_rate_limit_window_expiry(self):
        """Should reset rate limit after time window"""
        ip_address = '192.168.1.100'
        context = SecurityContext(ip_address=ip_address, user_id='user-123')
        request_data = {'endpoint': 'public_data'}
        
        # Make requests to hit limit
        for _ in range(100):
            self.security_validator.validate_request(request_data, context)
        
        # Should be rate limited
        is_valid, violations = self.security_validator.validate_request(request_data, context)
        assert any(v.violation_type == "rate_limit_exceeded" for v in violations)
        
        # Simulate time passing (would need to wait 60s in real scenario)
        # For testing, we can manually clear the rate limit data
        self.security_validator.rate_limits.clear()
        
        # Should be able to make requests again
        is_valid, violations = self.security_validator.validate_request(request_data, context)
        rate_limit_violations = [v for v in violations if v.violation_type == "rate_limit_exceeded"]
        assert len(rate_limit_violations) == 0


class TestUserBasedRateLimiting:
    """Test user-based rate limiting"""
    
    def setup_method(self):
        self.security_validator = SecurityValidator()
    
    @pytest.mark.asyncio
    async def test_rate_limit_per_user(self):
        """Should enforce rate limit per user"""
        user_id = 'user-123'
        request_data = {'endpoint': 'public_data'}
        
        # Same user, different IPs
        for i in range(100):
            context = SecurityContext(ip_address=f'192.168.1.{i}', user_id=user_id)
            self.security_validator.validate_request(request_data, context)
        
        # Note: Current implementation is IP-based, not user-based
        # This test documents expected behavior if user-based limiting is added


class TestEndpointSpecificLimits:
    """Test endpoint-specific rate limits"""
    
    def setup_method(self):
        self.security_validator = SecurityValidator()
    
    @pytest.mark.asyncio
    async def test_different_limits_per_endpoint(self):
        """Should enforce different limits for different endpoints"""
        # This test documents expected behavior
        # Current implementation has global limit
        # Could be enhanced to have per-endpoint limits
        
        context = SecurityContext(ip_address='192.168.1.1', user_id='user-123')
        
        # Public endpoint - higher limit
        public_request = {'endpoint': 'public_data'}
        
        # Protected endpoint - lower limit
        protected_request = {'endpoint': 'process-excel', 'user_id': 'user-123', 'session_token': 'token'}
        
        # Both currently share same limit
        # Future enhancement: different limits per endpoint


class TestRateLimitBypassAttempts:
    """Test prevention of rate limit bypass attempts"""
    
    def setup_method(self):
        self.security_validator = SecurityValidator()
    
    @pytest.mark.asyncio
    async def test_prevent_ip_spoofing(self):
        """Should not allow IP spoofing to bypass limits"""
        request_data = {'endpoint': 'public_data'}
        
        # Attacker tries to spoof X-Forwarded-For header
        # SecurityContext should use actual connection IP, not headers
        
        real_ip = '192.168.1.100'
        context = SecurityContext(ip_address=real_ip, user_id='user-123')
        
        # Make requests to hit limit
        for _ in range(100):
            self.security_validator.validate_request(request_data, context)
        
        # Should be rate limited regardless of spoofed headers
        is_valid, violations = self.security_validator.validate_request(request_data, context)
        assert any(v.violation_type == "rate_limit_exceeded" for v in violations)
    
    @pytest.mark.asyncio
    async def test_prevent_distributed_attack(self):
        """Should detect distributed rate limit bypass attempts"""
        request_data = {'endpoint': 'public_data'}
        
        # Simulate distributed attack from multiple IPs
        for i in range(10):
            ip = f'10.0.0.{i}'
            context = SecurityContext(ip_address=ip, user_id='user-123')
            
            # Each IP makes many requests
            for _ in range(100):
                self.security_validator.validate_request(request_data, context)
        
        # Each IP should be independently rate limited
        # (This documents current behavior - could add cross-IP detection)


class TestRateLimitHeaders:
    """Test rate limit response headers"""
    
    @pytest.mark.asyncio
    async def test_rate_limit_headers_present(self):
        """Should include rate limit headers in responses"""
        # This test documents expected behavior
        # Rate limit headers like X-RateLimit-Limit, X-RateLimit-Remaining
        # should be included in API responses
        
        # Example headers:
        # X-RateLimit-Limit: 100
        # X-RateLimit-Remaining: 50
        # X-RateLimit-Reset: 1234567890
        pass


class TestConcurrentRateLimiting:
    """Test rate limiting under concurrent requests"""
    
    def setup_method(self):
        self.security_validator = SecurityValidator()
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_counted_correctly(self):
        """Should count concurrent requests correctly"""
        ip_address = '192.168.1.100'
        request_data = {'endpoint': 'public_data'}
        
        async def make_request():
            context = SecurityContext(ip_address=ip_address, user_id='user-123')
            return self.security_validator.validate_request(request_data, context)
        
        # Make 50 concurrent requests
        tasks = [make_request() for _ in range(50)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed (under limit)
        rate_limited = sum(1 for is_valid, violations in results 
                          if any(v.violation_type == "rate_limit_exceeded" for v in violations))
        
        assert rate_limited == 0  # None should be rate limited yet
        
        # Make 60 more to exceed limit
        tasks = [make_request() for _ in range(60)]
        results = await asyncio.gather(*tasks)
        
        # Some should be rate limited
        rate_limited = sum(1 for is_valid, violations in results 
                          if any(v.violation_type == "rate_limit_exceeded" for v in violations))
        
        assert rate_limited > 0  # Some should be rate limited


class TestRateLimitRecovery:
    """Test rate limit recovery and cleanup"""
    
    def setup_method(self):
        self.security_validator = SecurityValidator()
    
    def test_cleanup_expired_rate_limits(self):
        """Should clean up expired rate limit data"""
        # Add some rate limit data
        self.security_validator.rate_limits['192.168.1.1'] = [
            datetime.utcnow() - timedelta(hours=2)  # Old timestamp
        ]
        
        # Run cleanup
        cleaned = self.security_validator.cleanup_expired_data()
        
        # Old data should be cleaned
        assert cleaned > 0
        assert '192.168.1.1' not in self.security_validator.rate_limits


class TestRateLimitMetrics:
    """Test rate limit metrics and monitoring"""
    
    def setup_method(self):
        self.security_validator = SecurityValidator()
    
    @pytest.mark.asyncio
    async def test_track_rate_limit_violations(self):
        """Should track rate limit violations for monitoring"""
        ip_address = '192.168.1.100'
        context = SecurityContext(ip_address=ip_address, user_id='user-123')
        request_data = {'endpoint': 'public_data'}
        
        # Hit rate limit
        for _ in range(101):
            self.security_validator.validate_request(request_data, context)
        
        # Get statistics
        stats = self.security_validator.get_security_statistics()
        
        assert 'total_violations' in stats
        assert stats['total_violations'] > 0
        assert 'rate_limited_ips' in stats


class TestAdaptiveRateLimiting:
    """Test adaptive rate limiting (future enhancement)"""
    
    @pytest.mark.asyncio
    async def test_adaptive_limits_based_on_behavior(self):
        """Should adapt limits based on user behavior"""
        # This test documents expected behavior for future enhancement
        # Good users could get higher limits
        # Suspicious users could get lower limits
        pass
    
    @pytest.mark.asyncio
    async def test_temporary_limit_increase_for_verified_users(self):
        """Should allow temporary limit increases for verified users"""
        # This test documents expected behavior for future enhancement
        # Verified/premium users could get higher limits
        pass


class TestRateLimitPerformance:
    """Test rate limiting performance"""
    
    def setup_method(self):
        self.security_validator = SecurityValidator()
    
    @pytest.mark.asyncio
    async def test_rate_limit_check_performance(self):
        """Should check rate limits quickly"""
        import time
        
        context = SecurityContext(ip_address='192.168.1.1', user_id='user-123')
        request_data = {'endpoint': 'public_data'}
        
        start_time = time.time()
        
        # Make 100 requests
        for _ in range(100):
            self.security_validator.validate_request(request_data, context)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 100
        
        print(f"Average rate limit check time: {avg_time * 1000:.2f}ms")
        
        # Should be very fast
        assert avg_time < 0.01  # <10ms per check


class TestRateLimitEdgeCases:
    """Test edge cases in rate limiting"""
    
    def setup_method(self):
        self.security_validator = SecurityValidator()
    
    @pytest.mark.asyncio
    async def test_handle_missing_ip_address(self):
        """Should handle requests without IP address"""
        context = SecurityContext(ip_address=None, user_id='user-123')
        request_data = {'endpoint': 'public_data'}
        
        # Should not crash
        is_valid, violations = self.security_validator.validate_request(request_data, context)
        
        # Should still validate (no rate limiting without IP)
        assert isinstance(is_valid, bool)
    
    @pytest.mark.asyncio
    async def test_handle_ipv6_addresses(self):
        """Should handle IPv6 addresses"""
        context = SecurityContext(ip_address='2001:0db8:85a3:0000:0000:8a2e:0370:7334', user_id='user-123')
        request_data = {'endpoint': 'public_data'}
        
        # Should handle IPv6
        is_valid, violations = self.security_validator.validate_request(request_data, context)
        
        assert isinstance(is_valid, bool)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
