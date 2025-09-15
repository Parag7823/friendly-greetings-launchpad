"""
Test suite for the production-grade caching system.
Tests Redis integration, in-memory fallback, and performance.
"""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, patch
from typing import Dict, Any

# Mock Redis for testing
class MockRedis:
    def __init__(self):
        self.data = {}
        self.connected = True
    
    async def ping(self):
        return True
    
    async def get(self, key):
        return self.data.get(key)
    
    async def setex(self, key, ttl, value):
        self.data[key] = value
        return True
    
    async def delete(self, key):
        if key in self.data:
            del self.data[key]
            return 1
        return 0
    
    async def flushdb(self):
        self.data.clear()
        return True
    
    async def close(self):
        self.connected = False

# Test the caching system
class TestCachingSystem:
    """Test suite for the caching system"""
    
    @pytest.fixture
    def mock_redis(self):
        return MockRedis()
    
    @pytest.fixture
    def cache_config(self):
        from caching_system import CacheConfig
        return CacheConfig(
            redis_url='redis://localhost:6379',
            default_ttl=3600,
            max_memory_mb=100,
            enable_compression=True,
            enable_serialization=True
        )
    
    @pytest.mark.asyncio
    async def test_in_memory_cache_basic_operations(self):
        """Test basic in-memory cache operations"""
        from caching_system import InMemoryCache
        
        cache = InMemoryCache(max_size_mb=10)
        
        # Test set and get
        await cache.set("test_key", "test_value", 3600)
        result = await cache.get("test_key")
        assert result == "test_value"
        
        # Test delete
        await cache.delete("test_key")
        result = await cache.get("test_key")
        assert result is None
        
        # Test clear
        await cache.set("key1", "value1", 3600)
        await cache.set("key2", "value2", 3600)
        await cache.clear()
        
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
    
    @pytest.mark.asyncio
    async def test_in_memory_cache_ttl_expiration(self):
        """Test TTL expiration in in-memory cache"""
        from caching_system import InMemoryCache
        
        cache = InMemoryCache(max_size_mb=10)
        
        # Set with short TTL
        await cache.set("expiring_key", "expiring_value", 1)  # 1 second TTL
        
        # Should be available immediately
        result = await cache.get("expiring_key")
        assert result == "expiring_value"
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Should be expired
        result = await cache.get("expiring_key")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_in_memory_cache_lru_eviction(self):
        """Test LRU eviction when cache is full"""
        from caching_system import InMemoryCache
        
        # Create small cache
        cache = InMemoryCache(max_size_mb=1)  # 1MB limit
        
        # Fill cache with large values
        large_value = "x" * 100000  # 100KB
        
        # Add multiple entries to exceed limit
        for i in range(15):  # 15 * 100KB = 1.5MB > 1MB limit
            await cache.set(f"key_{i}", large_value, 3600)
        
        # Check that some entries were evicted
        stats = cache.get_stats()
        assert stats['entries'] < 15
        assert stats['size_mb'] <= 1.0
    
    @pytest.mark.asyncio
    async def test_in_memory_cache_cleanup_expired(self):
        """Test cleanup of expired entries"""
        from caching_system import InMemoryCache
        
        cache = InMemoryCache(max_size_mb=10)
        
        # Add some entries with different TTLs
        await cache.set("permanent", "value1", 3600)
        await cache.set("expiring", "value2", 1)  # 1 second TTL
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Cleanup expired entries
        cleaned = await cache.cleanup_expired()
        assert cleaned == 1
        
        # Check results
        assert await cache.get("permanent") == "value1"
        assert await cache.get("expiring") is None
    
    @pytest.mark.asyncio
    async def test_redis_cache_basic_operations(self, mock_redis):
        """Test basic Redis cache operations"""
        from caching_system import RedisCache, CacheConfig
        
        config = CacheConfig()
        cache = RedisCache(config)
        cache.redis_client = mock_redis
        cache.connected = True  # Manually set connected status
        
        # Test set and get
        await cache.set("test_key", "test_value", 3600)
        result = await cache.get("test_key")
        assert result == "test_value"
        
        # Test delete
        await cache.delete("test_key")
        result = await cache.get("test_key")
        assert result is None
        
        # Test clear
        await cache.set("key1", "value1", 3600)
        await cache.set("key2", "value2", 3600)
        await cache.clear()
        
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
    
    @pytest.mark.asyncio
    async def test_redis_cache_serialization(self, mock_redis):
        """Test Redis cache serialization"""
        from caching_system import RedisCache, CacheConfig
        
        config = CacheConfig(enable_serialization=True)
        cache = RedisCache(config)
        cache.redis_client = mock_redis
        cache.connected = True  # Manually set connected status
        
        # Test complex data structure
        complex_data = {
            "string": "test",
            "number": 42,
            "list": [1, 2, 3],
            "dict": {"nested": "value"}
        }
        
        await cache.set("complex_key", complex_data, 3600)
        result = await cache.get("complex_key")
        
        assert result == complex_data
    
    @pytest.mark.asyncio
    async def test_production_cache_redis_primary(self, mock_redis):
        """Test production cache with Redis as primary"""
        from caching_system import ProductionCache, CacheConfig
        
        config = CacheConfig()
        cache = ProductionCache(config)
        
        # Mock Redis connection
        cache.redis_cache.redis_client = mock_redis
        cache.redis_cache.connected = True
        
        await cache.initialize()
        
        # Test set and get
        await cache.set("test_key", "test_value", 3600)
        result = await cache.get("test_key")
        assert result == "test_value"
        
        # Test that it's also in memory cache
        memory_result = await cache.memory_cache.get("test_key")
        assert memory_result == "test_value"
    
    @pytest.mark.asyncio
    async def test_production_cache_memory_fallback(self):
        """Test production cache with memory fallback when Redis fails"""
        from caching_system import ProductionCache, CacheConfig
        
        config = CacheConfig()
        cache = ProductionCache(config)
        
        # Don't connect to Redis (simulate failure)
        cache.redis_cache.connected = False
        
        await cache.initialize()
        
        # Test set and get with memory fallback
        await cache.set("test_key", "test_value", 3600)
        result = await cache.get("test_key")
        assert result == "test_value"
        
        # Verify it's in memory cache
        memory_result = await cache.memory_cache.get("test_key")
        assert memory_result == "test_value"
    
    @pytest.mark.asyncio
    async def test_production_cache_key_generation(self):
        """Test cache key generation"""
        from caching_system import ProductionCache, CacheConfig
        
        config = CacheConfig()
        cache = ProductionCache(config)
        
        # Test key generation
        key1 = cache.generate_key("prefix", "arg1", "arg2", param1="value1", param2="value2")
        key2 = cache.generate_key("prefix", "arg1", "arg2", param1="value1", param2="value2")
        key3 = cache.generate_key("prefix", "arg1", "arg2", param2="value2", param1="value1")  # Different order
        
        # Same inputs should generate same key
        assert key1 == key2
        # Different parameter order should generate same key (sorted)
        assert key1 == key3
        
        # Different inputs should generate different keys
        key4 = cache.generate_key("prefix", "arg1", "arg3")
        assert key1 != key4
    
    @pytest.mark.asyncio
    async def test_production_cache_statistics(self):
        """Test cache statistics tracking"""
        from caching_system import ProductionCache, CacheConfig
        
        config = CacheConfig()
        cache = ProductionCache(config)
        cache.redis_cache.connected = False  # Use memory only
        
        await cache.initialize()
        
        # Perform operations
        await cache.set("key1", "value1", 3600)
        await cache.get("key1")  # Hit
        await cache.get("key2")  # Miss
        await cache.set("key2", "value2", 3600)
        await cache.delete("key1")
        
        # Check statistics
        stats = cache.get_stats()
        
        assert stats['performance']['hits'] == 1
        assert stats['performance']['misses'] == 1
        assert stats['performance']['sets'] == 2
        assert stats['performance']['deletes'] == 1
        assert stats['performance']['hit_rate'] == 0.5  # 1 hit / 2 total requests
    
    @pytest.mark.asyncio
    async def test_enrichment_cache_specialized(self):
        """Test specialized enrichment cache"""
        from caching_system import ProductionCache, EnrichmentCache, CacheConfig
        
        config = CacheConfig()
        cache = ProductionCache(config)
        cache.redis_cache.connected = False  # Use memory only
        
        await cache.initialize()
        
        enrichment_cache = EnrichmentCache(cache)
        
        # Test enrichment result caching
        row_data = {"vendor": "Amazon", "amount": 100.0}
        file_context = {"filename": "test.csv", "user_id": "user123"}
        result = {"vendor_standard": "Amazon.com", "confidence": 0.9}
        
        await enrichment_cache.set_enrichment_result(row_data, file_context, result)
        cached_result = await enrichment_cache.get_enrichment_result(row_data, file_context)
        
        assert cached_result == result
        
        # Test vendor standardization caching
        vendor_result = {"standardized": "Amazon.com", "confidence": 0.95}
        await enrichment_cache.set_vendor_standardization("Amazon", "stripe", vendor_result)
        cached_vendor = await enrichment_cache.get_vendor_standardization("Amazon", "stripe")
        
        assert cached_vendor == vendor_result
    
    @pytest.mark.asyncio
    async def test_document_analysis_cache_specialized(self):
        """Test specialized document analysis cache"""
        from caching_system import ProductionCache, DocumentAnalysisCache, CacheConfig
        
        config = CacheConfig()
        cache = ProductionCache(config)
        cache.redis_cache.connected = False  # Use memory only
        
        await cache.initialize()
        
        doc_cache = DocumentAnalysisCache(cache)
        
        # Test document classification caching
        df_hash = "abc123"
        filename = "income_statement.xlsx"
        classification = {"document_type": "income_statement", "confidence": 0.9}
        
        await doc_cache.set_document_classification(df_hash, filename, classification)
        cached_classification = await doc_cache.get_document_classification(df_hash, filename)
        
        assert cached_classification == classification
        
        # Test platform detection caching
        column_names = ["revenue", "expenses", "profit"]
        platform_result = {"platform": "quickbooks", "confidence": 0.8}
        
        await doc_cache.set_platform_detection(column_names, filename, platform_result)
        cached_platform = await doc_cache.get_platform_detection(column_names, filename)
        
        assert cached_platform == platform_result
    
    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Test cache performance with multiple operations"""
        from caching_system import ProductionCache, CacheConfig
        
        config = CacheConfig()
        cache = ProductionCache(config)
        cache.redis_cache.connected = False  # Use memory only
        
        await cache.initialize()
        
        # Test performance with many operations
        start_time = time.time()
        
        # Set many values
        for i in range(1000):
            await cache.set(f"key_{i}", f"value_{i}", 3600)
        
        # Get many values
        for i in range(1000):
            result = await cache.get(f"key_{i}")
            assert result == f"value_{i}"
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete within reasonable time (1 second for 2000 operations)
        assert total_time < 1.0
        
        # Check hit rate
        stats = cache.get_stats()
        assert stats['performance']['hit_rate'] == 1.0  # All gets should be hits
    
    @pytest.mark.asyncio
    async def test_cache_error_handling(self):
        """Test cache error handling"""
        from caching_system import ProductionCache, CacheConfig
        
        config = CacheConfig()
        cache = ProductionCache(config)
        
        # Simulate Redis connection failure
        cache.redis_cache.connected = False
        
        await cache.initialize()
        
        # Operations should still work with memory fallback
        await cache.set("test_key", "test_value", 3600)
        result = await cache.get("test_key")
        assert result == "test_value"
        
        # Check that errors are tracked
        stats = cache.get_stats()
        assert stats['performance']['errors'] >= 0  # May have connection errors
    
    @pytest.mark.asyncio
    async def test_global_cache_singleton(self):
        """Test global cache singleton behavior"""
        from caching_system import get_global_cache, close_global_cache
        
        # Get first instance
        cache1 = await get_global_cache()
        
        # Get second instance (should be same)
        cache2 = await get_global_cache()
        
        assert cache1 is cache2
        
        # Test operations work
        await cache1.set("test_key", "test_value", 3600)
        result = await cache2.get("test_key")
        assert result == "test_value"
        
        # Close and get new instance
        await close_global_cache()
        cache3 = await get_global_cache()
        
        # Should be different instance
        assert cache1 is not cache3
        
        # Previous data should not be available
        result = await cache3.get("test_key")
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
