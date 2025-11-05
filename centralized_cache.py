"""
Centralized Redis Cache Configuration
======================================
Single source of truth for all caching across the application.
Replaces 5 different caching systems with one production-ready aiocache Redis backend.

Usage:
    from centralized_cache import get_cache, initialize_cache
    
    # Initialize once at startup
    cache = initialize_cache()
    
    # Use in any module
    cache = get_cache()
    await cache.set("key", value, ttl=3600)
    result = await cache.get("key")
"""

import os
import logging
from typing import Optional, Any
from aiocache import Cache
from aiocache.serializers import JsonSerializer
import structlog

logger = structlog.get_logger(__name__)

# Global cache instance
_cache_instance: Optional[Cache] = None

class CentralizedCache:
    """
    Production-ready centralized cache with Redis backend.
    
    Features:
    - Redis backend for distributed caching across workers/instances
    - JSON serialization for complex objects
    - Configurable TTL
    - Connection pooling
    - Automatic reconnection
    - Metrics tracking
    """
    
    def __init__(self, redis_url: Optional[str] = None, default_ttl: int = 3600):
        """
        Initialize centralized cache with Redis backend.
        
        Args:
            redis_url: Redis connection URL (defaults to env var ARQ_REDIS_URL or REDIS_URL)
            default_ttl: Default time-to-live in seconds (default: 1 hour)
        """
        # Get Redis URL from environment with fallbacks
        self.redis_url = redis_url or os.environ.get('ARQ_REDIS_URL') or os.environ.get('REDIS_URL', 'redis://localhost:6379')
        self.default_ttl = default_ttl
        
        # Initialize aiocache with Redis backend
        self.cache = Cache(
            Cache.REDIS,
            endpoint=self.redis_url.replace('redis://', '').split(':')[0],
            port=int(self.redis_url.replace('redis://', '').split(':')[1].split('/')[0]) if ':' in self.redis_url else 6379,
            serializer=JsonSerializer(),
            namespace="finely_ai",  # Namespace all keys
            timeout=5,  # Connection timeout
            pool_min_size=10,
            pool_max_size=100
        )
        
        # Metrics
        self.metrics = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        }
        
        logger.info("centralized_cache_initialized", redis_url=self.redis_url, default_ttl=default_ttl)
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        try:
            value = await self.cache.get(key)
            if value is not None:
                self.metrics['hits'] += 1
                logger.debug("cache_hit", key=key)
            else:
                self.metrics['misses'] += 1
                logger.debug("cache_miss", key=key)
            return value
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error("cache_get_error", key=key, error=str(e))
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.cache.set(key, value, ttl=ttl or self.default_ttl)
            self.metrics['sets'] += 1
            logger.debug("cache_set", key=key, ttl=ttl or self.default_ttl)
            return True
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error("cache_set_error", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.cache.delete(key)
            self.metrics['deletes'] += 1
            logger.debug("cache_delete", key=key)
            return True
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error("cache_delete_error", key=key, error=str(e))
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists, False otherwise
        """
        try:
            return await self.cache.exists(key)
        except Exception as e:
            logger.error("cache_exists_error", key=key, error=str(e))
            return False
    
    async def clear(self, namespace: Optional[str] = None) -> bool:
        """
        Clear cache entries.
        
        Args:
            namespace: Optional namespace to clear (clears all if None)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.cache.clear(namespace=namespace)
            logger.info("cache_cleared", namespace=namespace)
            return True
        except Exception as e:
            logger.error("cache_clear_error", namespace=namespace, error=str(e))
            return False
    
    def get_metrics(self) -> dict:
        """Get cache metrics."""
        total_requests = self.metrics['hits'] + self.metrics['misses']
        hit_rate = (self.metrics['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.metrics,
            'total_requests': total_requests,
            'hit_rate_percent': round(hit_rate, 2)
        }
    
    async def close(self):
        """Close cache connections."""
        try:
            await self.cache.close()
            logger.info("cache_closed")
        except Exception as e:
            logger.error("cache_close_error", error=str(e))


def initialize_cache(redis_url: Optional[str] = None, default_ttl: int = 3600) -> CentralizedCache:
    """
    Initialize global cache instance.
    
    Args:
        redis_url: Redis connection URL (defaults to env var)
        default_ttl: Default time-to-live in seconds
        
    Returns:
        CentralizedCache instance
    """
    global _cache_instance
    _cache_instance = CentralizedCache(redis_url=redis_url, default_ttl=default_ttl)
    logger.info("global_cache_initialized", redis_url=redis_url or "from_env", default_ttl=default_ttl)
    return _cache_instance


def get_cache() -> CentralizedCache:
    """
    Get global cache instance.
    
    Returns:
        CentralizedCache instance
        
    Raises:
        RuntimeError: If cache not initialized
    """
    if _cache_instance is None:
        raise RuntimeError("Cache not initialized. Call initialize_cache() first.")
    return _cache_instance


def safe_get_cache() -> Optional[CentralizedCache]:
    """
    Safely get cache instance without raising errors.
    
    Returns:
        CentralizedCache instance or None if not initialized
    """
    return _cache_instance
