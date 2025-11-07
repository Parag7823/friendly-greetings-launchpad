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
import asyncio
from typing import Optional, Any
from aiocache import Cache
from aiocache.serializers import JsonSerializer
import structlog

logger = structlog.get_logger(__name__)

# Global cache instance
_cache_instance: Optional[Cache] = None
_health_check_task = None
_circuit_breaker_open = False
_failure_count = 0
_last_failure_time = None

# Circuit breaker configuration
CIRCUIT_BREAKER_THRESHOLD = 5  # Open circuit after 5 failures
CIRCUIT_BREAKER_TIMEOUT = 60  # Reset after 60 seconds

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
        
        # CRITICAL FIX: Robust Redis URL parsing
        from urllib.parse import urlparse
        parsed = urlparse(self.redis_url)
        
        # Extract components
        endpoint = parsed.hostname or 'localhost'
        port = parsed.port or 6379
        password = parsed.password or os.environ.get('REDIS_PASSWORD')
        db = int(parsed.path.lstrip('/')) if parsed.path and parsed.path != '/' else 0
        
        # Check for TLS
        use_tls = parsed.scheme in ('rediss', 'redis+tls') or os.environ.get('REDIS_TLS', 'false').lower() == 'true'
        
        # Initialize aiocache with Redis backend
        cache_config = {
            'endpoint': endpoint,
            'port': port,
            'db': db,
            'serializer': JsonSerializer(),
            'namespace': "finely_ai",
            'timeout': 5,
            'pool_min_size': int(os.environ.get('REDIS_POOL_MIN', '10')),
            'pool_max_size': int(os.environ.get('REDIS_POOL_MAX', '100'))
        }
        
        if password:
            cache_config['password'] = password
        
        if use_tls:
            cache_config['ssl'] = True
        
        self.cache = Cache(Cache.REDIS, **cache_config)
        
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
        global _circuit_breaker_open, _failure_count, _last_failure_time
        
        # Check circuit breaker
        if _circuit_breaker_open:
            import time
            if time.time() - _last_failure_time > CIRCUIT_BREAKER_TIMEOUT:
                _circuit_breaker_open = False
                _failure_count = 0
                logger.info("circuit_breaker_closed")
            else:
                logger.warning("circuit_breaker_open", key=key)
                return None
        
        try:
            value = await self.cache.get(key)
            if value is not None:
                self.metrics['hits'] += 1
                logger.debug("cache_hit", key=key)
            else:
                self.metrics['misses'] += 1
                logger.debug("cache_miss", key=key)
            
            # Reset failure count on success
            _failure_count = 0
            return value
        except Exception as e:
            self.metrics['errors'] += 1
            _failure_count += 1
            
            if _failure_count >= CIRCUIT_BREAKER_THRESHOLD:
                import time
                _circuit_breaker_open = True
                _last_failure_time = time.time()
                logger.error("circuit_breaker_opened", failures=_failure_count, error=str(e))
            else:
                logger.error("cache_get_error", key=key, error=str(e), failures=_failure_count)
            
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
    Initialize global cache instance and configure aiocache global config.
    
    CRITICAL FIX: This ensures @cached decorators use the same Redis backend
    as the centralized cache, preventing cache divergence across workers.
    
    Args:
        redis_url: Redis connection URL (defaults to env var)
        default_ttl: Default time-to-live in seconds
        
    Returns:
        CentralizedCache instance
    """
    global _cache_instance
    _cache_instance = CentralizedCache(redis_url=redis_url, default_ttl=default_ttl)
    
    # CRITICAL FIX: Configure aiocache global config to use same Redis backend
    # This ensures @cached decorators point to centralized Redis, not separate cache
    from aiocache import caches
    from urllib.parse import urlparse
    
    parsed = urlparse(_cache_instance.redis_url)
    endpoint = parsed.hostname or 'localhost'
    port = parsed.port or 6379
    password = parsed.password or os.environ.get('REDIS_PASSWORD')
    db = int(parsed.path.lstrip('/')) if parsed.path and parsed.path != '/' else 0
    use_tls = parsed.scheme in ('rediss', 'redis+tls') or os.environ.get('REDIS_TLS', 'false').lower() == 'true'
    
    aiocache_config = {
        'default': {
            'cache': "aiocache.RedisCache",
            'endpoint': endpoint,
            'port': port,
            'db': db,
            'serializer': {
                'class': "aiocache.serializers.JsonSerializer"
            },
            'namespace': "finely_ai",
            'timeout': 5,
            'pool_min_size': int(os.environ.get('REDIS_POOL_MIN', '10')),
            'pool_max_size': int(os.environ.get('REDIS_POOL_MAX', '100'))
        }
    }
    
    if password:
        aiocache_config['default']['password'] = password
    
    if use_tls:
        aiocache_config['default']['ssl'] = True
    
    caches.set_config(aiocache_config)
    
    logger.info("global_cache_initialized", 
               redis_url=redis_url or "from_env", 
               default_ttl=default_ttl,
               aiocache_configured=True)
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


async def health_check() -> dict:
    """
    Check Redis cache health.
    
    Returns:
        dict with health status
    """
    global _circuit_breaker_open, _failure_count
    
    cache = safe_get_cache()
    if cache is None:
        return {
            'status': 'unavailable',
            'initialized': False,
            'circuit_breaker_open': False,
            'error': 'Cache not initialized'
        }
    
    try:
        # Try to set and get a test value
        test_key = 'health_check_test'
        await cache.set(test_key, 'ok', ttl=10)
        result = await cache.get(test_key)
        await cache.delete(test_key)
        
        if result == 'ok':
            return {
                'status': 'healthy',
                'initialized': True,
                'circuit_breaker_open': _circuit_breaker_open,
                'failure_count': _failure_count,
                'metrics': cache.get_metrics()
            }
        else:
            return {
                'status': 'degraded',
                'initialized': True,
                'circuit_breaker_open': _circuit_breaker_open,
                'failure_count': _failure_count,
                'error': 'Test value mismatch'
            }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'initialized': True,
            'circuit_breaker_open': _circuit_breaker_open,
            'failure_count': _failure_count,
            'error': str(e)
        }


async def validate_redis_connection(redis_url: str) -> bool:
    """
    Validate Redis connection before initializing cache.
    
    Args:
        redis_url: Redis connection URL
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        import redis.asyncio as redis
        
        # Parse Redis URL
        client = redis.from_url(redis_url, decode_responses=True)
        
        # Test connection with ping
        await client.ping()
        await client.close()
        
        logger.info("redis_connection_validated", url=redis_url)
        return True
    except Exception as e:
        logger.error("redis_connection_failed", url=redis_url, error=str(e))
        return False


def require_redis_cache() -> bool:
    """
    Check if Redis cache is required in production.
    
    Returns:
        True if Redis is required (production mode)
    """
    env = os.environ.get('ENVIRONMENT', 'development').lower()
    require_redis = os.environ.get('REQUIRE_REDIS_CACHE', 'true').lower() == 'true'
    
    return env == 'production' or require_redis


async def start_health_check_monitor(interval: int = 60):
    """
    Start background health check monitor.
    
    Args:
        interval: Health check interval in seconds
    """
    global _health_check_task
    
    async def _monitor():
        while True:
            try:
                health = await health_check()
                if health['status'] != 'healthy':
                    logger.warning("cache_health_degraded", health=health)
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("health_check_monitor_error", error=str(e))
                await asyncio.sleep(interval)
    
    _health_check_task = asyncio.create_task(_monitor())
    logger.info("health_check_monitor_started", interval=interval)


async def stop_health_check_monitor():
    """Stop background health check monitor."""
    global _health_check_task
    
    if _health_check_task:
        _health_check_task.cancel()
        try:
            await _health_check_task
        except asyncio.CancelledError:
            pass
        _health_check_task = None
        logger.info("health_check_monitor_stopped")
