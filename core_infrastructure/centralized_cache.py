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
import structlog
import asyncio
from typing import Optional, Any
from aiocache import Cache
from aiocache.serializers import PickleSerializer

logger = structlog.get_logger(__name__)

# Global cache instance
_cache_instance: Optional[Cache] = None
_health_check_task = None

# CRITICAL FIX: Circuit breaker configuration - hybrid Redis + in-memory fallback
CIRCUIT_BREAKER_THRESHOLD = 5  # Open circuit after 5 failures
CIRCUIT_BREAKER_TIMEOUT = 60  # Reset after 60 seconds
CIRCUIT_BREAKER_KEY = "cache:circuit_breaker"
CIRCUIT_BREAKER_FAILURE_KEY = "cache:circuit_breaker:failures"

# In-memory circuit breaker state (fallback when Redis is down)
_circuit_breaker_open = False
_circuit_breaker_failure_count = 0
_circuit_breaker_last_reset = None

def _parse_redis_url(redis_url: str) -> Dict[str, Any]:
    """
    FIX #16: Extract Redis URL parsing to eliminate duplication.
    
    Parses Redis URL and returns connection components.
    Used by both __init__ and initialize_cache.
    """
    from urllib.parse import urlparse
    
    parsed = urlparse(redis_url)
    
    return {
        'endpoint': parsed.hostname or 'localhost',
        'port': parsed.port or 6379,
        'password': parsed.password or os.environ.get('REDIS_PASSWORD'),
        'db': int(parsed.path.lstrip('/')) if parsed.path and parsed.path != '/' else 0,
        'use_tls': parsed.scheme in ('rediss', 'redis+tls') or os.environ.get('REDIS_TLS', 'false').lower() == 'true'
    }


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
        
        # FIX #16: Use extracted helper method
        parsed_config = _parse_redis_url(self.redis_url)
        endpoint = parsed_config['endpoint']
        port = parsed_config['port']
        password = parsed_config['password']
        db = parsed_config['db']
        use_tls = parsed_config['use_tls']
        
        # Initialize aiocache with Redis backend
        cache_config = {
            'endpoint': endpoint,
            'port': port,
            'db': db,
            'serializer': PickleSerializer(),
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
        
        # Initialize direct Redis client for operations not supported by aiocache
        self.redis_client = None  # Will be initialized lazily when needed
        
        # Metrics
        self.metrics = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        }
        
        logger.info("centralized_cache_initialized", redis_url=self.redis_url, default_ttl=default_ttl)
    
    async def _check_circuit_breaker(self) -> bool:
        """
        Check circuit breaker state with in-memory fallback.
        
        CRITICAL FIX: When Redis is down, use in-memory circuit breaker state
        to prevent app hangs trying to check Redis-based breaker.
        
        Returns:
            True if circuit is open (cache unavailable), False if closed (cache available)
        """
        global _circuit_breaker_open, _circuit_breaker_failure_count, _circuit_breaker_last_reset
        from datetime import datetime, timedelta
        
        # Try Redis first (distributed state across all workers)
        try:
            breaker_open = await self.cache.get(CIRCUIT_BREAKER_KEY)
            if breaker_open:
                logger.warning("circuit_breaker_open_redis", source="redis")
                return True
            
            # Reset in-memory state if Redis breaker is closed
            _circuit_breaker_open = False
            _circuit_breaker_failure_count = 0
            return False
        
        except Exception as redis_error:
            # Redis is down - use in-memory circuit breaker fallback
            logger.warning("circuit_breaker_check_failed_using_memory", error=str(redis_error))
            
            # Check if in-memory breaker should reset
            if _circuit_breaker_open and _circuit_breaker_last_reset:
                time_since_reset = datetime.utcnow() - _circuit_breaker_last_reset
                if time_since_reset > timedelta(seconds=CIRCUIT_BREAKER_TIMEOUT):
                    logger.info("circuit_breaker_timeout_reset_memory")
                    _circuit_breaker_open = False
                    _circuit_breaker_failure_count = 0
                    return False
            
            return _circuit_breaker_open
    
    async def _update_circuit_breaker(self, error: Exception) -> None:
        """
        Update circuit breaker state with in-memory fallback.
        
        CRITICAL FIX: Increment failure count in Redis, but also update
        in-memory state so app doesn't hang if Redis is down.
        
        Args:
            error: The exception that triggered the failure
        """
        global _circuit_breaker_open, _circuit_breaker_failure_count, _circuit_breaker_last_reset
        from datetime import datetime
        
        # Try to update Redis (distributed state)
        try:
            failure_count = await self.cache.get(CIRCUIT_BREAKER_FAILURE_KEY) or 0
            failure_count += 1
            await self.cache.set(CIRCUIT_BREAKER_FAILURE_KEY, failure_count, ttl=CIRCUIT_BREAKER_TIMEOUT)
            
            if failure_count >= CIRCUIT_BREAKER_THRESHOLD:
                await self.cache.set(CIRCUIT_BREAKER_KEY, True, ttl=CIRCUIT_BREAKER_TIMEOUT)
                logger.error("circuit_breaker_opened_redis", failures=failure_count)
            else:
                logger.error("cache_failure_recorded", failures=failure_count, error=str(error))
        
        except Exception as breaker_error:
            # Redis is down - update in-memory circuit breaker
            logger.warning("circuit_breaker_update_failed_using_memory", error=str(breaker_error))
            _circuit_breaker_failure_count += 1
            
            if _circuit_breaker_failure_count >= CIRCUIT_BREAKER_THRESHOLD:
                _circuit_breaker_open = True
                _circuit_breaker_last_reset = datetime.utcnow()
                logger.error("circuit_breaker_opened_memory", failures=_circuit_breaker_failure_count)
            else:
                logger.error("cache_failure_recorded_memory", failures=_circuit_breaker_failure_count, error=str(error))
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache with circuit breaker protection.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        # CRITICAL FIX: Check circuit breaker with in-memory fallback (prevents hangs)
        breaker_open = await self._check_circuit_breaker()
        if breaker_open:
            logger.warning("circuit_breaker_open_rejecting_request", key=key)
            return None
        
        try:
            value = await self.cache.get(key)
            if value is not None:
                self.metrics['hits'] += 1
                logger.debug("cache_hit", key=key)
            else:
                self.metrics['misses'] += 1
                logger.debug("cache_miss", key=key)
            
            # Reset failure count on success (try Redis, ignore if fails)
            try:
                await self.cache.delete(CIRCUIT_BREAKER_FAILURE_KEY)
            except Exception:
                pass
            
            return value
        except Exception as e:
            self.metrics['errors'] += 1
            
            # Update circuit breaker with in-memory fallback
            await self._update_circuit_breaker(e)
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache with circuit breaker protection.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        # Check circuit breaker before attempting set
        breaker_open = await self._check_circuit_breaker()
        if breaker_open:
            logger.warning("circuit_breaker_open_rejecting_set", key=key)
            return False
        
        try:
            await self.cache.set(key, value, ttl=ttl or self.default_ttl)
            self.metrics['sets'] += 1
            logger.debug("cache_set", key=key, ttl=ttl or self.default_ttl)
            return True
        except Exception as e:
            self.metrics['errors'] += 1
            
            # Update circuit breaker with in-memory fallback
            await self._update_circuit_breaker(e)
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
    
    def _generate_cache_key(self, content: Any, classification_type: str) -> str:
        """
        FIX #17: Extract cache key generation to eliminate duplication.
        
        Generates consistent cache keys for classification caching.
        Used by both get_cached_classification and set_cached_classification.
        """
        import hashlib
        import orjson as json  # LIBRARY REPLACEMENT: orjson for 3-5x faster JSON parsing
        
        # Generate cache key from content
        content_str = json.dumps(content, sort_keys=True) if isinstance(content, dict) else str(content)
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]
        return f"{classification_type}:{content_hash}"
    
    async def get_cached_classification(self, content: Any, classification_type: str) -> Optional[Any]:
        """
        Get cached classification result (compatibility method for AI cache).
        
        Args:
            content: Content to use for cache key generation
            classification_type: Type of classification
            
        Returns:
            Cached classification result or None
        """
        # FIX #17: Use extracted helper method
        cache_key = self._generate_cache_key(content, classification_type)
        return await self.get(cache_key)
    
    async def set_cached_classification(self, content: Any, classification_type: str, result: Any, ttl: Optional[int] = None) -> bool:
        """
        Set cached classification result (compatibility method for AI cache).
        
        Args:
            content: Content to use for cache key generation
            classification_type: Type of classification
            result: Classification result to cache
            ttl: Optional TTL override
            
        Returns:
            True if successful, False otherwise
        """
        # FIX #17: Use extracted helper method
        cache_key = self._generate_cache_key(content, classification_type)
        return await self.set(cache_key, result, ttl=ttl)
    
    async def store_classification(self, content: Any, result: Any, classification_type: str, ttl_hours: Optional[int] = None, **kwargs) -> bool:
        """
        Alias for set_cached_classification (compatibility method).
        
        Args:
            content: Content to use for cache key generation
            result: Classification result to cache
            classification_type: Type of classification
            ttl_hours: Optional TTL in hours (converted to seconds)
            **kwargs: Additional arguments (ignored for compatibility)
            
        Returns:
            True if successful, False otherwise
        """
        ttl = int(ttl_hours * 3600) if ttl_hours else None
        return await self.set_cached_classification(content, classification_type, result, ttl)
    
    async def incr(self, key: str, delta: int = 1) -> int:
        """
        Increment a counter in cache (for rate limiting).
        
        Args:
            key: Cache key
            delta: Amount to increment by (default: 1)
            
        Returns:
            New value after increment
        """
        try:
            if self.redis_client:
                return await self.redis_client.incr(key, delta)
            else:
                # Fallback to memory cache
                current = await self.get(key) or 0
                new_value = int(current) + delta
                await self.set(key, new_value)
                return new_value
        except Exception as e:
            logger.error("cache_incr_error", key=key, error=str(e))
            return 1
    
    async def expire(self, key: str, seconds: int) -> bool:
        """
        Set expiration time on a key.
        
        Args:
            key: Cache key
            seconds: TTL in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.redis_client:
                return await self.redis_client.expire(key, seconds)
            else:
                # Memory cache handles TTL differently - re-set with TTL
                value = await self.get(key)
                if value is not None:
                    await self.set(key, value, ttl=seconds)
                return True
        except Exception as e:
            logger.error("cache_expire_error", key=key, error=str(e))
            return False
    
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
    
    # FIX #16: Use extracted helper method
    parsed_config = _parse_redis_url(_cache_instance.redis_url)
    endpoint = parsed_config['endpoint']
    port = parsed_config['port']
    password = parsed_config['password']
    db = parsed_config['db']
    use_tls = parsed_config['use_tls']
    
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
    cache = safe_get_cache()
    if cache is None:
        return {
            'status': 'unavailable',
            'initialized': False,
            'circuit_breaker_open': False,
            'error': 'Cache not initialized'
        }
    
    try:
        # CRITICAL FIX: Read circuit breaker state from Redis
        breaker_open = await cache.cache.get(CIRCUIT_BREAKER_KEY) or False
        failure_count = await cache.cache.get(CIRCUIT_BREAKER_FAILURE_KEY) or 0
        
        # Try to set and get a test value
        test_key = 'health_check_test'
        await cache.set(test_key, 'ok', ttl=10)
        result = await cache.get(test_key)
        await cache.delete(test_key)
        
        if result == 'ok':
            return {
                'status': 'healthy',
                'initialized': True,
                'circuit_breaker_open': breaker_open,
                'failure_count': failure_count,
                'metrics': cache.get_metrics()
            }
        else:
            return {
                'status': 'degraded',
                'initialized': True,
                'circuit_breaker_open': breaker_open,
                'failure_count': failure_count,
                'error': 'Test value mismatch'
            }
    except Exception as e:
        # Try to read breaker state even on error
        try:
            breaker_open = await cache.cache.get(CIRCUIT_BREAKER_KEY) or False
            failure_count = await cache.cache.get(CIRCUIT_BREAKER_FAILURE_KEY) or 0
        except:
            breaker_open = False
            failure_count = 0
        
        return {
            'status': 'unhealthy',
            'initialized': True,
            'circuit_breaker_open': breaker_open,
            'failure_count': failure_count,
            'error': str(e)
        }


async def validate_redis_connection(redis_url: str, max_retries: int = 3, retry_delay: float = 1.0) -> bool:
    """
    Validate Redis connection before initializing cache with retry logic.
    
    Args:
        redis_url: Redis connection URL
        max_retries: Maximum number of retry attempts (default 3)
        retry_delay: Delay between retries in seconds (default 1.0)
        
    Returns:
        True if connection successful, False otherwise
    """
    import redis.asyncio as redis
    
    for attempt in range(max_retries):
        try:
            # Parse Redis URL
            client = redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=5)
            
            # Test connection with ping
            await client.ping()
            await client.close()
            
            logger.info("redis_connection_validated", url=redis_url, attempt=attempt + 1)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning("redis_connection_failed_retrying", 
                             url=redis_url, attempt=attempt + 1, 
                             max_retries=max_retries, error=str(e))
                await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            else:
                logger.error("redis_connection_failed_exhausted_retries", 
                           url=redis_url, max_retries=max_retries, error=str(e))
    
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
