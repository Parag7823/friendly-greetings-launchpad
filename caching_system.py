"""
Production-Grade Caching System
Implements Redis-based caching with in-memory fallback for high-performance data access.
"""

import json
import time
import hashlib
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pickle
import os

logger = logging.getLogger(__name__)

# Try to import Redis, fall back to in-memory if not available
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, using in-memory cache only")

@dataclass
class CacheConfig:
    """Configuration for caching system"""
    redis_url: str = os.getenv('REDIS_URL', 'redis://localhost:6379')
    redis_password: Optional[str] = os.getenv('REDIS_PASSWORD')
    redis_db: int = int(os.getenv('REDIS_DB', '0'))
    default_ttl: int = int(os.getenv('CACHE_DEFAULT_TTL', '3600'))  # 1 hour
    max_memory_mb: int = int(os.getenv('CACHE_MAX_MEMORY_MB', '512'))
    enable_compression: bool = os.getenv('CACHE_ENABLE_COMPRESSION', 'true').lower() == 'true'
    enable_serialization: bool = os.getenv('CACHE_ENABLE_SERIALIZATION', 'true').lower() == 'true'

@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    total_size_bytes: int = 0
    last_reset: datetime = None
    
    def __post_init__(self):
        if self.last_reset is None:
            self.last_reset = datetime.utcnow()
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def reset(self):
        """Reset statistics"""
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.errors = 0
        self.total_size_bytes = 0
        self.last_reset = datetime.utcnow()

class InMemoryCache:
    """In-memory cache implementation with LRU eviction"""
    
    def __init__(self, max_size_mb: int = 512):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.current_size = 0
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        async with self._lock:
            if key in self.cache:
                # Check if expired
                if self.cache[key]['expires_at'] < time.time():
                    # Remove expired entry
                    value_size = self._calculate_size(self.cache[key]['value'])
                    del self.cache[key]
                    del self.access_times[key]
                    self.current_size -= value_size
                    return None
                
                # Update access time for LRU
                self.access_times[key] = time.time()
                return self.cache[key]['value']
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL"""
        async with self._lock:
            try:
                # Calculate size
                value_size = self._calculate_size(value)
                
                # Check if we need to evict
                if self.current_size + value_size > self.max_size_bytes:
                    await self._evict_lru()
                
                # Store value
                self.cache[key] = {
                    'value': value,
                    'expires_at': time.time() + ttl,
                    'created_at': time.time()
                }
                self.access_times[key] = time.time()
                self.current_size += value_size
                
                return True
                
            except Exception as e:
                logger.error(f"In-memory cache set failed: {e}")
                return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        async with self._lock:
            if key in self.cache:
                value_size = self._calculate_size(self.cache[key]['value'])
                del self.cache[key]
                del self.access_times[key]
                self.current_size -= value_size
                return True
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        async with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.current_size = 0
            return True
    
    async def cleanup_expired(self) -> int:
        """Remove expired entries"""
        async with self._lock:
            current_time = time.time()
            expired_keys = []
            
            for key, data in self.cache.items():
                if data['expires_at'] < current_time:
                    expired_keys.append(key)
            
            for key in expired_keys:
                value_size = self._calculate_size(self.cache[key]['value'])
                del self.cache[key]
                del self.access_times[key]
                self.current_size -= value_size
            
            return len(expired_keys)
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes"""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, dict):
                return len(str(value))
            else:
                return len(pickle.dumps(value))
        except:
            return 1024  # Default estimate
    
    async def _evict_lru(self):
        """Evict least recently used entries"""
        if not self.access_times:
            return
        
        # Sort by access time and remove oldest
        sorted_keys = sorted(self.access_times.items(), key=lambda x: x[1])
        
        for key, _ in sorted_keys:
            if key in self.cache:
                value_size = self._calculate_size(self.cache[key]['value'])
                del self.cache[key]
                del self.access_times[key]
                self.current_size -= value_size
                
                # Stop if we've freed enough space
                if self.current_size < self.max_size_bytes * 0.8:
                    break
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'entries': len(self.cache),
            'size_bytes': self.current_size,
            'size_mb': self.current_size / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'utilization': self.current_size / self.max_size_bytes
        }

class RedisCache:
    """Redis cache implementation"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self.connected = False
    
    async def connect(self) -> bool:
        """Connect to Redis"""
        if not REDIS_AVAILABLE:
            return False
        
        try:
            self.redis_client = redis.from_url(
                self.config.redis_url,
                password=self.config.redis_password,
                db=self.config.redis_db,
                decode_responses=False  # We'll handle encoding ourselves
            )
            
            # Test connection
            await self.redis_client.ping()
            self.connected = True
            logger.info("Connected to Redis cache")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.connected = False
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        if not self.connected or not self.redis_client:
            return None
        
        try:
            data = await self.redis_client.get(key)
            if data:
                return self._deserialize(data)
            return None
            
        except Exception as e:
            logger.error(f"Redis get failed: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in Redis cache"""
        if not self.connected or not self.redis_client:
            return False
        
        try:
            serialized = self._serialize(value)
            result = await self.redis_client.setex(key, ttl, serialized)
            return result is True
            
        except Exception as e:
            logger.error(f"Redis set failed: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache"""
        if not self.connected or not self.redis_client:
            return False
        
        try:
            result = await self.redis_client.delete(key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Redis delete failed: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        if not self.connected or not self.redis_client:
            return False
        
        try:
            await self.redis_client.flushdb()
            return True
            
        except Exception as e:
            logger.error(f"Redis clear failed: {e}")
            return False
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage"""
        if self.config.enable_serialization:
            return pickle.dumps(value)
        else:
            return json.dumps(value, default=str).encode('utf-8')
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        if data is None:
            return None
        
        if self.config.enable_serialization:
            return pickle.loads(data)
        else:
            return json.loads(data.decode('utf-8'))
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            self.connected = False

class ProductionCache:
    """
    Production-grade caching system with Redis primary and in-memory fallback.
    Provides high availability and performance for data enrichment and document analysis.
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.redis_cache = RedisCache(self.config)
        self.memory_cache = InMemoryCache(self.config.max_memory_mb)
        self.stats = CacheStats()
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the caching system"""
        if self._initialized:
            return True
        
        try:
            # Try to connect to Redis
            redis_connected = await self.redis_cache.connect()
            
            if redis_connected:
                logger.info("✅ Redis cache initialized successfully")
            else:
                logger.warning("⚠️ Redis not available, using in-memory cache only")
            
            # Start cleanup task
            asyncio.create_task(self._cleanup_task())
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Cache initialization failed: {e}")
            self.stats.errors += 1
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with fallback strategy"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Try Redis first
            if self.redis_cache.connected:
                value = await self.redis_cache.get(key)
                if value is not None:
                    self.stats.hits += 1
                    # Update in-memory cache for faster access
                    await self.memory_cache.set(key, value, self.config.default_ttl)
                    return value
            
            # Fall back to in-memory cache
            value = await self.memory_cache.get(key)
            if value is not None:
                self.stats.hits += 1
                return value
            
            self.stats.misses += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            self.stats.errors += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with both Redis and in-memory"""
        if not self._initialized:
            await self.initialize()
        
        ttl = ttl or self.config.default_ttl
        
        try:
            success = True
            
            # Set in Redis
            if self.redis_cache.connected:
                redis_success = await self.redis_cache.set(key, value, ttl)
                if not redis_success:
                    success = False
            
            # Set in memory cache
            memory_success = await self.memory_cache.set(key, value, ttl)
            if not memory_success:
                success = False
            
            if success:
                self.stats.sets += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Cache set failed: {e}")
            self.stats.errors += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from both caches"""
        if not self._initialized:
            await self.initialize()
        
        try:
            success = True
            
            # Delete from Redis
            if self.redis_cache.connected:
                redis_success = await self.redis_cache.delete(key)
                if not redis_success:
                    success = False
            
            # Delete from memory cache
            memory_success = await self.memory_cache.delete(key)
            if not memory_success:
                success = False
            
            if success:
                self.stats.deletes += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Cache delete failed: {e}")
            self.stats.errors += 1
            return False
    
    async def clear(self) -> bool:
        """Clear all caches"""
        if not self._initialized:
            await self.initialize()
        
        try:
            success = True
            
            # Clear Redis
            if self.redis_cache.connected:
                redis_success = await self.redis_cache.clear()
                if not redis_success:
                    success = False
            
            # Clear memory cache
            memory_success = await self.memory_cache.clear()
            if not memory_success:
                success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            self.stats.errors += 1
            return False
    
    def generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from prefix and arguments"""
        # Create deterministic key from arguments
        key_data = {
            'prefix': prefix,
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else {}
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        
        return f"{prefix}:{key_hash}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        memory_stats = self.memory_cache.get_stats()
        
        return {
            'redis_connected': self.redis_cache.connected,
            'memory_cache': memory_stats,
            'performance': {
                'hits': self.stats.hits,
                'misses': self.stats.misses,
                'hit_rate': self.stats.hit_rate,
                'sets': self.stats.sets,
                'deletes': self.stats.deletes,
                'errors': self.stats.errors
            },
            'config': asdict(self.config)
        }
    
    async def _cleanup_task(self):
        """Background task to clean up expired entries"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self.memory_cache.cleanup_expired()
            except Exception as e:
                logger.error(f"Cache cleanup task failed: {e}")
    
    async def close(self):
        """Close cache connections"""
        if self.redis_cache.connected:
            await self.redis_cache.close()

# ============================================================================
# CACHE DECORATORS
# ============================================================================

def cached(ttl: int = 3600, key_prefix: str = "default"):
    """Decorator to cache function results"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cache = ProductionCache()
            await cache.initialize()
            
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator

# ============================================================================
# SPECIALIZED CACHE MANAGERS
# ============================================================================

class EnrichmentCache:
    """Specialized cache for data enrichment results"""
    
    def __init__(self, cache: ProductionCache):
        self.cache = cache
        self.prefix = "enrichment"
    
    async def get_enrichment_result(self, row_data: Dict, file_context: Dict) -> Optional[Dict]:
        """Get cached enrichment result"""
        key = self.cache.generate_key(
            f"{self.prefix}:result",
            row_data=row_data,
            file_context=file_context
        )
        return await self.cache.get(key)
    
    async def set_enrichment_result(self, row_data: Dict, file_context: Dict, result: Dict, ttl: int = 3600):
        """Cache enrichment result"""
        key = self.cache.generate_key(
            f"{self.prefix}:result",
            row_data=row_data,
            file_context=file_context
        )
        await self.cache.set(key, result, ttl)
    
    async def get_vendor_standardization(self, vendor_name: str, platform: str) -> Optional[Dict]:
        """Get cached vendor standardization"""
        key = self.cache.generate_key(
            f"{self.prefix}:vendor",
            vendor_name=vendor_name,
            platform=platform
        )
        return await self.cache.get(key)
    
    async def set_vendor_standardization(self, vendor_name: str, platform: str, result: Dict, ttl: int = 7200):
        """Cache vendor standardization"""
        key = self.cache.generate_key(
            f"{self.prefix}:vendor",
            vendor_name=vendor_name,
            platform=platform
        )
        await self.cache.set(key, result, ttl)

class DocumentAnalysisCache:
    """Specialized cache for document analysis results"""
    
    def __init__(self, cache: ProductionCache):
        self.cache = cache
        self.prefix = "document_analysis"
    
    async def get_document_classification(self, df_hash: str, filename: str) -> Optional[Dict]:
        """Get cached document classification"""
        key = self.cache.generate_key(
            f"{self.prefix}:classification",
            df_hash=df_hash,
            filename=filename
        )
        return await self.cache.get(key)
    
    async def set_document_classification(self, df_hash: str, filename: str, result: Dict, ttl: int = 3600):
        """Cache document classification"""
        key = self.cache.generate_key(
            f"{self.prefix}:classification",
            df_hash=df_hash,
            filename=filename
        )
        await self.cache.set(key, result, ttl)
    
    async def get_platform_detection(self, column_names: List[str], filename: str) -> Optional[Dict]:
        """Get cached platform detection"""
        key = self.cache.generate_key(
            f"{self.prefix}:platform",
            column_names=column_names,
            filename=filename
        )
        return await self.cache.get(key)
    
    async def set_platform_detection(self, column_names: List[str], filename: str, result: Dict, ttl: int = 7200):
        """Cache platform detection"""
        key = self.cache.generate_key(
            f"{self.prefix}:platform",
            column_names=column_names,
            filename=filename
        )
        await self.cache.set(key, result, ttl)

# ============================================================================
# GLOBAL CACHE INSTANCE
# ============================================================================

# Global cache instance
_global_cache: Optional[ProductionCache] = None

async def get_global_cache() -> ProductionCache:
    """Get or create global cache instance"""
    global _global_cache
    
    if _global_cache is None:
        _global_cache = ProductionCache()
        await _global_cache.initialize()
    
    return _global_cache

async def close_global_cache():
    """Close global cache instance"""
    global _global_cache
    
    if _global_cache:
        await _global_cache.close()
        _global_cache = None
