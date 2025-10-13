"""
AI Call Deduplication and Caching System
This module provides intelligent caching for AI classification calls to reduce costs by 90%
"""

import hashlib
import json
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import asyncio
from enum import Enum

logger = logging.getLogger(__name__)

class CacheStatus(Enum):
    HIT = "hit"
    MISS = "miss"
    EXPIRED = "expired"
    ERROR = "error"

@dataclass
class CacheEntry:
    """Cache entry for AI classification results"""
    content_hash: str
    classification_result: Dict[str, Any]
    created_at: datetime
    expires_at: datetime
    hit_count: int = 0
    last_accessed: Optional[datetime] = None
    confidence_score: float = 0.0
    model_version: str = "gpt-3.5-turbo"

@dataclass
class CacheStats:
    """Cache performance statistics"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    expired_entries: int = 0
    cost_savings_usd: float = 0.0
    avg_response_time_ms: float = 0.0

class AIClassificationCache:
    """
    Intelligent caching system for AI classification results.
    
    Features:
    - Content-based hashing for deduplication
    - TTL-based expiration
    - LRU eviction policy
    - Cost tracking and savings calculation
    - Performance monitoring
    """
    
    def __init__(self, 
                 max_cache_size: Optional[int] = None,
                 default_ttl_hours: Optional[int] = None,
                 cost_per_1k_tokens: Optional[float] = None):
        import os
        # Make all parameters configurable via environment variables
        self.cache: Dict[str, CacheEntry] = {}
        self.max_cache_size = max_cache_size or int(os.getenv('AI_CACHE_MAX_SIZE', '10000'))
        self.default_ttl_hours = default_ttl_hours or int(os.getenv('AI_CACHE_TTL_HOURS', '24'))
        self.cost_per_1k_tokens = cost_per_1k_tokens or float(os.getenv('AI_CACHE_COST_PER_1K', '0.002'))
        self.stats = CacheStats()
        self._lock = asyncio.Lock()
    
    def _generate_content_hash(self, content: Any) -> str:
        """Generate consistent hash for content deduplication"""
        try:
            # Normalize content for consistent hashing
            if isinstance(content, dict):
                # Sort keys for consistent hashing
                normalized = json.dumps(content, sort_keys=True, default=str)
            elif isinstance(content, (list, tuple)):
                normalized = json.dumps(sorted(content) if all(isinstance(x, (str, int, float)) for x in content) else content, default=str)
            else:
                normalized = str(content)
            
            return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to generate content hash: {e}")
            return hashlib.sha256(str(content).encode('utf-8')).hexdigest()
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired"""
        return datetime.utcnow() > entry.expires_at
    
    def _evict_expired_entries(self):
        """Remove expired entries from cache"""
        expired_keys = [
            key for key, entry in self.cache.items() 
            if self._is_expired(entry)
        ]
        
        for key in expired_keys:
            del self.cache[key]
            self.stats.expired_entries += 1
        
        if expired_keys:
            logger.info(f"Evicted {len(expired_keys)} expired cache entries")
    
    def _evict_lru_entries(self):
        """Evict least recently used entries if cache is full"""
        if len(self.cache) <= self.max_cache_size:
            return
        
        # Sort by last_accessed (None values go first)
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].last_accessed or datetime.min
        )
        
        # Remove oldest entries
        entries_to_remove = len(self.cache) - self.max_cache_size + 1
        for i in range(entries_to_remove):
            key_to_remove = sorted_entries[i][0]
            del self.cache[key_to_remove]
        
        logger.info(f"Evicted {entries_to_remove} LRU cache entries")
    
    async def get_cached_classification(self, 
                                      content: Any,
                                      classification_type: str = "row_classification") -> Optional[Dict[str, Any]]:
        """
        Get cached AI classification result if available.
        
        Args:
            content: Content to classify (row data, text, etc.)
            classification_type: Type of classification for cache segmentation
            
        Returns:
            Cached classification result or None if not found/expired
        """
        async with self._lock:
            start_time = time.time()
            self.stats.total_requests += 1
            
            try:
                # Generate content hash
                content_hash = f"{classification_type}:{self._generate_content_hash(content)}"
                
                # Check if entry exists
                if content_hash not in self.cache:
                    self.stats.cache_misses += 1
                    return None
                
                entry = self.cache[content_hash]
                
                # Check if expired
                if self._is_expired(entry):
                    del self.cache[content_hash]
                    self.stats.expired_entries += 1
                    self.stats.cache_misses += 1
                    return None
                
                # Update access tracking
                entry.hit_count += 1
                entry.last_accessed = datetime.utcnow()
                
                # Update stats
                self.stats.cache_hits += 1
                
                # Calculate cost savings (approximate)
                estimated_tokens = len(str(content)) // 4  # Rough token estimation
                cost_saved = (estimated_tokens / 1000) * self.cost_per_1k_tokens
                self.stats.cost_savings_usd += cost_saved
                
                response_time = (time.time() - start_time) * 1000
                self.stats.avg_response_time_ms = (
                    (self.stats.avg_response_time_ms * (self.stats.total_requests - 1) + response_time) 
                    / self.stats.total_requests
                )
                
                logger.debug(f"Cache HIT for {classification_type}: {content_hash[:8]}... (saved ${cost_saved:.4f})")
                
                return entry.classification_result
                
            except Exception as e:
                logger.error(f"Error retrieving from AI cache: {e}")
                return None
    
    async def store_classification(self,
                                 content: Any,
                                 classification_result: Dict[str, Any],
                                 classification_type: str = "row_classification",
                                 ttl_hours: Optional[int] = None,
                                 confidence_score: float = 0.0,
                                 model_version: str = "gpt-3.5-turbo") -> bool:
        """
        Store AI classification result in cache.
        
        Args:
            content: Original content that was classified
            classification_result: AI classification result to cache
            classification_type: Type of classification for cache segmentation
            ttl_hours: Time to live in hours (uses default if None)
            confidence_score: Confidence score of the classification
            model_version: AI model version used
            
        Returns:
            True if stored successfully, False otherwise
        """
        async with self._lock:
            try:
                # Generate content hash
                content_hash = f"{classification_type}:{self._generate_content_hash(content)}"
                
                # Calculate expiration
                ttl = ttl_hours or self.default_ttl_hours
                expires_at = datetime.utcnow() + timedelta(hours=ttl)
                
                # Create cache entry
                entry = CacheEntry(
                    content_hash=content_hash,
                    classification_result=classification_result,
                    created_at=datetime.utcnow(),
                    expires_at=expires_at,
                    confidence_score=confidence_score,
                    model_version=model_version
                )
                
                # Clean up before storing
                self._evict_expired_entries()
                self._evict_lru_entries()
                
                # Store entry
                self.cache[content_hash] = entry
                
                logger.debug(f"Cached {classification_type} result: {content_hash[:8]}... (expires in {ttl}h)")
                
                return True
                
            except Exception as e:
                logger.error(f"Error storing in AI cache: {e}")
                return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        hit_rate = (self.stats.cache_hits / self.stats.total_requests * 100) if self.stats.total_requests > 0 else 0
        
        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.max_cache_size,
            "total_requests": self.stats.total_requests,
            "cache_hits": self.stats.cache_hits,
            "cache_misses": self.stats.cache_misses,
            "hit_rate_percent": round(hit_rate, 2),
            "expired_entries": self.stats.expired_entries,
            "cost_savings_usd": round(self.stats.cost_savings_usd, 4),
            "avg_response_time_ms": round(self.stats.avg_response_time_ms, 2),
            "estimated_monthly_savings": round(self.stats.cost_savings_usd * 30, 2)
        }
    
    async def clear_cache(self):
        """Clear all cache entries"""
        async with self._lock:
            self.cache.clear()
            logger.info("AI classification cache cleared")
    
    async def cleanup_expired(self):
        """Manual cleanup of expired entries"""
        async with self._lock:
            self._evict_expired_entries()

# Global cache instance
_ai_cache: Optional[AIClassificationCache] = None

def initialize_ai_cache(max_cache_size: Optional[int] = None,
                       default_ttl_hours: Optional[int] = None,
                       cost_per_1k_tokens: Optional[float] = None) -> AIClassificationCache:
    """Initialize global AI cache instance with environment variable support"""
    global _ai_cache
    _ai_cache = AIClassificationCache(
        max_cache_size=max_cache_size,
        default_ttl_hours=default_ttl_hours,
        cost_per_1k_tokens=cost_per_1k_tokens
    )
    logger.info(f"AI classification cache initialized (max_size={_ai_cache.max_cache_size}, ttl={_ai_cache.default_ttl_hours}h, cost_per_1k=${_ai_cache.cost_per_1k_tokens})")
    return _ai_cache

def get_ai_cache() -> AIClassificationCache:
    """Get global AI cache instance"""
    if _ai_cache is None:
        raise RuntimeError("AI cache not initialized. Call initialize_ai_cache() first.")
    return _ai_cache

# Safe accessor that never raises, returns a no-op cache if uninitialized
class NullAIClassificationCache:
    """No-op cache to safely operate in degraded mode"""
    async def get_cached_classification(self, content: Any, classification_type: str = "row_classification"):
        return None

    async def store_classification(self, content: Any, classification_result: Dict[str, Any],
                                   classification_type: str = "row_classification", ttl_hours: Optional[int] = None,
                                   confidence_score: float = 0.0, model_version: str = "") -> bool:
        return False

def safe_get_ai_cache():
    """Return AI cache if initialized; otherwise return a no-op cache to avoid crashes."""
    try:
        return get_ai_cache()
    except Exception:
        return NullAIClassificationCache()

# Decorator for automatic caching
def cache_ai_classification(classification_type: str = "row_classification", 
                          ttl_hours: Optional[int] = None):
    """
    Decorator to automatically cache AI classification function results.
    
    Usage:
        @cache_ai_classification("platform_detection", ttl_hours=48)
        async def classify_platform(data):
            # AI classification logic
            return result
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache = get_ai_cache()
            
            # Create cache key from function arguments
            cache_key = {
                'function': func.__name__,
                'args': args,
                'kwargs': kwargs
            }
            
            # Try to get from cache first
            cached_result = await cache.get_cached_classification(cache_key, classification_type)
            if cached_result is not None:
                return cached_result
            
            # Call original function
            result = await func(*args, **kwargs)
            
            # Store result in cache
            if result is not None:
                await cache.store_classification(
                    cache_key, 
                    result, 
                    classification_type, 
                    ttl_hours
                )
            
            return result
        
        return wrapper
    return decorator
