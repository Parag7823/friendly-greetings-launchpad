"""
Rate limiting and concurrency control using aiometer library + Redis.
Replaces manual asyncio.Semaphore usage throughout the codebase.

CRITICAL FIX #1: Global rate limiting across all users
- Prevents 50+ users from overwhelming provider APIs
- Uses Redis for distributed rate limiting across workers
- Implements queue depth limits per provider and user
- Tracks active syncs globally
"""

import asyncio
import time
import structlog
from typing import Callable, Any, TypeVar, Coroutine, Optional, Tuple
from aiometer import aiohttp_limiter, AsyncRateLimiter
from core_infrastructure.config_manager import get_connector_config
from core_infrastructure.centralized_cache import safe_get_cache

T = TypeVar('T')
logger = structlog.get_logger(__name__)

# Singleton rate limiter instance
_rate_limiter: AsyncRateLimiter = None


def get_rate_limiter() -> AsyncRateLimiter:
    """Get or create singleton rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        config = get_connector_config()
        _rate_limiter = AsyncRateLimiter(
            max_rate=config.rate_limit_per_second,
            time_period=1
        )
    return _rate_limiter


# CRITICAL FIX #1: Global rate limiting using Redis
class GlobalRateLimiter:
    """
    Distributed rate limiter using Redis.
    Prevents multiple users from overwhelming provider APIs.
    
    Example:
        limiter = GlobalRateLimiter()
        can_sync, msg = await limiter.check_global_rate_limit('gmail', user_id)
        if not can_sync:
            raise HTTPException(status_code=429, detail=msg)
    """
    
    def __init__(self):
        self.cache = safe_get_cache()
        self.config = get_connector_config()
    
    async def check_global_rate_limit(
        self, 
        provider: str, 
        user_id: str
    ) -> Tuple[bool, str]:
        """
        Check if sync is allowed based on global rate limits.
        
        Returns:
            (can_sync: bool, message: str)
        """
        if not self.cache:
            logger.warning("Redis cache unavailable, skipping global rate limit check")
            return True, "OK"
        
        try:
            # Key for tracking active syncs per provider (across all users)
            active_syncs_key = f"sync:active:{provider}"
            
            # Key for tracking syncs per user per provider
            user_syncs_key = f"sync:active:{provider}:{user_id}"
            
            # Get current counts
            active_syncs = await self.cache.get(active_syncs_key) or 0
            user_syncs = await self.cache.get(user_syncs_key) or 0
            
            # Check global limit (max syncs per minute per provider)
            if active_syncs >= self.config.global_max_syncs_per_minute:
                msg = (
                    f"Global rate limit exceeded for {provider}. "
                    f"Active syncs: {active_syncs}/{self.config.global_max_syncs_per_minute}. "
                    f"Please try again in 1 minute."
                )
                logger.warning("global_rate_limit_exceeded", provider=provider, active_syncs=active_syncs)
                return False, msg
            
            # Check per-user limit
            if user_syncs >= self.config.global_max_queued_syncs_per_user:
                msg = (
                    f"User rate limit exceeded for {provider}. "
                    f"Your active syncs: {user_syncs}/{self.config.global_max_queued_syncs_per_user}. "
                    f"Please wait for current syncs to complete."
                )
                logger.warning("user_rate_limit_exceeded", provider=provider, user_id=user_id, user_syncs=user_syncs)
                return False, msg
            
            return True, "OK"
            
        except Exception as e:
            logger.error("rate_limit_check_failed", error=str(e), provider=provider)
            # Fail open - allow sync if Redis fails
            return True, "OK"
    
    async def acquire_sync_slot(
        self, 
        provider: str, 
        user_id: str
    ) -> bool:
        """
        Acquire a sync slot. Call this BEFORE queuing sync job.
        
        Returns:
            True if slot acquired, False if limit reached
        """
        if not self.cache:
            return True
        
        try:
            active_syncs_key = f"sync:active:{provider}"
            user_syncs_key = f"sync:active:{provider}:{user_id}"
            
            # Atomic increment with TTL (1 minute)
            active_syncs = await self.cache.incr(active_syncs_key)
            await self.cache.expire(active_syncs_key, 60)
            
            user_syncs = await self.cache.incr(user_syncs_key)
            await self.cache.expire(user_syncs_key, 60)
            
            logger.info(
                "sync_slot_acquired",
                provider=provider,
                user_id=user_id,
                active_syncs=active_syncs,
                user_syncs=user_syncs
            )
            return True
            
        except Exception as e:
            logger.error("sync_slot_acquisition_failed", error=str(e))
            return True  # Fail open
    
    async def release_sync_slot(
        self, 
        provider: str, 
        user_id: str
    ) -> None:
        """
        Release a sync slot. Call this AFTER sync completes (success or failure).
        """
        if not self.cache:
            return
        
        try:
            active_syncs_key = f"sync:active:{provider}"
            user_syncs_key = f"sync:active:{provider}:{user_id}"
            
            # Atomic decrement (don't go below 0)
            active_syncs = await self.cache.decr(active_syncs_key)
            if active_syncs < 0:
                await self.cache.set(active_syncs_key, 0)
            
            user_syncs = await self.cache.decr(user_syncs_key)
            if user_syncs < 0:
                await self.cache.set(user_syncs_key, 0)
            
            logger.info(
                "sync_slot_released",
                provider=provider,
                user_id=user_id,
                active_syncs=max(0, active_syncs),
                user_syncs=max(0, user_syncs)
            )
            
        except Exception as e:
            logger.error("sync_slot_release_failed", error=str(e))


# CRITICAL FIX #2: Distributed sync locking to prevent duplicate syncs
class DistributedSyncLock:
    """
    Distributed lock using Redis to prevent concurrent syncs for same connection.
    
    Prevents: User clicks "Sync Gmail" twice → TWO concurrent jobs
    
    Example:
        lock = DistributedSyncLock()
        acquired = await lock.acquire_sync_lock(user_id, provider, connection_id)
        if not acquired:
            raise HTTPException(status_code=409, detail="Sync already in progress")
        
        try:
            # Run sync...
        finally:
            await lock.release_sync_lock(user_id, provider, connection_id)
    """
    
    def __init__(self):
        self.cache = safe_get_cache()
        self.config = get_connector_config()
    
    def _get_lock_key(self, user_id: str, provider: str, connection_id: str) -> str:
        """Generate lock key for a specific sync."""
        return f"sync_lock:{user_id}:{provider}:{connection_id}"
    
    async def acquire_sync_lock(
        self, 
        user_id: str, 
        provider: str, 
        connection_id: str,
        timeout_seconds: Optional[int] = None
    ) -> bool:
        """
        Acquire lock for sync. Returns False if already locked.
        
        Args:
            user_id: User ID
            provider: Provider name (gmail, quickbooks, etc.)
            connection_id: Connection ID
            timeout_seconds: Lock timeout (default: config.sync_lock_expiry_seconds)
        
        Returns:
            True if lock acquired, False if already locked
        """
        if not self.cache:
            logger.warning("Redis cache unavailable, skipping sync lock")
            return True
        
        try:
            lock_key = self._get_lock_key(user_id, provider, connection_id)
            timeout = timeout_seconds or self.config.sync_lock_expiry_seconds
            
            # Try to set lock with NX (only if not exists) and EX (expiry)
            lock_value = f"{user_id}:{provider}:{connection_id}:{int(time.time())}"
            
            # Use Redis SET NX EX for atomic lock acquisition
            acquired = await self.cache.set(lock_key, lock_value, ex=timeout, nx=True)
            
            if acquired:
                logger.info(
                    "sync_lock_acquired",
                    user_id=user_id,
                    provider=provider,
                    connection_id=connection_id,
                    timeout=timeout
                )
                return True
            else:
                logger.warning(
                    "sync_lock_already_held",
                    user_id=user_id,
                    provider=provider,
                    connection_id=connection_id
                )
                return False
                
        except Exception as e:
            logger.error("sync_lock_acquisition_failed", error=str(e))
            # Fail open - allow sync if Redis fails
            return True
    
    async def release_sync_lock(
        self, 
        user_id: str, 
        provider: str, 
        connection_id: str
    ) -> None:
        """
        Release lock for sync.
        """
        if not self.cache:
            return
        
        try:
            lock_key = self._get_lock_key(user_id, provider, connection_id)
            await self.cache.delete(lock_key)
            
            logger.info(
                "sync_lock_released",
                user_id=user_id,
                provider=provider,
                connection_id=connection_id
            )
            
        except Exception as e:
            logger.error("sync_lock_release_failed", error=str(e))
    
    async def is_locked(
        self, 
        user_id: str, 
        provider: str, 
        connection_id: str
    ) -> bool:
        """Check if sync is currently locked."""
        if not self.cache:
            return False
        
        try:
            lock_key = self._get_lock_key(user_id, provider, connection_id)
            exists = await self.cache.exists(lock_key)
            return bool(exists)
        except Exception as e:
            logger.error("sync_lock_check_failed", error=str(e))
            return False


# Singleton instances
_global_rate_limiter: Optional[GlobalRateLimiter] = None
_sync_lock: Optional[DistributedSyncLock] = None


def get_global_rate_limiter() -> GlobalRateLimiter:
    """Get or create global rate limiter."""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = GlobalRateLimiter()
    return _global_rate_limiter


def get_sync_lock() -> DistributedSyncLock:
    """Get or create distributed sync lock."""
    global _sync_lock
    if _sync_lock is None:
        _sync_lock = DistributedSyncLock()
    return _sync_lock


class ConcurrencyLimiter:
    """
    Replaces manual asyncio.Semaphore usage.
    Provides cleaner API for rate-limited concurrent operations.
    """
    
    def __init__(self, max_concurrency: int = None):
        """Initialize with max concurrent operations."""
        if max_concurrency is None:
            config = get_connector_config()
            max_concurrency = config.concurrency
        
        self.max_concurrency = max(1, min(max_concurrency, 10))
        self.semaphore = asyncio.Semaphore(self.max_concurrency)
    
    async def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run a coroutine with concurrency control."""
        async with self.semaphore:
            return await coro
    
    async def run_batch(self, coros: list[Coroutine[Any, Any, T]]) -> list[T]:
        """Run multiple coroutines with concurrency control."""
        tasks = [asyncio.create_task(self.run(coro)) for coro in coros]
        return await asyncio.gather(*tasks, return_exceptions=True)


async def rate_limited_batch(
    items: list[Any],
    async_fn: Callable[[Any], Coroutine[Any, Any, T]],
    max_concurrency: int = None
) -> list[T]:
    """
    Execute async function on items with rate limiting and concurrency control.
    
    Example:
        results = await rate_limited_batch(
            files,
            async_fn=download_file,
            max_concurrency=5
        )
    """
    limiter = ConcurrencyLimiter(max_concurrency)
    coros = [async_fn(item) for item in items]
    return await limiter.run_batch(coros)


# ISSUE #8: Sync deduplication using idempotency keys
class SyncDeduplicator:
    """
    Prevents duplicate sync requests using idempotency keys.
    Tracks recent syncs (last 5 minutes) in Redis.
    
    Example:
        dedup = SyncDeduplicator()
        idempotency_key = dedup.generate_key(user_id, provider, connection_id, mode)
        
        # Check if sync already in progress
        cached_result = await dedup.get_cached_result(idempotency_key)
        if cached_result:
            return cached_result  # Return cached result
        
        # Run sync...
        await dedup.cache_result(idempotency_key, result)
    """
    
    def __init__(self):
        self.cache = safe_get_cache()
        self.config = get_connector_config()
    
    def generate_key(
        self, 
        user_id: str, 
        provider: str, 
        connection_id: str, 
        mode: str = 'incremental'
    ) -> str:
        """Generate idempotency key for sync request."""
        return f"sync:dedup:{user_id}:{provider}:{connection_id}:{mode}"
    
    async def get_cached_result(self, idempotency_key: str) -> Optional[dict]:
        """Get cached result if sync was recently completed."""
        if not self.cache:
            return None
        
        try:
            result = await self.cache.get(idempotency_key)
            if result:
                logger.info("sync_dedup_cache_hit", key=idempotency_key)
                return result
            return None
        except Exception as e:
            logger.error("sync_dedup_get_failed", error=str(e))
            return None
    
    async def cache_result(
        self, 
        idempotency_key: str, 
        result: dict,
        ttl_seconds: int = 300
    ) -> None:
        """Cache sync result for 5 minutes."""
        if not self.cache:
            return
        
        try:
            await self.cache.set(idempotency_key, result, ex=ttl_seconds)
            logger.info("sync_dedup_result_cached", key=idempotency_key, ttl=ttl_seconds)
        except Exception as e:
            logger.error("sync_dedup_cache_failed", error=str(e))


# ISSUE #9: Scheduler rate limiting with exponential backoff
class SchedulerRateLimiter:
    """
    Rate limits scheduler dispatch to prevent overwhelming workers.
    Implements exponential backoff on failures.
    
    Example:
        scheduler_limiter = SchedulerRateLimiter()
        can_dispatch, msg = await scheduler_limiter.check_scheduler_rate_limit(provider)
        if not can_dispatch:
            return {"status": "rate_limited", "message": msg}
        
        # Dispatch syncs...
        await scheduler_limiter.record_dispatch(provider, count=5)
        await scheduler_limiter.record_failure(provider)  # On error
    """
    
    def __init__(self):
        self.cache = safe_get_cache()
        self.config = get_connector_config()
    
    async def check_scheduler_rate_limit(self, provider: str) -> Tuple[bool, str]:
        """
        Check if scheduler can dispatch syncs for provider.
        Limit: 5 syncs per minute per provider
        
        Returns:
            (can_dispatch: bool, message: str)
        """
        if not self.cache:
            return True, "OK"
        
        try:
            # Key for tracking scheduler dispatches
            dispatch_key = f"scheduler:dispatch:{provider}"
            failure_key = f"scheduler:failure:{provider}"
            
            # Get current dispatch count and failure count
            dispatch_count = await self.cache.get(dispatch_key) or 0
            failure_count = await self.cache.get(failure_key) or 0
            
            # Calculate backoff: exponential backoff on failures
            # 0 failures: allow 5 syncs/min
            # 1 failure: allow 4 syncs/min
            # 2 failures: allow 3 syncs/min
            # 3+ failures: allow 1 sync/min
            max_dispatches = max(1, 5 - int(failure_count))
            
            if dispatch_count >= max_dispatches:
                msg = (
                    f"Scheduler rate limit for {provider}: "
                    f"{dispatch_count}/{max_dispatches} syncs dispatched this minute. "
                    f"Failures: {failure_count}. Backoff active."
                )
                logger.warning("scheduler_rate_limit_exceeded", provider=provider, 
                             dispatch_count=dispatch_count, failure_count=failure_count)
                return False, msg
            
            return True, "OK"
            
        except Exception as e:
            logger.error("scheduler_rate_limit_check_failed", error=str(e))
            return True, "OK"  # Fail open
    
    async def record_dispatch(self, provider: str, count: int = 1) -> None:
        """Record scheduler dispatch."""
        if not self.cache:
            return
        
        try:
            dispatch_key = f"scheduler:dispatch:{provider}"
            current = await self.cache.get(dispatch_key) or 0
            new_count = current + count
            
            # Reset counter every minute
            await self.cache.set(dispatch_key, new_count, ex=60)
            
            logger.info("scheduler_dispatch_recorded", provider=provider, count=count, total=new_count)
        except Exception as e:
            logger.error("scheduler_dispatch_record_failed", error=str(e))
    
    async def record_failure(self, provider: str) -> None:
        """Record scheduler failure for exponential backoff."""
        if not self.cache:
            return
        
        try:
            failure_key = f"scheduler:failure:{provider}"
            current = await self.cache.get(failure_key) or 0
            new_count = current + 1
            
            # Keep failure count for 5 minutes
            await self.cache.set(failure_key, new_count, ex=300)
            
            logger.warning("scheduler_failure_recorded", provider=provider, failure_count=new_count)
        except Exception as e:
            logger.error("scheduler_failure_record_failed", error=str(e))
    
    async def reset_failure_count(self, provider: str) -> None:
        """Reset failure count on successful dispatch."""
        if not self.cache:
            return
        
        try:
            failure_key = f"scheduler:failure:{provider}"
            await self.cache.delete(failure_key)
            logger.info("scheduler_failure_count_reset", provider=provider)
        except Exception as e:
            logger.error("scheduler_failure_reset_failed", error=str(e))


# Singleton instances
_sync_deduplicator: Optional[SyncDeduplicator] = None
_scheduler_rate_limiter: Optional[SchedulerRateLimiter] = None


def get_sync_deduplicator() -> SyncDeduplicator:
    """Get or create sync deduplicator."""
    global _sync_deduplicator
    if _sync_deduplicator is None:
        _sync_deduplicator = SyncDeduplicator()
    return _sync_deduplicator


def get_scheduler_rate_limiter() -> SchedulerRateLimiter:
    """Get or create scheduler rate limiter."""
    global _scheduler_rate_limiter
    if _scheduler_rate_limiter is None:
        _scheduler_rate_limiter = SchedulerRateLimiter()
    return _scheduler_rate_limiter
