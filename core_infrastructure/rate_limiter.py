"""
Rate limiting and concurrency control using aiometer library.
Replaces manual asyncio.Semaphore usage throughout the codebase.
"""

import asyncio
from typing import Callable, Any, TypeVar, Coroutine
from aiometer import aiohttp_limiter, AsyncRateLimiter
from core_infrastructure.config_manager import get_connector_config

T = TypeVar('T')

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
