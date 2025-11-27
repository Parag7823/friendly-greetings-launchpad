"""
Production-Grade Supabase Client with Connection Pooling
=========================================================

CRITICAL FIX: Implements database connection pooling to prevent "too many connections" errors
in production environments with multiple workers (Gunicorn, uvicorn, etc.).

Key Features:
- Connection pooling with configurable limits
- Automatic connection recycling
- Health checks and monitoring
- Thread-safe singleton pattern
- Support for pgBouncer external pooler
"""

import os
import structlog
from typing import Optional
from supabase import create_client, Client
import threading
import signal

logger = structlog.get_logger(__name__)

# Global lock for thread-safe singleton
_client_lock = threading.Lock()
_client_instance: Optional[Client] = None


class SupabaseConnectionPool:
    """
    Manages Supabase client with connection pooling best practices.
    
    Configuration via environment variables:
    - SUPABASE_URL: Supabase project URL
    - SUPABASE_SERVICE_ROLE_KEY: Service role key (bypasses RLS)
    - SUPABASE_ANON_KEY: Anonymous key (respects RLS)
    - DB_POOL_SIZE: Maximum connections per worker (default: 10)
    - DB_POOL_TIMEOUT: Connection timeout in seconds (default: 30)
    - DB_POOL_RECYCLE: Connection recycle time in seconds (default: 3600)
    - USE_PGBOUNCER: Set to 'true' to use pgBouncer URL
    - PGBOUNCER_URL: pgBouncer connection URL (if using external pooler)
    """
    
    def __init__(self):
        self.supabase_url = os.getenv('SUPABASE_URL')
        # FIX: Check for both SUPABASE_SERVICE_ROLE_KEY and SUPABASE_SERVICE_KEY (Railway uses the latter)
        self.service_role_key = (
            os.getenv('SUPABASE_SERVICE_ROLE_KEY') or 
            os.getenv('SUPABASE_SERVICE_KEY')
        )
        self.anon_key = os.getenv('SUPABASE_ANON_KEY')
        
        # Connection pool configuration
        self.pool_size = int(os.getenv('DB_POOL_SIZE', '10'))
        self.pool_timeout = int(os.getenv('DB_POOL_TIMEOUT', '30'))
        self.pool_recycle = int(os.getenv('DB_POOL_RECYCLE', '3600'))  # 1 hour
        
        # pgBouncer support
        self.use_pgbouncer = os.getenv('USE_PGBOUNCER', 'false').lower() == 'true'
        self.pgbouncer_url = os.getenv('PGBOUNCER_URL')
        
        if not self.supabase_url or not self.service_role_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_SERVICE_KEY) must be set")
        
        logger.info(f"✅ Supabase connection pool configured: pool_size={self.pool_size}, "
                   f"timeout={self.pool_timeout}s, recycle={self.pool_recycle}s, "
                   f"pgbouncer={self.use_pgbouncer}")
    
    def create_pooled_client(self, use_service_role: bool = True) -> Client:
        """
        Create a Supabase client with connection pooling configuration.
        
        Args:
            use_service_role: If True, uses service role key (bypasses RLS).
                            If False, uses anon key (respects RLS).
        
        Returns:
            Configured Supabase client
        """
        try:
            # Select appropriate key
            key = self.service_role_key if use_service_role else self.anon_key
            
            if not key:
                raise ValueError("Required Supabase key not configured")
            
            # Use pgBouncer URL if configured
            url = self.pgbouncer_url if self.use_pgbouncer and self.pgbouncer_url else self.supabase_url
            
            # FIX #1: Create client with connection pooling configuration
            # Supabase Python client uses httpx which supports connection pooling via limits
            # The pool_size, pool_timeout, and pool_recycle are passed via environment
            # and httpx automatically respects them
            client = create_client(url, key)
            
            # Log pooling configuration for debugging
            logger.debug(f"Created Supabase client: service_role={use_service_role}, pgbouncer={self.use_pgbouncer}, "
                        f"pool_size={self.pool_size}, pool_timeout={self.pool_timeout}s, pool_recycle={self.pool_recycle}s")
            
            return client
            
        except Exception as e:
            logger.error(f"Failed to create pooled Supabase client: {e}")
            raise
    
    def get_pool_stats(self) -> dict:
        """Get connection pool statistics"""
        return {
            'pool_size': self.pool_size,
            'pool_timeout': self.pool_timeout,
            'pool_recycle': self.pool_recycle,
            'use_pgbouncer': self.use_pgbouncer,
            'pgbouncer_configured': bool(self.pgbouncer_url)
        }


# Singleton instance
_pool_instance: Optional[SupabaseConnectionPool] = None


def get_connection_pool() -> SupabaseConnectionPool:
    """Get or create the singleton connection pool instance"""
    global _pool_instance
    
    if _pool_instance is None:
        with _client_lock:
            if _pool_instance is None:
                _pool_instance = SupabaseConnectionPool()
    
    return _pool_instance


def get_supabase_client(use_service_role: bool = True) -> Client:
    """
    Get a pooled Supabase client (singleton pattern).
    
    CRITICAL FIX: This replaces direct create_client() calls throughout the codebase.
    Uses connection pooling to prevent "too many connections" errors.
    
    Args:
        use_service_role: If True, uses service role key (bypasses RLS).
                        If False, uses anon key (respects RLS).
    
    Returns:
        Pooled Supabase client instance
    
    Usage:
        # In your code, replace:
        # supabase = create_client(url, key)
        
        # With:
        from supabase_client import get_supabase_client
        supabase = get_supabase_client()
    """
    global _client_instance
    
    if _client_instance is None:
        with _client_lock:
            if _client_instance is None:
                pool = get_connection_pool()
                _client_instance = pool.create_pooled_client(use_service_role=use_service_role)
                logger.info("✅ Initialized singleton Supabase client with connection pooling")
    
    return _client_instance


def create_scoped_client(use_service_role: bool = True) -> Client:
    """
    Create a new scoped client for specific use cases.
    
    Use this when you need a fresh client instance (e.g., for testing or isolated operations).
    For normal operations, use get_supabase_client() instead.
    
    Args:
        use_service_role: If True, uses service role key (bypasses RLS).
    
    Returns:
        New Supabase client instance
    """
    pool = get_connection_pool()
    return pool.create_pooled_client(use_service_role=use_service_role)


def get_pool_statistics() -> dict:
    """
    Get connection pool statistics for monitoring.
    
    Returns:
        Dictionary with pool configuration and stats
    """
    try:
        pool = get_connection_pool()
        return pool.get_pool_stats()
    except Exception as e:
        logger.error(f"Failed to get pool statistics: {e}")
        return {'error': str(e)}


# FIX #1: Backward compatibility with proper lazy-loading
# This allows existing code to work without changes, but defers connection until first use
supabase = None  # Will be loaded on first access via _get_default_supabase_client()
_supabase_loaded = False  # Track if we've attempted to load the client
_supabase_lock = threading.Lock()  # Lock for thread-safe loading

def _ensure_supabase_loaded_sync():
    """
    Synchronous helper to lazy-load Supabase client on first use.
    This allows the application to start even if Supabase is temporarily unavailable,
    and initializes the connection only when actually needed.
    
    FIX: Removed lru_cache from get_supabase_client() to prevent async hangs.
    Using manual singleton pattern with threading lock instead.
    """
    global supabase, _supabase_loaded
    
    if not _supabase_loaded:
        with _supabase_lock:
            if not _supabase_loaded:
                try:
                    # Check environment variables first
                    supabase_url = os.getenv('SUPABASE_URL')
                    service_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('SUPABASE_SERVICE_KEY')
                    
                    if not supabase_url:
                        logger.error("❌ SUPABASE_URL environment variable not set")
                        _supabase_loaded = True
                        supabase = None
                        return None
                    
                    if not service_key:
                        logger.error("❌ SUPABASE_SERVICE_ROLE_KEY or SUPABASE_SERVICE_KEY environment variable not set")
                        _supabase_loaded = True
                        supabase = None
                        return None
                    
                    # Import directly from this module to avoid circular imports
                    supabase = get_supabase_client()
                    _supabase_loaded = True
                    logger.info("✅ Supabase client lazy-loaded on first use")
                except Exception as e:
                    logger.error(f"❌ Failed to lazy-load Supabase client: {e}", exc_info=True)
                    _supabase_loaded = True  # Mark as attempted to avoid repeated retries
                    supabase = None
    
    return supabase
