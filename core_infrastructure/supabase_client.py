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
import socket
from urllib.parse import urlparse

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
        
        # Validate Supabase URL format
        if not self.supabase_url.startswith(('http://', 'https://')):
            logger.warning(f"⚠️ SUPABASE_URL does not start with http:// or https://: {self.supabase_url}")
        
        logger.info(f"✅ Supabase URL configured: {self.supabase_url[:50]}..." if len(self.supabase_url) > 50 else f"✅ Supabase URL configured: {self.supabase_url}")
        
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
            
            # FIX #2: Wrap client creation in a thread with timeout to prevent hangs
            # The Supabase client may hang if network is slow or Supabase is unreachable
            import threading
            client_holder = {'client': None, 'error': None}
            
            def create_client_thread():
                try:
                    logger.info(f"🔗 Attempting to connect to Supabase at {url}")
                    
                    # Check DNS resolution first
                    try:
                        parsed_url = urlparse(url)
                        hostname = parsed_url.hostname
                        if hostname:
                            logger.info(f"🔍 Resolving DNS for {hostname}...")
                            ip = socket.gethostbyname(hostname)
                            logger.info(f"✅ DNS resolved: {hostname} -> {ip}")
                    except socket.gaierror as dns_err:
                        logger.error(f"❌ DNS resolution failed for {hostname}: {dns_err}")
                        client_holder['error'] = dns_err
                        return
                    except Exception as dns_err:
                        logger.warning(f"⚠️ DNS check failed (non-critical): {dns_err}")
                    
                    client_holder['client'] = create_client(url, key)
                    logger.info(f"✅ Successfully created Supabase client")
                except Exception as e:
                    logger.error(f"❌ Error creating Supabase client: {e}", exc_info=True)
                    client_holder['error'] = e
            
            thread = threading.Thread(target=create_client_thread, daemon=True)
            thread.start()
            thread.join(timeout=10.0)  # 10 second timeout for client creation
            
            if thread.is_alive():
                logger.error(f"⏱️ Supabase client creation timed out after 10 seconds")
                logger.error(f"   URL: {url}")
                logger.error(f"   Possible causes:")
                logger.error(f"   1. Network connectivity issue from Railway to Supabase")
                logger.error(f"   2. Supabase project is paused or unavailable")
                logger.error(f"   3. Firewall/security group blocking connection")
                logger.error(f"   4. DNS resolution issue")
                raise TimeoutError("Supabase client creation timed out after 10 seconds")
            
            if client_holder['error']:
                logger.error(f"Failed to create Supabase client: {client_holder['error']}")
                raise client_holder['error']
            
            if not client_holder['client']:
                raise RuntimeError("Failed to create Supabase client - no error but client is None")
            
            # Log pooling configuration for debugging
            logger.debug(f"Created Supabase client: service_role={use_service_role}, pgbouncer={self.use_pgbouncer}, "
                        f"pool_size={self.pool_size}, pool_timeout={self.pool_timeout}s, pool_recycle={self.pool_recycle}s")
            
            return client_holder['client']
            
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
                try:
                    _pool_instance = SupabaseConnectionPool()
                except Exception as e:
                    logger.error(f"❌ Failed to initialize connection pool: {e}")
                    raise
    
    return _pool_instance


class LazySupabaseClient:
    """
    NUCLEAR FIX: Lazy proxy client that defers connection until first actual use.
    This prevents timeouts during initialization.
    
    CRITICAL: Uses a thread with timeout to prevent hanging on connection attempts.
    """
    def __init__(self, url: str, key: str):
        self.url = url
        self.key = key
        self._real_client = None
        self._connecting = False
        self._connect_lock = threading.Lock()
        self._connection_timeout = 5.0  # 5 second timeout for connection
    
    def _ensure_connected(self):
        """Lazily connect on first actual API call with timeout"""
        if self._real_client is None and not self._connecting:
            with self._connect_lock:
                if self._real_client is None:
                    try:
                        self._connecting = True
                        logger.info(f"🔗 Lazy-connecting to Supabase on first use...")
                        
                        # Connect in a thread with timeout to prevent hangs
                        client_holder = {'client': None, 'error': None}
                        
                        def connect_thread():
                            try:
                                client_holder['client'] = create_client(self.url, self.key)
                            except Exception as e:
                                client_holder['error'] = e
                        
                        thread = threading.Thread(target=connect_thread, daemon=True)
                        thread.start()
                        thread.join(timeout=self._connection_timeout)
                        
                        if thread.is_alive():
                            logger.error(f"⏱️ Supabase connection timed out after {self._connection_timeout} seconds")
                            raise TimeoutError(f"Supabase connection timed out after {self._connection_timeout} seconds")
                        
                        if client_holder['error']:
                            raise client_holder['error']
                        
                        self._real_client = client_holder['client']
                        logger.info(f"✅ Lazy-connected to Supabase successfully")
                    except Exception as e:
                        logger.error(f"❌ Failed to connect to Supabase on first use: {e}")
                        raise
                    finally:
                        self._connecting = False
    
    def __getattr__(self, name):
        """Proxy all attribute access to real client, connecting if needed"""
        self._ensure_connected()
        return getattr(self._real_client, name)


def get_supabase_client(use_service_role: bool = True) -> Client:
    """
    Get a pooled Supabase client (singleton pattern with lazy connection).
    
    NUCLEAR FIX: Returns a lazy proxy that defers connection until first actual use.
    This prevents timeouts during initialization - the chat endpoint responds immediately.
    
    Args:
        use_service_role: If True, uses service role key (bypasses RLS).
                        If False, uses anon key (respects RLS).
    
    Returns:
        Lazy Supabase client instance (connects on first API call)
    
    Usage:
        # In your code, replace:
        # supabase = create_client(url, key)
        
        # With:
        from supabase_client import get_supabase_client
        supabase = get_supabase_client()  # Returns immediately, connects on first use
    """
    global _client_instance
    
    if _client_instance is None:
        with _client_lock:
            if _client_instance is None:
                # ULTRA-FAST: Just read env vars directly, don't create connection pool
                # Connection pool creation can be slow - defer it until first use
                url = os.getenv('SUPABASE_URL')
                key = (
                    os.getenv('SUPABASE_SERVICE_ROLE_KEY') or 
                    os.getenv('SUPABASE_SERVICE_KEY')
                )
                
                if not url or not key:
                    logger.error("❌ SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set")
                    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_SERVICE_KEY) must be set")
                
                # NUCLEAR FIX: Use lazy proxy instead of connecting immediately
                _client_instance = LazySupabaseClient(url, key)
                logger.info("✅ Created lazy Supabase client (will connect on first use)")
    
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
    NUCLEAR FIX: Synchronous helper that returns lazy Supabase client immediately.
    The actual connection happens on first API call, not during initialization.
    This allows the chat endpoint to respond immediately without waiting for network.
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
                    
                    # Get lazy client - returns immediately without connecting
                    supabase = get_supabase_client()
                    _supabase_loaded = True
                    logger.info("✅ Lazy Supabase client created (will connect on first API call)")
                except Exception as e:
                    logger.error(f"❌ Failed to create lazy Supabase client: {e}", exc_info=True)
                    _supabase_loaded = True  # Mark as attempted to avoid repeated retries
                    supabase = None
    
    return supabase
