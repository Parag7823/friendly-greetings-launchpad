"""Thin wrapper around datasketch.MinHashLSH with Redis persistence via centralized_cache"""

from typing import List, Dict, Any, Optional
import structlog
from datasketch import MinHash, MinHashLSH

logger = structlog.get_logger(__name__)


class PersistentLSHService:
    """Datasketch MinHashLSH with per-user Redis-backed persistence"""
    
    def __init__(self, threshold: float = 0.8, num_perm: int = 128):
        self.threshold = threshold
        self.num_perm = num_perm
        logger.info("lsh_initialized", threshold=threshold, num_perm=num_perm)
    
    def _get_shard_key(self, user_id: str) -> str:
        return f"lsh:shard:{user_id}"
    
    async def _load_shard(self, user_id: str) -> Optional[MinHashLSH]:
        from core_infrastructure.centralized_cache import safe_get_cache
        cache = safe_get_cache()
        if not cache:
            return None
        
        try:
            shard_key = self._get_shard_key(user_id)
            cached_data = await cache.get(shard_key)
            
            if cached_data:
                import pickle
                lsh = pickle.loads(cached_data)
                logger.debug("lsh_shard_loaded", user_id=user_id)
                return lsh
            
            lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
            logger.debug("lsh_shard_created", user_id=user_id)
            return lsh
        except Exception as e:
            logger.error("lsh_load_failed", user_id=user_id, error=str(e))
            return None
    
    async def _save_shard(self, user_id: str, lsh: MinHashLSH) -> bool:
        from core_infrastructure.centralized_cache import safe_get_cache
        cache = safe_get_cache()
        if not cache:
            return False
        
        try:
            import pickle
            shard_key = self._get_shard_key(user_id)
            serialized = pickle.dumps(lsh)
            await cache.set(shard_key, serialized, ttl=86400 * 30)
            logger.debug("lsh_shard_saved", user_id=user_id)
            return True
        except Exception as e:
            logger.error("lsh_save_failed", user_id=user_id, error=str(e))
            return False
    
    async def insert(self, user_id: str, file_hash: str, content: str) -> bool:
        try:
            lsh = await self._load_shard(user_id)
            if not lsh:
                return False
            
            minhash = MinHash(num_perm=self.num_perm)
            for token in content.lower().split():
                minhash.update(token.encode('utf-8'))
            
            lsh.insert(file_hash, minhash)
            await self._save_shard(user_id, lsh)
            logger.debug("lsh_insert_success", user_id=user_id, file_hash=file_hash)
            return True
        except Exception as e:
            logger.error("lsh_insert_failed", user_id=user_id, error=str(e))
            return False
    
    async def query(self, user_id: str, content: str) -> List[str]:
        try:
            lsh = await self._load_shard(user_id)
            if not lsh:
                return []
            
            minhash = MinHash(num_perm=self.num_perm)
            for token in content.lower().split():
                minhash.update(token.encode('utf-8'))
            
            results = lsh.query(minhash)
            logger.debug("lsh_query_success", user_id=user_id, results=len(results))
            return results
        except Exception as e:
            logger.error("lsh_query_failed", user_id=user_id, error=str(e))
            return []
    
    async def remove(self, user_id: str, file_hash: str) -> bool:
        try:
            lsh = await self._load_shard(user_id)
            if not lsh or not hasattr(lsh, 'remove'):
                return False
            
            lsh.remove(file_hash)
            await self._save_shard(user_id, lsh)
            logger.debug("lsh_remove_success", user_id=user_id, file_hash=file_hash)
            return True
        except Exception as e:
            logger.error("lsh_remove_failed", user_id=user_id, error=str(e))
            return False
    
    async def get_shard_stats(self, user_id: str) -> Dict[str, Any]:
        try:
            lsh = await self._load_shard(user_id)
            if not lsh:
                return {'exists': False, 'entry_count': 0}
            
            return {
                'exists': True,
                'entry_count': len(lsh.keys) if hasattr(lsh, 'keys') else 0,
                'threshold': self.threshold,
                'num_perm': self.num_perm
            }
        except Exception as e:
            logger.error("lsh_stats_failed", user_id=user_id, error=str(e))
            return {'error': str(e)}


_lsh_service = None

def get_lsh_service() -> PersistentLSHService:
    global _lsh_service
    if _lsh_service is None:
        _lsh_service = PersistentLSHService()
    return _lsh_service


# ============================================================================
# PRELOAD PATTERN: Initialize LSH service at module-load time
# ============================================================================
# This runs automatically when the module is imported, eliminating the
# first-request latency that was caused by lazy-loading.
# 
# BENEFITS:
# - First request is instant (no cold-start delay)
# - Shared across all worker instances
# - Memory is allocated once, not per-instance

try:
    _lsh_service = PersistentLSHService()
    logger.info("✅ PRELOAD: PersistentLSHService initialized at module-load time")
except Exception as e:
    logger.warning(f"Module-level LSH service preload failed (will use fallback): {e}")
