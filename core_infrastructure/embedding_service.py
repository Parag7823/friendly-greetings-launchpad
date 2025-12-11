"""BGE Embedding Service - Free, open-source embeddings using BAAI's bge-large-en-v1.5 model."""

import structlog
from typing import List, Optional
import asyncio
from functools import lru_cache
import os
from aiocache import cached
from aiocache.serializers import JsonSerializer

logger = structlog.get_logger(__name__)

# Lazy load numpy to prevent import-time crashes
np = None

def _load_numpy():
    """Lazy load numpy C extension on first use"""
    global np
    if np is None:
        try:
            import numpy as numpy_module
            np = numpy_module
            logger.info("✅ numpy module loaded")
        except ImportError:
            logger.error("numpy not installed - numerical features unavailable")
            raise ImportError("numpy is required. Install with: pip install numpy")
    return np

# Global embedding model cache
_embedding_model = None
_model_lock = asyncio.Lock()


async def get_embedding_model():
    """Get or initialize the BGE embedding model (singleton pattern with lazy loading)."""
    global _embedding_model
    
    if _embedding_model is not None:
        return _embedding_model
    
    async with _model_lock:
        if _embedding_model is not None:
            return _embedding_model
        
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("🔄 Loading BGE embedding model (bge-large-en-v1.5)...")
            logger.warning("⚠️ FIX #6: Model loading in worker process. Consider using embedding microservice for production.")
            _embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
            logger.info("✅ BGE embedding model loaded successfully")
            return _embedding_model
        except ImportError:
            logger.error("❌ sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise RuntimeError("sentence-transformers required for embeddings")
        except Exception as e:
            logger.error(f"❌ Failed to load BGE model: {e}")
            raise


class EmbeddingService:
    """Production-grade embedding service using BGE with lazy loading and Redis caching."""
    
    def __init__(self, cache_client=None, enable_embeddings=True):
        self.model = None
        self.enable_embeddings = enable_embeddings
        # FIX #1: Use centralized Redis cache instead of unbounded dictionary
        self.cache = cache_client
        self.cache_hits = 0
        self.cache_misses = 0
    
    async def initialize(self):
        """Initialize the embedding model"""
        self.model = await get_embedding_model()
        
        # FIX #1: Initialize cache if not provided
        if self.cache is None:
            try:
                from core_infrastructure.centralized_cache import safe_get_cache, initialize_cache
                self.cache = safe_get_cache()
                # If cache is still None, try to initialize it
                if self.cache is None:
                    try:
                        self.cache = initialize_cache()
                        logger.info("Centralized cache initialized for embeddings")
                    except Exception as init_err:
                        logger.warning(f"Could not initialize cache: {init_err}")
            except ImportError as e:
                logger.warning(f"Centralized cache not available - embeddings will not be cached: {e}")
    
    @cached(ttl=86400, namespace="embeddings")
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text (1024 dimensions)."""
        if not self.enable_embeddings:
            logger.debug("Embeddings disabled - returning zero vector")
            return [0.0] * 1024

        if not self.model:
            await self.initialize()

        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            
            numpy_module = _load_numpy()
            if isinstance(embedding, numpy_module.ndarray):
                embedding = embedding.tolist()
            
            logger.debug(f"Generated embedding for text: {text[:50]}...")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (more efficient than individual calls)."""
        if not self.model:
            await self.initialize()
        
        try:
            embeddings = []
            for text in texts:
                embedding = await self.embed_text(text)
                embeddings.append(embedding)
            
            logger.info(f"Generated {len(embeddings)} embeddings in batch")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise
    
    @staticmethod
    def similarity(embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings (0 to 1)."""
        try:
            # Convert to numpy arrays (lazy load numpy)
            numpy_module = _load_numpy()
            e1 = numpy_module.array(embedding1)
            e2 = numpy_module.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = numpy_module.dot(e1, e2)
            norm1 = numpy_module.linalg.norm(e1)
            norm2 = numpy_module.linalg.norm(e2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0
    
    @staticmethod
    def batch_similarity(embeddings1: List[List[float]], embeddings2: List[List[float]]) -> List[List[float]]:
        """Calculate similarity matrix between two sets of embeddings."""
        try:
            # Lazy load numpy
            numpy_module = _load_numpy()
            e1 = numpy_module.array(embeddings1)
            e2 = numpy_module.array(embeddings2)
            
            # Normalize embeddings
            e1_norm = e1 / numpy_module.linalg.norm(e1, axis=1, keepdims=True)
            e2_norm = e2 / numpy_module.linalg.norm(e2, axis=1, keepdims=True)
            
            # Calculate similarity matrix
            similarity_matrix = numpy_module.dot(e1_norm, e2_norm.T)
            
            return similarity_matrix.tolist()
            
        except Exception as e:
            logger.error(f"Failed to calculate batch similarity: {e}")
            return []
    
    def get_cache_stats(self) -> dict:
        """Get embedding cache statistics from Redis."""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_requests': total,
            'hit_rate_percent': hit_rate,
            'backend': 'redis' if self.cache else 'none'
        }
    
    async def clear_cache(self):
        """Clear the embedding cache from Redis."""
        if self.cache:
            try:
                await self.cache.clear()
                logger.info("Embedding cache cleared from Redis")
            except Exception as e:
                logger.error(f"Failed to clear cache: {e}")
        else:
            logger.warning("No cache client available to clear")


# Global singleton instance
_embedding_service = None


async def get_embedding_service(cache_client=None, enable_embeddings=True) -> EmbeddingService:
    """Get or create the global embedding service singleton."""
    global _embedding_service
    
    if _embedding_service is None:
        _embedding_service = EmbeddingService(cache_client=cache_client, enable_embeddings=enable_embeddings)
        await _embedding_service.initialize()
    
    return _embedding_service


# Preload pattern: Initialize heavy dependencies at module-load time
_PRELOAD_COMPLETED = False

def _preload_all_modules():
    """Initialize heavy modules at module-load time to eliminate first-request latency."""
    global _PRELOAD_COMPLETED
    
    if _PRELOAD_COMPLETED:
        return
    
    # Preload numpy (lazy-loaded above, but trigger it now)
    try:
        _load_numpy()
        logger.info("✅ PRELOAD: numpy loaded at module-load time")
    except Exception as e:
        logger.warning(f"⚠️ PRELOAD: numpy load failed: {e}")
    
    # Preload sentence-transformers (HEAVY - only import, don't load model)
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("✅ PRELOAD: sentence-transformers loaded at module-load time")
    except Exception as e:
        logger.warning(f"⚠️ PRELOAD: sentence-transformers load failed: {e}")
    
    # Preload aiocache decorator
    try:
        from aiocache import cached
        from aiocache.serializers import JsonSerializer
        logger.info("✅ PRELOAD: aiocache loaded at module-load time")
    except Exception as e:
        logger.warning(f"⚠️ PRELOAD: aiocache load failed: {e}")
    
    # Preload BGE model (default: true, set to false for limited RAM)
    if os.environ.get('PRELOAD_EMBEDDING_MODEL', 'true').lower() != 'false':
        try:
            import asyncio
            # Try to use existing event loop or create new one
            try:
                loop = asyncio.get_running_loop()
                # If there's a running loop, just log that we'll preload later
                logger.info("✅ PRELOAD: BGE embedding model will load on first async call")
            except RuntimeError:
                # No running loop, create one for preloading
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(get_embedding_model())
                logger.info("✅ PRELOAD: BGE embedding model (1.5GB) loaded at module-load time")
        except Exception as e:
            logger.warning(f"⚠️ PRELOAD: BGE model load failed: {e}")
    
    _PRELOAD_COMPLETED = True

try:
    _preload_all_modules()
except Exception as e:
    logger.warning(f"Module-level embedding_service preload failed (will use fallback): {e}")



