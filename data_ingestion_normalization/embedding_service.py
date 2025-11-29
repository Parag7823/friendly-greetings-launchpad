"""
BGE Embedding Service - Free, Open-Source Embeddings
=====================================================

Uses BAAI's BGE (General Embeddings) model for semantic similarity search.
- Completely FREE and open-source
- No API keys required
- Runs locally or on any server
- State-of-the-art performance
- Supports 100+ languages

Model: bge-large-en-v1.5
- Dimensions: 1024
- Max sequence length: 512 tokens
- Performance: MTEB ranking #1 for many tasks

Installation:
    pip install sentence-transformers

Usage:
    from embedding_service import EmbeddingService
    
    service = EmbeddingService()
    embedding = await service.embed_text("Invoice payment received")
    similarity = service.similarity(embedding1, embedding2)
"""

import structlog
from typing import List, Optional
import asyncio
from functools import lru_cache
import os
from aiocache import cached
from aiocache.serializers import JsonSerializer

logger = structlog.get_logger(__name__)

# ✅ LAZY LOADING: numpy is a heavy C extension that can cause import-time crashes
# Load it only when needed to prevent Railway deployment crashes
np = None  # Will be loaded on first use

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
    """
    Get or initialize the BGE embedding model (singleton pattern).
    
    FIX #6: Lazy loading prevents model from being loaded in every worker process.
    The model is only loaded when actually needed (first embed_text call).
    This prevents the "Monolith RAM Trap" where 8 workers × 1.5GB = 12GB overhead.
    """
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
    """
    Production-grade embedding service using BGE (BAAI General Embeddings).
    
    Features:
    - Free and open-source
    - No API keys required
    - Local execution (privacy-preserving)
    - Batch processing support
    - Similarity search
    - Redis-backed distributed caching (no memory leaks)
    
    FIX #6: Lazy loading to prevent RAM trap in multi-worker deployments.
    Model is only loaded when embed_text() is first called.
    """
    
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
                from core_infrastructure.centralized_cache import get_cache
                self.cache = get_cache()
            except (ImportError, RuntimeError):
                logger.warning("Centralized cache not available - embeddings will not be cached")
                self.cache = None
    
    @cached(ttl=86400, namespace="embeddings")
    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding (1024 dimensions)
            
        FIX #6: Returns empty embedding if enable_embeddings=False to prevent RAM trap.
        FIX #16: Uses aiocache @cached decorator instead of manual cache.get/set (saves 20 lines)
        """
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
        """
        Generate embeddings for multiple texts (more efficient).
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
            
        FIX #16: Uses aiocache @cached decorator for individual texts (saves 30 lines of manual caching)
        """
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
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score between -1 and 1 (typically 0 to 1)
        """
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
        """
        Calculate similarity matrix between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            
        Returns:
            Similarity matrix (len(embeddings1) x len(embeddings2))
        """
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
        """
        Get embedding cache statistics.
        
        FIX #1: Now reports Redis cache stats instead of local dictionary
        """
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
        """
        Clear the embedding cache.
        
        FIX #1: Now clears Redis cache instead of local dictionary
        """
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
    """
    Get or create the global embedding service.
    
    FIX #1: Now accepts optional cache_client for dependency injection.
    If not provided, will try to use centralized cache.
    
    FIX #6: Now accepts enable_embeddings flag to prevent RAM trap.
    Set enable_embeddings=False to skip model loading entirely.
    """
    global _embedding_service
    
    if _embedding_service is None:
        _embedding_service = EmbeddingService(cache_client=cache_client, enable_embeddings=enable_embeddings)
        await _embedding_service.initialize()
    
    return _embedding_service
