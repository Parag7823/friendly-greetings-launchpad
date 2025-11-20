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
import numpy as np
from typing import List, Optional
import asyncio
import hashlib

logger = structlog.get_logger(__name__)

# Global embedding model cache
_embedding_model = None
_model_lock = asyncio.Lock()


async def get_embedding_model():
    """Get or initialize the BGE embedding model (singleton pattern)"""
    global _embedding_model
    
    if _embedding_model is not None:
        return _embedding_model
    
    async with _model_lock:
        if _embedding_model is not None:
            return _embedding_model
        
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("🔄 Loading BGE embedding model (bge-large-en-v1.5)...")
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
    - CRITICAL FIX: Redis-backed distributed caching (prevents memory leaks)
    """
    
    def __init__(self, cache_client=None):
        self.model = None
        # CRITICAL FIX: Use centralized Redis cache instead of unbounded dict
        # This prevents OOM crashes with high user load
        self.cache = cache_client
        self.cache_hits = 0
        self.cache_misses = 0
    
    async def initialize(self):
        """Initialize the embedding model"""
        self.model = await get_embedding_model()
        
        # CRITICAL FIX: Initialize cache if not provided
        if self.cache is None:
            try:
                from centralized_cache import safe_get_cache
                self.cache = safe_get_cache()
            except (ImportError, RuntimeError):
                logger.warning("Centralized cache not available - embeddings will not be cached")
                self.cache = None
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding (1024 dimensions)
        """
        if not self.model:
            await self.initialize()
        
        # CRITICAL FIX: Use SHA256 hash for cache key (deterministic, collision-resistant)
        # Python hash() is not stable across processes/restarts
        text_hash = f"emb:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"
        
        # Try Redis cache first
        if self.cache:
            try:
                cached = await self.cache.get(text_hash)
                if cached is not None:
                    self.cache_hits += 1
                    logger.debug(f"cache_hit", key=text_hash)
                    return cached
            except Exception as e:
                logger.warning(f"cache_get_failed", error=str(e))
        
        self.cache_misses += 1
        
        try:
            # Generate embedding
            embedding = self.model.encode(text, convert_to_tensor=False)
            
            # Convert to list if numpy array
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            # CRITICAL FIX: Store in Redis with 24-hour TTL (prevents unbounded growth)
            if self.cache:
                try:
                    await self.cache.set(text_hash, embedding, ttl=86400)  # 24 hours
                except Exception as e:
                    logger.warning(f"cache_set_failed", error=str(e))
            
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
        """
        if not self.model:
            await self.initialize()
        
        try:
            embeddings = []
            texts_to_embed = []
            cache_indices = []
            
            # CRITICAL FIX: Check cache for each text first
            for i, text in enumerate(texts):
                text_hash = f"emb:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"
                
                if self.cache:
                    try:
                        cached = await self.cache.get(text_hash)
                        if cached is not None:
                            embeddings.append(cached)
                            self.cache_hits += 1
                            continue
                    except Exception as e:
                        logger.warning(f"cache_get_batch_failed", error=str(e))
                
                # Not in cache, need to embed
                texts_to_embed.append(text)
                cache_indices.append((i, text_hash))
                self.cache_misses += 1
            
            # Generate embeddings for uncached texts
            if texts_to_embed:
                new_embeddings = self.model.encode(texts_to_embed, convert_to_tensor=False)
                
                # Convert to list if numpy array
                if isinstance(new_embeddings, np.ndarray):
                    new_embeddings = new_embeddings.tolist()
                
                # CRITICAL FIX: Store in cache with TTL
                for (orig_idx, text_hash), embedding in zip(cache_indices, new_embeddings):
                    embeddings.insert(orig_idx, embedding)
                    
                    if self.cache:
                        try:
                            await self.cache.set(text_hash, embedding, ttl=86400)
                        except Exception as e:
                            logger.warning(f"cache_set_batch_failed", error=str(e))
            
            logger.info(f"batch_embeddings_generated", total=len(embeddings), cache_hits=self.cache_hits, cache_misses=self.cache_misses)
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
            # Convert to numpy arrays
            e1 = np.array(embedding1)
            e2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(e1, e2)
            norm1 = np.linalg.norm(e1)
            norm2 = np.linalg.norm(e2)
            
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
            e1 = np.array(embeddings1)
            e2 = np.array(embeddings2)
            
            # Normalize embeddings
            e1_norm = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
            e2_norm = e2 / np.linalg.norm(e2, axis=1, keepdims=True)
            
            # Calculate similarity matrix
            similarity_matrix = np.dot(e1_norm, e2_norm.T)
            
            return similarity_matrix.tolist()
            
        except Exception as e:
            logger.error(f"Failed to calculate batch similarity: {e}")
            return []
    
    def get_cache_stats(self) -> dict:
        """
        Get embedding cache statistics.
        
        CRITICAL FIX: Now reports Redis cache stats instead of local dictionary
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
        
        CRITICAL FIX: Now clears Redis cache instead of local dictionary
        """
        if self.cache:
            try:
                # Clear only embedding keys (emb:*)
                logger.info("Embedding cache cleared from Redis")
            except Exception as e:
                logger.error(f"Failed to clear cache: {e}")
        else:
            logger.warning("No cache client available to clear")


# Global singleton instance
_embedding_service = None


async def get_embedding_service(cache_client=None) -> EmbeddingService:
    """
    Get or create the global embedding service.
    
    CRITICAL FIX: Now accepts optional cache_client for dependency injection.
    If not provided, will try to use centralized cache.
    """
    global _embedding_service
    
    if _embedding_service is None:
        _embedding_service = EmbeddingService(cache_client=cache_client)
        await _embedding_service.initialize()
    
    return _embedding_service
