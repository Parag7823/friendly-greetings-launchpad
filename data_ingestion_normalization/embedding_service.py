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
from functools import lru_cache

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
    - Caching for performance
    """
    
    def __init__(self):
        self.model = None
        self.embedding_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    async def initialize(self):
        """Initialize the embedding model"""
        self.model = await get_embedding_model()
    
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
        
        # Check cache
        text_hash = hash(text)
        if text_hash in self.embedding_cache:
            self.cache_hits += 1
            return self.embedding_cache[text_hash]
        
        self.cache_misses += 1
        
        try:
            # Generate embedding
            embedding = self.model.encode(text, convert_to_tensor=False)
            
            # Convert to list if numpy array
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            # Cache result
            self.embedding_cache[text_hash] = embedding
            
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
            # Generate embeddings in batch
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            
            # Convert to list if numpy array
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            
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
        """Get embedding cache statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_requests': total,
            'hit_rate_percent': hit_rate,
            'cached_embeddings': len(self.embedding_cache)
        }
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self.embedding_cache.clear()
        logger.info("Embedding cache cleared")


# Global singleton instance
_embedding_service = None


async def get_embedding_service() -> EmbeddingService:
    """Get or create the global embedding service"""
    global _embedding_service
    
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
        await _embedding_service.initialize()
    
    return _embedding_service
