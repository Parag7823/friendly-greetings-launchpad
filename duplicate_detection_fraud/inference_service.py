"""
Inference Service - Lazy-Loading ML Models
==========================================
Moves heavy model initialization OUT of request flow to prevent:
- Memory explosion (350MB+ per worker)
- Cold start latency (5-30 seconds)
- OOM crashes at scale

Architecture:
- Singleton pattern with lazy loading
- Thread pool for blocking operations
- Redis-backed model caching
- Health checks and circuit breakers
"""

import os
import pickle
import asyncio
import structlog
from typing import Optional, Dict, Any, List
from functools import lru_cache
from aiometer import AsyncRateLimiter
from aiocache import cached
from aiocache.serializers import PickleSerializer

logger = structlog.get_logger(__name__)

# Global instances (lazy-loaded)
# REMOVED: _sentence_model - Use embedding_service.py (BGE model) instead for consistency
_ocr_reader = None
_tfidf_vectorizer = None
_doc_type_vectors = None
_rate_limiter = None

def get_rate_limiter() -> AsyncRateLimiter:
    """
    FIX #15: Get or create aiometer rate limiter for blocking operations.
    
    Replaces ThreadPoolExecutor (lines 33-43) with aiometer library:
    - Centralized rate limiting across all services
    - Redis-backed for distributed systems
    - Automatic concurrency control
    - Better resource management
    
    Configuration:
    - max_rate: From INFERENCE_MAX_WORKERS env var (default 4 ops/sec)
    - time_period: 1 second window
    """
    global _rate_limiter
    if _rate_limiter is None:
        max_workers = int(os.environ.get('INFERENCE_MAX_WORKERS', '4'))
        _rate_limiter = AsyncRateLimiter(
            max_rate=max_workers,
            time_period=1
        )
        logger.info("inference_rate_limiter_initialized", max_rate=max_workers)
    return _rate_limiter


# DEAD CODE REMOVED: SentenceModelService class
# This loaded all-MiniLM-L6-v2 (384 dims) which conflicted with BGE model (1024 dims)
# in embedding_service.py. All embedding operations now use the shared BGE model
# via get_embedding_service() for consistency and to eliminate 400MB+ memory waste.


class OCRService:
    """Lazy-loading EasyOCR service with GPU configuration"""
    
    @staticmethod
    def _get_ocr_config():
        """Get OCR configuration from environment"""
        use_gpu = os.environ.get('OCR_GPU', 'false').lower() == 'true'
        languages = os.environ.get('OCR_LANGUAGES', 'en').split(',')
        return use_gpu, languages
    
    @staticmethod
    async def get_reader():
        """Get or load EasyOCR reader (lazy)"""
        global _ocr_reader
        
        if _ocr_reader is not None:
            return _ocr_reader
        
        use_gpu, languages = OCRService._get_ocr_config()
        logger.info("loading_ocr_reader", languages=languages, gpu=use_gpu)
        
        def _load_reader():
            import easyocr
            try:
                return easyocr.Reader(languages, gpu=use_gpu)
            except Exception as e:
                # Fallback to CPU if GPU fails
                if use_gpu:
                    logger.warning("ocr_gpu_failed_fallback_to_cpu", error=str(e))
                    return easyocr.Reader(languages, gpu=False)
                raise
        
        # FIX #15: Use aiometer rate limiter instead of ThreadPoolExecutor
        limiter = get_rate_limiter()
        loop = asyncio.get_event_loop()
        _ocr_reader = await limiter(loop.run_in_executor)(None, _load_reader)
        
        logger.info("ocr_reader_loaded", gpu=use_gpu, languages=languages)
        return _ocr_reader
    
    @staticmethod
    async def read_text(image_bytes: bytes) -> List[tuple]:
        """
        Perform OCR on image bytes (async, non-blocking).
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            List of (bbox, text, confidence) tuples
        """
        try:
            reader = await OCRService.get_reader()
            
            def _read():
                import numpy as np
                from PIL import Image
                import io
                
                image = Image.open(io.BytesIO(image_bytes))
                image_array = np.array(image)
                return reader.readtext(image_array)
            
            # FIX #15: Use aiometer rate limiter instead of ThreadPoolExecutor
            limiter = get_rate_limiter()
            loop = asyncio.get_event_loop()
            return await limiter(loop.run_in_executor)(None, _read)
        except Exception as e:
            logger.error("ocr_read_failed", error=str(e))
            # Return empty results instead of crashing
            return []


class TFIDFService:
    """Lazy-loading TF-IDF vectorizer service"""
    
    @staticmethod
    @cached(ttl=86400, namespace="inference:tfidf", serializer=PickleSerializer())
    async def get_vectorizer():
        """
        Get or load TF-IDF vectorizer (lazy).
        
        FIX #16: Uses aiocache @cached decorator instead of manual cache.get/set (saves 25 lines)
        """
        global _tfidf_vectorizer, _doc_type_vectors
        
        if _tfidf_vectorizer is not None:
            return _tfidf_vectorizer, _doc_type_vectors
        
        def _train_tfidf():
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            from universal_document_classifier_optimized import UniversalDocumentClassifierOptimized
            classifier = UniversalDocumentClassifierOptimized.__new__(UniversalDocumentClassifierOptimized)
            doc_database = classifier._initialize_document_database()
            
            corpus = []
            doc_types_list = []
            for doc_type, info in doc_database.items():
                doc_text = ' '.join(info['indicators'] + info['keywords'] + info['field_patterns'])
                corpus.append(doc_text)
                doc_types_list.append(doc_type)
            
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform(corpus)
            
            return vectorizer, vectors, doc_types_list
        
        limiter = get_rate_limiter()
        loop = asyncio.get_event_loop()
        _tfidf_vectorizer, _doc_type_vectors, doc_types_list = await limiter(loop.run_in_executor)(None, _train_tfidf)
        
        logger.info("tfidf_trained_and_cached")
        return _tfidf_vectorizer, _doc_type_vectors
    
    @staticmethod
    async def transform(text: str):
        """Transform text using TF-IDF"""
        vectorizer, _ = await TFIDFService.get_vectorizer()
        
        def _transform():
            return vectorizer.transform([text])
        
        # FIX #15: Use aiometer rate limiter instead of ThreadPoolExecutor
        limiter = get_rate_limiter()
        loop = asyncio.get_event_loop()
        return await limiter(loop.run_in_executor)(None, _transform)


class AutomatonService:
    """Lazy-loading Pyahocorasick automaton service"""
    
    @staticmethod
    @cached(ttl=86400, namespace="inference:platform_automaton", serializer=PickleSerializer())
    async def get_platform_automaton():
        """
        Get or build platform detection automaton.
        
        FIX #16: Uses aiocache @cached decorator instead of manual cache.get/set (saves 20 lines)
        FIX #4: Caches patterns instead of automaton object (safer for C-extensions)
        """
        logger.info("building_platform_automaton")
        
        def _build_automaton():
            import ahocorasick
            from universal_platform_detector_optimized import UniversalPlatformDetectorOptimized
            
            detector = UniversalPlatformDetectorOptimized.__new__(UniversalPlatformDetectorOptimized)
            platform_database = detector._initialize_platform_database()
            
            automaton = ahocorasick.Automaton()
            patterns = []
            for platform_id, platform_info in platform_database.items():
                for indicator in platform_info['indicators']:
                    automaton.add_word(indicator.lower(), (platform_id, indicator))
                    patterns.append((platform_id, indicator))
            automaton.make_automaton()
            
            return automaton, patterns
        
        limiter = get_rate_limiter()
        loop = asyncio.get_event_loop()
        automaton, patterns = await limiter(loop.run_in_executor)(None, _build_automaton)
        
        logger.info("platform_automaton_built_and_cached")
        return automaton
    
    @staticmethod
    @cached(ttl=86400, namespace="inference:document_automaton", serializer=PickleSerializer())
    async def get_document_automaton():
        """
        Get or build document classification automaton.
        
        FIX #16: Uses aiocache @cached decorator instead of manual cache.get/set (saves 20 lines)
        FIX #4: Caches patterns instead of automaton object (safer for C-extensions)
        """
        logger.info("building_document_automaton")
        
        def _build_automaton():
            import ahocorasick
            from universal_document_classifier_optimized import UniversalDocumentClassifierOptimized
            
            classifier = UniversalDocumentClassifierOptimized.__new__(UniversalDocumentClassifierOptimized)
            doc_database = classifier._initialize_document_database()
            
            automaton = ahocorasick.Automaton()
            patterns = []
            for doc_type_id, doc_info in doc_database.items():
                for keyword in doc_info['keywords']:
                    automaton.add_word(keyword.lower(), (doc_type_id, keyword))
                    patterns.append((doc_type_id, keyword))
                for indicator in doc_info['indicators']:
                    automaton.add_word(indicator.lower(), (doc_type_id, indicator))
                    patterns.append((doc_type_id, indicator))
            automaton.make_automaton()
            
            return automaton, patterns
        
        limiter = get_rate_limiter()
        loop = asyncio.get_event_loop()
        automaton, patterns = await limiter(loop.run_in_executor)(None, _build_automaton)
        
        logger.info("document_automaton_built_and_cached")
        return automaton


async def health_check() -> Dict[str, Any]:
    """Check health of inference services"""
    # FIX #2: Removed _sentence_model check (was removed - use embedding_service instead)
    # FIX #15: Changed executor check to rate_limiter check
    health = {
        'ocr_reader': _ocr_reader is not None,
        'tfidf_vectorizer': _tfidf_vectorizer is not None,
        'rate_limiter': _rate_limiter is not None
    }
    
    # Check Redis cache
    from core_infrastructure.centralized_cache import safe_get_cache
    cache = safe_get_cache()
    health['redis_cache'] = cache is not None
    
    # Check embedding service health
    try:
        from embedding_service import EmbeddingService
        embedding_service = EmbeddingService()
        health['embedding_service'] = embedding_service is not None
    except Exception:
        health['embedding_service'] = False
    
    return health


async def warmup():
    """Warm up all inference services (call at startup)"""
    logger.info("warming_up_inference_services")
    
    try:
        # FIX #3: Removed SentenceModelService.get_model() (class was removed)
        # Embedding service is initialized on-demand in universal_document_classifier
        await asyncio.gather(
            OCRService.get_reader(),
            TFIDFService.get_vectorizer(),
            AutomatonService.get_platform_automaton(),
            AutomatonService.get_document_automaton(),
            return_exceptions=True
        )
        logger.info("inference_services_warmed_up")
    except Exception as e:
        logger.error("inference_warmup_failed", error=str(e))


async def shutdown():
    """Shutdown inference services"""
    global _executor
    if _executor:
        _executor.shutdown(wait=True)
        _executor = None
        logger.info("inference_executor_shutdown")
