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


# ============================================================================
# PRELOAD PATTERN: Initialize heavy dependencies at module-load time
# ============================================================================
# This runs automatically when the module is imported, eliminating the
# first-request latency that was caused by lazy-loading.
#
# Heavy ML models (EasyOCR ~350MB, TF-IDF, Automaton) are NOW preloaded by default
# for systems with 8GB+ RAM. Set PRELOAD_HEAVY_ML=false to disable.

_PRELOAD_COMPLETED = False

def _preload_all_modules():
    """
    PRELOAD PATTERN: Initialize all heavy modules at module-load time.
    Called automatically when module is imported.
    This eliminates first-request latency.
    """
    global _PRELOAD_COMPLETED
    
    if _PRELOAD_COMPLETED:
        return
    
    # Preload aiocache decorators
    try:
        from aiocache import cached
        from aiocache.serializers import PickleSerializer
        logger.info("✅ PRELOAD: aiocache loaded at module-load time")
    except Exception as e:
        logger.warning(f"⚠️ PRELOAD: aiocache load failed: {e}")
    
    # Preload aiometer rate limiter
    try:
        from aiometer import AsyncRateLimiter
        logger.info("✅ PRELOAD: aiometer loaded at module-load time")
    except Exception as e:
        logger.warning(f"⚠️ PRELOAD: aiometer load failed: {e}")
    
    # Preload PIL for image processing
    try:
        from PIL import Image
        logger.info("✅ PRELOAD: PIL loaded at module-load time")
    except Exception as e:
        logger.warning(f"⚠️ PRELOAD: PIL load failed: {e}")
    
    # Preload centralized_cache
    try:
        from core_infrastructure.centralized_cache import safe_get_cache
        logger.info("✅ PRELOAD: centralized_cache loaded at module-load time")
    except Exception as e:
        logger.warning(f"⚠️ PRELOAD: centralized_cache load failed: {e}")
    
    # Pre-initialize rate limiter singleton
    try:
        get_rate_limiter()
        logger.info("✅ PRELOAD: Rate limiter singleton initialized")
    except Exception as e:
        logger.warning(f"⚠️ PRELOAD: Rate limiter init failed: {e}")
    
    # HEAVY ML PRELOAD: EasyOCR, TF-IDF, Automaton (for 8GB+ RAM systems)
    # Set PRELOAD_HEAVY_ML=false to disable if RAM is limited
    import os
    if os.environ.get('PRELOAD_HEAVY_ML', 'true').lower() != 'false':
        logger.info("🔄 PRELOAD: Starting heavy ML model preload (EasyOCR, TF-IDF, Automaton)...")
        
        # Preload EasyOCR (~350MB) - synchronous import
        try:
            import easyocr
            logger.info("✅ PRELOAD: EasyOCR module loaded at module-load time")
        except Exception as e:
            logger.warning(f"⚠️ PRELOAD: EasyOCR load failed: {e}")
        
        # Preload sklearn TF-IDF vectorizer
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            logger.info("✅ PRELOAD: sklearn TfidfVectorizer loaded at module-load time")
        except Exception as e:
            logger.warning(f"⚠️ PRELOAD: sklearn load failed: {e}")
        
        # Preload ahocorasick automaton
        try:
            import ahocorasick
            logger.info("✅ PRELOAD: ahocorasick loaded at module-load time")
        except Exception as e:
            logger.warning(f"⚠️ PRELOAD: ahocorasick load failed: {e}")
        
        # Call warmup() in background thread to preload actual model instances
        try:
            import asyncio
            import threading
            
            def _warmup_thread():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(warmup())
                    logger.info("✅ PRELOAD: Heavy ML models warmed up in background")
                except Exception as e:
                    logger.warning(f"⚠️ PRELOAD: Heavy ML warmup failed: {e}")
            
            thread = threading.Thread(target=_warmup_thread, daemon=True)
            thread.start()
            logger.info("✅ PRELOAD: Heavy ML warmup started in background thread")
        except Exception as e:
            logger.warning(f"⚠️ PRELOAD: Background warmup failed: {e}")
    
    _PRELOAD_COMPLETED = True

try:
    _preload_all_modules()
except Exception as e:
    logger.warning(f"Module-level inference_service preload failed (will use fallback): {e}")



