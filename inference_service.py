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
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

logger = structlog.get_logger(__name__)

# Global instances (lazy-loaded)
# REMOVED: _sentence_model - Use embedding_service.py (BGE model) instead for consistency
_ocr_reader = None
_tfidf_vectorizer = None
_doc_type_vectors = None
_executor = None

def get_executor() -> ThreadPoolExecutor:
    """Get or create thread pool executor for blocking operations"""
    global _executor
    if _executor is None:
        max_workers = int(os.environ.get('INFERENCE_MAX_WORKERS', '4'))
        _executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix='inference_'
        )
        logger.info("inference_executor_initialized", max_workers=max_workers)
    return _executor


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
        
        loop = asyncio.get_event_loop()
        _ocr_reader = await loop.run_in_executor(get_executor(), _load_reader)
        
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
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(get_executor(), _read)
        except Exception as e:
            logger.error("ocr_read_failed", error=str(e))
            # Return empty results instead of crashing
            return []


class TFIDFService:
    """Lazy-loading TF-IDF vectorizer service"""
    
    @staticmethod
    async def get_vectorizer():
        """Get or load TF-IDF vectorizer (lazy)"""
        global _tfidf_vectorizer, _doc_type_vectors
        
        if _tfidf_vectorizer is not None:
            return _tfidf_vectorizer, _doc_type_vectors
        
        # Check Redis cache first
        from centralized_cache import safe_get_cache
        cache = safe_get_cache()
        if cache:
            cached_data = await cache.get('inference:tfidf_vectorizer')
            if cached_data:
                logger.info("tfidf_loaded_from_cache")
                _tfidf_vectorizer = cached_data['vectorizer']
                _doc_type_vectors = cached_data['vectors']
                return _tfidf_vectorizer, _doc_type_vectors
        
        # Train TF-IDF (should be pre-computed and cached)
        logger.warning("tfidf_not_cached_training_now")
        
        def _train_tfidf():
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Load document database
            from universal_document_classifier_optimized import UniversalDocumentClassifierOptimized
            classifier = UniversalDocumentClassifierOptimized.__new__(UniversalDocumentClassifierOptimized)
            doc_database = classifier._initialize_document_database()
            
            # Build corpus
            corpus = []
            doc_types_list = []
            for doc_type, info in doc_database.items():
                doc_text = ' '.join(info['indicators'] + info['keywords'] + info['field_patterns'])
                corpus.append(doc_text)
                doc_types_list.append(doc_type)
            
            # Train vectorizer
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform(corpus)
            
            return vectorizer, vectors, doc_types_list
        
        loop = asyncio.get_event_loop()
        _tfidf_vectorizer, _doc_type_vectors, doc_types_list = await loop.run_in_executor(
            get_executor(), _train_tfidf
        )
        
        # Cache in Redis
        if cache:
            await cache.set('inference:tfidf_vectorizer', {
                'vectorizer': _tfidf_vectorizer,
                'vectors': _doc_type_vectors,
                'doc_types': doc_types_list
            }, ttl=86400)  # 24 hours
        
        logger.info("tfidf_trained_and_cached")
        return _tfidf_vectorizer, _doc_type_vectors
    
    @staticmethod
    async def transform(text: str):
        """Transform text using TF-IDF"""
        vectorizer, _ = await TFIDFService.get_vectorizer()
        
        def _transform():
            return vectorizer.transform([text])
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(get_executor(), _transform)


class AutomatonService:
    """Lazy-loading Pyahocorasick automaton service"""
    
    @staticmethod
    async def get_platform_automaton():
        """Get or build platform detection automaton"""
        from centralized_cache import safe_get_cache
        cache = safe_get_cache()
        
        if cache:
            cached = await cache.get('inference:platform_automaton')
            if cached:
                logger.info("platform_automaton_loaded_from_cache")
                return pickle.loads(cached)
        
        logger.info("building_platform_automaton")
        
        def _build_automaton():
            import ahocorasick
            from universal_platform_detector_optimized import UniversalPlatformDetectorOptimized
            
            detector = UniversalPlatformDetectorOptimized.__new__(UniversalPlatformDetectorOptimized)
            platform_database = detector._initialize_platform_database()
            
            automaton = ahocorasick.Automaton()
            for platform_id, platform_info in platform_database.items():
                for indicator in platform_info['indicators']:
                    automaton.add_word(indicator.lower(), (platform_id, indicator))
            automaton.make_automaton()
            
            return automaton
        
        loop = asyncio.get_event_loop()
        automaton = await loop.run_in_executor(get_executor(), _build_automaton)
        
        # Cache in Redis
        if cache:
            await cache.set('inference:platform_automaton', pickle.dumps(automaton), ttl=86400)
        
        logger.info("platform_automaton_built_and_cached")
        return automaton
    
    @staticmethod
    async def get_document_automaton():
        """Get or build document classification automaton"""
        from centralized_cache import safe_get_cache
        cache = safe_get_cache()
        
        if cache:
            cached = await cache.get('inference:document_automaton')
            if cached:
                logger.info("document_automaton_loaded_from_cache")
                return pickle.loads(cached)
        
        logger.info("building_document_automaton")
        
        def _build_automaton():
            import ahocorasick
            from universal_document_classifier_optimized import UniversalDocumentClassifierOptimized
            
            classifier = UniversalDocumentClassifierOptimized.__new__(UniversalDocumentClassifierOptimized)
            doc_database = classifier._initialize_document_database()
            
            automaton = ahocorasick.Automaton()
            for doc_type_id, doc_info in doc_database.items():
                for keyword in doc_info['keywords']:
                    automaton.add_word(keyword.lower(), (doc_type_id, keyword))
                for indicator in doc_info['indicators']:
                    automaton.add_word(indicator.lower(), (doc_type_id, indicator))
            automaton.make_automaton()
            
            return automaton
        
        loop = asyncio.get_event_loop()
        automaton = await loop.run_in_executor(get_executor(), _build_automaton)
        
        # Cache in Redis
        if cache:
            await cache.set('inference:document_automaton', pickle.dumps(automaton), ttl=86400)
        
        logger.info("document_automaton_built_and_cached")
        return automaton


async def health_check() -> Dict[str, Any]:
    """Check health of inference services"""
    health = {
        'sentence_model': _sentence_model is not None,
        'ocr_reader': _ocr_reader is not None,
        'tfidf_vectorizer': _tfidf_vectorizer is not None,
        'executor': _executor is not None
    }
    
    # Check Redis cache
    from centralized_cache import safe_get_cache
    cache = safe_get_cache()
    health['redis_cache'] = cache is not None
    
    return health


async def warmup():
    """Warm up all inference services (call at startup)"""
    logger.info("warming_up_inference_services")
    
    try:
        # Load models in parallel
        await asyncio.gather(
            SentenceModelService.get_model(),
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
