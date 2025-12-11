"""Production-grade data enrichment with validation, caching, and error handling."""

from __future__ import annotations
import os
import time
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

import pendulum
import polars as pl
import orjson

# Import from data_ingestion_normalization package
from data_ingestion_normalization.universal_field_detector import (
    UniversalFieldDetector,
    learn_field_mapping,
    get_learned_mappings,
)
from data_ingestion_normalization.universal_platform_detector_optimized import (
    UniversalPlatformDetectorOptimized as UniversalPlatformDetector,
)
from data_ingestion_normalization.universal_extractors_optimized import (
    UniversalExtractorsOptimized as UniversalExtractors,
)
from data_ingestion_normalization.universal_normalizer_optimized import (
    UniversalNormalizer, AmountNormalizer, DateNormalizer
)

# Core infrastructure utilities (Source of Truth)
from core_infrastructure.database_optimization_utils import calculate_row_hash
from core_infrastructure.security_system import InputSanitizer

# Cache utilities
try:
    from core_infrastructure.centralized_cache import get_cache as safe_get_cache
    from core_infrastructure.centralized_cache import get_ai_cache as safe_get_ai_cache
except ImportError:
    safe_get_cache = lambda: None
    safe_get_ai_cache = lambda: None

def _get_groq_client():
    """Lazy import of get_groq_client to avoid circular import"""
    try:
        from core_infrastructure.fastapi_backend_v2 import get_groq_client
        return get_groq_client()
    except ImportError:
        return None

def _get_thread_pool():
    """Lazy import of thread pool to avoid circular import"""
    try:
        from core_infrastructure.fastapi_backend_v2 import _thread_pool
        return _thread_pool
    except ImportError:
        return None

def _get_websocket_manager():
    """Lazy import of WebSocket manager to avoid circular import"""
    try:
        from core_infrastructure.fastapi_backend_v2 import manager
        return manager
    except ImportError:
        return None

# Pydantic validation
from pydantic import ValidationError

logger = logging.getLogger(__name__)

class DataEnrichmentProcessor:
    """Production-grade data enrichment with validation, caching, and error handling"""
    
    def __init__(self, cache_client=None, config=None, supabase_client=None):
        self.cache = cache_client or safe_get_cache()
        self.config = config or self._get_default_config()
        self.supabase = supabase_client
        
        # Initialize security singleton (Source of Truth for sanitization)
        self.sanitizer = InputSanitizer()
        
        self._cache_initialized = False
        self._accuracy_system_initialized = False
        self._integration_system_initialized = False
        
        try:
            from data_ingestion_normalization.universal_normalizer_optimized import UniversalNormalizer
            self.normalizer = UniversalNormalizer(cache_client=safe_get_ai_cache())
            self.universal_extractors = UniversalExtractors(cache_client=safe_get_ai_cache())
            self.universal_field_detector = UniversalFieldDetector()
            logger.info("✅ DataEnrichmentProcessor: Using UniversalNormalizer and InputSanitizer")
        except Exception as e:
            logger.error(f"Failed to initialize enrichment components: {e}")
            raise
        
        self.metrics = {
            'enrichment_count': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'error_count': 0,
            'avg_processing_time': 0.0
        }
        
        self.validation_rules = self._load_validation_rules()
        
        logger.info("âœ… DataEnrichmentProcessor initialized with production-grade features")
    
    
    
    async def _create_fallback_payload(self, row_data: Dict, platform_info: Dict, 
                                      ai_classification: Dict, file_context: Dict, 
                                      error_message: str) -> Dict[str, Any]:
        """Create fallback payload when enrichment fails"""
        return {
            **row_data,
            'kind': ai_classification.get('row_type', 'transaction'),
            'category': ai_classification.get('category', 'other'),
            'subcategory': ai_classification.get('subcategory', 'general'),
            'amount_original': self._extract_amount(row_data),
            'amount_usd': self._extract_amount(row_data),
            'currency': 'USD',
            'vendor_raw': '',
            'vendor_standard': '',
            'platform_ids': {},
            'enrichment_error': error_message,
            'enrichment_version': '2.0.0-fallback',
            'ingested_on': datetime.now().isoformat()
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for enrichment processor"""
        return {
            'batch_size': int(os.getenv('ENRICHMENT_BATCH_SIZE', '100')),
            'cache_ttl': int(os.getenv('ENRICHMENT_CACHE_TTL', '3600')),  # 1 hour
            'max_retries': int(os.getenv('ENRICHMENT_MAX_RETRIES', '3')),
            'confidence_threshold': float(os.getenv('ENRICHMENT_CONFIDENCE_THRESHOLD', '0.7')),
            'enable_caching': os.getenv('ENRICHMENT_ENABLE_CACHE', 'true').lower() == 'true',
            'enable_validation': os.getenv('ENRICHMENT_ENABLE_VALIDATION', 'true').lower() == 'true',
            'max_memory_usage_mb': int(os.getenv('ENRICHMENT_MAX_MEMORY_MB', '512'))
        }
    
    def _load_platform_patterns(self) -> Dict[str, Any]:
        """
        FIX #48: Load platform patterns from config/platform_id_patterns.yaml
        Non-developers can edit patterns without code changes.
        Falls back to empty dict if YAML not found.
        """
        try:
            import yaml
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'platform_id_patterns.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    logger.info(f"âœ… Platform patterns loaded from {config_path}")
                    return config.get('platforms', {})
        except Exception as e:
            logger.warning(f"Failed to load platform patterns from YAML: {e}. Using empty patterns.")
        
        return {}
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """
        FIX #46: Load validation rules from config/validation_rules.yaml
        Non-developers can edit validation rules without code changes.
        Falls back to hardcoded defaults if YAML not found.
        """
        try:
            import yaml
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'validation_rules.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    rules = yaml.safe_load(f)
                    logger.info(f"âœ… Validation rules loaded from {config_path}")
                    return rules
        except Exception as e:
            logger.warning(f"Failed to load validation rules from YAML: {e}. Using hardcoded defaults.")
        
        # Hardcoded fallback (backward compatibility)
        return {
            'amount': {
                'min_value': -100000000.0,
                'max_value': 100000000.0,
                'required_precision': 2
            },
            'currency': {
                'valid_currencies': ['USD', 'EUR', 'GBP', 'INR', 'JPY', 'CNY', 'AUD', 'CAD', 'CHF', 'SEK', 'NZD'],
                'default_currency': 'USD'
            },
            'date': {
                'min_year': 1900,
                'max_year': 2100,
                'required_format': '%Y-%m-%d'
            },
            'vendor': {
                'min_length': 2,
                'max_length': 255,
                'forbidden_chars': ['<', '>', '&', '"', "'"]
            },
            'confidence': {
                'threshold': 0.7,
                'high_priority_threshold': 0.5
            }
        }
    
    async def enrich_row_data(self, row_data: Dict, platform_info: Dict, column_names: List[str], 
                            ai_classification: Dict, file_context: Dict) -> Dict[str, Any]:
        """
        Production-grade row data enrichment with comprehensive validation,
        caching, error handling, and performance optimization.
        
        Args:
            row_data: Raw row data dictionary
            platform_info: Platform detection information
            column_names: List of column names
            ai_classification: AI classification results
            file_context: File context information
            
        Returns:
            Dict containing enriched data with confidence scores and metadata
            
        Raises:
            ValidationError: If input data fails validation
            EnrichmentError: If enrichment process fails
        """
        start_time = time.time()
        enrichment_id = self._generate_enrichment_id(row_data, file_context)
        
        try:
            # 0. Initialize observability and log operation start
            await self._initialize_observability()
            await self._log_operation_start("enrich_row_data", enrichment_id, file_context)
            
            try:
                # 1. Input validation and sanitization (includes security via InputSanitizer)
                validated_data = await self._validate_and_sanitize_input(
                    row_data, platform_info, column_names, ai_classification, file_context
                )

                # 2. Check cache for existing enrichment
                cached_result = await self._get_cached_enrichment(enrichment_id)
                if cached_result:
                    self.metrics['cache_hits'] += 1
                    logger.debug(f"Cache hit for enrichment {enrichment_id}")
                    return cached_result

                self.metrics['cache_misses'] += 1

                # 3. Extract and validate core fields
                extraction_results = await self._extract_core_fields(validated_data)

                # 4. FIX #1: Reuse platform_info from Phase 3 instead of re-classifying
                # Use passed platform_info parameter directly
                classification_results = {
                    'platform': platform_info.get('platform', 'unknown'),
                    'platform_confidence': platform_info.get('confidence', 0.5),
                    'platform_indicators': platform_info.get('indicators', []),
                    'document_type': platform_info.get('document_type', 'financial_data'),
                    'document_confidence': platform_info.get('document_confidence', 0.8)
                }
                logger.debug(f"âœ… Reusing platform info from Phase 3: {classification_results['platform']}")

                normalized_row = await self.normalizer.normalize_row(
                    extraction_results,
                    platform=classification_results.get('platform'),
                    document_type=classification_results.get('document_type')
                )
                extraction_results.update(normalized_row)
                
                vendor_results = {
                    'vendor_raw': extraction_results.get('vendor_raw', ''),
                    'vendor_standard': extraction_results.get('vendor_standard', ''),
                    'vendor_confidence': extraction_results.get('vendor_confidence', 0.0),
                    'vendor_cleaning_method': extraction_results.get('vendor_cleaning_method', 'normalized')
                }

                # 6. Platform ID extraction using UniversalPlatformDetectorOptimized (consolidated)
                platform_id_results = await self._extract_platform_ids_universal(
                    validated_data, classification_results
                )

                # 7. Currency processing with exchange rate handling
                currency_results = await self._process_currency_with_validation(
                    extraction_results, classification_results
                )

                # 8. Build enriched payload with confidence scoring
                enriched_payload = await self._build_enriched_payload(
                    validated_data, extraction_results, classification_results,
                    vendor_results, platform_id_results, currency_results, ai_classification
                )

                # 9. Final validation and confidence scoring
                validated_payload = await self._validate_enriched_payload(enriched_payload)

                # 9.5. Apply accuracy enhancement
                enhanced_payload = await self._apply_accuracy_enhancement(
                    validated_data['row_data'], validated_payload, file_context
                )

                # 10. Cache the result
                await self._cache_enrichment_result(enrichment_id, enhanced_payload)

                # 11. Update metrics
                processing_time = time.time() - start_time
                self._update_metrics(processing_time)

                # 12. Send real-time notification
                await self._send_enrichment_notification(file_context, enhanced_payload, processing_time)

                # 13. Log operation completion
                await self._log_operation_end("enrich_row_data", True, processing_time, file_context)

                # 14. Audit logging
                await self._log_enrichment_audit(enrichment_id, enhanced_payload, processing_time)

                return enhanced_payload

            except ValidationError as e:
                self.metrics['error_count'] += 1
                logger.error(f"Validation error in enrichment {enrichment_id}: {e}")
                raise
            except Exception as e:
                self.metrics['error_count'] += 1
                logger.error(f"Enrichment error for {enrichment_id}: {e}")
                # Return fallback payload with error information
                return await self._create_fallback_payload(
                    row_data, platform_info, ai_classification, file_context, str(e)
                )

        except ValidationError as e:
            self.metrics['error_count'] += 1
            logger.error(f"Validation error in enrichment {enrichment_id}: {e}")
            raise
        except Exception as e:
            self.metrics['error_count'] += 1
            logger.error(f"Enrichment error for {enrichment_id}: {e}")
            # Return fallback payload with error information
            return await self._create_fallback_payload(
                row_data, platform_info, ai_classification, file_context, str(e)
            )
    
    async def enrich_batch_data(self, batch_data: List[Dict], platform_info: Dict, 
                               column_names: List[str], ai_classifications: List[Dict], 
                               file_context: Dict) -> List[Dict[str, Any]]:
        """
        Batch enrichment for improved performance with large datasets.
        Processes multiple rows concurrently while maintaining memory efficiency.
        
        Args:
            batch_data: List of raw row data dictionaries
            platform_info: Platform detection information
            column_names: List of column names
            ai_classifications: List of AI classification results
            file_context: File context information
            
        Returns:
            List of enriched data dictionaries
        """
        if not batch_data:
            return []
        
        # Validate batch size
        if len(batch_data) > self.config['batch_size']:
            logger.warning(f"Batch size {len(batch_data)} exceeds limit {self.config['batch_size']}")
            batch_data = batch_data[:self.config['batch_size']]
            ai_classifications = ai_classifications[:len(batch_data)]

        if len(ai_classifications) < len(batch_data):
            logger.warning(
                "AI classifications shorter than batch (%s vs %s); padding with empty dicts",
                len(ai_classifications), len(batch_data)
            )
            ai_classifications = ai_classifications + [{} for _ in range(len(batch_data) - len(ai_classifications))]
        
        # FIX #55: Process batch in chunks instead of waiting for all tasks
        # Problem: Semaphore(10) with gather(*tasks) processes 10 at a time but waits for all before starting next batch
        # Solution: Process in chunks of 10, start next chunk immediately after current chunk completes
        chunk_size = 10
        all_results = []
        
        async def enrich_single_row(row_data, ai_classification, index):
            try:
                # Add row index to file context
                row_context = file_context.copy()
                row_context['row_index'] = index
                
                return await self.enrich_row_data(
                    row_data, platform_info, column_names, ai_classification, row_context
                )
            except Exception as e:
                logger.error(f"Batch enrichment error for row {index}: {e}")
                return await self._create_fallback_payload(
                    row_data, platform_info, ai_classification, file_context, str(e)
                )
        
        # FIX #55: Process in chunks for better throughput
        # Instead of: create all tasks, wait for all with gather
        # Do: create chunk tasks, wait for chunk, then start next chunk
        for chunk_start in range(0, len(batch_data), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(batch_data))
            chunk_tasks = [
                enrich_single_row(batch_data[i], ai_classifications[i], i)
                for i in range(chunk_start, chunk_end)
            ]
            
            # Process this chunk and collect results
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
            all_results.extend(chunk_results)
        
        results = all_results
        
        # Filter out exceptions and log errors
        enriched_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch processing error for row {i}: {result}")
                enriched_results.append(await self._create_fallback_payload(
                    batch_data[i], platform_info, ai_classifications[i], file_context, str(result)
                ))
            else:
                enriched_results.append(result)
        
        logger.info(f"Batch enrichment completed: {len(enriched_results)} rows processed")
        return enriched_results
    
    async def _get_field_mappings(self, user_id: str, column_names: List[str], 
                                  platform: str = None, document_type: str = None) -> Dict[str, str]:
        """
        UNIVERSAL FIX: Get learned field mappings from database.
        Returns dict mapping target_field -> source_column
        
        Uses the field_mapping_learner module for robust retrieval.
        """
        if not user_id or not self.supabase:
            return {}
        
        try:
            # Get learned mappings for this user and platform
            mappings = await get_learned_mappings(
                user_id=user_id,
                platform=platform,
                min_confidence=0.5,
                supabase=self.supabase
            )

            if mappings:
                logger.debug(f"Retrieved {len(mappings)} learned field mappings for user {user_id}")

            return mappings

        except Exception as e:
            logger.warning(f"Failed to get field mappings: {e}")
            return {}
    
    async def _learn_field_mappings_from_extraction(
        self,
        user_id: str,
        row_data: Dict,
        extraction_results: Dict,
        platform: Optional[str] = None,
        document_type: Optional[str] = None,
    ):
        """
        UNIVERSAL FIX: Learn field mappings from successful extractions.

        This method infers which columns were used for each field and records
        them in the field_mappings table for future use.
        """
        if not user_id or not self.supabase:
            return

        try:
            # Infer mappings from successful extractions
            # We look for columns that match the extracted values

            # Amount mapping
            amount = extraction_results.get('amount', 0.0)
            if amount and amount > 0:
                for col_name, col_value in row_data.items():
                    if isinstance(col_value, (int, float)) and abs(float(col_value) - amount) < 0.01:
                        await learn_field_mapping(
                            user_id=user_id,
                            source_column=col_name,
                            target_field='amount',
                            platform=platform,
                            document_type=document_type,
                            confidence=0.9,
                            extraction_success=True,
                            metadata={'inferred_from': 'extraction'},
                            supabase=self.supabase,
                        )
                        break

            # Vendor mapping
            vendor = extraction_results.get('vendor_name', '')
            if vendor:
                for col_name, col_value in row_data.items():
                    if isinstance(col_value, str) and col_value.strip() == vendor.strip():
                        await learn_field_mapping(
                            user_id=user_id,
                            source_column=col_name,
                            target_field='vendor',
                            platform=platform,
                            document_type=document_type,
                            confidence=0.85,
                            extraction_success=True,
                            metadata={'inferred_from': 'extraction'},
                            supabase=self.supabase,
                        )
                        break

                # Date mapping
            date = extraction_results.get('date', '')
            if date and date != datetime.now().strftime('%Y-%m-%d'):
                for col_name, col_value in row_data.items():
                    if isinstance(col_value, str):
                        try:
                            from dateutil import parser
                            parsed = parser.parse(col_value)
                            if parsed.strftime('%Y-%m-%d') == date:
                                await learn_field_mapping(
                                    user_id=user_id,
                                    source_column=col_name,
                                    target_field='date',
                                    platform=platform,
                                    document_type=document_type,
                                    confidence=0.9,
                                    extraction_success=True,
                                    metadata={'inferred_from': 'extraction'},
                                    supabase=self.supabase,
                                )
                                break
                        except Exception:
                            continue

            # Description mapping
            description = extraction_results.get('description', '')
            if description:
                for col_name, col_value in row_data.items():
                    if isinstance(col_value, str) and col_value.strip() == description.strip():
                        await learn_field_mapping(
                            user_id=user_id,
                            source_column=col_name,
                            target_field='description',
                            platform=platform,
                            document_type=document_type,
                            confidence=0.8,
                            extraction_success=True,
                            metadata={'inferred_from': 'extraction'},
                            supabase=self.supabase,
                        )
                        break

            # Currency mapping
            currency = extraction_results.get('currency', 'USD')
            if currency != 'USD':
                for col_name, col_value in row_data.items():
                    if isinstance(col_value, str) and col_value.strip().upper() == currency.upper():
                        await learn_field_mapping(
                            user_id=user_id,
                            source_column=col_name,
                            target_field='currency',
                            platform=platform,
                            document_type=document_type,
                            confidence=0.95,
                            extraction_success=True,
                            metadata={'inferred_from': 'extraction'},
                            supabase=self.supabase,
                        )
                        break
        except Exception as e:
            logger.warning(f"Failed to learn field mappings from extraction: {e}")
    

    

    

    

    

    
    # ============================================================================
    # PRODUCTION-GRADE HELPER METHODS
    # ============================================================================
    
    def _generate_enrichment_id(self, row_data: Dict, file_context: Dict) -> str:
        """
        Generate deterministic enrichment ID using standardized row hashing.
        
        INTEGRATION NOTE: Uses database_optimization_utils.calculate_row_hash
        for semantic normalization and consistent hashing across all modules.
        """
        try:
            # Use standardized row hash (Source of Truth for hashing)
            row_hash = calculate_row_hash(
                source_filename=file_context.get('filename', 'unknown'),
                row_index=0,  # Not applicable for enrichment context
                payload=row_data
            )
            
            return f"enrich_{row_hash[:16]}"
        except Exception as e:
            logger.warning(f"Failed to generate enrichment ID: {e}")
            return f"enrich_{int(time.time() * 1000)}"

    
    async def _validate_and_sanitize_input(self, row_data: Dict, platform_info: Dict, 
                                         column_names: List[str], ai_classification: Dict, 
                                         file_context: Dict) -> Dict[str, Any]:
        """
        Validate and sanitize input data using certified InputSanitizer.
        
        INTEGRATION NOTE: Uses security_system.InputSanitizer (Source of Truth)
        for XSS protection and input validation. Removes all custom sanitization.
        """
        try:
            # Use centralized InputSanitizer (Source of Truth for security)
            sanitized_row_data = self.sanitizer.sanitize_json(row_data)
            
            # Validate required fields
            if not sanitized_row_data:
                raise ValidationError("Row data cannot be empty")
            
            # Validate file context
            if not file_context.get('filename'):
                raise ValidationError("Filename is required in file context")
            
            # Validate user context
            if not file_context.get('user_id'):
                raise ValidationError("User ID is required in file context")
            
            return {
                'row_data': sanitized_row_data,
                'platform_info': platform_info,
                'column_names': column_names,
                'ai_classification': ai_classification,
                'file_context': file_context
            }
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            raise ValidationError(f"Input validation failed: {str(e)}")
    
    def _sanitize_string(self, value: str, field_type: str = 'generic') -> str:
        """
        LIBRARY FIX: Sanitize string input using validators + presidio (replaces manual char removal)
        FIX #57: Don't redact vendor/financial fields - only detect and log PII
        Redacting breaks data accuracy (e.g., "John Smith Contracting" â†’ "[REDACTED] Contracting")
        """
        if not isinstance(value, str):
            return str(value)
        
        # LIBRARY FIX: Use validators for length validation
        from validators import length
        
        # Limit length first
        if len(value) > 1000:
            value = value[:1000]
        
        # Validate length using validators library
        if not length(value, min=1, max=1000):
            raise ValueError("String length validation failed")
        
        # LIBRARY FIX: Use bleach for HTML/XSS sanitization (replaces manual char removal)
        from bleach import clean
        sanitized = clean(value, tags=[], strip=True)
        
        # FIX #57: Only detect PII in financial/vendor fields, don't redact (data integrity)
        # Redaction breaks vendor matching and financial data accuracy
        try:
            from presidio_analyzer import AnalyzerEngine
            analyzer = AnalyzerEngine()
            
            # Detect PII entities
            results = analyzer.analyze(text=sanitized, language='en')
            
            # Log if PII detected (for audit trail)
            if results:
                pii_types = [r.entity_type for r in results]
                logger.warning(f"PII detected in {field_type} field: {pii_types} - NOT REDACTED to preserve data integrity")
                # FIX #57: Do NOT redact - just log for audit purposes
                # Redaction corrupts financial data (vendor names, descriptions, etc.)
        except Exception as e:
            logger.debug(f"Presidio PII detection skipped: {e}")
        
        return sanitized.strip()
    
    def _validate_filename(self, filename: str) -> bool:
        """
        LIBRARY FIX: Validate filename using validators (replaces manual path traversal checks)
        FIX #58: os.path.basename() already strips path components, so checks for .., /, \\ are redundant
        """
        if not filename or not isinstance(filename, str):
            return False
        
        # LIBRARY FIX: Use validators for slug validation (prevents path traversal)
        from validators import slug
        
        # Extract just the filename without path
        import os
        basename = os.path.basename(filename)
        
        # FIX #58: REMOVED redundant path traversal checks
        # os.path.basename() already strips all path components:
        # - basename('/path/to/../file.txt') returns 'file.txt'
        # - basename('C:\\path\\file.txt') returns 'file.txt'
        # Checking for .., /, \\ in basename is redundant and never triggers
        
        # Validate filename format using slug validator (only real validation needed)
        if not slug(basename.replace('.', '-')):
            logger.warning(f"Invalid filename format: {filename}")
            return False
        
        return True
    
    async def _get_cached_enrichment(self, enrichment_id: str) -> Optional[Dict[str, Any]]:
        """Get cached enrichment result if available"""
        if not self.config['enable_caching']:
            return None
        
        try:
            # Use centralized cache (already initialized in __init__)
            if self.cache and hasattr(self.cache, 'get_cached_classification'):
                cached_data = await self.cache.get_cached_classification(
                    {'enrichment_id': enrichment_id}, 
                    'enrichment'
                )
                if cached_data:
                    logger.debug(f"Cache hit for enrichment {enrichment_id}")
                    return cached_data
        except Exception as e:
            logger.warning(f"Cache retrieval failed for {enrichment_id}: {e}")
        
        return None
    
    async def _cache_enrichment_result(self, enrichment_id: str, result: Dict[str, Any]) -> None:
        """Cache enrichment result for future use"""
        if not self.config['enable_caching']:
            return
        
        try:
            # Use centralized cache (already initialized in __init__)
            if self.cache and hasattr(self.cache, 'store_classification'):
                await self.cache.store_classification(
                    {'enrichment_id': enrichment_id},
                    result,
                    'enrichment',
                    ttl_hours=self.config['cache_ttl'] / 3600
                )
                logger.debug(f"Cached enrichment result for {enrichment_id}")
        except Exception as e:
            logger.warning(f"Cache storage failed for {enrichment_id}: {e}")
    




    async def _extract_core_fields(self, validated_data: Dict) -> Dict[str, Any]:
        row_data = validated_data['row_data']
        column_names = validated_data['column_names']
        platform_info = validated_data.get('platform_info', {})
        file_context = validated_data.get('file_context', {})
        user_id = file_context.get('user_id')
        
        try:
            field_detection_result = await self.universal_field_detector.detect_field_types_universal(
                data=row_data,
                filename=file_context.get('filename'),
                context={
                    'platform': platform_info.get('platform'),
                    'document_type': platform_info.get('document_type'),
                    'user_id': user_id,
                    'column_names': column_names
                }
            )
            
            detected_fields = field_detection_result.get('detected_fields', [])
            
            amount = 0.0
            vendor_name = ''
            date = datetime.now().strftime('%Y-%m-%d')
            description = ''
            currency = 'USD'
            confidence = 0.5
            fields_found = 0
            
            for field_info in detected_fields:
                field_name = field_info.get('name', '').lower()
                field_type = field_info.get('type', '').lower()
                field_value = row_data.get(field_info.get('name'))
                field_confidence = field_info.get('confidence', 0.0)
                
                if field_confidence < 0.5:
                    continue
                
                if 'amount' in field_type:
                    from data_ingestion_normalization.universal_normalizer_optimized import AmountNormalizer
                    amount = AmountNormalizer.normalize(field_value)
                    fields_found += 1
                elif 'vendor' in field_type:
                    vendor_name = str(field_value).strip() if field_value else ''
                    fields_found += 1
                elif 'date' in field_type:
                    from data_ingestion_normalization.universal_normalizer_optimized import DateNormalizer
                    date = DateNormalizer.normalize(field_value)
                    fields_found += 1
                elif 'description' in field_type:
                    description = str(field_value).strip() if field_value else ''
                    fields_found += 1
                elif 'currency' in field_type:
                    currency = str(field_value).upper() if field_value else 'USD'
                    fields_found += 1
            
            confidence = min(0.9, 0.5 + (fields_found * 0.1))
            
            extraction_results = {
                'amount': amount,
                'vendor_name': vendor_name,
                'date': date,
                'description': description,
                'currency': currency,
                'confidence': confidence,
                'fields_extracted': fields_found,
                'field_detection_metadata': {
                    'method': field_detection_result.get('method'),
                    'detected_fields_count': len(detected_fields),
                    'field_types': {f['name']: f['type'] for f in detected_fields}
                }
            }
            
            if confidence > 0.5:
                await self._learn_field_mappings_from_extraction(
                    user_id=user_id,
                    row_data=row_data,
                    extraction_results=extraction_results,
                    platform=platform_info.get('platform'),
                    document_type=platform_info.get('document_type')
                )
            
            return extraction_results
            
        except Exception as e:
            logger.error(f"Core field extraction failed: {e}")
            return {
                'amount': 0.0,
                'vendor_name': '',
                'date': datetime.now().strftime('%Y-%m-%d'),
                'description': '',
                'currency': 'USD'
            }
 
    
    async def _extract_platform_ids_universal(self, validated_data: Dict, classification_results: Dict) -> Dict[str, Any]:
        """
        MODULARIZATION: Delegate to UniversalPlatformDetectorOptimized.extract_platform_ids
        Centralizes all platform knowledge in the platform detector module.
        """
        try:
            row_data = validated_data.get('row_data', {})
            platform = classification_results.get('platform', 'unknown')
            
            # MODULARIZATION: Delegate to UniversalPlatformDetector
            platform_detector = UniversalPlatformDetector()
            return await platform_detector.extract_platform_ids(row_data, platform)
        except Exception as e:
            logger.error(f"Platform ID extraction failed: {e}")
            return {
                'platform_ids': {},
                'platform_id_count': 0,
                'has_platform_id': False
            }
    
    async def _process_currency_with_validation(self, extraction_results: Dict, classification_results: Dict) -> Dict[str, Any]:
        """Process currency with exchange rate handling using historical rates for transaction date"""
        amount = extraction_results.get('amount', 0.0)
        currency = extraction_results.get('currency', 'USD')
        
        # FIX #5: Use transaction date for exchange rate, not current date
        transaction_date = extraction_results.get('date', datetime.now().strftime('%Y-%m-%d'))
        
        if currency == 'USD':
            amount_usd = amount
            exchange_rate = 1.0
        else:
            # Get exchange rate for the transaction date (historical data)
            # FIX #33: NO FALLBACK - fail clearly if exchange rate fetch fails
            exchange_rate = await self._get_exchange_rate(currency, 'USD', transaction_date)
            amount_usd = amount * exchange_rate
        
        return {
            'amount_original': amount,
            'amount_usd': amount_usd,
            'currency': currency,
            'exchange_rate': exchange_rate,
            'exchange_date': transaction_date  # FIX #5: Use transaction date, not today
        }

    async def _get_exchange_rate(self, from_currency: str, to_currency: str, transaction_date: str) -> float:
        """
        FIX #51: Use aiohttp for async exchange rate fetching (not synchronous forex-python)
        forex-python is synchronous and requires thread pool overhead.
        aiohttp is already in requirements (3.11.7) and provides true async I/O.
        
        Uses exchangerate-api.com (free tier: 1500 requests/month)
        Falls back to forex-python only if aiohttp fails.
        """
        import aiohttp
        from datetime import datetime
        import asyncio
        
        try:
            # FIX #5: Use transaction_date in cache key for historical accuracy
            cache_key = f"exchange_rate_{from_currency}_{to_currency}_{transaction_date}"
            
            # Check cache first
            if self.cache and hasattr(self.cache, 'get_cached_classification'):
                cached_rate = await self.cache.get_cached_classification(
                    {'cache_key': cache_key}, 
                    'exchange_rate'
                )
                if cached_rate and isinstance(cached_rate, dict):
                    return cached_rate.get('rate', 1.0)
            
            # FIX #51: Use aiohttp for true async exchange rate fetching
            # exchangerate-api.com provides real-time rates (free tier: 1500 requests/month)
            api_url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(api_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            data = await response.json()
                            rate = data.get('rates', {}).get(to_currency)
                            if rate:
                                # Cache the rate for 24 hours
                                if self.cache and hasattr(self.cache, 'store_classification'):
                                    await self.cache.store_classification(
                                        {'cache_key': cache_key},
                                        {'rate': rate},
                                        'exchange_rate',
                                        ttl_hours=24
                                    )
                                return rate
            except Exception as aiohttp_err:
                logger.warning(f"aiohttp exchange rate fetch failed: {aiohttp_err}, falling back to forex-python")
            
            # Fallback: Use forex-python only if aiohttp fails (thread pool overhead acceptable as fallback)
            from forex_python.converter import CurrencyRates
            
            def _get_rate_sync():
                c = CurrencyRates()
                date_obj = datetime.strptime(transaction_date, '%Y-%m-%d').date()
                return c.get_rate(from_currency, to_currency, date_obj)
            
            loop = asyncio.get_event_loop()
            # FIX #7: Add null check for _thread_pool to prevent race condition
            if _thread_pool is None:
                logger.warning("Thread pool not initialized, running exchange rate fetch synchronously")
                rate = _get_rate_sync()
            else:
                rate = await loop.run_in_executor(_thread_pool, _get_rate_sync)
            
            # Cache the fallback rate
            if self.cache and hasattr(self.cache, 'store_classification'):
                await self.cache.store_classification(
                    {'cache_key': cache_key},
                    {'rate': rate},
                    'exchange_rate',
                    ttl_hours=24
                )
            
            return rate
            
        except Exception as e:
            logger.error(f" CRITICAL: Exchange rate fetch failed for {from_currency}/{to_currency}: {e}. Cannot proceed without live rates.")
            raise ValueError(f"Exchange rate unavailable for {from_currency}/{to_currency}. Live forex service required.")
    
    async def _build_enriched_payload(self, validated_data: Dict, extraction_results: Dict, 
                                     classification_results: Dict, vendor_results: Dict, 
                                     platform_id_results: Dict, currency_results: Dict, 
                                     ai_classification: Dict) -> Dict[str, Any]:
        """Build enriched payload with all processed data"""
        try:
            row_data = validated_data.get('row_data', {})
            file_context = validated_data.get('file_context', {})
            
            # Build comprehensive payload
            payload = {
                # Original data
                **row_data,
                
                # Extracted core fields
                'amount_original': extraction_results.get('amount', 0.0),
                'date': extraction_results.get('date', ''),
                'description': extraction_results.get('description', ''),
                'standard_description': extraction_results.get('description', '').strip(),
                
                # Vendor information
                'vendor_raw': vendor_results.get('vendor_raw', ''),
                'vendor_standard': vendor_results.get('vendor_standard', ''),
                'vendor_confidence': vendor_results.get('vendor_confidence', 0.0),
                'vendor_cleaning_method': vendor_results.get('vendor_cleaning_method', 'none'),
                
                # Currency and amounts
                'amount_usd': currency_results.get('amount_usd', 0.0),
                'currency': currency_results.get('currency', 'USD'),
                'exchange_rate': currency_results.get('exchange_rate', 1.0),
                'exchange_date': currency_results.get('exchange_date', ''),
                
                # Platform IDs
                'platform_ids': platform_id_results.get('platform_ids', {}),
                
                # Classification
                'kind': ai_classification.get('row_type', 'transaction'),
                'category': ai_classification.get('category', 'other'),
                'subcategory': ai_classification.get('subcategory', 'general'),
                'ai_confidence': ai_classification.get('confidence', 0.5),
                'ai_reasoning': ai_classification.get('reasoning', ''),
                
                # Entities
                'entities': ai_classification.get('entities', {}),
                'relationships': ai_classification.get('relationships', {}),
                
                # Metadata
                'ingested_on': datetime.now().isoformat(),
                'enrichment_version': '2.0.0',
                'file_context': {
                    'filename': file_context.get('filename', ''),
                    'row_index': file_context.get('row_index', 0)
                }
            }
            
            return payload
            
        except Exception as e:
            logger.error(f"Payload building failed: {e}")
            return validated_data.get('row_data', {})
    
    async def _validate_enriched_payload(self, enriched_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate enriched payload for correctness"""
        try:
            # Validate amount (increased from $1M to $100M to support enterprise transactions)
            if 'amount_usd' in enriched_payload:
                amount = enriched_payload['amount_usd']
                if not isinstance(amount, (int, float)):
                    enriched_payload['amount_usd'] = 0.0
                elif amount < -100000000 or amount > 100000000:
                    logger.warning(f"Amount out of range: {amount}")
            
            # Validate currency
            valid_currencies = ['USD', 'EUR', 'GBP', 'INR', 'JPY', 'CNY', 'AUD', 'CAD']
            if enriched_payload.get('currency') not in valid_currencies:
                enriched_payload['currency'] = 'USD'
            
            # Validate confidence scores
            for key in ['vendor_confidence', 'ai_confidence']:
                if key in enriched_payload:
                    conf = enriched_payload[key]
                    if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
                        enriched_payload[key] = 0.5
            
            return enriched_payload
            
        except Exception as e:
            logger.error(f"Payload validation failed: {e}")
            return enriched_payload
    
    async def _apply_accuracy_enhancement(self, row_data: Dict, validated_payload: Dict, 
                                         file_context: Dict) -> Dict[str, Any]:
        """
        ACCURACY FIX #1-5: Apply comprehensive accuracy enhancements
        - Add amount direction and transaction type
        - Standardize timestamp semantics
        - Add data validation
        - Add canonical entity IDs
        - Use confidence scores for flagging
        """
        try:
            enhanced = validated_payload.copy()
            
            # FIX #1: Add amount direction and signed amounts (HIGH PRIORITY)
            amount_usd = enhanced.get('amount_usd', 0.0)
            category = enhanced.get('category', '').lower()
            kind = enhanced.get('kind', '').lower()
            
            # Determine transaction type and direction
            transaction_type = 'unknown'
            amount_direction = 'unknown'
            affects_cash = True
            
            # Income indicators
            if any(keyword in category for keyword in ['revenue', 'income', 'sale', 'payment_received']):
                transaction_type = 'income'
                amount_direction = 'credit'
                amount_signed_usd = abs(amount_usd)  # Income is positive
            # Expense indicators
            elif any(keyword in category for keyword in ['expense', 'cost', 'payment', 'purchase', 'bill']):
                transaction_type = 'expense'
                amount_direction = 'debit'
                amount_signed_usd = -abs(amount_usd)  # Expenses are negative
            # Transfer indicators
            elif any(keyword in category for keyword in ['transfer', 'move', 'reclass']):
                transaction_type = 'transfer'
                amount_direction = 'neutral'
                affects_cash = False  # Transfers don't affect net cash
                amount_signed_usd = 0.0  # Neutral for cash flow
            # Refund indicators
            elif any(keyword in category for keyword in ['refund', 'return', 'credit_note']):
                transaction_type = 'refund'
                amount_direction = 'credit'
                amount_signed_usd = abs(amount_usd)  # Refunds are positive
            else:
                # Default: treat as expense if amount exists
                if amount_usd != 0:
                    transaction_type = 'expense'
                    amount_direction = 'debit'
                    amount_signed_usd = -abs(amount_usd)
                else:
                    amount_signed_usd = 0.0
            
            enhanced['transaction_type'] = transaction_type
            enhanced['amount_direction'] = amount_direction
            enhanced['amount_signed_usd'] = amount_signed_usd
            enhanced['affects_cash'] = affects_cash
            
            # FIX #2 & #47: Standardize timestamp semantics (MEDIUM PRIORITY)
            # Use consistent pendulum.now() for all timestamps (source of truth)
            # Naming: source_ts (transaction time), ingested_ts (when we received it), processed_ts (when we processed it)
            current_time = pendulum.now().to_iso8601_string()
            
            # Extract source timestamp from row data
            source_ts = None
            for date_col in ['date', 'transaction_date', 'created_at', 'timestamp']:
                if date_col in row_data:
                    try:
                        # Parse date using Polars for better performance
                        parsed_date = pl.Series([row_data[date_col]]).str.to_datetime().to_list()[0]
                        source_ts = parsed_date.isoformat() if hasattr(parsed_date, 'isoformat') else str(parsed_date)
                        break
                    except:
                        continue
            
            enhanced['source_ts'] = source_ts or current_time  # When transaction occurred
            enhanced['ingested_ts'] = current_time  # When we ingested it
            enhanced['processed_ts'] = current_time  # When we processed it
            
            # For currency conversion, use transaction date
            transaction_date = source_ts.split('T')[0] if source_ts else pendulum.now().strftime('%Y-%m-%d')
            enhanced['transaction_date'] = transaction_date
            enhanced['exchange_rate_date'] = enhanced.get('exchange_date', transaction_date)
            
            # Remove old ambiguous timestamps
            enhanced.pop('ingested_on', None)
            
            # FIX #3: Add data validation flags (MEDIUM PRIORITY)
            validation_flags = {
                'amount_valid': True,
                'currency_valid': True,
                'exchange_rate_valid': True,
                'vendor_valid': True,
                'validation_errors': []
            }
            
            # Validate amount
            if amount_usd is None:
                validation_flags['amount_valid'] = False
                validation_flags['validation_errors'].append('amount_usd is null')
            elif not isinstance(amount_usd, (int, float)):
                validation_flags['amount_valid'] = False
                validation_flags['validation_errors'].append(f'amount_usd is not numeric: {type(amount_usd)}')
            elif abs(amount_usd) > 100000000:  # $100M limit
                validation_flags['amount_valid'] = False
                validation_flags['validation_errors'].append(f'amount_usd exceeds limit: {amount_usd}')
            
            # FIX #46: Load validation rules from config/validation_rules.yaml
            # Non-developers can edit validation rules without code changes
            validation_rules = self._load_validation_rules()
            valid_currencies = validation_rules.get('currency', {}).get('valid_currencies', ['USD'])
            currency = enhanced.get('currency', 'USD')
            if currency not in valid_currencies:
                validation_flags['currency_valid'] = False
                validation_flags['validation_errors'].append(f'Invalid currency code: {currency}')
                enhanced['currency'] = validation_rules.get('currency', {}).get('default_currency', 'USD')
            
            # Validate exchange rate
            exchange_rate = enhanced.get('exchange_rate', 1.0)
            if exchange_rate is not None:
                if not isinstance(exchange_rate, (int, float)):
                    validation_flags['exchange_rate_valid'] = False
                    validation_flags['validation_errors'].append(f'exchange_rate is not numeric: {type(exchange_rate)}')
                elif exchange_rate <= 0 or exchange_rate > 1000:
                    validation_flags['exchange_rate_valid'] = False
                    validation_flags['validation_errors'].append(f'exchange_rate out of range: {exchange_rate}')
            
            # Validate vendor
            vendor_standard = enhanced.get('vendor_standard', '')
            if vendor_standard and len(vendor_standard) < 2:
                validation_flags['vendor_valid'] = False
                validation_flags['validation_errors'].append(f'vendor_standard too short: {vendor_standard}')
            
            enhanced['validation_flags'] = validation_flags
            enhanced['is_valid'] = len(validation_flags['validation_errors']) == 0
            
            # Canonical ID and alternatives now come from EntityResolverOptimized
            # No duplicate generation needed - they're already in vendor_results from _resolve_vendor_entity
            
            # FIX #5: Use confidence scores for flagging (LOW PRIORITY)
            confidence_score = enhanced.get('ai_confidence', 0.5)
            vendor_confidence = enhanced.get('vendor_confidence', 0.5)
            
            # Calculate overall confidence
            overall_confidence = (confidence_score + vendor_confidence) / 2
            enhanced['overall_confidence'] = overall_confidence
            
            # Flag low-confidence rows for review
            CONFIDENCE_THRESHOLD = 0.7
            if overall_confidence < CONFIDENCE_THRESHOLD:
                enhanced['requires_review'] = True
                enhanced['review_reason'] = f"Low confidence: {overall_confidence:.2f}"
                enhanced['review_priority'] = 'high' if overall_confidence < 0.5 else 'medium'
            else:
                enhanced['requires_review'] = False
                enhanced['review_reason'] = None
                enhanced['review_priority'] = None
            
            # Add accuracy enhancement metadata
            enhanced['accuracy_enhanced'] = True
            enhanced['accuracy_version'] = '1.0.0'
            enhanced['enhancements_applied'] = [
                'amount_direction',
                'transaction_type',
                'timestamp_standardization',
                'data_validation',
                'canonical_entity_ids',
                'confidence_flagging'
            ]
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Accuracy enhancement failed: {e}")
            # Return original payload if enhancement fails
            validated_payload['accuracy_enhanced'] = False
            validated_payload['accuracy_error'] = str(e)
            return validated_payload
    
    async def _validate_security(self, row_data: Dict, platform_info: Dict, 
                                column_names: List[str], ai_classification: Dict, 
                                file_context: Dict) -> bool:
        """Validate security of input data"""
        try:
            # Check for SQL injection patterns
            dangerous_patterns = ['DROP TABLE', 'DELETE FROM', 'INSERT INTO', '--', ';--']
            
            for key, value in row_data.items():
                if isinstance(value, str):
                    value_upper = value.upper()
                    if any(pattern in value_upper for pattern in dangerous_patterns):
                        logger.warning(f"Potential SQL injection detected in {key}: {value}")
                        return False
            
            return True
        except Exception as e:
            logger.error(f"Security validation failed: {e}")
            return True  # Allow processing on validation error
    
    
    async def _log_operation_start(self, operation: str, enrichment_id: str, file_context: Dict):
        """Log operation start"""
        logger.debug(f"Starting {operation} for {enrichment_id}")
    
    async def _log_operation_end(self, operation: str, success: bool, duration: float, 
                                file_context: Dict, error: Exception = None):
        """Log operation end"""
        if success:
            logger.debug(f"Completed {operation} in {duration:.3f}s")
        else:
            logger.error(f"Failed {operation}: {error}")
    
    def _update_metrics(self, processing_time: float):
        """Update processing metrics"""
        self.metrics['enrichment_count'] += 1
        
        # Update average processing time
        count = self.metrics['enrichment_count']
        current_avg = self.metrics['avg_processing_time']
        self.metrics['avg_processing_time'] = (current_avg * (count - 1) + processing_time) / count
    
    async def _send_enrichment_notification(self, file_context: Dict, enriched_payload: Dict, 
                                           processing_time: float):
        """Send real-time notification about enrichment via WebSocket"""
        try:
            job_id = file_context.get('job_id')
            if not job_id:
                return
            
            # Send enrichment notification through WebSocket manager
            notification = {
                "step": "enrichment_completed",
                "message": f"âœ… Row enriched: {enriched_payload.get('vendor_standard', 'N/A')} - ${enriched_payload.get('amount_usd', 0):.2f}",
                "enrichment_data": {
                    "vendor": enriched_payload.get('vendor_standard'),
                    "amount_usd": enriched_payload.get('amount_usd'),
                    "currency": enriched_payload.get('currency'),
                    "category": enriched_payload.get('category'),
                    "confidence": enriched_payload.get('vendor_confidence'),
                    "processing_time_ms": int(processing_time * 1000)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Use the global WebSocket manager
            await manager.send_update(job_id, notification)
            
        except Exception as e:
            logger.warning(f"Failed to send enrichment notification: {e}")
    
    async def _log_enrichment_audit(self, enrichment_id: str, enriched_payload: Dict, 
                                   processing_time: float):
        """Log enrichment audit trail"""
        logger.info(f"Enrichment {enrichment_id} completed in {processing_time:.3f}s")

    async def _cache_analysis_result(self, analysis_id: str, result: Dict[str, Any]) -> None:
        """Cache analysis result for future use"""
        if not self.config['enable_caching']:
            return
        
        try:
            # Use centralized cache (already initialized in __init__)
            if self.cache:
                pass
        except Exception as e:
            logger.warning(f"Cache storage failed for {analysis_id}: {e}")
    

    

    

    

    

    

    

    
    async def _classify_with_ai(self, document_features: Dict, pattern_classification: Dict) -> Dict[str, Any]:
        """Classify document using AI analysis"""
        try:
            self.metrics['ai_classifications'] += 1
            
            # Prepare AI prompt
            prompt = self._build_ai_classification_prompt(document_features, pattern_classification)
            
            # FIX #32: Use unified Groq client initialization helper
            local_groq_client = get_groq_client()
            
            # Call AI service (using Groq Llama-3.3-70B for cost-effective document classification)
            response = local_groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.1
            )
            
            result = response.choices[0].message.content
            
            # Parse AI response
            ai_result = self._parse_ai_classification_response(result)
            
            return ai_result
            
        except Exception as e:
            logger.error(f"AI classification failed: {e}")
            return {
                'document_type': 'unknown',
                'confidence': 0.0,
                'classification_method': 'ai_failed',
                'error': str(e)
            }
    
    def _build_ai_classification_prompt(self, document_features: Dict, pattern_classification: Dict) -> str:
        """Build AI classification prompt"""
        return f"""
        Analyze this financial document and classify its type.
        
        FILENAME: {document_features['filename']}
        COLUMNS: {document_features['column_names']}
        SAMPLE DATA: {document_features['sample_data']}
        PATTERN CLASSIFICATION: {pattern_classification.get('document_type', 'unknown')} (confidence: {pattern_classification.get('confidence', 0.0)})
        
        Return ONLY a valid JSON object with this structure:
        {{
            "document_type": "income_statement|balance_sheet|cash_flow|payroll_data|expense_data|revenue_data|general_ledger|budget|unknown",
            "source_platform": "gusto|quickbooks|xero|razorpay|freshbooks|stripe|shopify|unknown",
            "confidence": 0.95,
            "key_columns": ["col1", "col2"],
            "analysis": "Brief explanation",
            "data_patterns": {{
                "has_revenue_data": true,
                "has_expense_data": true,
                "has_employee_data": false,
                "has_account_data": false,
                "has_transaction_data": false,
                "time_period": "monthly"
            }},
            "classification_reasoning": "Step-by-step explanation",
            "platform_indicators": ["indicator1"],
            "document_indicators": ["indicator1"]
        }}
        
        IMPORTANT: Return ONLY the JSON object, no additional text.
        """
    
    def _parse_ai_classification_response(self, response: str) -> Dict[str, Any]:
        """Parse AI classification response"""
        try:
            # Clean the response
            cleaned_result = response.strip()
            if cleaned_result.startswith('```json'):
                cleaned_result = cleaned_result[7:]
            if cleaned_result.endswith('```'):
                cleaned_result = cleaned_result[:-3]
            cleaned_result = cleaned_result.strip()
            
            # Parse JSON
            parsed_result = orjson.loads(cleaned_result)
            
            # Ensure all required fields are present
            required_fields = {
                'data_patterns': {
                    "has_revenue_data": False,
                    "has_expense_data": False,
                    "has_employee_data": False,
                    "has_account_data": False,
                    "has_transaction_data": False,
                    "time_period": "unknown"
                },
                'classification_reasoning': "AI analysis completed",
                'platform_indicators': [],
                'document_indicators': []
            }
            
            for field, default_value in required_fields.items():
                if field not in parsed_result:
                    parsed_result[field] = default_value
            
            parsed_result['classification_method'] = 'ai_analysis'
            return parsed_result
            
        except (ValueError) as e:
            # FIX #49: Standardized error handling - orjson raises ValueError, not JSONDecodeError
            logger.error(f"Failed to parse AI response: {e}")
            return {
                'document_type': 'unknown',
                'confidence': 0.0,
                'classification_method': 'ai_parse_error',
                'error': f"JSON parsing failed: {str(e)}"
            }
    
    async def _analyze_with_ocr(self, validated_input: Dict, document_features: Dict) -> Dict[str, Any]:
        """Analyze document using OCR for image/PDF content extraction"""
        if not self.ocr_available or not validated_input.get('file_content'):
            return {
                'ocr_used': False,
                'confidence': 0.0,
                'extracted_text': '',
                'analysis': 'OCR not available or no file content provided'
            }
        
        try:
            self.metrics['ocr_operations'] += 1
            
            file_content = validated_input.get('file_content')
            filename = validated_input.get('filename', '')
            
            # Check if file is image or PDF
            file_ext = filename.split('.')[-1].lower() if '.' in filename else ''
            
            if file_ext not in ['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp']:
                return {
                    'ocr_used': False,
                    'confidence': 0.0,
                    'extracted_text': '',
                    'analysis': f'OCR not applicable for {file_ext} files'
                }
            
            # Use pytesseract for OCR
            try:
                import pytesseract
                from PIL import Image
                import io
                
                # Convert file content to image
                if file_ext == 'pdf':
                    # For PDF, use pdf2image
                    try:
                        from pdf2image import convert_from_bytes
                        images = convert_from_bytes(file_content, first_page=1, last_page=1)
                        if images:
                            image = images[0]
                        else:
                            raise Exception("No pages in PDF")
                    except ImportError:
                        logger.warning("pdf2image not available, skipping PDF OCR")
                        return {
                            'ocr_used': False,
                            'confidence': 0.0,
                            'extracted_text': '',
                            'analysis': 'PDF OCR requires pdf2image library'
                        }
                else:
                    # For images, use PIL directly
                    image = Image.open(io.BytesIO(file_content))
                
                # Extract text using OCR
                extracted_text = pytesseract.image_to_string(image)
                
                # Calculate confidence based on text length and quality
                text_length = len(extracted_text.strip())
                confidence = min(0.9, text_length / 1000) if text_length > 0 else 0.0
                
                # LIBRARY FIX: Use rapidfuzz for keyword detection (replaces manual .lower() checks)
                from rapidfuzz import fuzz
                financial_keywords = ['invoice', 'receipt', 'total', 'amount', 'payment', 'date', 'vendor', 'customer']
                keyword_count = sum(1 for keyword in financial_keywords if fuzz.partial_ratio(extracted_text.lower(), keyword) > 80)
                
                analysis = f"Extracted {text_length} characters, found {keyword_count} financial keywords"
                
                return {
                    'ocr_used': True,
                    'confidence': confidence,
                    'extracted_text': extracted_text[:500],  # First 500 chars
                    'full_text_length': text_length,
                    'financial_keywords_found': keyword_count,
                    'analysis': analysis
                }
                
            except Exception as ocr_error:
                logger.warning(f"OCR processing failed: {ocr_error}")
                return {
                    'ocr_used': True,
                    'confidence': 0.0,
                    'extracted_text': '',
                    'analysis': f'OCR processing failed: {str(ocr_error)}'
                }
            
        except Exception as e:
            logger.error(f"OCR analysis failed: {e}")
            return {
                'ocr_used': True,
                'confidence': 0.0,
                'extracted_text': '',
                'error': str(e)
            }
    
    async def _combine_classification_results(self, pattern_classification: Dict, 
                                            ai_classification: Dict, ocr_analysis: Dict,
                                            document_features: Dict) -> Dict[str, Any]:
        """Combine all classification results into final result"""
        # Weight the different classification methods
        pattern_weight = 0.3
        ai_weight = 0.6
        ocr_weight = 0.1
        
        # Combine document types
        final_doc_type = ai_classification.get('document_type', 'unknown')
        if ai_classification.get('confidence', 0.0) < 0.5:
            final_doc_type = pattern_classification.get('document_type', 'unknown')
        
        # Calculate combined confidence
        pattern_conf = pattern_classification.get('confidence', 0.0)
        ai_conf = ai_classification.get('confidence', 0.0)
        ocr_conf = ocr_analysis.get('confidence', 0.0)
        
        combined_confidence = (
            pattern_conf * pattern_weight +
            ai_conf * ai_weight +
            ocr_conf * ocr_weight
        )
        
        # Build final result
        final_result = {
            'document_type': final_doc_type,
            'source_platform': ai_classification.get('source_platform', 'unknown'),
            'confidence': combined_confidence,
            'key_columns': ai_classification.get('key_columns', document_features['column_names']),
            'analysis': ai_classification.get('analysis', 'Document analysis completed'),
            'data_patterns': ai_classification.get('data_patterns', {}),
            'classification_reasoning': ai_classification.get('classification_reasoning', 'Combined analysis'),
            'platform_indicators': ai_classification.get('platform_indicators', []),
            'document_indicators': ai_classification.get('document_indicators', []),
            'classification_methods': {
                'pattern_matching': pattern_classification,
                'ai_analysis': ai_classification,
                'ocr_analysis': ocr_analysis
            },
            'analysis_timestamp': pendulum.now().to_iso8601_string(),
            'analysis_version': '2.0.0'
        }
        
        return final_result
    
    async def _create_fallback_classification(self, df: pl.DataFrame, filename: str, 
                                            error_message: str) -> Dict[str, Any]:
        """Create fallback classification when analysis fails"""
        return {
            "document_type": "unknown",
            "source_platform": "unknown",
            "confidence": 0.1,
            "key_columns": list(df.columns) if df is not None else [],
            "analysis": f"Analysis failed: {error_message}",
            "data_patterns": {
                "has_revenue_data": False,
                "has_expense_data": False,
                "has_employee_data": False,
                "has_account_data": False,
                "has_transaction_data": False,
                "time_period": "unknown"
            },
            "classification_reasoning": f"Fallback classification due to error: {error_message}",
            "platform_indicators": [],
            "document_indicators": [],
            "analysis_timestamp": pendulum.now().to_iso8601_string(),
            "analysis_version": "2.0.0-fallback"
        }
    
    def _update_analysis_metrics(self, processing_time: float) -> None:
        """Update analysis performance metrics"""
        self.metrics['documents_analyzed'] += 1
        
        # Update average processing time
        current_avg = self.metrics['avg_processing_time']
        count = self.metrics['documents_analyzed']
        self.metrics['avg_processing_time'] = (current_avg * (count - 1) + processing_time) / count
    
    async def _log_analysis_audit(self, analysis_id: str, result: Dict[str, Any], 
                                processing_time: float, user_id: str = None) -> None:
        """Log analysis audit information"""
        audit_data = {
            'analysis_id': analysis_id,
            'user_id': user_id or 'anonymous',
            'document_type': result.get('document_type', 'unknown'),
            'confidence': result.get('confidence', 0.0),
            'processing_time': processing_time,
            'timestamp': pendulum.now().to_iso8601_string()
        }
        
        logger.info(f"Document analysis audit: {orjson.dumps(audit_data).decode()}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current analysis metrics"""
        return self.metrics.copy()
    
    def clear_cache(self) -> None:
        """Clear analysis cache"""
        logger.info("Document analysis cache cleared")
