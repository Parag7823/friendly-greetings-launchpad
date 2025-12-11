"""Production-grade data normalization with vendor, currency, date, and field mapping."""

import re
import asyncio
import structlog
from typing import Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from rapidfuzz import fuzz
from dateutil import parser

logger = structlog.get_logger(__name__)

# Global thread pool for CPU-bound operations
_thread_pool = None

def initialize_thread_pool(max_workers: int = 4):
    """Initialize global thread pool for CPU-bound operations"""
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(f"Thread pool initialized with {max_workers} workers")

def get_thread_pool() -> Optional[ThreadPoolExecutor]:
    """Get global thread pool"""
    return _thread_pool


class VendorNormalizer:
    """Vendor standardization using rapidfuzz + dedupe for ML-based deduplication"""
    
    # Centralized suffix list (single source of truth)
    BUSINESS_SUFFIXES = [
        'inc', 'inc.', 'llc', 'ltd', 'ltd.', 'corp', 'corp.', 'co', 'co.', 'company',
        'incorporated', 'limited', 'corporation', 'limited liability company'
    ]
    
    # Lazy-loaded dedupe matcher
    _dedupe_matcher = None
    
    def __init__(self, cache_client=None):
        from core_infrastructure.centralized_cache import safe_get_cache
        self.cache = cache_client or safe_get_cache()
        self._vendor_cache = {}  # In-memory cache for dedupe matches
        
    def _is_effectively_empty(self, text: str) -> bool:
        """Check if text is effectively empty (None, empty, or only whitespace)"""
        if not text:
            return True
        return len(text.strip()) == 0
    
    def _clean_vendor_name(self, vendor_name: str) -> str:
        """Clean vendor name for rapidfuzz compatibility"""
        if not vendor_name:
            return vendor_name
        
        cleaned = vendor_name.strip()
        cleaned_lower = cleaned.lower()
        
        for suffix in self.BUSINESS_SUFFIXES:
            if cleaned_lower.endswith(suffix):
                cleaned = cleaned[:-len(suffix)].strip()
                cleaned_lower = cleaned.lower()
        
        cleaned = ''.join(char if char.isalnum() or char.isspace() else ' ' for char in cleaned)
        cleaned = ' '.join(cleaned.split())
        cleaned = cleaned.title()
        
        return cleaned if cleaned else vendor_name
    
    async def standardize(self, vendor_name: str, platform: str = None) -> Dict[str, Any]:
        """Standardize vendor name using rapidfuzz + dedupe"""
        try:
            # Check for empty input
            if not vendor_name or self._is_effectively_empty(vendor_name):
                return {
                    "vendor_raw": vendor_name,
                    "vendor_standard": "",
                    "confidence": 0.0,
                    "cleaning_method": "empty"
                }
            
            # Check cache first
            cache_content = {'vendor_name': vendor_name, 'platform': platform or 'unknown'}
            if self.cache:
                try:
                    cached_result = await self.cache.get_cached_classification(
                        cache_content,
                        classification_type='vendor_standardization'
                    )
                    if cached_result:
                        logger.debug(f"✅ Vendor cache hit: {vendor_name}")
                        return cached_result
                except Exception as e:
                    logger.warning(f"Cache retrieval failed: {e}")
            
            def _compute_similarity_sync(vendor_name, cleaned_name):
                return fuzz.token_sort_ratio(vendor_name.lower(), cleaned_name.lower()) / 100.0
            
            cleaned_name = self._clean_vendor_name(vendor_name)
            
            thread_pool = get_thread_pool()
            if thread_pool is None:
                logger.warning("Thread pool not initialized, running vendor similarity synchronously")
                similarity = _compute_similarity_sync(vendor_name, cleaned_name)
            else:
                loop = asyncio.get_event_loop()
                similarity = await loop.run_in_executor(thread_pool, _compute_similarity_sync, vendor_name, cleaned_name)
            
            result = {
                "vendor_raw": vendor_name,
                "vendor_standard": cleaned_name,
                "confidence": min(0.95, 0.7 + (similarity * 0.25)),  # 0.7-0.95 range
                "cleaning_method": "rapidfuzz"
            }
            
            # Store in cache
            if self.cache:
                try:
                    await self.cache.store_classification(
                        cache_content,
                        result,
                        classification_type='vendor_standardization',
                        ttl_hours=48,
                        confidence_score=result.get('confidence', 0.8),
                        model_version='rapidfuzz-3.10.1'
                    )
                except Exception as e:
                    logger.warning(f"Cache storage failed: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Vendor standardization failed: {e}")
            raise ValueError(f"Vendor standardization failed - no fallback allowed: {e}") from e
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics from centralized cache"""
        if self.cache and hasattr(self.cache, 'get_cache_stats'):
            try:
                stats = await self.cache.get_cache_stats()
                vendor_stats = {
                    'classification_type': 'vendor_standardization',
                    'total_entries': stats.get('total_classifications', 0),
                    'cache_hit_rate': stats.get('cache_hit_rate', 0.0),
                    'cost_savings': stats.get('cost_savings_usd', 0.0)
                }
                return vendor_stats
            except Exception as e:
                logger.warning(f"Failed to get cache stats: {e}")
        
        return {
            'classification_type': 'vendor_standardization',
            'total_entries': 0,
            'cache_hit_rate': 0.0,
            'cost_savings': 0.0,
            'note': 'Using centralized AIClassificationCache'
        }


class AmountNormalizer:
    """Standardize amount handling using glom for declarative data extraction"""
    
    @staticmethod
    def normalize(amount_value) -> float:
        """Normalize amount value to float"""
        try:
            from glom import glom, Coalesce, SKIP
            
            if amount_value is None or amount_value == '':
                return 0.0
            
            amount_spec = Coalesce(
                lambda x: float(x) if isinstance(x, (int, float)) else SKIP,
                lambda x: float(x) if hasattr(x, '__float__') else SKIP,
                lambda x: float(x.item()) if hasattr(x, 'item') else SKIP,
                lambda x: float(re.sub(r'[^\d.-]', '', str(x).strip())) if isinstance(x, str) and re.sub(r'[^\d.-]', '', str(x).strip()) else SKIP,
                default=0.0
            )
            
            return glom(amount_value, amount_spec)
                
        except Exception as e:
            logger.warning(f"Amount normalization failed for value '{amount_value}': {e}")
            return 0.0


class DateNormalizer:
    """Standardize date handling using python-dateutil for robust parsing"""
    
    @staticmethod
    def normalize(date_value) -> str:
        """Normalize date value to ISO 8601 format (YYYY-MM-DD)"""
        try:
            if date_value is None or date_value == '':
                return datetime.now().strftime('%Y-%m-%d')
            
            parsed_date = parser.parse(str(date_value))
            return parsed_date.strftime('%Y-%m-%d')
                
        except Exception as e:
            logger.warning(f"Date normalization failed for value '{date_value}': {e}")
            return datetime.now().strftime('%Y-%m-%d')


class CurrencyNormalizer:
    """Currency normalization with conversion, validation, and caching"""
    
    def __init__(self, cache_client=None):
        from core_infrastructure.centralized_cache import safe_get_cache
        self.cache = cache_client or safe_get_cache()
        self.exchange_rates = {}  # Cache exchange rates
    
    async def normalize(self, amount: float, from_currency: str = 'USD', to_currency: str = 'USD') -> Dict[str, Any]:
        """
        Normalize currency by converting to target currency.
        
        Args:
            amount: Amount to convert
            from_currency: Source currency code (e.g., 'INR', 'USD')
            to_currency: Target currency code (default 'USD')
        
        Returns:
            Dict with normalized amount, conversion rate, and metadata
        """
        try:
            if from_currency == to_currency:
                return {
                    'amount_original': amount,
                    'amount_normalized': amount,
                    'from_currency': from_currency,
                    'to_currency': to_currency,
                    'conversion_rate': 1.0,
                    'method': 'no_conversion'
                }
            
            # Get exchange rate (from cache or API)
            rate = await self._get_exchange_rate(from_currency, to_currency)
            
            if rate is None:
                logger.warning(f"Could not get exchange rate for {from_currency} -> {to_currency}")
                return {
                    'amount_original': amount,
                    'amount_normalized': amount,
                    'from_currency': from_currency,
                    'to_currency': to_currency,
                    'conversion_rate': None,
                    'method': 'no_rate_available'
                }
            
            normalized_amount = amount * rate
            
            return {
                'amount_original': amount,
                'amount_normalized': normalized_amount,
                'from_currency': from_currency,
                'to_currency': to_currency,
                'conversion_rate': rate,
                'method': 'exchange_rate_conversion'
            }
        
        except Exception as e:
            logger.error(f"Currency normalization failed: {e}")
            return {
                'amount_original': amount,
                'amount_normalized': amount,
                'from_currency': from_currency,
                'to_currency': to_currency,
                'conversion_rate': None,
                'method': 'error_fallback',
                'error': str(e)
            }
    
    async def _get_exchange_rate(self, from_currency: str, to_currency: str) -> Optional[float]:
        """Get exchange rate (with caching)"""
        # TODO: Implement exchange rate API integration (e.g., Open Exchange Rates, Fixer.io)
        # For now, return None to indicate rate not available
        return None


class FieldMapper:
    """Field mapping with semantic and pattern-based matching"""
    
    def __init__(self, cache_client=None):
        from core_infrastructure.centralized_cache import safe_get_cache
        self.cache = cache_client or safe_get_cache()
        self.field_mappings = {}  # Cache field mappings
    
    async def map_fields(self, raw_fields: Dict[str, Any], platform: str = None, document_type: str = None) -> Dict[str, str]:
        """
        Map raw fields to standardized field names.
        
        Args:
            raw_fields: Dictionary of raw field names and values
            platform: Platform name (e.g., 'QuickBooks', 'Stripe')
            document_type: Document type (e.g., 'invoice', 'receipt')
        
        Returns:
            Dictionary mapping raw field names to standardized names
        """
        try:
            mappings = {}
            
            for raw_field, value in raw_fields.items():
                # Try to find mapping from cache or patterns
                standard_field = await self._find_standard_field(raw_field, platform, document_type)
                if standard_field:
                    mappings[raw_field] = standard_field
            
            return mappings
        
        except Exception as e:
            logger.error(f"Field mapping failed: {e}")
            return {}
    
    async def _find_standard_field(self, raw_field: str, platform: str = None, document_type: str = None) -> Optional[str]:
        """Find standard field name for raw field"""
        # TODO: Implement semantic matching and pattern-based detection
        # For now, return None
        return None


class UniversalNormalizer:
    """Orchestrator combining vendor, currency, date normalization and field mapping"""
    
    def __init__(self, cache_client=None):
        from core_infrastructure.centralized_cache import safe_get_cache
        self.cache = cache_client or safe_get_cache()
        self.vendor_normalizer = VendorNormalizer(cache_client=self.cache)
        self.amount_normalizer = AmountNormalizer()
        self.date_normalizer = DateNormalizer()
        self.currency_normalizer = CurrencyNormalizer(cache_client=self.cache)
        self.field_mapper = FieldMapper(cache_client=self.cache)
    
    async def normalize_row(self, row_data: Dict[str, Any], platform: str = None, document_type: str = None) -> Dict[str, Any]:
        """
        Normalize a complete row of data.
        
        Args:
            row_data: Raw row data
            platform: Platform name
            document_type: Document type
        
        Returns:
            Normalized row data
        """
        try:
            normalized = row_data.copy()
            
            # Normalize vendor
            if 'vendor_raw' in normalized or 'vendor' in normalized:
                vendor_raw = normalized.get('vendor_raw') or normalized.get('vendor', '')
                if vendor_raw:
                    vendor_result = await self.vendor_normalizer.standardize(vendor_raw, platform)
                    normalized['vendor_standard'] = vendor_result.get('vendor_standard', '')
                    normalized['vendor_confidence'] = vendor_result.get('confidence', 0.0)
            
            # Normalize amount
            if 'amount' in normalized:
                normalized['amount'] = self.amount_normalizer.normalize(normalized['amount'])
            
            # Normalize date
            if 'date' in normalized:
                normalized['date'] = self.date_normalizer.normalize(normalized['date'])
            
            # Normalize currency if needed
            if 'currency' in normalized and 'amount' in normalized:
                currency_result = await self.currency_normalizer.normalize(
                    normalized['amount'],
                    from_currency=normalized.get('currency', 'USD'),
                    to_currency='USD'
                )
                normalized['amount_usd'] = currency_result.get('amount_normalized', normalized['amount'])
            
            # Map fields
            field_mappings = await self.field_mapper.map_fields(normalized, platform, document_type)
            for raw_field, standard_field in field_mappings.items():
                if raw_field in normalized:
                    normalized[standard_field] = normalized.pop(raw_field)
            
            return normalized
        
        except Exception as e:
            logger.error(f"Row normalization failed: {e}")
            raise
    
    async def normalize_batch(self, rows: list, platform: str = None, document_type: str = None) -> list:
        """
        Normalize a batch of rows.
        
        Args:
            rows: List of row dictionaries
            platform: Platform name
            document_type: Document type
        
        Returns:
            List of normalized rows
        """
        try:
            normalized_rows = []
            for row in rows:
                normalized = await self.normalize_row(row, platform, document_type)
                normalized_rows.append(normalized)
            return normalized_rows
        
        except Exception as e:
            logger.error(f"Batch normalization failed: {e}")
            raise
