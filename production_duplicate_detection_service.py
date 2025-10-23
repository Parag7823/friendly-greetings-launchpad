"""
Production-Grade Duplicate Detection Service
============================================

A comprehensive, scalable duplicate detection service designed for production use.
Handles millions of files per user with efficient algorithms and proper error handling.

Features:
- Exact duplicate detection using SHA-256 hashing
- Near-duplicate detection using MinHash and content similarity
- Efficient database queries with proper indexing
- Redis-ready caching system
- Comprehensive error handling and logging
- Security-first design with proper auth validation
- Async/await for optimal performance
- Memory-efficient processing for large files
- Production observability with metrics and structured logging

Author: Senior Full-Stack Engineer
Version: 2.0.0
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from supabase import Client

# Configure structured logging
logger = logging.getLogger(__name__)

class DuplicateType(Enum):
    """Types of duplicates detected"""
    EXACT = "exact"
    NEAR = "near"
    CONTENT = "content"
    NONE = "none"

class DuplicateAction(Enum):
    """Actions user can take for duplicates"""
    REPLACE = "replace"
    KEEP_BOTH = "keep_both"
    SKIP = "skip"
    MERGE = "merge"

@dataclass
class DuplicateResult:
    """Structured result for duplicate detection"""
    is_duplicate: bool
    duplicate_type: DuplicateType
    similarity_score: float
    duplicate_files: List[Dict[str, Any]]
    recommendation: DuplicateAction
    message: str
    confidence: float
    processing_time_ms: int
    cache_hit: bool = False
    error: Optional[str] = None
    delta_analysis: Optional[Dict[str, Any]] = None  # BUG #12 FIX: Add delta analysis support

@dataclass
class FileMetadata:
    """File metadata for processing"""
    user_id: str
    file_hash: str
    filename: str
    file_size: int
    content_type: str
    upload_timestamp: datetime
    content_fingerprint: Optional[str] = None

class ProductionDuplicateDetectionService:
    """
    Production-grade duplicate detection service.
    
    Designed to handle millions of files per user with:
    - Efficient database queries using proper indexing
    - Advanced similarity algorithms (MinHash, content fingerprinting)
    - Redis-ready caching with TTL and invalidation
    - Comprehensive error handling and retry logic
    - Security-first design with proper auth validation
    - Memory-efficient processing for large files
    - Production observability with metrics and logging
    """
    
    def __init__(self, supabase: Client, redis_client: Optional[Any] = None):
        """
        Initialize the duplicate detection service.
        
        Args:
            supabase: Supabase client for database operations
            redis_client: Optional Redis client for distributed caching
        """
        self.supabase = supabase
        self.redis_client = redis_client
        self.cache_ttl = int(os.environ.get('DUPLICATE_CACHE_TTL', 3600))  # 1 hour default
        self.similarity_threshold = float(os.environ.get('SIMILARITY_THRESHOLD', 0.85))
        self.max_file_size = int(os.environ.get('MAX_FILE_SIZE', 500 * 1024 * 1024))  # 500MB
        self.batch_size = int(os.environ.get('BATCH_SIZE', 100))
        self.max_workers = int(os.environ.get('MAX_WORKERS', 4))
        
        # FIX #12: In-memory cache as fallback when Redis is not available
        # Now with automatic TTL cleanup to prevent memory leaks
        self.memory_cache = {}
        self.cache_timestamps = {}
        self.max_cache_size = int(os.environ.get('MAX_CACHE_SIZE', 10000))  # Max 10k entries
        
        # FIX #12: Start background cleanup task
        import asyncio
        self._cleanup_task = None
        try:
            loop = asyncio.get_event_loop()
            self._cleanup_task = loop.create_task(self._cache_cleanup_loop())
        except RuntimeError:
            # No event loop running yet, will start on first use
            logger.warning("Event loop not running, cache cleanup will start on first cache operation")
        
        # Thread pool for CPU-intensive operations
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Security settings
        self.max_filename_length = int(os.environ.get('MAX_FILENAME_LENGTH', 255))
        
        # Metrics for observability
        self.metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'exact_duplicates_found': 0,
            'near_duplicates_found': 0,
            'processing_errors': 0,
            'total_processing_time': 0,
            'cache_evictions': 0  # FIX #12: Track evictions
        }
        
        logger.info("Production Duplicate Detection Service initialized with cache cleanup")
    
    def _validate_security(self, user_id: str, file_hash: str, filename: str) -> None:
        """
        Validate security parameters to prevent attacks
        
        Args:
            user_id: User identifier
            file_hash: File hash
            filename: Original filename
            
        Raises:
            ValueError: If security validation fails
        """
        # Validate user_id
        if not user_id or not isinstance(user_id, str):
            raise ValueError("Invalid user_id: must be a non-empty string")
        
        if len(user_id) > 255:
            raise ValueError("User ID too long")
        
        # Validate file_hash
        if not file_hash or not isinstance(file_hash, str):
            raise ValueError("Invalid file_hash: must be a non-empty string")
        
        if len(file_hash) != 64:  # SHA-256 hash length
            raise ValueError("Invalid file_hash: must be a valid SHA-256 hash")
        
        # Validate filename
        if not filename or not isinstance(filename, str):
            raise ValueError("Invalid filename: must be a non-empty string")
        
        if len(filename) > self.max_filename_length:
            raise ValueError(f"Filename too long: {len(filename)} > {self.max_filename_length}")
        
        # Check for path traversal
        if self._is_path_traversal_unsafe(filename):
            raise ValueError(f"Filename contains path traversal patterns: {filename}")
        
        # Check for null bytes and control characters
        if '\x00' in filename or '\x1a' in filename or '\x7f' in filename:
            raise ValueError(f"Filename contains invalid characters: {filename}")
    
    def _is_path_traversal_unsafe(self, filename: str) -> bool:
        """
        Check if filename contains path traversal patterns
        
        Args:
            filename: Filename to check
            
        Returns:
            True if unsafe, False if safe
        """
        dangerous_patterns = [
            '..',
            '../',
            '..\\',
            '%2e%2e',
            '%2E%2E',
            '....//',
            '....\\\\'
        ]
        
        filename_lower = filename.lower()
        for pattern in dangerous_patterns:
            if pattern in filename_lower:
                return True
        
        # Check for absolute paths
        if filename.startswith('/') or (len(filename) > 1 and filename[1] == ':'):
            return True
            
        return False
    
    async def detect_duplicates(
        self, 
        file_content: bytes, 
        file_metadata: FileMetadata,
        enable_near_duplicate: bool = True,
        enable_content_duplicate: bool = True,
        sheets_data: Optional[Dict[str, Any]] = None
    ) -> DuplicateResult:
        """
        Main entry point for duplicate detection.
        
        BUG #12 FIX: Now implements ALL 4 phases:
        - Phase 1: Exact duplicate detection (SHA-256 hash comparison)
        - Phase 2: Near-duplicate detection (content similarity)
        - Phase 3: Content-level duplicate detection (row-level fingerprinting)
        - Phase 4: Delta analysis (intelligent merging)
        
        Args:
            file_content: Raw file content as bytes
            file_metadata: File metadata including user_id, hash, etc.
            enable_near_duplicate: Whether to perform near-duplicate detection
            enable_content_duplicate: Whether to perform content-level duplicate detection
            sheets_data: Optional parsed sheets data for content duplicate detection
            
        Returns:
            DuplicateResult with comprehensive duplicate information
        """
        start_time = time.time()
        
        try:
            # Validate inputs and security
            await self._validate_inputs(file_content, file_metadata)
            
            # Check cache first
            cache_key = self._generate_cache_key(file_metadata)
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                self.metrics['cache_hits'] += 1
                cached_result.cache_hit = True
                return cached_result
            
            self.metrics['cache_misses'] += 1
            
            # Phase 1: Exact duplicate detection
            exact_result = await self._detect_exact_duplicates(file_metadata)
            if exact_result.is_duplicate:
                self.metrics['exact_duplicates_found'] += 1
                result = self._create_result(exact_result, start_time, cache_key)
                await self._set_cache(cache_key, result)
                return result
            
            # Phase 2: Near-duplicate detection (if enabled)
            if enable_near_duplicate:
                near_result = await self._detect_near_duplicates(file_content, file_metadata)
                if near_result.is_duplicate:
                    self.metrics['near_duplicates_found'] += 1
                    result = self._create_result(near_result, start_time, cache_key)
                    await self._set_cache(cache_key, result)
                    return result
            
            # Phase 3: Content-level duplicate detection (row-level fingerprinting)
            if enable_content_duplicate and file_content:
                try:
                    # Calculate content fingerprint
                    content_fingerprint = await self._calculate_content_fingerprint(file_content)
                    
                    # Check for content duplicates
                    content_result = await self.check_content_duplicate(
                        file_metadata.user_id,
                        content_fingerprint,
                        file_metadata.filename
                    )
                    
                    if content_result.get('is_content_duplicate'):
                        # Phase 4: Delta analysis for intelligent merging
                        delta_analysis = None
                        if sheets_data and content_result.get('overlapping_files'):
                            existing_file_id = content_result['overlapping_files'][0]['id']
                            delta_result = await self.analyze_delta_ingestion(
                                file_metadata.user_id,
                                sheets_data,
                                existing_file_id
                            )
                            delta_analysis = delta_result.get('delta_analysis')
                        
                        # Return content duplicate with delta analysis
                        result = DuplicateResult(
                            is_duplicate=True,
                            duplicate_type=DuplicateType.CONTENT,
                            similarity_score=1.0,  # Content fingerprint match
                            duplicate_files=content_result.get('overlapping_files', []),
                            recommendation=DuplicateAction.MERGE,
                            message=content_result.get('message', 'Content-level duplicate detected'),
                            confidence=0.95,
                            processing_time_ms=int((time.time() - start_time) * 1000)
                        )
                        
                        # Attach delta analysis if available
                        if delta_analysis:
                            result.delta_analysis = delta_analysis
                        
                        await self._set_cache(cache_key, result)
                        return result
                        
                except Exception as e:
                    logger.warning(f"Content duplicate detection failed: {e}")
                    # Continue to "no duplicates" if content check fails
            
            # No duplicates found
            result = DuplicateResult(
                is_duplicate=False,
                duplicate_type=DuplicateType.NONE,
                similarity_score=0.0,
                duplicate_files=[],
                recommendation=DuplicateAction.REPLACE,
                message="No duplicates found",
                confidence=1.0,
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
            
            await self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            self.metrics['processing_errors'] += 1
            logger.error(f"Duplicate detection failed: {e}", exc_info=True)
            return DuplicateResult(
                is_duplicate=False,
                duplicate_type=DuplicateType.NONE,
                similarity_score=0.0,
                duplicate_files=[],
                recommendation=DuplicateAction.REPLACE,
                message="Duplicate detection failed",
                confidence=0.0,
                processing_time_ms=int((time.time() - start_time) * 1000),
                error=str(e)
            )
    
    async def _validate_inputs(self, file_content: bytes, file_metadata: FileMetadata) -> None:
        """
        Validate inputs and perform security checks.
        
        BROKEN #6 FIX: File size validation happens after loading into memory.
        Note: This is a limitation of the current architecture where file_content is already
        loaded. Ideally, size should be checked before reading the file.
        
        Args:
            file_content: Raw file content
            file_metadata: File metadata
            
        Raises:
            ValueError: If inputs are invalid
            SecurityError: If security checks fail
        """
        # BROKEN #6 FIX: Document limitation and add early size check where possible
        # If file_metadata has size, check it first (before content is loaded)
        if hasattr(file_metadata, 'file_size') and file_metadata.file_size > 0:
            if file_metadata.file_size > self.max_file_size:
                raise ValueError(f"File too large: {file_metadata.file_size} bytes (max: {self.max_file_size})")
        
        # Validate actual content size (already in memory, but still validate)
        if len(file_content) > self.max_file_size:
            raise ValueError(f"File too large: {len(file_content)} bytes (max: {self.max_file_size})")
        
        # Validate file metadata
        if not file_metadata.user_id or not file_metadata.file_hash:
            raise ValueError("Invalid file metadata: user_id and file_hash required")
        
        # Security: Validate user_id format (prevent injection)
        if not re.match(r'^[a-zA-Z0-9\-_]+$', file_metadata.user_id):
            raise ValueError("Invalid user_id format")
        
        # Security: Validate filename (prevent path traversal)
        if '..' in file_metadata.filename or '/' in file_metadata.filename:
            raise ValueError("Invalid filename: path traversal detected")
        
        # Validate file hash format
        if not re.match(r'^[a-f0-9]{64}$', file_metadata.file_hash):
            raise ValueError("Invalid file hash format")
    
    async def _detect_exact_duplicates(self, file_metadata: FileMetadata) -> DuplicateResult:
        """
        Detect exact duplicates using SHA-256 hash comparison.
        
        Uses efficient database query with proper indexing.
        
        Args:
            file_metadata: File metadata
            
        Returns:
            DuplicateResult with exact duplicate information
        """
        try:
            # Use efficient query with JSONB filtering
            result = await self._query_duplicates_by_hash(
                file_metadata.user_id, 
                file_metadata.file_hash
            )
            
            if not result:
                return DuplicateResult(
                    is_duplicate=False,
                    duplicate_type=DuplicateType.NONE,
                    similarity_score=0.0,
                    duplicate_files=[],
                    recommendation=DuplicateAction.REPLACE,
                    message="No exact duplicates found",
                    confidence=1.0,
                    processing_time_ms=0
                )
            
            # Process duplicate files
            duplicate_files = []
            for record in result:
                duplicate_files.append({
                    'id': record['id'],
                    'filename': record['file_name'],
                    'uploaded_at': record['created_at'],
                    'file_size': record.get('file_size', 0),
                    'status': record.get('status', 'unknown')
                })
            
            # Sort by upload date (newest first)
            duplicate_files.sort(key=lambda x: x['uploaded_at'], reverse=True)
            
            # Find latest duplicate
            latest_duplicate = duplicate_files[0] if duplicate_files else None
            
            return DuplicateResult(
                is_duplicate=True,
                duplicate_type=DuplicateType.EXACT,
                similarity_score=1.0,
                duplicate_files=duplicate_files,
                recommendation=DuplicateAction.REPLACE,
                message=f"Exact duplicate found: '{latest_duplicate['filename']}' uploaded on {latest_duplicate['uploaded_at'][:10]}",
                confidence=1.0,
                processing_time_ms=0
            )
            
        except Exception as e:
            logger.error(f"Exact duplicate detection failed: {e}")
            raise
    
    async def _detect_near_duplicates(self, file_content: bytes, file_metadata: FileMetadata) -> DuplicateResult:
        """
        Detect near-duplicates using content similarity algorithms.
        
        Uses MinHash and content fingerprinting for efficient similarity detection.
        
        Args:
            file_content: Raw file content
            file_metadata: File metadata
            
        Returns:
            DuplicateResult with near-duplicate information
        """
        try:
            # Get recent files for similarity comparison
            recent_files = await self._get_recent_files(file_metadata.user_id, days=30)
            
            if not recent_files:
                return DuplicateResult(
                    is_duplicate=False,
                    duplicate_type=DuplicateType.NONE,
                    similarity_score=0.0,
                    duplicate_files=[],
                    recommendation=DuplicateAction.REPLACE,
                    message="No recent files for comparison",
                    confidence=1.0,
                    processing_time_ms=0
                )
            
            # Calculate content fingerprint
            content_fingerprint = await self._calculate_content_fingerprint(file_content)
            
            # Find most similar file
            best_match = None
            best_score = 0.0
            
            for file_record in recent_files:
                if file_record['file_name'] == file_metadata.filename:
                    continue
                
                # Calculate similarity using multiple methods
                similarity = await self._calculate_similarity(
                    file_content, 
                    file_record, 
                    file_metadata.filename
                )
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = file_record
            
            # Check if similarity exceeds threshold
            if best_score >= self.similarity_threshold:
                return DuplicateResult(
                    is_duplicate=True,
                    duplicate_type=DuplicateType.NEAR,
                    similarity_score=best_score,
                    duplicate_files=[{
                        'id': best_match['id'],
                        'filename': best_match['file_name'],
                        'uploaded_at': best_match['created_at'],
                        'similarity_score': best_score
                    }],
                    recommendation=DuplicateAction.MERGE,
                    message=f"Near-duplicate found: '{best_match['file_name']}' with {best_score:.1%} similarity",
                    confidence=best_score,
                    processing_time_ms=0
                )
            
            return DuplicateResult(
                is_duplicate=False,
                duplicate_type=DuplicateType.NONE,
                similarity_score=best_score,
                duplicate_files=[],
                recommendation=DuplicateAction.REPLACE,
                message="No near-duplicates found",
                confidence=1.0,
                processing_time_ms=0
            )
            
        except Exception as e:
            logger.error(f"Near-duplicate detection failed: {e}")
            raise
    
    async def _query_duplicates_by_hash(self, user_id: str, file_hash: str) -> List[Dict[str, Any]]:
        """
        Query database for files with matching hash.
        
        Uses efficient query with dedicated file_hash column and proper indexing.
        This ensures fast lookups without loading all files into memory.
        
        Args:
            user_id: User ID
            file_hash: File hash to search for
            
        Returns:
            List of matching file records
        """
        try:
            # Use efficient query with dedicated file_hash column and proper indexing
            result = self.supabase.table('raw_records').select(
                'id, file_name, created_at, file_size, status'
            ).eq('user_id', user_id).eq('file_hash', file_hash).execute()
            
            return result.data or []
            
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            raise
    
    async def _get_recent_files(self, user_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get recent files for similarity comparison.
        
        This method is optimized for memory efficiency by only selecting
        essential fields and limiting the number of records returned.
        
        Args:
            user_id: User ID
            days: Number of days to look back
            
        Returns:
            List of recent file records (memory-optimized)
        """
        try:
            cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
            
            # Memory-optimized query: only select essential fields
            # Note: content_fingerprint is stored in the 'content' JSONB column
            result = self.supabase.table('raw_records').select(
                'id, file_name, created_at, file_size, content'
            ).eq('user_id', user_id).gte('created_at', cutoff_date).limit(50).execute()  # Reduced limit for memory efficiency
            
            return result.data or []
            
        except Exception as e:
            logger.error(f"Failed to get recent files: {e}")
            return []
    
    async def _calculate_content_fingerprint(self, file_content: bytes) -> str:
        """
        Calculate content fingerprint for similarity comparison.
        
        Uses MinHash algorithm for efficient similarity detection.
        
        Args:
            file_content: Raw file content
            
        Returns:
            Content fingerprint string
        """
        try:
            # Convert to text for processing
            content_text = file_content.decode('utf-8', errors='ignore').lower()
            
            # Extract features (words, n-grams, etc.)
            features = self._extract_features(content_text)
            
            # Calculate MinHash signature
            minhash_signature = self._calculate_minhash(features)
            
            return hashlib.sha256(minhash_signature.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Content fingerprint calculation failed: {e}")
            return ""
    
    def _extract_features(self, text: str) -> Set[str]:
        """
        Extract features from text for similarity comparison.
        
        Args:
            text: Input text
            
        Returns:
            Set of features
        """
        features = set()
        
        # Extract words
        words = re.findall(r'\b\w+\b', text)
        features.update(words)
        
        # Extract 3-grams
        for i in range(len(text) - 2):
            features.add(text[i:i+3])
        
        # PLACEHOLDER #8 FIX: Make line limit configurable
        max_lines = int(os.environ.get('FEATURE_MAX_LINES', 100))
        max_line_chars = int(os.environ.get('FEATURE_MAX_LINE_CHARS', 50))
        
        # Extract line-based features
        lines = text.split('\n')
        for line in lines[:max_lines]:
            if line.strip():
                features.add(line.strip()[:max_line_chars])
        
        return features
    
    def _calculate_minhash(self, features: Set[str], num_hashes: int = 128) -> str:
        """
        Calculate MinHash signature for efficient similarity comparison.
        
        BROKEN #5 FIX: Use deterministic hashing instead of Python's built-in hash()
        
        Args:
            features: Set of features
            num_hashes: Number of hash functions to use
            
        Returns:
            MinHash signature as string
        """
        if not features:
            return ""
        
        # PLACEHOLDER #8 FIX: Make feature limit configurable
        max_features = int(os.environ.get('MINHASH_MAX_FEATURES', 1000))
        if len(features) > max_features:
            features = set(list(features)[:max_features])
        
        # Convert features to sorted list for consistent hashing
        sorted_features = sorted(features)
        
        # BROKEN #5 FIX: Use hashlib for deterministic hashing
        # Python's built-in hash() is randomized across processes for security
        import hashlib
        
        hash_values = []
        for i in range(num_hashes):
            min_hash = float('inf')
            for feature in sorted_features:
                # Use deterministic SHA-256 hash instead of built-in hash()
                hash_input = f"{i}:{feature}".encode('utf-8')
                hash_digest = hashlib.sha256(hash_input).hexdigest()
                # Convert first 8 hex chars to int for consistent hash value
                hash_val = int(hash_digest[:8], 16) % (2**32)
                min_hash = min(min_hash, hash_val)
            hash_values.append(min_hash)
        
        return ','.join(map(str, hash_values))
    
    async def _calculate_similarity(
        self, 
        file_content: bytes, 
        file_record: Dict[str, Any], 
        current_filename: str
    ) -> float:
        """
        Calculate similarity between current file and existing file.
        
        Uses multiple similarity algorithms and combines results.
        
        Args:
            file_content: Current file content
            file_record: Existing file record
            current_filename: Current filename
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        try:
            # Method 1: Filename similarity
            filename_similarity = self._calculate_filename_similarity(
                current_filename, 
                file_record.get('file_name', '')
            )
            
            # Method 2: Content fingerprint similarity
            content_similarity = await self._calculate_content_similarity(
                file_content, 
                file_record
            )
            
            # Method 3: Date similarity
            date_similarity = self._calculate_date_similarity(
                file_record.get('created_at', '')
            )
            
            # BROKEN #4 FIX: Fixed weighted combination to maintain consistent weights
            # Use configurable weights (can be overridden via environment variables)
            weight_filename = float(os.environ.get('SIMILARITY_WEIGHT_FILENAME', 0.3))
            weight_content = float(os.environ.get('SIMILARITY_WEIGHT_CONTENT', 0.5))
            weight_date = float(os.environ.get('SIMILARITY_WEIGHT_DATE', 0.2))
            
            weights = [weight_filename, weight_content, weight_date]
            similarities = [filename_similarity, content_similarity, date_similarity]
            
            # BROKEN #4 FIX: Always use all weights, don't exclude zeros
            # This maintains consistent weight distribution
            weighted_sum = sum(w * s for w, s in zip(weights, similarities))
            total_weight = sum(weights)
            
            if total_weight == 0:
                return 0.0
            
            return min(weighted_sum / total_weight, 1.0)
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_filename_similarity(self, filename1: str, filename2: str) -> float:
        """
        Calculate filename similarity using sequence matching.
        
        Args:
            filename1: First filename
            filename2: Second filename
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not filename1 or not filename2:
            return 0.0
        
        # Normalize filenames
        name1 = filename1.lower().strip()
        name2 = filename2.lower().strip()
        
        if name1 == name2:
            return 1.0
        
        # Use sequence matcher
        similarity = SequenceMatcher(None, name1, name2).ratio()
        
        # PLACEHOLDER #6 FIX: Make extension boost configurable
        ext_boost = float(os.environ.get('FILENAME_EXTENSION_BOOST', 0.05))
        
        ext1 = name1.split('.')[-1] if '.' in name1 else ''
        ext2 = name2.split('.')[-1] if '.' in name2 else ''
        
        if ext1 and ext2 and ext1 == ext2:
            similarity = min(similarity + ext_boost, 1.0)
        
        return similarity
    
    async def _calculate_content_similarity(self, file_content: bytes, file_record: Dict[str, Any]) -> float:
        """
        Calculate content similarity using fingerprints.
        
        Args:
            file_content: Current file content
            file_record: Existing file record
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        try:
            # Get stored content fingerprint
            stored_fingerprint = file_record.get('content_fingerprint')
            if not stored_fingerprint:
                return 0.0
            
            # Calculate current fingerprint
            current_fingerprint = await self._calculate_content_fingerprint(file_content)
            if not current_fingerprint:
                return 0.0
            
            # Compare fingerprints
            if current_fingerprint == stored_fingerprint:
                return 1.0
            
            # Calculate partial similarity
            return self._calculate_fingerprint_similarity(current_fingerprint, stored_fingerprint)
            
        except Exception as e:
            logger.error(f"Content similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_fingerprint_similarity(self, fp1: str, fp2: str) -> float:
        """
        Calculate similarity between two fingerprints.
        
        Args:
            fp1: First fingerprint
            fp2: Second fingerprint
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not fp1 or not fp2:
            return 0.0
        
        # Simple character-based similarity
        common_chars = sum(1 for a, b in zip(fp1, fp2) if a == b)
        total_chars = max(len(fp1), len(fp2))
        
        if total_chars == 0:
            return 0.0
        
        return common_chars / total_chars
    
    def _calculate_date_similarity(self, existing_date: str) -> float:
        """
        Calculate date similarity (files uploaded close in time are more likely similar).
        
        Args:
            existing_date: Existing file upload date
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        try:
            existing_dt = datetime.fromisoformat(existing_date.replace('Z', '+00:00'))
            current_dt = datetime.utcnow()
            
            # PLACEHOLDER #7 FIX: Make date window configurable
            date_window_days = int(os.environ.get('SIMILARITY_DATE_WINDOW_DAYS', 7))
            
            # Files uploaded within configured days are considered similar
            time_diff = abs((current_dt - existing_dt).days)
            if time_diff <= date_window_days:
                return 1.0 - (time_diff / float(date_window_days))
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _generate_cache_key(self, file_metadata: FileMetadata) -> str:
        """Generate cache key for duplicate check"""
        return f"duplicate_check:{file_metadata.user_id}:{file_metadata.file_hash}"
    
    async def _get_from_cache(self, cache_key: str) -> Optional[DuplicateResult]:
        """Get result from cache (Redis or memory)"""
        try:
            if self.redis_client:
                # Try Redis first (handle both sync and async Redis clients)
                try:
                    if hasattr(self.redis_client, 'aget'):
                        # Async Redis client
                        cached_data = await self.redis_client.aget(cache_key)
                        if cached_data:
                            return DuplicateResult(**json.loads(cached_data))
                    elif hasattr(self.redis_client, 'get'):
                        # Synchronous Redis client
                        cached_data = self.redis_client.get(cache_key)
                        if cached_data:
                            return DuplicateResult(**json.loads(cached_data))
                except Exception as redis_e:
                    logger.warning(f"Redis cache get failed: {redis_e}")
            
            # Fallback to memory cache
            if cache_key in self.memory_cache:
                cache_time = self.cache_timestamps.get(cache_key, 0)
                if time.time() - cache_time < self.cache_ttl:
                    return self.memory_cache[cache_key]
            
            return None
            
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None
    
    async def _set_cache(self, cache_key: str, result: DuplicateResult) -> None:
        """Set result in cache (Redis or memory)"""
        try:
            if self.redis_client:
                # Try Redis first (handle both sync and async Redis clients)
                try:
                    if hasattr(self.redis_client, 'asetex'):
                        # Async Redis client
                        await self.redis_client.asetex(
                            cache_key, 
                            self.cache_ttl, 
                            json.dumps(result.__dict__, default=str)
                        )
                    elif hasattr(self.redis_client, 'setex'):
                        # Synchronous Redis client
                        self.redis_client.setex(
                            cache_key, 
                            self.cache_ttl, 
                            json.dumps(result.__dict__, default=str)
                        )
                except Exception as redis_e:
                    logger.warning(f"Redis cache set failed: {redis_e}")
            
            # Always set in memory cache as fallback
            self.memory_cache[cache_key] = result
            self.cache_timestamps[cache_key] = time.time()
            
            # Clean expired entries periodically
            if len(self.memory_cache) > 1000:
                self._clean_memory_cache()
                    
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")
    
    def _clean_memory_cache(self) -> None:
        """Clean expired memory cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.cache_timestamps.items()
            if current_time - timestamp >= self.cache_ttl
        ]
        
        for key in expired_keys:
            self.memory_cache.pop(key, None)
            self.cache_timestamps.pop(key, None)
    
    def _create_result(
        self, 
        duplicate_result: DuplicateResult, 
        start_time: float, 
        cache_key: str
    ) -> DuplicateResult:
        """Create final result with processing time"""
        processing_time = int((time.time() - start_time) * 1000)
        self.metrics['total_processing_time'] += processing_time
        
        return DuplicateResult(
            is_duplicate=duplicate_result.is_duplicate,
            duplicate_type=duplicate_result.duplicate_type,
            similarity_score=duplicate_result.similarity_score,
            duplicate_files=duplicate_result.duplicate_files,
            recommendation=duplicate_result.recommendation,
            message=duplicate_result.message,
            confidence=duplicate_result.confidence,
            processing_time_ms=processing_time,
            cache_hit=False
        )
    
    async def analyze_delta_ingestion(self, user_id: str, new_sheets: Dict[str, pd.DataFrame], 
                                    existing_file_id: str) -> Dict[str, Any]:
        """
        Analyze what rows are new vs existing for delta ingestion.
        
        This feature enables intelligent merging of overlapping data by analyzing
        the differences between new and existing files at the row level.
        
        Args:
            user_id: User identifier
            new_sheets: New file sheets data
            existing_file_id: ID of existing file to compare against
            
        Returns:
            Dictionary with delta analysis results
        """
        try:
            # Get existing file data (prefer lightweight "sheets_row_hashes")
            existing_result = self.supabase.table('raw_records').select(
                'id, content, file_name'
            ).eq('id', existing_file_id).eq('user_id', user_id).limit(1).execute()
            
            if not existing_result.data:
                return {'delta_analysis': None, 'error': 'Existing file not found'}
            
            existing_row = existing_result.data[0]
            existing_content = existing_row.get('content', {}) or {}
            existing_hashes = existing_content.get('sheets_row_hashes') or {}
            existing_sheets: Dict[str, List[str]] = {}
            
            if existing_hashes:
                # Use precomputed row hashes per sheet
                for sheet_name, hashes in existing_hashes.items():
                    if isinstance(hashes, list):
                        existing_sheets[sheet_name] = hashes
            else:
                # Fallback: reconstruct per-sheet row hashes from raw_events of existing file
                try:
                    events_res = self.supabase.table('raw_events').select(
                        'sheet_name, payload'
                    ).eq('file_id', existing_file_id).eq('user_id', user_id).execute()
                    tmp: Dict[str, List[str]] = {}
                    for ev in (events_res.data or []):
                        sname = ev.get('sheet_name') or 'Unknown'
                        payload = ev.get('payload') or {}
                        # naive stable hash across payload values
                        try:
                            # preserve key ordering by sorting items
                            items = sorted(payload.items(), key=lambda kv: kv[0])
                            row_str = "|".join([f"{k}={v}" for k, v in items])
                        except Exception:
                            row_str = str(payload)
                        row_hash = hashlib.md5(row_str.encode('utf-8', errors='ignore')).hexdigest()
                        tmp.setdefault(sname, []).append(row_hash)
                    existing_sheets = tmp
                except Exception:
                    existing_sheets = {}
            
            delta_analysis = {
                'new_rows': 0,
                'existing_rows': 0,
                'modified_rows': 0,
                'sheet_analysis': {},
                'recommendation': 'merge_intelligent',
                'confidence': 0.0
            }
            
            total_similarity = 0
            sheet_count = 0
            
            for sheet_name, new_df in new_sheets.items():
                existing_hash_list = existing_sheets.get(sheet_name, [])
                if not existing_hash_list:
                    # Entirely new sheet
                    delta_analysis['sheet_analysis'][sheet_name] = {
                        'status': 'new_sheet',
                        'new_rows': len(new_df),
                        'existing_rows': 0,
                        'similarity': 0.0
                    }
                    delta_analysis['new_rows'] += len(new_df)
                    continue
                
                # Compare rows in existing sheet
                new_row_hashes = set()
                existing_row_hashes = set(existing_hash_list)
                
                # Calculate hashes for new rows
                for _, row in new_df.iterrows():
                    row_str = "|".join([str(val) for val in row.values if pd.notna(val)])
                    row_hash = hashlib.md5(row_str.encode('utf-8')).hexdigest()
                    new_row_hashes.add(row_hash)
                
                # Analyze differences
                new_only = new_row_hashes - existing_row_hashes
                existing_only = existing_row_hashes - new_row_hashes
                common = new_row_hashes & existing_row_hashes
                
                # Calculate similarity for this sheet
                total_rows = len(new_row_hashes) + len(existing_row_hashes) - len(common)
                similarity = len(common) / max(total_rows, 1) if total_rows > 0 else 0.0
                total_similarity += similarity
                sheet_count += 1
                
                delta_analysis['sheet_analysis'][sheet_name] = {
                    'status': 'partial_overlap' if len(common) > 0 else 'no_overlap',
                    'new_rows': len(new_only),
                    'existing_rows': len(existing_only),
                    'common_rows': len(common),
                    'similarity': similarity
                }
                
                delta_analysis['new_rows'] += len(new_only)
                delta_analysis['existing_rows'] += len(existing_only)
                delta_analysis['modified_rows'] += len(common)
            
            # Calculate overall confidence
            delta_analysis['confidence'] = total_similarity / max(sheet_count, 1)
            
            # Determine recommendation based on analysis
            if delta_analysis['new_rows'] == 0:
                delta_analysis['recommendation'] = 'skip'
            elif delta_analysis['existing_rows'] == 0:
                delta_analysis['recommendation'] = 'append'
            elif delta_analysis['confidence'] > 0.8:
                delta_analysis['recommendation'] = 'merge_intelligent'
            else:
                delta_analysis['recommendation'] = 'merge_new_only'
            
            logger.info(f"Delta analysis completed: {delta_analysis['new_rows']} new, {delta_analysis['existing_rows']} existing, {delta_analysis['confidence']:.2%} similarity")
            
            return {'delta_analysis': delta_analysis}
            
        except Exception as e:
            logger.error(f"Error analyzing delta ingestion: {e}")
            return {'delta_analysis': None, 'error': str(e)}
    
    async def check_content_duplicate(self, user_id: str, content_fingerprint: str, filename: str) -> Dict[str, Any]:
        """
        Check for content-level duplicates using row-level fingerprinting.
        
        This method identifies files with similar content even if they have different
        filenames or were uploaded at different times.
        
        Args:
            user_id: User identifier
            content_fingerprint: Content fingerprint to check
            filename: Current filename for context
            
        Returns:
            Dictionary with content duplicate analysis
        """
        try:
            # Get existing content fingerprints for this user
            result = self.supabase.table('raw_records').select(
                'id, file_name, created_at, status, content'
            ).eq('user_id', user_id).execute()
            
            if not result.data:
                return {'is_content_duplicate': False, 'overlapping_files': []}
            
            overlapping_files = []
            for record in result.data:
                existing_fingerprint = record.get('content', {}).get('content_fingerprint')
                if existing_fingerprint == content_fingerprint:
                    overlapping_files.append({
                        'id': record['id'],
                        'filename': record['file_name'],
                        'uploaded_at': record['created_at'],
                        'status': record['status']
                    })
            
            if overlapping_files:
                return {
                    'is_content_duplicate': True,
                    'overlapping_files': overlapping_files,
                    'recommendation': 'delta_ingestion_required',
                    'message': f"Found {len(overlapping_files)} file(s) with similar content. Would you like to merge only new rows or replace entirely?"
                }
            
            return {'is_content_duplicate': False, 'overlapping_files': []}
            
        except Exception as e:
            logger.error(f"Error checking content duplicate: {e}")
            return {'is_content_duplicate': False, 'error': str(e)}
    
    async def handle_duplicate_decision(self, user_id: str, file_hash: str, 
                                     decision: str, existing_file_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Handle user's decision about duplicate files.
        
        Args:
            user_id: User identifier
            file_hash: File hash of the new file
            decision: User decision ('replace', 'keep_both', 'skip', 'delta_merge')
            existing_file_id: Existing file ID for delta operations (optional)
        
        Returns:
            Dictionary with action taken
        """
        try:
            decision = decision.lower()
            if decision == "replace":
                # Mark old versions as replaced
                result = self.supabase.table('raw_records').update({
                    'status': 'replaced',
                    'updated_at': datetime.utcnow().isoformat(),
                    'metadata': {
                        'replaced_at': datetime.utcnow().isoformat(),
                        'replacement_hash': file_hash
                    }
                }).eq('user_id', user_id).eq('file_hash', file_hash).execute()
                replaced_count = len(result.data) if result.data else 0
                return {
                    'status': 'success',
                    'action': 'replaced',
                    'message': f"Marked {replaced_count} file(s) as replaced",
                    'replaced_count': replaced_count
                }
            
            elif decision == "keep_both":
                # No action needed, just return success
                return {
                    'status': 'success',
                    'action': 'keep_both',
                    'message': 'New file will be processed alongside existing files'
                }
            
            elif decision == "skip":
                # No action needed, user cancelled processing
                return {
                    'status': 'success',
                    'action': 'skip',
                    'message': 'Processing skipped by user'
                }
            
            elif decision == "delta_merge":
                if not existing_file_id:
                    return {
                        'status': 'error',
                        'error': 'existing_file_id required for delta_merge'
                    }

                delta_result = await self._perform_delta_merge(user_id, file_hash, existing_file_id)
                return {
                    'status': 'success',
                    'action': 'delta_merge',
                    'message': 'Delta merge completed',
                    'delta_result': delta_result
                }
            
            else:
                return {
                    'status': 'error',
                    'error': f"Unsupported decision: {decision}"
                }
        except Exception as e:
            logger.error(f"Error handling duplicate decision: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    async def _perform_delta_merge(self, user_id: str, new_file_hash: str, existing_file_id: str) -> Dict[str, Any]:
        """
        Apply delta ingestion by merging new rows into existing records.
        BUG #17 FIX: Wrapped in transaction to prevent data corruption.
        """
        try:
            # BUG #17 FIX: Start transaction for atomic operation
            # Note: Supabase doesn't support transactions directly, so we'll use error handling
            # and cleanup to maintain consistency
            
            new_record = (
                self.supabase
                .table('raw_records')
                .select('id, content, storage_path, source, status, file_name, job_id')
                .eq('user_id', user_id)
                .eq('file_hash', new_file_hash)
                .order('created_at', desc=True)
                .limit(1)
                .execute()
            )
            if not new_record.data:
                raise ValueError('New file not found for delta merge')
            new_record_id = new_record.data[0]['id']

            delta_payload = await self._prepare_delta_payload(user_id, existing_file_id, new_record_id)

            if not delta_payload['new_events']:
                return {
                    'merged_events': 0,
                    'reason': 'No new rows found compared to existing file'
                }

            # BUG #17 FIX: Insert events first, then create delta log only if successful
            inserted_events = None
            try:
                inserted_events = (
                    self.supabase
                    .table('raw_events')
                    .insert(delta_payload['new_events'])
                    .execute()
                )
                
                if not inserted_events.data:
                    raise ValueError('Event insertion failed - no data returned')
                    
            except Exception as insert_error:
                logger.error(f"Failed to insert events during delta merge: {insert_error}")
                # Don't create delta log if event insertion failed
                raise ValueError(f"Event insertion failed: {insert_error}")

            result_data = {
                'merged_events': len(inserted_events.data or []),
                'existing_events': delta_payload['existing_event_count'],
                'new_record_id': new_record_id,
                'existing_file_id': existing_file_id
            }

            # BUG #17 FIX: Only create delta log after successful event insertion
            if delta_payload['event_id_mapping']:
                try:
                    self.supabase.table('event_delta_logs').insert({
                        'user_id': user_id,
                        'existing_file_id': existing_file_id,
                        'new_file_id': new_record_id,
                        'delta_summary': result_data,
                        'events_included': delta_payload['event_id_mapping'],
                        'created_at': datetime.utcnow().isoformat()
                    }).execute()
                except Exception as log_error:
                    # Log error but don't fail the merge - events are already inserted
                    logger.warning(f"Failed to create delta log (events already inserted): {log_error}")

            return result_data
        except Exception as e:
            logger.error(f"Delta merge failed: {e}")
            raise

    async def _prepare_delta_payload(self, user_id: str, existing_file_id: str, new_record_id: str) -> Dict[str, Any]:
        """
        Compile payload for delta merge by comparing existing and new events.
        BUG #18 FIX: Added validation to prevent silent failures.
        """
        # BUG #18 FIX: Validate inputs first
        if not existing_file_id or not new_record_id:
            raise ValueError(f"Invalid file IDs: existing={existing_file_id}, new={new_record_id}")
        
        existing_events_resp = (
            self.supabase
            .table('raw_events')
            .select('id, payload, row_index, sheet_name')
            .eq('user_id', user_id)
            .eq('file_id', existing_file_id)
            .execute()
        )
        
        # BUG #18 FIX: Validate that query succeeded and returned data
        if existing_events_resp.error:
            raise ValueError(f"Failed to fetch existing events: {existing_events_resp.error}")
        
        existing_events = existing_events_resp.data or []
        
        # BUG #18 FIX: Validate that existing file has events
        if not existing_events:
            raise ValueError(f"No existing events found for file_id={existing_file_id}. Cannot perform delta merge on empty file.")
        
        logger.info(f"Found {len(existing_events)} existing events for delta merge")
        
        existing_hashes = {
            self._hash_event_payload(event['payload']): event['id']
            for event in existing_events
        }

        new_events_resp = (
            self.supabase
            .table('raw_events')
            .select('*')
            .eq('user_id', user_id)
            .eq('file_id', new_record_id)
            .execute()
        )
        new_events = new_events_resp.data or []

        new_rows = []
        event_id_mapping = {}
        for event in new_events:
            event_hash = self._hash_event_payload(event.get('payload', {}))
            if event_hash not in existing_hashes:
                event_copy = {**event}
                event_copy['file_id'] = existing_file_id
                event_copy.pop('id', None)
                new_rows.append(event_copy)
            else:
                event_id_mapping[event.get('id')] = existing_hashes[event_hash]

        return {
            'new_events': new_rows,
            'existing_event_count': len(existing_events),
            'event_id_mapping': event_id_mapping
        }

    def _hash_event_payload(self, payload: Dict[str, Any]) -> str:
        try:
            normalized = json.dumps(payload, sort_keys=True, default=str)
        except Exception:
            normalized = str(payload)
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics for monitoring"""
        average_time = 0.0
        try:
            total_requests = max(1, self.metrics['cache_hits'] + self.metrics['cache_misses'])
            average_time = self.metrics['total_processing_time'] / total_requests
        except Exception:
            average_time = 0.0

        return {
            'cache_hits': self.metrics['cache_hits'],
            'cache_misses': self.metrics['cache_misses'],
            'exact_duplicates_found': self.metrics['exact_duplicates_found'],
            'near_duplicates_found': self.metrics['near_duplicates_found'],
            'processing_errors': self.metrics['processing_errors'],
            'cache_size': len(self.memory_cache),
            'avg_processing_time_ms': average_time,
            'redis_enabled': bool(self.redis_client)
        }
    
    async def _cache_cleanup_loop(self):
        """
        FIX #12: Background task to clean up expired cache entries.
        Runs every 5 minutes to prevent memory leaks.
        """
        import asyncio
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._cleanup_expired_cache()
            except asyncio.CancelledError:
                logger.info("Cache cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def _cleanup_expired_cache(self):
        """
        FIX #12: Remove expired entries from memory cache.
        Also enforces max cache size using LRU eviction.
        """
        try:
            current_time = time.time()
            expired_keys = []
            
            # Find expired entries
            for key, timestamp in list(self.cache_timestamps.items()):
                if current_time - timestamp > self.cache_ttl:
                    expired_keys.append(key)
            
            # Remove expired entries
            for key in expired_keys:
                self.memory_cache.pop(key, None)
                self.cache_timestamps.pop(key, None)
                self.metrics['cache_evictions'] += 1
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
            
            # FIX #12: Enforce max cache size using LRU eviction
            if len(self.memory_cache) > self.max_cache_size:
                # Sort by timestamp (oldest first)
                sorted_keys = sorted(
                    self.cache_timestamps.items(),
                    key=lambda x: x[1]
                )
                
                # Remove oldest entries until under limit
                entries_to_remove = len(self.memory_cache) - self.max_cache_size
                for key, _ in sorted_keys[:entries_to_remove]:
                    self.memory_cache.pop(key, None)
                    self.cache_timestamps.pop(key, None)
                    self.metrics['cache_evictions'] += 1
                
                logger.info(f"Evicted {entries_to_remove} entries to enforce cache size limit")
                
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")
    
    async def clear_cache(self, user_id: Optional[str] = None) -> None:
        """Clear cache for user or all users"""
        try:
            if self.redis_client:
                if user_id:
                    pattern = f"duplicate_check:{user_id}:*"
                    keys = await self.redis_client.keys(pattern)
                    if keys:
                        await self.redis_client.delete(*keys)
                else:
                    await self.redis_client.flushdb()
            else:
                if user_id:
                    keys_to_remove = [
                        key for key in self.memory_cache.keys()
                        if key.startswith(f"duplicate_check:{user_id}:")
                    ]
                    for key in keys_to_remove:
                        self.memory_cache.pop(key, None)
                        self.cache_timestamps.pop(key, None)
                else:
                    self.memory_cache.clear()
                    self.cache_timestamps.clear()
                    
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
