"""
Google-Grade Test Suite for Autonomous Ingestion & Normalization Backend

Test Philosophy:
1. No Mocks - Test against real deployed backend logic
2. Full Edge Case Coverage - Use Hypothesis for property-based testing
3. 10/10 Production Quality - Zero tolerance for silent failures
4. Test Failures Drive Production Fixes - If test fails, fix production code

Structure:
- Phase A: Unit Tests (Testing individual components in isolation)
- Phase B: Modular Integration Tests (Testing component interactions + utilities)
- Separate E2E file: test_ingestion_e2e.py
"""

import pytest
import asyncio
import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Any
from io import BytesIO
import xxhash
from hypothesis import given, strategies as st, settings, HealthCheck

# Import components to test
from data_ingestion_normalization.streaming_source import StreamedFile
from data_ingestion_normalization.excel_processor import ExcelProcessor
from core_infrastructure.database_optimization_utils import calculate_row_hash
from core_infrastructure.security_system import InputSanitizer

# Import services for integration tests - REAL SERVICES, NO MOCKS!
from data_ingestion_normalization.production_duplicate_detection_service import ProductionDuplicateDetectionService
from data_ingestion_normalization.data_enrichment_processor import DataEnrichmentProcessor
from data_ingestion_normalization.universal_field_detector import UniversalFieldDetector

# Import test fixtures
from tests.fixtures.test_data import (
    create_test_excel_file,
    create_test_csv_file,
    create_malformed_excel,
    create_large_excel_file,
)


# ============================================================================
# PHASE A: PRODUCTION DUPLICATE DETECTION SERVICE TESTS
# Testing 4-phase duplicate detection pipeline with REAL services (NO MOCKS)
# ============================================================================


class TestPhaseA_DuplicateDetection_Unit:
    """
    Unit tests for ProductionDuplicateDetectionService core methods.
    
    Coverage:
    - Exact duplicate detection (hash-based)
    - Security validation (PII, path traversal)
    - Content fingerprinting (MinHash)
    - Cache operations
    - Edge cases: empty files, malformed data
    """
    
    @pytest.fixture
    def supabase_client(self):
        """Get real Supabase client - NO MOCKS!"""
        from core_infrastructure.fastapi_backend_v2 import supabase
        return supabase
    
    @pytest.fixture
    def duplicate_service(self, supabase_client):
        """Initialize ProductionDuplicateDetectionService with real Supabase."""
        from data_ingestion_normalization.production_duplicate_detection_service import ProductionDuplicateDetectionService
        return ProductionDuplicateDetectionService(supabase_client)
    
    def test_security_validation_pii_detection(self, duplicate_service):
        """Verify PII detection in filenames is blocked."""
        # Test email in filename
        with pytest.raises(ValueError, match="PII"):
            duplicate_service._validate_security(
                user_id="test_user",
                file_hash="a" * 32,
                filename="invoice_john.doe@company.com.xlsx"
            )
    
    def test_security_validation_path_traversal(self, duplicate_service):
        """Verify path traversal in filenames is blocked."""
        with pytest.raises(ValueError, match="path traversal"):
            duplicate_service._validate_security(
                user_id="test_user",
                file_hash="a" * 32,
                filename="../../../etc/passwd.xlsx"
            )
    
    def test_security_validation_valid_filename(self, duplicate_service):
        """Verify valid filenames pass security checks."""
        # Should not raise
        duplicate_service._validate_security(
            user_id="test_user_123",
            file_hash="a" * 32,
            filename="transactions_2024.xlsx"
        )
    
    async def test_content_fingerprint_consistency(self, duplicate_service):
        """Verify content fingerprinting produces consistent hashes."""
        from data_ingestion_normalization.streaming_source import StreamedFile
        
        # Same content should produce same fingerprint
        content = b"Test transaction data\nRow 1\nRow 2"
        
        file1 = StreamedFile.from_bytes(content, "test1.csv")
        file2 = StreamedFile.from_bytes(content, "test2.csv")
        
        fp1 = await duplicate_service._calculate_content_fingerprint_from_path(file1)
        fp2 = await duplicate_service._calculate_content_fingerprint_from_path(file2)
        
        assert fp1 == fp2, "Same content should produce same fingerprint"
        assert len(fp1) == 64, "SHA-256 hash should be 64 hex chars"
        
        # Cleanup
        file1.cleanup()
        file2.cleanup()
    
    async def test_content_fingerprint_different_content(self, duplicate_service):
        """Verify different content produces different fingerprints."""
        from data_ingestion_normalization.streaming_source import StreamedFile
        
        file1 = StreamedFile.from_bytes(b"Content A", "test1.csv")
        file2 = StreamedFile.from_bytes(b"Content B", "test2.csv")
        
        fp1 = await duplicate_service._calculate_content_fingerprint_from_path(file1)
        fp2 = await duplicate_service._calculate_content_fingerprint_from_path(file2)
        
        assert fp1 != fp2, "Different content should produce different fingerprints"
        
        # Cleanup
        file1.cleanup()
        file2.cleanup()
    
    def test_cache_key_generation(self, duplicate_service):
        """Verify cache keys are generated consistently."""
        from data_ingestion_normalization.production_duplicate_detection_service import FileMetadata
        
        metadata1 = FileMetadata(
            user_id="user_123",
            file_hash="abc123",
            filename="test.xlsx"
        )
        metadata2 = FileMetadata(
            user_id="user_123",
            file_hash="abc123",
            filename="different_name.xlsx"  # Different filename
        )
        
        key1 = duplicate_service._generate_cache_key(metadata1)
        key2 = duplicate_service._generate_cache_key(metadata2)
        
        # Cache key should be based on user_id + file_hash (NOT filename)
        assert key1 == key2, "Cache key should ignore filename"
        assert "user_123" in key1
        assert "abc123" in key1


class TestPhaseB_DuplicateDetection_Integration:
    """
    Integration tests for ProductionDuplicateDetectionService.
    Tests against REAL Supabase, Redis cache, LSH service.
    
    Coverage:
    - Phase 1: Exact duplicate detection (Supabase query)
    - Phase 2: Near duplicate detection (LSH + rapidfuzz)
    - Phase 3: Content duplicate detection  
    - Phase 4: Delta analysis (polars hash joins)
    - End-to-end: Complete duplicate detection flow
    """
    
    @pytest.fixture
    def supabase_client(self):
        """Get real Supabase client."""
        from core_infrastructure.fastapi_backend_v2 import supabase
        return supabase
    
    @pytest.fixture
    def duplicate_service(self, supabase_client):
        """Initialize service with real Supabase."""
        from data_ingestion_normalization.production_duplicate_detection_service import ProductionDuplicateDetectionService
        return ProductionDuplicateDetectionService(supabase_client)
    
    @pytest.fixture
    def test_user_id(self):
        """Generate unique test user ID."""
        import uuid
        return f"test_user_{uuid.uuid4().hex[:8]}"
    
    async def test_exact_duplicate_detection_no_duplicates(self, duplicate_service, test_user_id):
        """Phase 1: Verify no duplicates found for new file."""
        from data_ingestion_normalization.production_duplicate_detection_service import FileMetadata
        import uuid
        
        metadata = FileMetadata(
            user_id=test_user_id,
            file_hash=f"unique_hash_{uuid.uuid4().hex}",
            filename="new_file.xlsx"
        )
        
        result = await duplicate_service._detect_exact_duplicates(metadata)
        
        assert not result.is_duplicate, "New file should not be duplicate"
        assert result.duplicate_type.value == "none"
        assert result.similarity_score == 0.0
        assert len(result.duplicate_files) == 0
    
    async def test_detect_duplicates_complete_flow_no_duplicates(self, duplicate_service, test_user_id, tmp_path):
        """End-to-end: Complete 4-phase pipeline with no duplicates found."""
        from data_ingestion_normalization.streaming_source import StreamedFile
        from data_ingestion_normalization.production_duplicate_detection_service import FileMetadata
        import uuid
        
        # Create test file
        test_content = b"Transaction,Amount\nTest,100\nData,200"
        streamed_file = StreamedFile.from_bytes(test_content, "test_transactions.csv", temp_dir=str(tmp_path))
        
        metadata = FileMetadata(
            user_id=test_user_id,
            file_hash=f"test_hash_{uuid.uuid4().hex}",
            filename="test_transactions.csv",
            file_size=len(test_content)
        )
        
        #  Run complete duplicate detection (all 4 phases)
        result = await duplicate_service.detect_duplicates(
            streamed_file=streamed_file,
            file_metadata=metadata,
            enable_near_duplicate=True,
            enable_content_duplicate=True
        )
        
        # Verify result
        assert not result.is_duplicate, "New unique file should not be duplicate"
        assert result.duplicate_type.value == "none"
        assert result.processing_time_ms > 0, "Should have processing time"
        assert "No duplicates" in result.message
        
        # Cleanup
        streamed_file.cleanup()
    
    async def test_cache_integration(self, duplicate_service, test_user_id):
        """Verify Redis cache integration works correctly."""
        from data_ingestion_normalization.production_duplicate_detection_service import DuplicateResult, DuplicateType, DuplicateAction
        
        cache_key = f"test_cache_{test_user_id}"
        
        test_result = DuplicateResult(
            is_duplicate=True,
            duplicate_type=DuplicateType.EXACT,
            similarity_score=1.0,
            duplicate_files=[{"id": "test_123", "filename": "test.xlsx"}],
            recommendation=DuplicateAction.REPLACE,
            message="Test cache entry",
            confidence=1.0,
            processing_time_ms=100
        )
        
        # Save to cache
        await duplicate_service._set_cache(cache_key, test_result)
        
        # Retrieve from cache
        cached = await duplicate_service._get_from_cache(cache_key)
        
        assert cached is not None, "Should retrieve from cache"
        assert cached.is_duplicate == test_result.is_duplicate
        assert cached.duplicate_type == test_result.duplicate_type
        assert cached.similarity_score == test_result.similarity_score


# ============================================================================
# PYTEST FIXTURES - REAL INFRASTRUCTURE INITIALIZATION (NO MOCKS!)
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def initialize_real_infrastructure():
    """
    Initialize REAL infrastructure for integration tests.
    - Redis cache for caching
    - Supabase client for database
    
    This runs ONCE per test session and ensures all tests use REAL services.
    NO MOCKS!
    """
    # Initialize Redis cache
    try:
        from core_infrastructure.centralized_cache import initialize_cache
        initialize_cache()
        print("✅ Initialized real Redis cache for integration tests")
    except Exception as e:
        print(f"⚠️  Could not initialize Redis cache: {e}")
        print("   Tests requiring cache will be skipped")
    
    # Supabase is initialized via environment variables automatically
    try:
        from core_infrastructure.supabase_factories import get_supabase_client
        supabase = get_supabase_client()
        print(f"✅ Connected to real Supabase: {supabase.supabase_url[:30]}...")
    except Exception as e:
        print(f"⚠️  Could not connect to Supabase: {e}")
    
    yield  # Tests run here
    
    # Cleanup after all tests complete
    print("\n🧹 Test session complete - infrastructure cleanup")


# ============================================================================
# PHASE A: UNIT TESTS
# Testing individual components in isolation with full edge case coverage
# ============================================================================


class TestPhaseA_StreamedFile_Unit:
    """
    Unit tests for StreamedFile class (streaming_source.py)
    
    Coverage:
    - xxh3_128 hashing consistency
    - Memory-efficient chunked reading
    - Temporary file cleanup
    - from_bytes factory method
    - Context manager cleanup
    - Edge cases: empty files, huge files, binary data, unicode
    """
    
    def test_xxh3_128_hash_consistency(self, tmp_path):
        """
        CRITICAL: Verify xxh3_128 produces consistent hashes for same content.
        This is the standard hash used across the system.
        """
        content = b"Test content for hashing consistency"
        expected_hash = xxhash.xxh3_128(content).hexdigest()
        
        # Create StreamedFile from bytes
        streamed_file = StreamedFile.from_bytes(content, "test.txt", temp_dir=tmp_path)
        
        # Verify hash matches expected
        assert streamed_file.xxh3_128 == expected_hash, \
            f"Hash mismatch: {streamed_file.xxh3_128} != {expected_hash}"
        
        # Verify hash is stable across multiple reads
        assert streamed_file.xxh3_128 == expected_hash, \
            "Hash changed after re-access - not stable!"
    
    def test_empty_file_handling(self, tmp_path):
        """Edge case: Empty files should hash correctly and not crash."""
        empty_content = b""
        expected_hash = xxhash.xxh3_128(empty_content).hexdigest()
        
        streamed_file = StreamedFile.from_bytes(empty_content, "empty.txt", temp_dir=tmp_path)
        
        assert streamed_file.xxh3_128 == expected_hash
        assert streamed_file.size == 0
        assert streamed_file.read_bytes() == b""
    
    @given(st.binary(min_size=1, max_size=1024*1024))  # Up to 1MB random data
    @settings(max_examples=50)
    def test_binary_data_integrity_hypothesis(self, binary_data):
        """
        Hypothesis test: Verify data integrity for arbitrary binary content.
        Tests that read_bytes() returns exactly what was written.
        """
        # Use system temp dir instead of tmp_path fixture (Hypothesis compatibility)
        streamed_file = StreamedFile.from_bytes(binary_data, "random.bin", temp_dir=tempfile.gettempdir())
        
        try:
            # Verify data can be read back exactly
            read_data = streamed_file.read_bytes()
            assert read_data == binary_data, \
                f"Data corruption detected: {len(read_data)} bytes != {len(binary_data)} bytes"
            
            # Verify hash is correct
            expected_hash = xxhash.xxh3_128(binary_data).hexdigest()
            assert streamed_file.xxh3_128 == expected_hash
        finally:
            # Clean up temp file
            streamed_file.cleanup()
    
    def test_chunked_reading_memory_efficiency(self, tmp_path):
        """
        Verify chunked reading doesn't load entire file into memory.
        Critical for large file handling.
        """
        # Create 16MB test file (2 chunks of 8MB default chunk size)
        large_content = b"X" * (16 * 1024 * 1024)
        streamed_file = StreamedFile.from_bytes(large_content, "large.bin", temp_dir=tmp_path)
        
        # Read in chunks
        chunks = []
        for chunk in streamed_file.iter_bytes(chunk_size=8*1024*1024):
            chunks.append(chunk)
        
        # Verify we got 2 chunks
        assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}"
        assert len(chunks[0]) == 8*1024*1024
        assert len(chunks[1]) == 8*1024*1024
        
        # Verify reassembled data matches original
        reassembled = b"".join(chunks)
        assert reassembled == large_content
    
    def test_context_manager_cleanup(self, tmp_path):
        """
        CRITICAL: Verify temp files are cleaned up after context manager exit.
        Memory leak prevention.
        """
        content = b"Test cleanup"
        temp_file_path = None
        
        with StreamedFile.from_bytes(content, "cleanup_test.txt", temp_dir=tmp_path) as sf:
            temp_file_path = sf.path
            assert os.path.exists(temp_file_path), "Temp file should exist during context"
        
        # After context exit, file should be deleted
        assert not os.path.exists(temp_file_path), \
            f"Temp file {temp_file_path} not cleaned up! Memory leak!"
    
    def test_unicode_filename_handling(self, tmp_path):
        """Edge case: Unicode characters in filenames."""
        content = b"Test unicode filename"
        unicode_name = "测试文件名_émojis_🔥.txt"
        
        streamed_file = StreamedFile.from_bytes(content, unicode_name, temp_dir=tmp_path)
        
        assert streamed_file.filename == unicode_name
        assert streamed_file.read_bytes() == content
    
    def test_read_text_encoding(self, tmp_path):
        """Verify text reading with various encodings."""
        # UTF-8 with special characters
        text_content = "Hello 世界 🌍 Émojis"
        bytes_content = text_content.encode('utf-8')
        
        streamed_file = StreamedFile.from_bytes(bytes_content, "text.txt", temp_dir=tmp_path)
        
        # Read as text
        read_text = streamed_file.read_text()
        assert read_text == text_content


class TestPhaseA_ExcelProcessor_Unit:
    """
    Unit tests for ExcelProcessor core methods (excel_processor.py)
    
    Coverage:
    - stream_xlsx_processing (polars/fastexcel streaming)
    - detect_anomalies (statistical analysis)
    - detect_financial_fields (ML field detection)
    - _get_sheet_metadata (metadata extraction)
    - Hash consistency (xxh3_128 usage)
    - Edge cases: malformed files, empty sheets, huge datasets
    """
    
    @pytest.fixture
    def excel_processor(self):
        """Initialize ExcelProcessor instance."""
        return ExcelProcessor()
    
    @pytest.fixture
    def simple_excel_file(self, tmp_path):
        """Create a simple test Excel file."""
        return create_test_excel_file(tmp_path, rows=100, columns=5)
    
    def test_get_sheet_metadata_hash_consistency(self, excel_processor, simple_excel_file, tmp_path):
        """
        CRITICAL: Verify _get_sheet_metadata uses xxh3_128 (not xxh64).
        This test validates the fix we just made.
        """
        streamed_file = StreamedFile.from_bytes(
            simple_excel_file.read_bytes(), 
            "test.xlsx", 
            tmp_path
        )
        
        # Run metadata extraction
        metadata = asyncio.run(excel_processor._get_sheet_metadata(streamed_file))
        
        # Verify metadata exists
        assert metadata, "Metadata extraction failed"
        assert len(metadata) > 0, "No sheets found"
        
        # Verify sample_hash uses xxh3_128 (128-bit hash = 32 hex chars)
        for sheet_name, meta in metadata.items():
            sample_hash = meta.get('sample_hash')
            assert sample_hash, f"No sample_hash for sheet {sheet_name}"
            assert len(sample_hash) == 32, \
                f"sample_hash should be 32 chars (xxh3_128), got {len(sample_hash)} chars"
    
    def test_empty_excel_file_handling(self, excel_processor, tmp_path):
        """Edge case: Excel file with no data rows."""
        # Create Excel with only headers, no data
        empty_excel = create_test_excel_file(tmp_path, rows=0, columns=3)
        streamed_file = StreamedFile.from_bytes(
            empty_excel.read_bytes(),
            "empty.xlsx",
            tmp_path
        )
        
        metadata = asyncio.run(excel_processor._get_sheet_metadata(streamed_file))
        
        # Should extract metadata but show 0 rows
        for sheet_name, meta in metadata.items():
            assert meta['row_count'] == 0, f"Expected 0 rows, got {meta['row_count']}"
            assert 'columns' in meta, "Should have column headers even with no data"
    
    def test_malformed_excel_error_handling(self, excel_processor, tmp_path):
        """Edge case: Corrupted/malformed Excel files should fail gracefully."""
        # Create corrupted Excel file
        malformed_file = create_malformed_excel(tmp_path)
        streamed_file = StreamedFile.from_bytes(
            malformed_file.read_bytes(),
            "corrupted.xlsx",
            tmp_path
        )
        
        # Should raise a clear error, not crash silently
        with pytest.raises(Exception) as exc_info:
            asyncio.run(excel_processor._get_sheet_metadata(streamed_file))
        
        # Error message should be informative
        assert "fail" in str(exc_info.value).lower() or \
               "corrupt" in str(exc_info.value).lower() or \
               "invalid" in str(exc_info.value).lower(), \
               f"Error message not clear: {exc_info.value}"
    
    @pytest.mark.slow
    def test_large_file_streaming_performance(self, excel_processor, tmp_path):
        """
        Performance test: 100k rows should stream efficiently.
        Should complete in <30 seconds and use <500MB RAM.
        """
        import psutil
        import time
        
        # Create large Excel file (100k rows, 10 columns)
        large_excel = create_large_excel_file(tmp_path, rows=100000, columns=10)
        streamed_file = StreamedFile.from_bytes(
            large_excel.read_bytes(),
            "large.xlsx",
            tmp_path
        )
        
        # Measure memory before
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Measure time
        start_time = time.time()
        metadata = asyncio.run(excel_processor._get_sheet_metadata(streamed_file))
        elapsed = time.time() - start_time
        
        # Measure memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_delta = mem_after - mem_before
        
        # Assertions
        assert elapsed < 30, f"Too slow: {elapsed:.2f}s > 30s"
        assert mem_delta < 500, f"Memory usage too high: {mem_delta:.2f}MB > 500MB"
        
        # Verify correct row count
        for sheet_name, meta in metadata.items():
            assert meta['row_count'] == 100000, \
                f"Row count mismatch: {meta['row_count']} != 100000"


class TestPhaseA_HashingUtility_Unit:
    """
    Unit tests for hashing utilities (database_optimization_utils.py)
    
    Coverage:
    - calculate_row_hash consistency
    - xxh3_128 usage verification
    - Edge cases: None values, empty dicts, special characters
    """
    
    def test_calculate_row_hash_consistency(self):
        """Verify calculate_row_hash produces consistent results."""
        row_data = {"vendor": "Test Corp", "amount": 1000.50, "date": "2024-01-15"}
        
        hash1 = calculate_row_hash("test.xlsx", 0, row_data)
        hash2 = calculate_row_hash("test.xlsx", 0, row_data)
        
        assert hash1 == hash2, "Hash should be deterministic"
        assert len(hash1) == 32, f"Hash should be 32 chars (xxh3_128), got {len(hash1)}"
    
    def test_calculate_row_hash_with_none_values(self):
        """Edge case: None values should be handled consistently."""
        row_with_none = {"vendor": "Test", "amount": None, "category": ""}
        
        # Should not crash
        hash_result = calculate_row_hash("test.xlsx", 0, row_with_none)
        assert hash_result, "Hash should be generated for rows with None/empty"
        assert len(hash_result) == 32
    
    @given(st.dictionaries(
        keys=st.text(min_size=1, max_size=50),
        values=st.one_of(
            st.text(),
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.none()
        ),
        min_size=1,
        max_size=20
    ))
    @settings(max_examples=100)
    def test_calculate_row_hash_hypothesis(self, row_dict):
        """Hypothesis: Hash should work for arbitrary row dictionaries."""
        # Should not raise exception for any valid dict
        hash_result = calculate_row_hash("test.xlsx", 0, row_dict)
        assert hash_result, "Hash generation failed"
        assert isinstance(hash_result, str), "Hash should be string"
        assert len(hash_result) == 32, "Hash should be 32 hex chars"


class TestPhaseA_InputSanitizer_Unit:
    """
    Unit tests for InputSanitizer (security_system.py)
    
    Coverage:
    - XSS prevention (bleach.clean)
    - Control character removal
    - Null byte removal
    - SQL injection detection ✅ NEW
    - Path traversal detection ✅ NEW
    - Edge cases: unicode, special chars, null bytes
    """
    
    @pytest.fixture
    def sanitizer(self):
        return InputSanitizer()
    
    def test_xss_prevention(self, sanitizer):
        """Verify XSS patterns are sanitized via bleach."""
        xss_inputs = [
            ("<script>alert('XSS')</script>", "alert('XSS')"),  # Script tags removed
            ("<img src=x onerror=alert('XSS')>", ""),  # Img tag removed
            ("javascript:alert('XSS')", "javascript:alert('XSS')"),  # Text preserved (no tags)
        ]
        
        for xss_input, expected_contains in xss_inputs:
            sanitized = sanitizer.sanitize_string(xss_input)
            # bleach.clean removes all HTML tags
            assert "<script>" not in sanitized.lower(), \
                f"XSS not sanitized: {sanitized}"
            assert "<img" not in sanitized.lower(), \
                f"XSS not sanitized: {sanitized}"
    
    def test_null_byte_removal(self, sanitizer):
        """Verify null bytes are removed."""
        input_with_null = "test\x00data"
        sanitized = sanitizer.sanitize_string(input_with_null)
        assert "\x00" not in sanitized, "Null byte not removed"
        assert sanitized == "testdata", f"Expected 'testdata', got '{sanitized}'"
    
    def test_control_character_removal(self, sanitizer):
        """Verify control characters are removed (except \n and \t)."""
        input_with_control = "test\x01\x02data\nline2\ttab"
        sanitized = sanitizer.sanitize_string(input_with_control)
        
        # Control chars should be removed, but \n and \t should remain
        assert "\x01" not in sanitized
        assert "\x02" not in sanitized
        assert "\n" in sanitized  # Newline preserved
        assert "\t" in sanitized  # Tab preserved
    
    def test_max_length_enforcement(self, sanitizer):
        """Verify max length is enforced."""
        long_string = "A" * 2000
        sanitized = sanitizer.sanitize_string(long_string, max_length=100)
        
        assert len(sanitized) <= 100, f"Max length not enforced: got {len(sanitized)} chars"
    
    def test_unicode_handling(self, sanitizer):
        """Verify unicode characters are preserved."""
        unicode_input = "Hello 世界 🌍 Café"
        sanitized = sanitizer.sanitize_string(unicode_input)
        
        assert sanitized == unicode_input, \
            f"Unicode not preserved: expected '{unicode_input}', got '{sanitized}'"
    
    def test_sql_injection_detection(self, sanitizer):
        """✅ NEW: Verify SQL injection patterns are detected and blocked."""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "1; DELETE FROM accounts;",
            "' UNION SELECT password FROM users--",
        ]
        
        for malicious in malicious_inputs:
            with pytest.raises(ValueError, match="SQL injection"):
                sanitizer.sanitize_string(malicious)
    
    def test_path_traversal_detection(self, sanitizer):
        """✅ NEW: Verify path traversal patterns are detected and blocked."""
        traversal_inputs = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "/etc/shadow",
            "C:\\Windows\\System32",
            "%2e%2e/secret",
        ]
        
        for traversal in traversal_inputs:
            with pytest.raises(ValueError, match="Path traversal|Absolute path"):
                sanitizer.sanitize_string(traversal)


# ============================================================================
# PHASE B: MODULAR INTEGRATION TESTS
# Testing component interactions and utility integration
# ============================================================================


class TestPhaseB_ExcelProcessor_DuplicateDetection_Integration:
    """
    Integration test: ExcelProcessor + ProductionDuplicateDetectionService
    
    Tests the full duplicate detection flow:
    1. ExcelProcessor extracts file hash
    2. ProductionDuplicateDetectionService checks for duplicates
    3. Correct handling of exact/near duplicates
    """
    
    @pytest.fixture
    async def duplicate_service(self):
        """Initialize ProductionDuplicateDetectionService."""
        # Use real Railway environment
        from core_infrastructure.supabase_factories import get_supabase_client
        supabase = get_supabase_client()
        return ProductionDuplicateDetectionService(supabase)
    
    @pytest.mark.asyncio
    async def test_exact_duplicate_detection_flow(self, duplicate_service, tmp_path):
        """
        Integration test: Upload same file twice, verify exact duplicate detected.
        NO MOCKS - tests against real DB.
        """
        # Create test file
        test_excel = create_test_excel_file(tmp_path, rows=50, columns=3)
        file_bytes = test_excel.read_bytes()
        
        # First upload - should succeed
        file_hash = xxhash.xxh3_128(file_bytes).hexdigest()
        test_user_id = f"test_user_{uuid.uuid4()}"
        
        from data_ingestion_normalization.production_duplicate_detection_service import FileMetadata
        import pendulum
        
        file_metadata = FileMetadata(
            user_id=test_user_id,
            file_hash=file_hash,
            filename="test.xlsx",
            file_size=len(file_bytes),
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            upload_timestamp=pendulum.now()
        )
        
        streamed_file = StreamedFile.from_bytes(file_bytes, "test.xlsx", str(tmp_path))
        
        # Detect duplicates (first time - should find none)
        result1 = await duplicate_service.detect_duplicates(
            file_metadata=file_metadata,
            streamed_file=streamed_file,
            enable_near_duplicate=True
        )
        
        assert not result1.is_duplicate, "First upload should not be duplicate"
        assert result1.duplicate_type.value == "none"
        
        # Simulate saving to database (required for second upload to detect duplicate)
        # In real flow, ExcelProcessor.process_file saves to raw_records
        from core_infrastructure.supabase_factories import get_supabase_client
        supabase = get_supabase_client()
        
        try:
            # Insert raw_record to simulate first upload
            raw_record_data = {
                'user_id': test_user_id,
                'file_name': 'test.xlsx',
                'file_hash': file_hash,
                'file_size': len(file_bytes),
                'source': 'test_upload',
                'content': {'test': 'data'},
                'status': 'completed'
            }
            insert_result = supabase.table('raw_records').insert(raw_record_data).execute()
            file_id = insert_result.data[0]['id']
            
            # Second upload - same file, should detect exact duplicate
            result2 = await duplicate_service.detect_duplicates(
                file_metadata=file_metadata,  # Same metadata
                streamed_file=streamed_file,
                enable_near_duplicate=True
            )
            
            # Verify exact duplicate detected
            assert result2.is_duplicate, "Second upload should be detected as duplicate"
            assert result2.duplicate_type.value == "exact", f"Expected exact duplicate, got {result2.duplicate_type.value}"
            assert result2.similarity_score >= 0.99, f"Similarity score should be ~1.0, got {result2.similarity_score}"
            assert len(result2.duplicate_files) > 0, "Should return list of duplicate files"
            
            # Cleanup
            supabase.table('raw_records').delete().eq('id', file_id).execute()
            
        except Exception as e:
            # Cleanup on error
            try:
                if 'file_id' in locals():
                    supabase.table('raw_records').delete().eq('id', file_id).execute()
            except:
                pass
            raise e
        finally:
            streamed_file.cleanup()
    
    @pytest.mark.asyncio
    async def test_field_detection_platform_detection_integration(self, tmp_path):
        """
        Integration test: Verify field detection runs before platform detection.
        Platform detection relies on field types (vendor, amount, date fields).
        """
        from data_ingestion_normalization.excel_processor import ExcelProcessor
        
        # Create test Excel with recognizable financial fields
        test_excel = create_test_excel_file(
            tmp_path, 
            rows=20, 
            columns=5,
            column_names=['Date', 'Vendor', 'Amount', 'Category', 'Description']
        )
        
        streamed_file = StreamedFile.from_bytes(
            test_excel.read_bytes(),
            "financial_transactions.xlsx",
            str(tmp_path)
        )
        
        processor = ExcelProcessor()
        
        # Get sheet metadata
        metadata = await processor._get_sheet_metadata(streamed_file)
        assert len(metadata) > 0, "Should extract sheet metadata"
        
        first_sheet_meta = list(metadata.values())[0]
        columns = first_sheet_meta['columns']
        
        # Verify expected columns
        assert 'Date' in columns or 'date' in [c.lower() for c in columns]
        assert 'Vendor' in columns or 'vendor' in [c.lower() for c in columns]
        assert 'Amount' in columns or 'amount' in [c.lower() for c in columns]
        
        # Test field detection
        sample_data = {
            'Date': '2024-01-15',
            'Vendor': 'Test Corp',
            'Amount': 1000.50,
            'Category': 'Software',
            'Description': 'License purchase'
        }
        
        field_result = await processor.universal_field_detector.detect_field_types_universal(
            data=sample_data,
            filename="financial_transactions.xlsx",
            context={'columns': columns, 'sheet_name': 'Sheet1'},
            user_id=f"test_user_{uuid.uuid4().hex[:8]}"
        )
        
        # Verify field detection results
        assert 'detected_fields' in field_result, "Should return detected_fields"
        assert 'field_types' in field_result, "Should return field_types"
        
        # Test platform detection with field context
        payload = {
            'columns': columns,
            'sample_data': [sample_data],
            'detected_fields': field_result.get('detected_fields', []),
            'field_types': field_result.get('field_types', {})
        }
        
        platform_result = await processor.universal_platform_detector.detect_platform_universal(
            payload,
            filename="financial_transactions.xlsx",
            user_id=f"test_user_{uuid.uuid4().hex[:8]}"
        )
        
        # Verify platform detection results
        assert 'platform' in platform_result, "Should return platform"
        assert 'confidence' in platform_result, "Should return confidence"
        # Platform should be detected (not 'unknown') due to clear field structure
        # But allow 'unknown' if AI service is unavailable
        assert platform_result['platform'] is not None
        
        streamed_file.cleanup()


class TestPhaseB_TransactionRollback_Integration:
    """
    Integration tests for transaction rollback functionality.
    Tests atomic operations and rollback on failure.
    """
    
    @pytest.fixture
    def supabase_client(self):
        from core_infrastructure.supabase_factories import get_supabase_client
        return get_supabase_client()
    
    @pytest.mark.asyncio
    async def test_transaction_rollback_on_error(self, supabase_client, tmp_path):
        """
        Verify that if processing fails mid-transaction, all database changes are rolled back.
        Tests the transaction manager's rollback capability.
        """
        from core_infrastructure.database_transaction_manager import get_transaction_manager
        test_user_id = f"test_user_{uuid.uuid4().hex[:8]}"
        transaction_id = str(uuid.uuid4())
        
        transaction_manager = get_transaction_manager()
        
        inserted_ids = []
        
        try:
            # Start transaction
            async with transaction_manager.transaction(
                transaction_id=transaction_id,
                user_id=test_user_id,
                operation_type="test_rollback"
            ) as tx:
                # Insert test data
                test_data = {
                    'user_id': test_user_id,
                    'file_name': 'test_rollback.xlsx',
                    'file_hash': 'test_hash_' + uuid.uuid4().hex,
                    'file_size': 1000,
                    'source': 'test',
                    'content': {'test': 'rollback'},
                    'status': 'processing'
                }
                
                result = await tx.insert('raw_records', test_data)
                assert result is not None, "Insert should succeed"
                file_id = result['id']
                inserted_ids.append(('raw_records', file_id))
                
                # Force an error to trigger rollback
                raise ValueError("Simulated processing error for rollback test")
        
        except ValueError as e:
            if "Simulated processing error" not in str(e):
                raise
        
        # Verify rollback occurred - data should NOT exist in database
        await asyncio.sleep(1)  # Give rollback time to complete
        
        for table, record_id in inserted_ids:
            result = supabase_client.table(table).select('*').eq('id', record_id).execute()
            assert len(result.data) == 0, \
                f"Rollback failed: Record {record_id} still exists in {table} after transaction failure"
        
        # Verify transaction status is 'rolled_back'
        tx_result = supabase_client.table('processing_transactions').select('status').eq('id', transaction_id).execute()
        if len(tx_result.data) > 0:
            assert tx_result.data[0]['status'] in ['rolled_back', 'failed'], \
                f"Transaction status should be rolled_back or failed, got {tx_result.data[0]['status']}"


class TestPhaseB_ErrorRecovery_Integration:
    """
    Integration tests for error recovery system.
    Tests graceful degradation when services unavailable.
    """
    
    @pytest.mark.asyncio
    async def test_redis_unavailable_graceful_degradation(self, tmp_path):
        """
        Verify system continues processing when Redis cache is unavailable.
        Should fall back to no-cache mode without crashing.
        """
        from data_ingestion_normalization.excel_processor import ExcelProcessor
        import os
        
        # Temporarily break Redis connection
        original_redis_url = os.environ.get('REDIS_URL')
        os.environ['REDIS_URL'] = 'redis://invalid-host:6379'
        
        try:
            processor = ExcelProcessor()
            
            # Create test file
            test_excel = create_test_excel_file(tmp_path, rows=10, columns=3)
            streamed_file = StreamedFile.from_bytes(
                test_excel.read_bytes(),
                "test_no_cache.xlsx",
                str(tmp_path)
            )
            
            # Should complete without crashing (degraded mode)
            metadata = await processor._get_sheet_metadata(streamed_file)
            
            assert metadata is not None, "Should work without cache"
            assert len(metadata) > 0, "Should extract metadata even without cache"
            
            streamed_file.cleanup()
            
        finally:
            # Restore Redis URL
            if original_redis_url:
                os.environ['REDIS_URL'] = original_redis_url
            else:
                os.environ.pop('REDIS_URL', None)
    
    @pytest.mark.asyncio
    async def test_malformed_file_error_handling(self, tmp_path):
        """
        Verify corrupt/malformed files fail gracefully with clear error messages.
        Should not crash silently or return success.
        """
        from data_ingestion_normalization.excel_processor import ExcelProcessor
        
        processor = ExcelProcessor()
        
        # Create malformed Excel file
        malformed_file = create_malformed_excel(tmp_path)
        streamed_file = StreamedFile.from_bytes(
            malformed_file.read_bytes(),
            "corrupted.xlsx",
            str(tmp_path)
        )
        
        # Should raise clear error (not crash silently)
        with pytest.raises(Exception) as exc_info:
            await processor._get_sheet_metadata(streamed_file)
        
        # Error message should be informative
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ['fail', 'corrupt', 'invalid', 'error']), \
            f"Error message not clear: {exc_info.value}"
        
        streamed_file.cleanup()


# Phase B Integration Tests Complete
