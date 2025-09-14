"""
Comprehensive Unit Tests for Production Duplicate Detection Service
================================================================

Tests cover all scenarios including edge cases, error conditions, and performance.
Designed to ensure 100% reliability in production environments.

Test Categories:
- Exact duplicate detection
- Near-duplicate detection
- Content similarity algorithms
- Database operations
- Caching behavior
- Error handling
- Security validation
- Performance benchmarks
- Edge cases and boundary conditions

Author: Senior Full-Stack Engineer
Version: 2.0.0
"""

import asyncio
import hashlib
import json
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any

from production_duplicate_detection_service import (
    ProductionDuplicateDetectionService,
    DuplicateType,
    DuplicateAction,
    DuplicateResult,
    FileMetadata
)

class TestProductionDuplicateDetectionService:
    """Test suite for Production Duplicate Detection Service"""
    
    @pytest.fixture
    def mock_supabase(self):
        """Mock Supabase client"""
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_client.table.return_value = mock_table
        return mock_client, mock_table
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client"""
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        mock_redis.setex.return_value = True
        mock_redis.aget.return_value = None
        mock_redis.asetex.return_value = True
        mock_redis.keys.return_value = []
        mock_redis.delete.return_value = 0
        mock_redis.flushdb.return_value = True
        return mock_redis
    
    @pytest.fixture
    def service(self, mock_supabase, mock_redis):
        """Create service instance with mocked dependencies"""
        supabase_client, _ = mock_supabase
        return ProductionDuplicateDetectionService(supabase_client, mock_redis)
    
    @pytest.fixture
    def sample_file_metadata(self):
        """Sample file metadata for testing"""
        return FileMetadata(
            user_id="test_user_123",
            file_hash="a" * 64,  # Valid SHA-256 hash
            filename="test_file.xlsx",
            file_size=1024,
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            upload_timestamp=datetime.utcnow()
        )
    
    @pytest.fixture
    def sample_file_content(self):
        """Sample file content for testing"""
        return b"Sample Excel file content for testing duplicate detection"
    
    # ============================================================================
    # EXACT DUPLICATE DETECTION TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_exact_duplicate_detection_no_duplicates(self, service, mock_supabase, sample_file_metadata):
        """Test exact duplicate detection when no duplicates exist"""
        _, mock_table = mock_supabase
        mock_table.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = []
        
        result = await service._detect_exact_duplicates(sample_file_metadata)
        
        assert not result.is_duplicate
        assert result.duplicate_type == DuplicateType.NONE
        assert result.similarity_score == 0.0
        assert result.duplicate_files == []
        assert result.recommendation == DuplicateAction.REPLACE
    
    @pytest.mark.asyncio
    async def test_exact_duplicate_detection_with_duplicates(self, service, mock_supabase, sample_file_metadata):
        """Test exact duplicate detection when duplicates exist"""
        _, mock_table = mock_supabase
        mock_duplicates = [
            {
                'id': 'dup_1',
                'file_name': 'duplicate_file.xlsx',
                'created_at': '2024-01-01T10:00:00Z',
                'file_size': 1024,
                'status': 'processed'
            }
        ]
        mock_table.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = mock_duplicates
        
        result = await service._detect_exact_duplicates(sample_file_metadata)
        
        assert result.is_duplicate
        assert result.duplicate_type == DuplicateType.EXACT
        assert result.similarity_score == 1.0
        assert len(result.duplicate_files) == 1
        assert result.recommendation == DuplicateAction.REPLACE
        assert "Exact duplicate found" in result.message
    
    @pytest.mark.asyncio
    async def test_exact_duplicate_detection_multiple_duplicates(self, service, mock_supabase, sample_file_metadata):
        """Test exact duplicate detection with multiple duplicates"""
        _, mock_table = mock_supabase
        mock_duplicates = [
            {
                'id': 'dup_1',
                'file_name': 'old_file.xlsx',
                'created_at': '2024-01-01T10:00:00Z',
                'file_size': 1024,
                'status': 'processed'
            },
            {
                'id': 'dup_2',
                'file_name': 'newer_file.xlsx',
                'created_at': '2024-01-02T10:00:00Z',
                'file_size': 1024,
                'status': 'processed'
            }
        ]
        mock_table.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = mock_duplicates
        
        result = await service._detect_exact_duplicates(sample_file_metadata)
        
        assert result.is_duplicate
        assert len(result.duplicate_files) == 2
        # Should find the latest duplicate
        assert result.duplicate_files[0]['filename'] == 'newer_file.xlsx'
    
    # ============================================================================
    # NEAR-DUPLICATE DETECTION TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_near_duplicate_detection_no_similar_files(self, service, mock_supabase, sample_file_content, sample_file_metadata):
        """Test near-duplicate detection when no similar files exist"""
        _, mock_table = mock_supabase
        mock_table.select.return_value.eq.return_value.gte.return_value.limit.return_value.execute.return_value.data = []
        
        result = await service._detect_near_duplicates(sample_file_content, sample_file_metadata)
        
        assert not result.is_duplicate
        assert result.duplicate_type == DuplicateType.NONE
        assert result.duplicate_files == []
    
    @pytest.mark.asyncio
    async def test_near_duplicate_detection_with_similar_files(self, service, mock_supabase, sample_file_content, sample_file_metadata):
        """Test near-duplicate detection when similar files exist"""
        _, mock_table = mock_supabase
        mock_recent_files = [
            {
                'id': 'similar_1',
                'file_name': 'similar_file.xlsx',
                'created_at': '2024-01-01T10:00:00Z',
                'file_size': 1024,
                'content': {'content_fingerprint': 'similar_fingerprint'}
            }
        ]
        mock_table.select.return_value.eq.return_value.gte.return_value.limit.return_value.execute.return_value.data = mock_recent_files
        
        # Mock similarity calculation to return high similarity
        with patch.object(service, '_calculate_similarity', return_value=0.9):
            result = await service._detect_near_duplicates(sample_file_content, sample_file_metadata)
        
        assert result.is_duplicate
        assert result.duplicate_type == DuplicateType.NEAR
        assert result.similarity_score == 0.9
        assert len(result.duplicate_files) == 1
        assert result.recommendation == DuplicateAction.MERGE
    
    @pytest.mark.asyncio
    async def test_near_duplicate_detection_low_similarity(self, service, mock_supabase, sample_file_content, sample_file_metadata):
        """Test near-duplicate detection with low similarity files"""
        _, mock_table = mock_supabase
        mock_recent_files = [
            {
                'id': 'different_1',
                'file_name': 'different_file.xlsx',
                'created_at': '2024-01-01T10:00:00Z',
                'file_size': 1024,
                'content': {'content_fingerprint': 'different_fingerprint'}
            }
        ]
        mock_table.select.return_value.eq.return_value.gte.return_value.limit.return_value.execute.return_value.data = mock_recent_files
        
        # Mock similarity calculation to return low similarity
        with patch.object(service, '_calculate_similarity', return_value=0.3):
            result = await service._detect_near_duplicates(sample_file_content, sample_file_metadata)
        
        assert not result.is_duplicate
        assert result.similarity_score == 0.3
    
    # ============================================================================
    # CONTENT SIMILARITY ALGORITHM TESTS
    # ============================================================================
    
    def test_filename_similarity_identical(self, service):
        """Test filename similarity for identical files"""
        similarity = service._calculate_filename_similarity("file.xlsx", "file.xlsx")
        assert similarity == 1.0
    
    def test_filename_similarity_similar(self, service):
        """Test filename similarity for similar files"""
        similarity = service._calculate_filename_similarity("file_v1.xlsx", "file_v2.xlsx")
        assert 0.0 < similarity < 1.0
    
    def test_filename_similarity_different(self, service):
        """Test filename similarity for different files"""
        similarity = service._calculate_filename_similarity("completely_different_file.xlsx", "totally_unrelated_document.pdf")
        assert similarity < 0.5
    
    def test_filename_similarity_same_extension_boost(self, service):
        """Test filename similarity boost for same extension"""
        similarity1 = service._calculate_filename_similarity("file1.xlsx", "file2.xlsx")
        similarity2 = service._calculate_filename_similarity("file1.xlsx", "file2.csv")
        assert similarity1 > similarity2
    
    def test_filename_similarity_empty_inputs(self, service):
        """Test filename similarity with empty inputs"""
        assert service._calculate_filename_similarity("", "file.xlsx") == 0.0
        assert service._calculate_filename_similarity("file.xlsx", "") == 0.0
        assert service._calculate_filename_similarity("", "") == 0.0
    
    def test_date_similarity_recent(self, service):
        """Test date similarity for recent files"""
        recent_date = (datetime.utcnow() - timedelta(days=1)).isoformat()
        similarity = service._calculate_date_similarity(recent_date)
        assert similarity > 0.8
    
    def test_date_similarity_old(self, service):
        """Test date similarity for old files"""
        old_date = (datetime.utcnow() - timedelta(days=30)).isoformat()
        similarity = service._calculate_date_similarity(old_date)
        assert similarity == 0.0
    
    def test_fingerprint_similarity_identical(self, service):
        """Test fingerprint similarity for identical fingerprints"""
        fp = "test_fingerprint_123"
        similarity = service._calculate_fingerprint_similarity(fp, fp)
        assert similarity == 1.0
    
    def test_fingerprint_similarity_partial(self, service):
        """Test fingerprint similarity for partial matches"""
        fp1 = "test_fingerprint_123"
        fp2 = "test_fingerprint_456"
        similarity = service._calculate_fingerprint_similarity(fp1, fp2)
        assert 0.0 < similarity < 1.0
    
    def test_fingerprint_similarity_different(self, service):
        """Test fingerprint similarity for different fingerprints"""
        fp1 = "test_fingerprint_123"
        fp2 = "different_fingerprint_456"
        similarity = service._calculate_fingerprint_similarity(fp1, fp2)
        assert similarity < 0.5
    
    # ============================================================================
    # CACHING TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_cache_miss(self, service, mock_redis):
        """Test cache miss behavior"""
        cache_key = "test_key"
        result = await service._get_from_cache(cache_key)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_hit_redis(self, service, mock_redis):
        """Test cache hit with Redis"""
        cache_key = "test_key"
        test_result = DuplicateResult(
            is_duplicate=True,
            duplicate_type=DuplicateType.EXACT,
            similarity_score=1.0,
            duplicate_files=[],
            recommendation=DuplicateAction.REPLACE,
            message="Test",
            confidence=1.0,
            processing_time_ms=100
        )
        
        mock_redis.aget.return_value = json.dumps(test_result.__dict__, default=str)
        
        result = await service._get_from_cache(cache_key)
        assert result is not None
        assert result.is_duplicate == test_result.is_duplicate
    
    @pytest.mark.asyncio
    async def test_cache_set_redis(self, service, mock_redis):
        """Test cache set with Redis"""
        cache_key = "test_key"
        test_result = DuplicateResult(
            is_duplicate=True,
            duplicate_type=DuplicateType.EXACT,
            similarity_score=1.0,
            duplicate_files=[],
            recommendation=DuplicateAction.REPLACE,
            message="Test",
            confidence=1.0,
            processing_time_ms=100
        )
        
        await service._set_cache(cache_key, test_result)
        
        mock_redis.asetex.assert_called_once()
        call_args = mock_redis.asetex.call_args
        assert call_args[0][0] == cache_key
        assert call_args[0][1] == service.cache_ttl
    
    @pytest.mark.asyncio
    async def test_cache_fallback_memory(self, service):
        """Test cache fallback to memory when Redis unavailable"""
        service.redis_client = None
        cache_key = "test_key"
        test_result = DuplicateResult(
            is_duplicate=True,
            duplicate_type=DuplicateType.EXACT,
            similarity_score=1.0,
            duplicate_files=[],
            recommendation=DuplicateAction.REPLACE,
            message="Test",
            confidence=1.0,
            processing_time_ms=100
        )
        
        await service._set_cache(cache_key, test_result)
        result = await service._get_from_cache(cache_key)
        
        assert result is not None
        assert result.is_duplicate == test_result.is_duplicate
    
    # ============================================================================
    # SECURITY VALIDATION TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_validate_inputs_valid(self, service, sample_file_content, sample_file_metadata):
        """Test input validation with valid inputs"""
        # Should not raise any exception
        await service._validate_inputs(sample_file_content, sample_file_metadata)
    
    @pytest.mark.asyncio
    async def test_validate_inputs_file_too_large(self, service, sample_file_metadata):
        """Test input validation with file too large"""
        large_content = b"x" * (service.max_file_size + 1)
        
        with pytest.raises(ValueError, match="File too large"):
            await service._validate_inputs(large_content, sample_file_metadata)
    
    @pytest.mark.asyncio
    async def test_validate_inputs_invalid_user_id(self, service, sample_file_content):
        """Test input validation with invalid user ID"""
        invalid_metadata = FileMetadata(
            user_id="invalid@user#id",
            file_hash="a" * 64,
            filename="test.xlsx",
            file_size=1024,
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            upload_timestamp=datetime.utcnow()
        )
        
        with pytest.raises(ValueError, match="Invalid user_id format"):
            await service._validate_inputs(sample_file_content, invalid_metadata)
    
    @pytest.mark.asyncio
    async def test_validate_inputs_path_traversal_filename(self, service, sample_file_content):
        """Test input validation with path traversal in filename"""
        invalid_metadata = FileMetadata(
            user_id="test_user",
            file_hash="a" * 64,
            filename="../../../etc/passwd",
            file_size=1024,
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            upload_timestamp=datetime.utcnow()
        )
        
        with pytest.raises(ValueError, match="Invalid filename"):
            await service._validate_inputs(sample_file_content, invalid_metadata)
    
    @pytest.mark.asyncio
    async def test_validate_inputs_invalid_hash(self, service, sample_file_content):
        """Test input validation with invalid file hash"""
        invalid_metadata = FileMetadata(
            user_id="test_user",
            file_hash="invalid_hash",
            filename="test.xlsx",
            file_size=1024,
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            upload_timestamp=datetime.utcnow()
        )
        
        with pytest.raises(ValueError, match="Invalid file hash format"):
            await service._validate_inputs(sample_file_content, invalid_metadata)
    
    # ============================================================================
    # ERROR HANDLING TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_database_error_handling(self, service, mock_supabase, sample_file_metadata):
        """Test error handling when database query fails"""
        _, mock_table = mock_supabase
        mock_table.select.return_value.eq.return_value.eq.return_value.execute.side_effect = Exception("Database error")
        
        with pytest.raises(Exception, match="Database error"):
            await service._detect_exact_duplicates(sample_file_metadata)
    
    @pytest.mark.asyncio
    async def test_cache_error_handling(self, service, sample_file_content, sample_file_metadata):
        """Test error handling when cache operations fail"""
        # Mock Redis to raise exception
        service.redis_client = AsyncMock()
        service.redis_client.get.side_effect = Exception("Redis error")
        
        # Should not raise exception, should fallback gracefully
        result = await service._get_from_cache("test_key")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_similarity_calculation_error_handling(self, service, sample_file_content, sample_file_metadata):
        """Test error handling when similarity calculation fails"""
        # Test that the method handles errors gracefully
        similarity = await service._calculate_similarity(sample_file_content, {}, "test.xlsx")
        assert similarity == 0.0  # Should return 0.0 for invalid input
    
    # ============================================================================
    # PERFORMANCE TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_large_file_processing(self, service, sample_file_metadata):
        """Test processing of large files"""
        # Create large content
        large_content = b"x" * (100 * 1024)  # 100KB
        
        # Should not raise memory error
        fingerprint = await service._calculate_content_fingerprint(large_content)
        assert fingerprint is not None
        assert len(fingerprint) > 0
    
    def test_feature_extraction_performance(self, service):
        """Test feature extraction performance with large text"""
        large_text = "word " * 10000  # 50,000 words
        
        start_time = time.time()
        features = service._extract_features(large_text)
        end_time = time.time()
        
        assert len(features) > 0
        assert end_time - start_time < 1.0  # Should complete within 1 second
    
    def test_minhash_calculation_performance(self, service):
        """Test MinHash calculation performance"""
        large_features = {f"feature_{i}" for i in range(10000)}
        
        start_time = time.time()
        signature = service._calculate_minhash(large_features)
        end_time = time.time()
        
        assert signature is not None
        assert end_time - start_time < 1.0  # Should complete within 1 second
    
    # ============================================================================
    # INTEGRATION TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_full_duplicate_detection_flow(self, service, mock_supabase, sample_file_content, sample_file_metadata):
        """Test complete duplicate detection flow"""
        _, mock_table = mock_supabase
        
        # Mock no exact duplicates
        mock_table.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = []
        
        # Mock no near duplicates
        mock_table.select.return_value.eq.return_value.gte.return_value.limit.return_value.execute.return_value.data = []
        
        result = await service.detect_duplicates(sample_file_content, sample_file_metadata)
        
        assert not result.is_duplicate
        assert result.duplicate_type == DuplicateType.NONE
        assert result.error is None
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, service):
        """Test metrics collection"""
        metrics = await service.get_metrics()
        
        assert 'cache_hits' in metrics
        assert 'cache_misses' in metrics
        assert 'exact_duplicates_found' in metrics
        assert 'near_duplicates_found' in metrics
        assert 'processing_errors' in metrics
        assert 'total_processing_time' in metrics
        assert 'cache_size' in metrics
        assert 'avg_processing_time' in metrics
    
    @pytest.mark.asyncio
    async def test_cache_clear(self, service, mock_redis):
        """Test cache clearing functionality"""
        await service.clear_cache("test_user")
        mock_redis.keys.assert_called_once()
    
    # ============================================================================
    # EDGE CASES AND BOUNDARY CONDITIONS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_empty_file_content(self, service, sample_file_metadata):
        """Test handling of empty file content"""
        empty_content = b""
        
        # Empty content should be handled gracefully (no error)
        await service._validate_inputs(empty_content, sample_file_metadata)
        
        # Test that empty content works with duplicate detection
        result = await service.detect_duplicates(empty_content, sample_file_metadata)
        assert result is not None
    
    def test_unicode_filename_handling(self, service):
        """Test handling of Unicode filenames"""
        unicode_filename = "测试文件.xlsx"
        similarity = service._calculate_filename_similarity(unicode_filename, unicode_filename)
        assert similarity == 1.0
    
    def test_very_long_filename_handling(self, service):
        """Test handling of very long filenames"""
        long_filename = "a" * 1000 + ".xlsx"
        similarity = service._calculate_filename_similarity(long_filename, long_filename)
        assert similarity == 1.0
    
    @pytest.mark.asyncio
    async def test_concurrent_duplicate_detection(self, service, mock_supabase, sample_file_content, sample_file_metadata):
        """Test concurrent duplicate detection requests"""
        _, mock_table = mock_supabase
        mock_table.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = []
        mock_table.select.return_value.eq.return_value.gte.return_value.limit.return_value.execute.return_value.data = []
        
        # Run multiple concurrent requests
        tasks = [
            service.detect_duplicates(sample_file_content, sample_file_metadata)
            for _ in range(10)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All results should be consistent
        for result in results:
            assert not result.is_duplicate
            assert result.duplicate_type == DuplicateType.NONE

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
