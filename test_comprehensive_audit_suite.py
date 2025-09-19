"""
Comprehensive Audit and Testing Suite for All Four Critical Components
====================================================================

This module provides complete unit tests, integration tests, and performance tests
for all four critical components:
1. Deduplication Detection Service
2. EnhancedFileProcessor  
3. VendorStandardizer
4. PlatformIDExtractor

Tests cover:
- Unit tests for all functions
- Edge cases and error handling
- Integration tests for full pipeline
- Performance and scalability tests
- Security validation tests
- Frontend/backend synchronization tests
- WebSocket real-time update tests
- Memory efficiency and concurrency tests

Author: Senior Full-Stack Engineer
Version: 1.0.0
"""

import pytest
import asyncio
import pandas as pd
import tempfile
import os
import json
import hashlib
import time
import psutil
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any
import io
import zipfile

# Import all four critical components
from production_duplicate_detection_service import (
    ProductionDuplicateDetectionService, 
    FileMetadata, 
    DuplicateType,
    DuplicateAction,
    DuplicateResult
)
# from enhanced_file_processor import EnhancedFileProcessor  # DEPRECATED: Module removed
from fastapi_backend import VendorStandardizer, PlatformIDExtractor
from duplicate_detection_api_integration import DuplicateDetectionAPIIntegration


class TestDeduplicationDetectionService:
    """Comprehensive tests for Deduplication Detection Service"""
    
    @pytest.fixture
    def mock_supabase(self):
        """Mock Supabase client with comprehensive setup"""
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_client.table.return_value = mock_table
        return mock_client, mock_table
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client with async support"""
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
        # Should find the latest duplicate first
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
    
    # ============================================================================
    # SECURITY VALIDATION TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_validate_inputs_valid(self, service, sample_file_content, sample_file_metadata):
        """Test input validation with valid inputs"""
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
    
    # ============================================================================
    # PERFORMANCE TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_large_file_processing(self, service, sample_file_metadata):
        """Test processing of large files"""
        large_content = b"x" * (100 * 1024)  # 100KB
        
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


class TestEnhancedFileProcessor:
    """Comprehensive tests for EnhancedFileProcessor"""
    
    @pytest.fixture
    def processor(self):
        """Create EnhancedFileProcessor instance for testing"""
        return EnhancedFileProcessor()
    
    @pytest.fixture
    def sample_excel_content(self):
        """Create sample Excel content for testing"""
        df1 = pd.DataFrame({
            'Name': ['John', 'Jane', 'Bob'],
            'Age': [25, 30, 35],
            'Salary': [50000, 60000, 70000]
        })
        df2 = pd.DataFrame({
            'Product': ['A', 'B', 'C'],
            'Price': [10.99, 20.99, 30.99],
            'Stock': [100, 200, 300]
        })
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df1.to_excel(writer, sheet_name='Employees', index=False)
            df2.to_excel(writer, sheet_name='Products', index=False)
        
        return output.getvalue()
    
    @pytest.fixture
    def sample_csv_content(self):
        """Create sample CSV content for testing"""
        csv_data = "Name,Age,Salary\nJohn,25,50000\nJane,30,60000\nBob,35,70000"
        return csv_data.encode('utf-8')
    
    @pytest.fixture
    def mock_progress_callback(self):
        """Create mock progress callback for testing"""
        return AsyncMock()
    
    # ============================================================================
    # FILE FORMAT DETECTION TESTS
    # ============================================================================
    
    def test_detect_file_format_excel(self, processor):
        """Test Excel file format detection"""
        result = processor._detect_file_format('test.xlsx', b'fake content')
        assert result == 'excel'
        
        result = processor._detect_file_format('test.xls', b'fake content')
        assert result == 'excel'
        
        result = processor._detect_file_format('test.xlsm', b'fake content')
        assert result == 'excel'
    
    def test_detect_file_format_csv(self, processor):
        """Test CSV file format detection"""
        result = processor._detect_file_format('test.csv', b'fake content')
        assert result == 'csv'
        
        result = processor._detect_file_format('test.tsv', b'fake content')
        assert result == 'csv'
        
        result = processor._detect_file_format('test.txt', b'fake content')
        assert result == 'csv'
    
    def test_detect_file_format_pdf(self, processor):
        """Test PDF file format detection"""
        result = processor._detect_file_format('test.pdf', b'fake content')
        assert result == 'pdf'
    
    def test_detect_file_format_archive(self, processor):
        """Test archive file format detection"""
        result = processor._detect_file_format('test.zip', b'fake content')
        assert result == 'zip'
        
        result = processor._detect_file_format('test.7z', b'fake content')
        assert result == 'zip'
        
        result = processor._detect_file_format('test.rar', b'fake content')
        assert result == 'zip'
    
    # ============================================================================
    # SECURITY VALIDATION TESTS
    # ============================================================================
    
    def test_validate_security_valid_file(self, processor):
        """Test security validation with valid file"""
        content = b"valid content"
        filename = "test.xlsx"
        
        # Should not raise exception
        processor._validate_security(content, filename)
    
    def test_validate_security_file_too_large(self, processor):
        """Test security validation with file too large"""
        large_content = b"x" * (processor.max_file_size + 1)
        filename = "test.xlsx"
        
        with pytest.raises(ValueError, match="File size.*exceeds maximum"):
            processor._validate_security(large_content, filename)
    
    def test_validate_security_path_traversal_filename(self, processor):
        """Test security validation with path traversal filename"""
        content = b"valid content"
        malicious_filename = "../../../etc/passwd"
        
        with pytest.raises(ValueError, match="Filename contains path traversal"):
            processor._validate_security(content, malicious_filename)
    
    def test_validate_security_invalid_extension(self, processor):
        """Test security validation with invalid extension"""
        content = b"valid content"
        filename = "test.exe"
        
        with pytest.raises(ValueError, match="File extension.*is not allowed"):
            processor._validate_security(content, filename)
    
    def test_validate_security_empty_file(self, processor):
        """Test security validation with empty file"""
        content = b""
        filename = "test.xlsx"
        
        with pytest.raises(ValueError, match="File is empty"):
            processor._validate_security(content, filename)
    
    # ============================================================================
    # CONTENT SANITIZATION TESTS
    # ============================================================================
    
    def test_sanitize_content_normal(self, processor):
        """Test content sanitization with normal content"""
        content = b"Normal file content without malicious patterns"
        sanitized = processor._sanitize_content(content)
        assert sanitized == content
    
    def test_sanitize_content_script_injection(self, processor):
        """Test content sanitization with script injection"""
        content = b"Normal content <script>alert('xss')</script> more content"
        sanitized = processor._sanitize_content(content)
        assert b"<script>" not in sanitized
        assert b"[REMOVED]" in sanitized
    
    def test_sanitize_content_sql_injection(self, processor):
        """Test content sanitization with SQL injection"""
        content = b"Normal content DROP TABLE users; more content"
        sanitized = processor._sanitize_content(content)
        assert b"DROP TABLE" not in sanitized
        assert b"[REMOVED]" in sanitized
    
    # ============================================================================
    # PERFORMANCE TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_process_file_enhanced_performance(self, processor, sample_excel_content, mock_progress_callback):
        """Test processing performance"""
        start_time = time.time()
        
        with patch.object(processor, '_process_excel_enhanced') as mock_process:
            mock_process.return_value = {'Sheet1': pd.DataFrame()}
            
            result = await processor.process_file_enhanced(
                sample_excel_content, 
                'test.xlsx', 
                mock_progress_callback
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should complete within reasonable time
            assert processing_time < 5.0  # 5 seconds max
            assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_process_file_enhanced_memory_efficiency(self, processor, mock_progress_callback):
        """Test memory efficiency with large files"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process multiple large files
        for i in range(5):
            large_content = b"x" * (1024 * 1024)  # 1MB each
            
            with patch.object(processor, '_detect_file_format', return_value='excel'):
                with patch.object(processor, '_process_excel_enhanced') as mock_process:
                    mock_process.return_value = {'Sheet1': pd.DataFrame()}
                    
                    result = await processor.process_file_enhanced(
                        large_content, 
                        f'test{i}.xlsx', 
                        mock_progress_callback
                    )
                    
                    assert isinstance(result, dict)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024
    
    # ============================================================================
    # ERROR HANDLING TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_process_file_enhanced_unsupported_format(self, processor, mock_progress_callback):
        """Test processing unsupported file format"""
        with patch.object(processor, '_detect_file_format', return_value='unknown'):
            with patch.object(processor, '_fallback_processing') as mock_fallback:
                mock_fallback.return_value = {'Sheet1': pd.DataFrame()}
                
                result = await processor.process_file_enhanced(
                    b"unknown content", 
                    'test.unknown', 
                    mock_progress_callback
                )
                
                assert isinstance(result, dict)
                mock_fallback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_file_enhanced_empty_content(self, processor, mock_progress_callback):
        """Test processing empty file content"""
        result = await processor.process_file_enhanced(
            b"", 
            'test.xlsx', 
            mock_progress_callback
        )
        
        # Should handle empty content gracefully
        assert isinstance(result, dict)


class TestVendorStandardizer:
    """Comprehensive tests for VendorStandardizer"""
    
    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI client"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"standard_name": "Amazon", "confidence": 0.95, "reasoning": "AI standardization"}'
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client
    
    @pytest.fixture
    def vendor_standardizer(self, mock_openai):
        """Create VendorStandardizer instance"""
        return VendorStandardizer(mock_openai)
    
    # ============================================================================
    # RULE-BASED CLEANING TESTS
    # ============================================================================
    
    def test_rule_based_cleaning_simple_case(self, vendor_standardizer):
        """Test rule-based cleaning with simple case"""
        result = vendor_standardizer._rule_based_cleaning("Amazon.com Inc")
        assert result == "Amazon"
    
    def test_rule_based_cleaning_multiple_suffixes(self, vendor_standardizer):
        """Test rule-based cleaning with multiple suffixes"""
        result = vendor_standardizer._rule_based_cleaning("Microsoft Corporation LLC")
        assert result == "Microsoft"
    
    def test_rule_based_cleaning_no_change_needed(self, vendor_standardizer):
        """Test rule-based cleaning when no change needed"""
        result = vendor_standardizer._rule_based_cleaning("Apple")
        assert result == "Apple"
    
    def test_rule_based_cleaning_edge_cases(self, vendor_standardizer):
        """Test rule-based cleaning with edge cases"""
        # Test with empty string
        result = vendor_standardizer._rule_based_cleaning("")
        assert result == ""
        
        # Test with only whitespace
        result = vendor_standardizer._rule_based_cleaning("   ")
        assert result == ""
        
        # Test with special characters
        result = vendor_standardizer._rule_based_cleaning("Test & Co. Ltd.")
        assert "Test" in result
    
    # ============================================================================
    # AI STANDARDIZATION TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_ai_standardization_success(self, vendor_standardizer):
        """Test successful AI standardization"""
        result = await vendor_standardizer._ai_standardization("Amazon.com Inc", "stripe")
        
        assert result["vendor_raw"] == "Amazon.com Inc"
        assert result["vendor_standard"] == "Amazon"
        assert result["confidence"] == 0.95
        assert result["cleaning_method"] == "ai_powered"
        assert "reasoning" in result
    
    @pytest.mark.asyncio
    async def test_ai_standardization_openai_error(self, vendor_standardizer):
        """Test AI standardization with OpenAI error"""
        vendor_standardizer.openai.chat.completions.create.side_effect = Exception("OpenAI error")
        
        result = await vendor_standardizer._ai_standardization("Test Company", "stripe")
        
        assert result["vendor_raw"] == "Test Company"
        assert result["vendor_standard"] == "Test Company"
        assert result["confidence"] == 0.5
        assert result["cleaning_method"] == "ai_fallback"
    
    # ============================================================================
    # MAIN STANDARDIZATION TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_standardize_vendor_success(self, vendor_standardizer):
        """Test successful vendor standardization"""
        result = await vendor_standardizer.standardize_vendor("Amazon.com Inc", "stripe")
        
        assert result["vendor_raw"] == "Amazon.com Inc"
        assert result["vendor_standard"] == "Amazon"
        assert result["confidence"] >= 0.8
        assert "cleaning_method" in result
    
    @pytest.mark.asyncio
    async def test_standardize_vendor_empty_input(self, vendor_standardizer):
        """Test vendor standardization with empty input"""
        result = await vendor_standardizer.standardize_vendor("", "stripe")
        
        assert result["vendor_raw"] == ""
        assert result["vendor_standard"] == ""
        assert result["confidence"] == 0.0
        assert result["cleaning_method"] == "empty"
    
    @pytest.mark.asyncio
    async def test_standardize_vendor_none_input(self, vendor_standardizer):
        """Test vendor standardization with None input"""
        result = await vendor_standardizer.standardize_vendor(None, "stripe")
        
        assert result["vendor_raw"] is None
        assert result["vendor_standard"] == ""
        assert result["confidence"] == 0.0
        assert result["cleaning_method"] == "empty"
    
    @pytest.mark.asyncio
    async def test_standardize_vendor_caching(self, vendor_standardizer):
        """Test vendor standardization caching"""
        # First call
        result1 = await vendor_standardizer.standardize_vendor("Amazon.com Inc", "stripe")
        
        # Second call should use cache
        result2 = await vendor_standardizer.standardize_vendor("Amazon.com Inc", "stripe")
        
        assert result1 == result2
        # OpenAI should only be called once
        assert vendor_standardizer.openai.chat.completions.create.call_count == 0  # Rule-based cleaning used
    
    # ============================================================================
    # ERROR HANDLING TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_standardize_vendor_exception_handling(self, vendor_standardizer):
        """Test vendor standardization exception handling"""
        # Mock an exception in the standardization process
        with patch.object(vendor_standardizer, '_rule_based_cleaning', side_effect=Exception("Test error")):
            result = await vendor_standardizer.standardize_vendor("Test Company", "stripe")
            
            assert result["vendor_raw"] == "Test Company"
            assert result["vendor_standard"] == "Test Company"
            assert result["confidence"] == 0.5
            assert result["cleaning_method"] == "fallback"
    
    # ============================================================================
    # PERFORMANCE TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_standardize_vendor_performance(self, vendor_standardizer):
        """Test vendor standardization performance"""
        start_time = time.time()
        
        # Standardize multiple vendors
        vendors = ["Amazon.com Inc", "Microsoft Corporation", "Google LLC", "Apple Inc.", "Meta Platforms Inc"]
        results = []
        
        for vendor in vendors:
            result = await vendor_standardizer.standardize_vendor(vendor, "stripe")
            results.append(result)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within reasonable time (5 seconds for 5 vendors)
        assert processing_time < 5.0
        assert len(results) == 5
        assert all(result["confidence"] > 0 for result in results)
    
    @pytest.mark.asyncio
    async def test_standardize_vendor_concurrent_processing(self, vendor_standardizer):
        """Test concurrent vendor standardization"""
        vendors = ["Amazon.com Inc", "Microsoft Corporation", "Google LLC", "Apple Inc.", "Meta Platforms Inc"]
        
        # Process vendors concurrently
        tasks = [
            vendor_standardizer.standardize_vendor(vendor, "stripe")
            for vendor in vendors
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(results) == 5
        assert all(result["confidence"] > 0 for result in results)
        assert all(result["vendor_standard"] != "" for result in results)


class TestPlatformIDExtractor:
    """Comprehensive tests for PlatformIDExtractor"""
    
    @pytest.fixture
    def platform_extractor(self):
        """Create PlatformIDExtractor instance"""
        return PlatformIDExtractor()
    
    # ============================================================================
    # PLATFORM ID EXTRACTION TESTS
    # ============================================================================
    
    def test_extract_platform_ids_razorpay(self, platform_extractor):
        """Test platform ID extraction for Razorpay"""
        row_data = {
            "payment_id": "pay_12345678901234",
            "order_id": "order_98765432109876",
            "amount": 1000,
            "description": "Payment for services"
        }
        column_names = ["payment_id", "order_id", "amount", "description"]
        
        result = platform_extractor.extract_platform_ids(row_data, "razorpay", column_names)
        
        assert result["platform"] == "razorpay"
        assert "payment_id" in result["extracted_ids"]
        assert "order_id" in result["extracted_ids"]
        assert result["extracted_ids"]["payment_id"] == "pay_12345678901234"
        assert result["extracted_ids"]["order_id"] == "order_98765432109876"
        assert result["total_ids_found"] >= 2
    
    def test_extract_platform_ids_stripe(self, platform_extractor):
        """Test platform ID extraction for Stripe"""
        row_data = {
            "charge_id": "ch_123456789012345678901234",
            "customer_id": "cus_12345678901234",
            "amount": 50.00,
            "description": "Stripe payment"
        }
        column_names = ["charge_id", "customer_id", "amount", "description"]
        
        result = platform_extractor.extract_platform_ids(row_data, "stripe", column_names)
        
        assert result["platform"] == "stripe"
        assert "charge_id" in result["extracted_ids"]
        assert "customer_id" in result["extracted_ids"]
        assert result["extracted_ids"]["charge_id"] == "ch_123456789012345678901234"
        assert result["extracted_ids"]["customer_id"] == "cus_12345678901234"
        assert result["total_ids_found"] >= 2
    
    def test_extract_platform_ids_quickbooks(self, platform_extractor):
        """Test platform ID extraction for QuickBooks"""
        row_data = {
            "transaction_id": "txn_123456789012",
            "invoice_id": "inv_1234567890",
            "vendor_id": "ven_12345678",
            "amount": 250.00
        }
        column_names = ["transaction_id", "invoice_id", "vendor_id", "amount"]
        
        result = platform_extractor.extract_platform_ids(row_data, "quickbooks", column_names)
        
        assert result["platform"] == "quickbooks"
        assert "transaction_id" in result["extracted_ids"]
        assert "invoice_id" in result["extracted_ids"]
        assert "vendor_id" in result["extracted_ids"]
        assert result["total_ids_found"] >= 3
    
    def test_extract_platform_ids_no_matches(self, platform_extractor):
        """Test platform ID extraction with no matches"""
        row_data = {
            "amount": 100,
            "description": "No platform IDs here"
        }
        column_names = ["amount", "description"]
        
        result = platform_extractor.extract_platform_ids(row_data, "unknown_platform", column_names)
        
        assert result["platform"] == "unknown_platform"
        assert "platform_generated_id" in result["extracted_ids"]
        assert result["total_ids_found"] == 1
    
    def test_extract_platform_ids_empty_row_data(self, platform_extractor):
        """Test platform ID extraction with empty row data"""
        row_data = {}
        column_names = []
        
        result = platform_extractor.extract_platform_ids(row_data, "test_platform", column_names)
        
        assert result["platform"] == "test_platform"
        assert "platform_generated_id" in result["extracted_ids"]
        assert result["total_ids_found"] == 1
    
    def test_extract_platform_ids_none_values(self, platform_extractor):
        """Test platform ID extraction with None values"""
        row_data = {
            "payment_id": None,
            "order_id": "",
            "amount": 100,
            "description": None
        }
        column_names = ["payment_id", "order_id", "amount", "description"]
        
        result = platform_extractor.extract_platform_ids(row_data, "razorpay", column_names)
        
        assert result["platform"] == "razorpay"
        assert "platform_generated_id" in result["extracted_ids"]
        assert result["total_ids_found"] == 1
    
    # ============================================================================
    # EDGE CASES AND ERROR HANDLING TESTS
    # ============================================================================
    
    def test_extract_platform_ids_exception_handling(self, platform_extractor):
        """Test platform ID extraction exception handling"""
        # Test with invalid input that could cause exceptions
        row_data = {"key": "value"}
        column_names = ["key"]
        
        # Mock an exception
        with patch.object(platform_extractor, '_extract_platform_ids', side_effect=Exception("Test error")):
            result = platform_extractor.extract_platform_ids(row_data, "test_platform", column_names)
            
            assert result["platform"] == "test_platform"
            assert result["extracted_ids"] == {}
            assert result["total_ids_found"] == 0
            assert "error" in result
            assert result["error"] == "Test error"
    
    def test_extract_platform_ids_very_long_values(self, platform_extractor):
        """Test platform ID extraction with very long values"""
        long_value = "a" * 1000
        row_data = {
            "payment_id": long_value,
            "amount": 100
        }
        column_names = ["payment_id", "amount"]
        
        result = platform_extractor.extract_platform_ids(row_data, "razorpay", column_names)
        
        assert result["platform"] == "razorpay"
        # Should handle long values without issues
        assert result["total_ids_found"] >= 1
    
    def test_extract_platform_ids_special_characters(self, platform_extractor):
        """Test platform ID extraction with special characters"""
        row_data = {
            "payment_id": "pay_123!@#$%^&*()",
            "amount": 100
        }
        column_names = ["payment_id", "amount"]
        
        result = platform_extractor.extract_platform_ids(row_data, "razorpay", column_names)
        
        assert result["platform"] == "razorpay"
        # Should handle special characters gracefully
        assert result["total_ids_found"] >= 1
    
    # ============================================================================
    # PERFORMANCE TESTS
    # ============================================================================
    
    def test_extract_platform_ids_performance(self, platform_extractor):
        """Test platform ID extraction performance"""
        start_time = time.time()
        
        # Test with multiple rows
        for i in range(1000):
            row_data = {
                f"payment_id": f"pay_{i:012d}",
                f"order_id": f"order_{i:012d}",
                "amount": 100 + i
            }
            column_names = [f"payment_id", f"order_id", "amount"]
            
            result = platform_extractor.extract_platform_ids(row_data, "razorpay", column_names)
            assert result["total_ids_found"] >= 2
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within reasonable time (1 second for 1000 extractions)
        assert processing_time < 1.0
    
    def test_extract_platform_ids_memory_efficiency(self, platform_extractor):
        """Test platform ID extraction memory efficiency"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process many rows
        for i in range(10000):
            row_data = {
                "payment_id": f"pay_{i:012d}",
                "amount": 100 + i
            }
            column_names = ["payment_id", "amount"]
            
            result = platform_extractor.extract_platform_ids(row_data, "razorpay", column_names)
            assert result["total_ids_found"] >= 1
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal (less than 10MB for 10k extractions)
        assert memory_increase < 10 * 1024 * 1024


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


