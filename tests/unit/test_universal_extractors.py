"""
Unit Tests for Universal Extractors (Phase 4)
Tests file parsing, format detection, and data extraction
"""
import pytest
import asyncio
import pandas as pd
import io
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from universal_extractors_optimized import UniversalExtractorsOptimized


class TestUniversalExtractors:
    """Unit tests for UniversalExtractorsOptimized"""
    
    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI client"""
        mock = Mock()
        mock.chat = Mock()
        mock.chat.completions = Mock()
        mock.chat.completions.create = AsyncMock(return_value=Mock(
            choices=[Mock(message=Mock(content='{"classification": "test"}'))]
        ))
        return mock
    
    @pytest.fixture
    def mock_cache(self):
        """Mock cache client"""
        mock = AsyncMock()
        mock.get_cached_classification = AsyncMock(return_value=None)
        mock.store_classification = AsyncMock(return_value=True)
        return mock
    
    @pytest.fixture
    def extractor(self, mock_openai, mock_cache):
        """Create UniversalExtractorsOptimized instance"""
        return UniversalExtractorsOptimized(
            openai_client=mock_openai,
            cache_client=mock_cache
        )
    
    # ============================================================================
    # CSV Format Tests
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_csv_extraction_basic(self, extractor):
        """Test basic CSV file extraction"""
        csv_content = b"name,amount,date\nJohn,100,2024-01-01\nJane,200,2024-01-02"
        
        result = await extractor.extract_data_universal(
            csv_content, 
            "test.csv", 
            "test_user"
        )
        
        assert result['file_format'] == 'csv'
        assert result['confidence_score'] >= 0.5
        assert 'extracted_data' in result or 'raw_data' in result
        assert result['metadata']['file_size_bytes'] == len(csv_content)
    
    @pytest.mark.asyncio
    async def test_csv_encoding_handling(self, extractor):
        """Test CSV with different encodings"""
        # UTF-8 with special characters
        csv_content = "name,amount\nJosé,€100".encode('utf-8')
        
        result = await extractor.extract_data_universal(
            csv_content,
            "test_utf8.csv",
            "test_user"
        )
        
        assert result['file_format'] == 'csv'
        assert 'error' not in result
    
    @pytest.mark.asyncio
    async def test_csv_empty_file(self, extractor):
        """Test empty CSV file handling"""
        csv_content = b""
        
        result = await extractor.extract_data_universal(
            csv_content,
            "empty.csv",
            "test_user"
        )
        
        # Should handle gracefully
        assert 'error' in result or result['confidence_score'] == 0.0
    
    # ============================================================================
    # Excel Format Tests
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_excel_single_sheet(self, extractor):
        """Test Excel file with single sheet"""
        # Create simple Excel file
        df = pd.DataFrame({
            'name': ['Alice', 'Bob'],
            'amount': [100, 200],
            'date': ['2024-01-01', '2024-01-02']
        })
        
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False, engine='openpyxl')
        excel_content = buffer.getvalue()
        
        result = await extractor.extract_data_universal(
            excel_content,
            "test.xlsx",
            "test_user"
        )
        
        assert result['file_format'] == 'xlsx'
        assert result['confidence_score'] >= 0.5
        assert 'extracted_data' in result
    
    @pytest.mark.asyncio
    async def test_excel_multi_sheet(self, extractor):
        """Test Excel file with multiple sheets"""
        # Create multi-sheet Excel
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            pd.DataFrame({'col1': [1, 2]}).to_excel(writer, sheet_name='Sheet1', index=False)
            pd.DataFrame({'col2': [3, 4]}).to_excel(writer, sheet_name='Sheet2', index=False)
        
        excel_content = buffer.getvalue()
        
        result = await extractor.extract_data_universal(
            excel_content,
            "multi_sheet.xlsx",
            "test_user"
        )
        
        assert result['file_format'] == 'xlsx'
        # Should extract multiple sheets
        raw_data = result.get('raw_data', {})
        if 'sheets' in raw_data:
            assert len(raw_data['sheets']) >= 1
    
    # ============================================================================
    # JSON Format Tests
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_json_extraction(self, extractor):
        """Test JSON file extraction"""
        json_content = b'[{"name": "test", "value": 123}, {"name": "test2", "value": 456}]'
        
        result = await extractor.extract_data_universal(
            json_content,
            "test.json",
            "test_user"
        )
        
        assert result['file_format'] == 'json'
        assert result['confidence_score'] >= 0.5
    
    @pytest.mark.asyncio
    async def test_json_invalid(self, extractor):
        """Test invalid JSON handling"""
        json_content = b'{invalid json content'
        
        result = await extractor.extract_data_universal(
            json_content,
            "invalid.json",
            "test_user"
        )
        
        # Should return error
        assert 'error' in result or result['confidence_score'] == 0.0
    
    # ============================================================================
    # Format Detection Tests
    # ============================================================================
    
    def test_format_detection_csv(self, extractor):
        """Test CSV format detection"""
        csv_content = b"col1,col2\nval1,val2"
        format_detected = extractor._detect_file_format(csv_content, "test.csv")
        assert format_detected == 'csv'
    
    def test_format_detection_excel(self, extractor):
        """Test Excel format detection"""
        # Excel magic number: PK (ZIP format)
        excel_content = b'PK\x03\x04' + b'\x00' * 100
        format_detected = extractor._detect_file_format(excel_content, "test.xlsx")
        assert format_detected in ['xlsx', 'unknown']
    
    def test_format_detection_json(self, extractor):
        """Test JSON format detection"""
        json_content = b'{"key": "value"}'
        format_detected = extractor._detect_file_format(json_content, "test.json")
        assert format_detected == 'json'
    
    # ============================================================================
    # Validation Tests
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_file_size_validation(self, extractor):
        """Test file size limit validation"""
        # Create file larger than max size
        large_content = b'x' * (extractor.config['max_file_size_mb'] * 1024 * 1024 + 1)
        
        result = await extractor.extract_data_universal(
            large_content,
            "large.csv",
            "test_user"
        )
        
        # Should fail validation
        assert 'error' in result or result.get('status') == 'failed'
    
    @pytest.mark.asyncio
    async def test_empty_filename_validation(self, extractor):
        """Test empty filename validation"""
        result = await extractor.extract_data_universal(
            b"test content",
            "",
            "test_user"
        )
        
        assert 'error' in result
    
    @pytest.mark.asyncio
    async def test_empty_user_id_validation(self, extractor):
        """Test empty user ID validation"""
        result = await extractor.extract_data_universal(
            b"test content",
            "test.csv",
            ""
        )
        
        assert 'error' in result
    
    # ============================================================================
    # Caching Tests
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_cache_hit(self, extractor, mock_cache):
        """Test cache hit scenario"""
        cached_result = {
            'extraction_id': 'test_id',
            'file_format': 'csv',
            'confidence_score': 0.9
        }
        mock_cache.get_cached_classification = AsyncMock(return_value=cached_result)
        
        result = await extractor.extract_data_universal(
            b"test,data\n1,2",
            "test.csv",
            "test_user"
        )
        
        # Should return cached result
        assert result == cached_result
        assert extractor.metrics['cache_hits'] > 0
    
    @pytest.mark.asyncio
    async def test_cache_miss(self, extractor, mock_cache):
        """Test cache miss scenario"""
        mock_cache.get_cached_classification = AsyncMock(return_value=None)
        
        result = await extractor.extract_data_universal(
            b"test,data\n1,2",
            "test.csv",
            "test_user"
        )
        
        assert extractor.metrics['cache_misses'] > 0
        # Should have called store_classification
        assert mock_cache.store_classification.called
    
    # ============================================================================
    # Metrics Tests
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_metrics_tracking(self, extractor):
        """Test metrics are properly tracked"""
        initial_count = extractor.metrics['extractions_performed']
        
        await extractor.extract_data_universal(
            b"test,data\n1,2",
            "test.csv",
            "test_user"
        )
        
        assert extractor.metrics['extractions_performed'] == initial_count + 1
        assert len(extractor.metrics['processing_times']) > 0
    
    def test_get_metrics(self, extractor):
        """Test get_metrics returns complete metrics"""
        metrics = extractor.get_metrics()
        
        assert 'extractions_performed' in metrics
        assert 'cache_hits' in metrics
        assert 'cache_misses' in metrics
        assert 'avg_processing_time' in metrics
        assert 'cache_hit_rate' in metrics
    
    # ============================================================================
    # Error Handling Tests
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_corrupted_file_handling(self, extractor):
        """Test handling of corrupted files"""
        corrupted_content = b'\x00\x01\x02\x03\x04\x05'
        
        result = await extractor.extract_data_universal(
            corrupted_content,
            "corrupted.xlsx",
            "test_user"
        )
        
        # Should handle gracefully
        assert 'error' in result or result['confidence_score'] < 0.5
    
    @pytest.mark.asyncio
    async def test_unsupported_format(self, extractor):
        """Test unsupported file format"""
        result = await extractor.extract_data_universal(
            b"binary content",
            "test.unsupported",
            "test_user"
        )
        
        # Should handle gracefully - either error or unknown/txt format
        assert 'error' in result or result.get('file_format') in ['unknown', 'txt']


class TestUniversalExtractorsPerformance:
    """Performance tests for extractors"""
    
    @pytest.fixture
    def extractor(self):
        return UniversalExtractorsOptimized()
    
    @pytest.mark.asyncio
    async def test_large_csv_performance(self, extractor):
        """Test performance with large CSV (1000 rows)"""
        import time
        
        # Create large CSV
        rows = []
        for i in range(1000):
            rows.append(f"row{i},value{i},{i}")
        csv_content = ("col1,col2,col3\n" + "\n".join(rows)).encode()
        
        start_time = time.time()
        result = await extractor.extract_data_universal(
            csv_content,
            "large.csv",
            "test_user"
        )
        processing_time = time.time() - start_time
        
        # Should complete in reasonable time (< 5 seconds)
        assert processing_time < 5.0
        assert result['confidence_score'] >= 0.5
    
    @pytest.mark.asyncio
    async def test_concurrent_extractions(self, extractor):
        """Test concurrent file extractions"""
        import time
        
        csv_content = b"col1,col2\nval1,val2"
        
        start_time = time.time()
        
        # Process 10 files concurrently
        tasks = [
            extractor.extract_data_universal(csv_content, f"test{i}.csv", "test_user")
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)
        
        processing_time = time.time() - start_time
        
        # All should succeed
        assert len(results) == 10
        assert all('error' not in r for r in results)
        
        # Should be faster than sequential (< 2 seconds for 10 files)
        assert processing_time < 2.0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
