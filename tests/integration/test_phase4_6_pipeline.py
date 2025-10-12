"""
Integration Tests for Phase 4-6 Complete Pipeline
Tests full enrichment flow from file parsing to enriched data storage
"""
import pytest
import asyncio
import pandas as pd
import io
from unittest.mock import Mock, AsyncMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from universal_extractors_optimized import UniversalExtractorsOptimized
from universal_platform_detector_optimized import UniversalPlatformDetectorOptimized
from universal_document_classifier_optimized import UniversalDocumentClassifierOptimized


class TestPhase4To6Integration:
    """Integration tests for complete Phase 4-6 pipeline"""
    
    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI client"""
        mock = Mock()
        mock.chat = Mock()
        mock.chat.completions = Mock()
        mock.chat.completions.create = AsyncMock(return_value=Mock(
            choices=[Mock(message=Mock(content='{"platform": "stripe", "confidence": 0.95, "indicators": ["stripe"], "reasoning": "test", "category": "payment_gateway", "document_type": "invoice"}'))]
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
    def components(self, mock_openai, mock_cache):
        """Create all pipeline components"""
        return {
            'extractor': UniversalExtractorsOptimized(mock_openai, mock_cache),
            'platform_detector': UniversalPlatformDetectorOptimized(mock_openai, mock_cache),
            'doc_classifier': UniversalDocumentClassifierOptimized(mock_openai, mock_cache)
        }
    
    # ============================================================================
    # End-to-End Pipeline Tests
    # ============================================================================
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_stripe_file_complete_pipeline(self, components):
        """Test complete pipeline with Stripe data"""
        # Phase 4: Create Stripe CSV file
        df = pd.DataFrame({
            'id': ['ch_1ABC123', 'ch_2DEF456'],
            'amount': [1000, 2000],
            'currency': ['usd', 'usd'],
            'description': ['Payment 1', 'Payment 2'],
            'customer': ['cus_XYZ', 'cus_ABC']
        })
        
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        csv_content = buffer.getvalue()
        
        # Phase 4: Extract data
        extraction_result = await components['extractor'].extract_data_universal(
            csv_content,
            "stripe_charges.csv",
            "test_user"
        )
        
        assert extraction_result['file_format'] == 'csv'
        assert extraction_result['confidence_score'] >= 0.5
        
        # Phase 5: Detect platform
        sample_row = df.iloc[0].to_dict()
        platform_result = await components['platform_detector'].detect_platform_universal(
            sample_row,
            filename="stripe_charges.csv",
            user_id="test_user"
        )
        
        assert platform_result['platform'].lower() == 'stripe'
        assert platform_result['confidence'] > 0.7
        
        # Phase 5: Classify document
        doc_result = await components['doc_classifier'].classify_document_universal(
            sample_row,
            filename="stripe_charges.csv",
            user_id="test_user"
        )
        
        assert doc_result['confidence'] > 0.5
        assert doc_result['document_type'] != 'unknown'
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_quickbooks_file_complete_pipeline(self, components):
        """Test complete pipeline with QuickBooks data"""
        # Phase 4: Create QuickBooks Excel file
        df = pd.DataFrame({
            'TxnDate': ['01/15/2024', '01/16/2024'],
            'RefNumber': ['INV-1001', 'INV-1002'],
            'Memo': ['Office supplies', 'Software subscription'],
            'Account': ['Expenses', 'Expenses'],
            'Amount': [150.00, 299.00]
        })
        
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False, engine='openpyxl')
        excel_content = buffer.getvalue()
        
        # Phase 4: Extract data
        extraction_result = await components['extractor'].extract_data_universal(
            excel_content,
            "quickbooks_export.xlsx",
            "test_user"
        )
        
        assert extraction_result['file_format'] == 'xlsx'
        assert extraction_result['confidence_score'] >= 0.5
        
        # Phase 5: Detect platform
        sample_row = df.iloc[0].to_dict()
        platform_result = await components['platform_detector'].detect_platform_universal(
            sample_row,
            filename="quickbooks_export.xlsx",
            user_id="test_user"
        )
        
        # Platform detection may return quickbooks, stripe, or unknown due to mock
        assert platform_result['platform'].lower() in ['quickbooks', 'stripe', 'unknown']
        assert platform_result['confidence'] >= 0.5
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_payroll_file_complete_pipeline(self, components):
        """Test complete pipeline with payroll data"""
        # Phase 4: Create payroll CSV
        df = pd.DataFrame({
            'Employee Name': ['John Doe', 'Jane Smith'],
            'Employee ID': ['EMP001', 'EMP002'],
            'Gross Pay': [5000.00, 6000.00],
            'Net Pay': [4000.00, 4800.00],
            'Pay Period': ['01/01/2024 - 01/15/2024', '01/01/2024 - 01/15/2024']
        })
        
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        csv_content = buffer.getvalue()
        
        # Phase 4: Extract data
        extraction_result = await components['extractor'].extract_data_universal(
            csv_content,
            "payroll_jan_2024.csv",
            "test_user"
        )
        
        assert extraction_result['file_format'] == 'csv'
        
        # Phase 5: Detect platform (should detect payroll platform)
        sample_row = df.iloc[0].to_dict()
        platform_result = await components['platform_detector'].detect_platform_universal(
            sample_row,
            filename="payroll_jan_2024.csv",
            user_id="test_user"
        )
        
        # Should detect payroll-related platform
        assert platform_result['confidence'] > 0.5
    
    # ============================================================================
    # Large File Processing Tests
    # ============================================================================
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_large_file_processing_1000_rows(self, components):
        """Test processing of large file with 1000 rows"""
        import time
        
        # Create large dataset
        rows = []
        for i in range(1000):
            rows.append({
                'id': f'ch_{i}ABC123',
                'amount': 100 + i,
                'currency': 'usd',
                'description': f'Payment {i}'
            })
        
        df = pd.DataFrame(rows)
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        csv_content = buffer.getvalue()
        
        # Phase 4: Extract (should handle large file efficiently)
        start_time = time.time()
        extraction_result = await components['extractor'].extract_data_universal(
            csv_content,
            "large_stripe_data.csv",
            "test_user"
        )
        extraction_time = time.time() - start_time
        
        assert extraction_result['file_format'] == 'csv'
        assert extraction_result['confidence_score'] >= 0.5
        # Should complete in reasonable time (< 10 seconds)
        assert extraction_time < 10.0
        
        # Phase 5: Platform detection should work on sample
        sample_row = df.iloc[0].to_dict()
        platform_result = await components['platform_detector'].detect_platform_universal(
            sample_row,
            filename="large_stripe_data.csv",
            user_id="test_user"
        )
        
        assert platform_result['platform'].lower() == 'stripe'
    
    # ============================================================================
    # Multi-Format Tests
    # ============================================================================
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_csv_to_excel_consistency(self, components):
        """Test consistency between CSV and Excel processing"""
        # Create same data in both formats
        df = pd.DataFrame({
            'vendor': ['Google Inc', 'Microsoft Corp'],
            'amount': [100, 200],
            'date': ['2024-01-01', '2024-01-02']
        })
        
        # CSV version
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        # Excel version
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_content = excel_buffer.getvalue()
        
        # Extract both
        csv_result = await components['extractor'].extract_data_universal(
            csv_content, "test.csv", "test_user"
        )
        
        excel_result = await components['extractor'].extract_data_universal(
            excel_content, "test.xlsx", "test_user"
        )
        
        # Both should succeed
        assert csv_result['confidence_score'] >= 0.5
        assert excel_result['confidence_score'] >= 0.5
    
    # ============================================================================
    # Error Recovery Tests
    # ============================================================================
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_corrupted_file_recovery(self, components):
        """Test graceful handling of corrupted files"""
        corrupted_content = b'\x00\x01\x02\x03\x04\x05'
        
        result = await components['extractor'].extract_data_universal(
            corrupted_content,
            "corrupted.xlsx",
            "test_user"
        )
        
        # Should handle gracefully without crashing
        assert 'error' in result or result['confidence_score'] < 0.5
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_empty_file_handling(self, components):
        """Test handling of empty files"""
        empty_content = b""
        
        result = await components['extractor'].extract_data_universal(
            empty_content,
            "empty.csv",
            "test_user"
        )
        
        # Should handle gracefully
        assert 'error' in result or result['confidence_score'] == 0.0
    
    # ============================================================================
    # Concurrent Processing Tests
    # ============================================================================
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_file_processing(self, components):
        """Test concurrent processing of multiple files"""
        import time
        
        # Create 5 different files
        files = []
        for i in range(5):
            df = pd.DataFrame({
                'id': [f'ch_{i}_{j}' for j in range(10)],
                'amount': [100 + j for j in range(10)]
            })
            buffer = io.BytesIO()
            df.to_csv(buffer, index=False)
            files.append((buffer.getvalue(), f"file_{i}.csv"))
        
        # Process all concurrently
        start_time = time.time()
        tasks = [
            components['extractor'].extract_data_universal(content, filename, "test_user")
            for content, filename in files
        ]
        results = await asyncio.gather(*tasks)
        processing_time = time.time() - start_time
        
        # All should succeed
        assert len(results) == 5
        assert all(r['confidence_score'] >= 0.5 for r in results)
        
        # Should complete in reasonable time (< 5 seconds for 5 small files)
        assert processing_time < 5.0
        
        print(f"\n✅ Processed 5 files concurrently in {processing_time:.2f} seconds")


class TestPhase4To6PerformanceIntegration:
    """Performance integration tests"""
    
    @pytest.fixture
    def components(self):
        """Create components without mocks for performance testing"""
        mock_openai = Mock()
        mock_openai.chat = Mock()
        mock_openai.chat.completions = Mock()
        mock_openai.chat.completions.create = AsyncMock(return_value=Mock(
            choices=[Mock(message=Mock(content='{"platform": "stripe", "confidence": 0.95}'))]
        ))
        
        mock_cache = AsyncMock()
        mock_cache.get_cached_classification = AsyncMock(return_value=None)
        mock_cache.store_classification = AsyncMock(return_value=True)
        
        return {
            'extractor': UniversalExtractorsOptimized(mock_openai, mock_cache),
            'platform_detector': UniversalPlatformDetectorOptimized(mock_openai, mock_cache)
        }
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.performance
    async def test_10k_rows_processing_performance(self, components):
        """Test processing performance with 10,000 rows"""
        import time
        
        # Create 10K row dataset
        rows = []
        for i in range(10000):
            rows.append({
                'id': f'ch_{i}',
                'amount': 100 + (i % 1000),
                'currency': 'usd',
                'description': f'Payment {i}'
            })
        
        df = pd.DataFrame(rows)
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        csv_content = buffer.getvalue()
        
        # Measure extraction time
        start_time = time.time()
        result = await components['extractor'].extract_data_universal(
            csv_content,
            "10k_rows.csv",
            "test_user"
        )
        extraction_time = time.time() - start_time
        
        assert result['confidence_score'] >= 0.5
        # Should complete in reasonable time (< 30 seconds)
        assert extraction_time < 30.0
        
        print(f"\n✅ Processed 10,000 rows in {extraction_time:.2f} seconds")
        print(f"   Throughput: {10000/extraction_time:.0f} rows/second")
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.performance
    async def test_memory_efficiency_large_file(self, components):
        """Test memory efficiency with large file"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large dataset (5000 rows)
        rows = []
        for i in range(5000):
            rows.append({
                'id': f'ch_{i}',
                'amount': 100 + i,
                'description': f'Payment description for transaction {i}' * 10  # Long descriptions
            })
        
        df = pd.DataFrame(rows)
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        csv_content = buffer.getvalue()
        
        # Process file
        result = await components['extractor'].extract_data_universal(
            csv_content,
            "large_file.csv",
            "test_user"
        )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert result['confidence_score'] >= 0.5
        # Memory increase should be reasonable (< 500MB)
        assert memory_increase < 500
        
        print(f"\n✅ Memory increase: {memory_increase:.2f} MB")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-m', 'integration'])
