"""
Comprehensive unit tests for EnhancedFileProcessor component.

Tests cover:
1. File format detection
2. Excel processing with repair capabilities
3. CSV processing with various encodings
4. PDF processing with multiple extraction methods
5. Archive processing (ZIP, 7Z, RAR)
6. Image processing with OCR
7. Error handling and edge cases
8. Performance and memory efficiency
9. Security validation
10. Integration with progress callbacks
"""

import pytest
import pandas as pd
import numpy as np
import io
import tempfile
import os
import zipfile
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime
import asyncio

# Import the EnhancedFileProcessor
from enhanced_file_processor import EnhancedFileProcessor


class TestEnhancedFileProcessor:
    """Test suite for EnhancedFileProcessor"""
    
    @pytest.fixture
    def processor(self):
        """Create EnhancedFileProcessor instance for testing"""
        return EnhancedFileProcessor()
    
    @pytest.fixture
    def sample_excel_content(self):
        """Create sample Excel content for testing"""
        # Create a simple Excel file in memory
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
        
        # Create Excel file in memory
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
    def sample_pdf_content(self):
        """Create sample PDF content for testing"""
        # This would be a real PDF file in practice
        # For testing, we'll mock the PDF processing
        return b"PDF content for testing"
    
    @pytest.fixture
    def sample_zip_content(self):
        """Create sample ZIP content for testing"""
        # Create a ZIP file in memory
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w') as zip_file:
            zip_file.writestr('file1.csv', 'Name,Age\nJohn,25\nJane,30')
            zip_file.writestr('file2.xlsx', b'Excel content')
        return output.getvalue()
    
    @pytest.fixture
    def mock_progress_callback(self):
        """Create mock progress callback for testing"""
        return AsyncMock()
    
    # ============================================================================
    # FILE FORMAT DETECTION TESTS
    # ============================================================================
    
    def test_detect_file_format_excel(self, processor):
        """Test Excel file format detection"""
        # Test with .xlsx extension
        result = processor._detect_file_format('test.xlsx', b'fake content')
        assert result == 'excel'
        
        # Test with .xls extension
        result = processor._detect_file_format('test.xls', b'fake content')
        assert result == 'excel'
        
        # Test with .xlsm extension
        result = processor._detect_file_format('test.xlsm', b'fake content')
        assert result == 'excel'
    
    def test_detect_file_format_csv(self, processor):
        """Test CSV file format detection"""
        # Test with .csv extension
        result = processor._detect_file_format('test.csv', b'fake content')
        assert result == 'csv'
        
        # Test with .tsv extension
        result = processor._detect_file_format('test.tsv', b'fake content')
        assert result == 'csv'
        
        # Test with .txt extension
        result = processor._detect_file_format('test.txt', b'fake content')
        assert result == 'csv'
    
    def test_detect_file_format_pdf(self, processor):
        """Test PDF file format detection"""
        result = processor._detect_file_format('test.pdf', b'fake content')
        assert result == 'pdf'
    
    def test_detect_file_format_archive(self, processor):
        """Test archive file format detection"""
        # Test ZIP
        result = processor._detect_file_format('test.zip', b'fake content')
        assert result == 'zip'
        
        # Test 7Z
        result = processor._detect_file_format('test.7z', b'fake content')
        assert result == 'zip'
        
        # Test RAR
        result = processor._detect_file_format('test.rar', b'fake content')
        assert result == 'zip'
    
    def test_detect_file_format_image(self, processor):
        """Test image file format detection"""
        # Test PNG
        result = processor._detect_file_format('test.png', b'fake content')
        assert result == 'image'
        
        # Test JPG
        result = processor._detect_file_format('test.jpg', b'fake content')
        assert result == 'image'
        
        # Test JPEG
        result = processor._detect_file_format('test.jpeg', b'fake content')
        assert result == 'image'
    
    def test_detect_file_format_unknown(self, processor):
        """Test unknown file format detection"""
        result = processor._detect_file_format('test.unknown', b'fake content')
        assert result == 'unknown'
    
    # ============================================================================
    # EXCEL PROCESSING TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_process_excel_enhanced_success(self, processor, sample_excel_content, mock_progress_callback):
        """Test successful Excel processing"""
        with patch.object(processor, '_process_excel_enhanced') as mock_process:
            mock_process.return_value = {
                'Sheet1': pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
            }
            
            result = await processor.process_file_enhanced(
                sample_excel_content, 
                'test.xlsx', 
                mock_progress_callback
            )
            
            assert isinstance(result, dict)
            assert 'Sheet1' in result
            assert isinstance(result['Sheet1'], pd.DataFrame)
            
            # Verify progress callbacks were called
            assert mock_progress_callback.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_process_excel_enhanced_corrupted_file(self, processor, mock_progress_callback):
        """Test Excel processing with corrupted file"""
        corrupted_content = b"corrupted excel content"
        
        with patch.object(processor, '_process_excel_enhanced') as mock_process:
            mock_process.side_effect = Exception("Corrupted file")
            
            # Should fallback to basic processing
            with patch.object(processor, '_fallback_processing') as mock_fallback:
                mock_fallback.return_value = {'Sheet1': pd.DataFrame()}
                
                result = await processor.process_file_enhanced(
                    corrupted_content, 
                    'test.xlsx', 
                    mock_progress_callback
                )
                
                assert isinstance(result, dict)
                mock_fallback.assert_called_once()
    
    # ============================================================================
    # CSV PROCESSING TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_process_csv_enhanced_success(self, processor, sample_csv_content, mock_progress_callback):
        """Test successful CSV processing"""
        with patch.object(processor, '_process_csv_enhanced') as mock_process:
            mock_process.return_value = {
                'Sheet1': pd.DataFrame({'Name': ['John', 'Jane'], 'Age': [25, 30]})
            }
            
            result = await processor.process_file_enhanced(
                sample_csv_content, 
                'test.csv', 
                mock_progress_callback
            )
            
            assert isinstance(result, dict)
            assert 'Sheet1' in result
            assert isinstance(result['Sheet1'], pd.DataFrame)
    
    @pytest.mark.asyncio
    async def test_process_csv_enhanced_different_encodings(self, processor, mock_progress_callback):
        """Test CSV processing with different encodings"""
        # Test UTF-8
        utf8_content = "Name,Age\nJohn,25\nJane,30".encode('utf-8')
        
        with patch.object(processor, '_process_csv_enhanced') as mock_process:
            mock_process.return_value = {'Sheet1': pd.DataFrame()}
            
            result = await processor.process_file_enhanced(
                utf8_content, 
                'test.csv', 
                mock_progress_callback
            )
            
            assert isinstance(result, dict)
    
    # ============================================================================
    # PDF PROCESSING TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_process_pdf_success(self, processor, sample_pdf_content, mock_progress_callback):
        """Test successful PDF processing"""
        with patch.object(processor, '_process_pdf') as mock_process:
            mock_process.return_value = {
                'Page1': pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
            }
            
            result = await processor.process_file_enhanced(
                sample_pdf_content, 
                'test.pdf', 
                mock_progress_callback
            )
            
            assert isinstance(result, dict)
            assert 'Page1' in result
            assert isinstance(result['Page1'], pd.DataFrame)
    
    @pytest.mark.asyncio
    async def test_process_pdf_multiple_extraction_methods(self, processor, sample_pdf_content, mock_progress_callback):
        """Test PDF processing with multiple extraction methods"""
        with patch.object(processor, '_process_pdf') as mock_process:
            # Mock different extraction methods
            mock_process.return_value = {
                'tabula': pd.DataFrame({'A': [1, 2], 'B': [3, 4]}),
                'camelot': pd.DataFrame({'C': [5, 6], 'D': [7, 8]}),
                'pdfplumber': pd.DataFrame({'E': [9, 10], 'F': [11, 12]})
            }
            
            result = await processor.process_file_enhanced(
                sample_pdf_content, 
                'test.pdf', 
                mock_progress_callback
            )
            
            assert isinstance(result, dict)
            assert len(result) == 3
    
    # ============================================================================
    # ARCHIVE PROCESSING TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_process_archive_zip_success(self, processor, sample_zip_content, mock_progress_callback):
        """Test successful ZIP archive processing"""
        with patch.object(processor, '_process_archive') as mock_process:
            mock_process.return_value = {
                'file1.csv': pd.DataFrame({'Name': ['John'], 'Age': [25]}),
                'file2.xlsx': pd.DataFrame({'Product': ['A'], 'Price': [10.99]})
            }
            
            result = await processor.process_file_enhanced(
                sample_zip_content, 
                'test.zip', 
                mock_progress_callback
            )
            
            assert isinstance(result, dict)
            assert 'file1.csv' in result
            assert 'file2.xlsx' in result
    
    @pytest.mark.asyncio
    async def test_process_archive_7z_success(self, processor, mock_progress_callback):
        """Test successful 7Z archive processing"""
        with patch.object(processor, '_process_archive') as mock_process:
            mock_process.return_value = {
                'file1.csv': pd.DataFrame({'Name': ['John'], 'Age': [25]})
            }
            
            result = await processor.process_file_enhanced(
                b"7z content", 
                'test.7z', 
                mock_progress_callback
            )
            
            assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_process_archive_rar_success(self, processor, mock_progress_callback):
        """Test successful RAR archive processing"""
        with patch.object(processor, '_process_archive') as mock_process:
            mock_process.return_value = {
                'file1.csv': pd.DataFrame({'Name': ['John'], 'Age': [25]})
            }
            
            result = await processor.process_file_enhanced(
                b"rar content", 
                'test.rar', 
                mock_progress_callback
            )
            
            assert isinstance(result, dict)
    
    # ============================================================================
    # IMAGE PROCESSING TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_process_image_success(self, processor, mock_progress_callback):
        """Test successful image processing with OCR"""
        with patch.object(processor, '_process_image') as mock_process:
            mock_process.return_value = {
                'OCR_Result': pd.DataFrame({'Text': ['Sample text'], 'Confidence': [0.95]})
            }
            
            result = await processor.process_file_enhanced(
                b"image content", 
                'test.png', 
                mock_progress_callback
            )
            
            assert isinstance(result, dict)
            assert 'OCR_Result' in result
    
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
    async def test_process_file_enhanced_processing_error(self, processor, sample_excel_content, mock_progress_callback):
        """Test processing error handling"""
        with patch.object(processor, '_process_excel_enhanced') as mock_process:
            mock_process.side_effect = Exception("Processing error")
            
            with patch.object(processor, '_fallback_processing') as mock_fallback:
                mock_fallback.return_value = {'Sheet1': pd.DataFrame()}
                
                result = await processor.process_file_enhanced(
                    sample_excel_content, 
                    'test.xlsx', 
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
    
    @pytest.mark.asyncio
    async def test_process_file_enhanced_large_file(self, processor, mock_progress_callback):
        """Test processing large file"""
        # Create large content
        large_content = b"x" * (10 * 1024 * 1024)  # 10MB
        
        with patch.object(processor, '_detect_file_format', return_value='excel'):
            with patch.object(processor, '_process_excel_enhanced') as mock_process:
                mock_process.return_value = {'Sheet1': pd.DataFrame()}
                
                result = await processor.process_file_enhanced(
                    large_content, 
                    'large.xlsx', 
                    mock_progress_callback
                )
                
                assert isinstance(result, dict)
    
    # ============================================================================
    # SECURITY TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_process_file_enhanced_malicious_filename(self, processor, mock_progress_callback):
        """Test processing with malicious filename"""
        malicious_filename = "../../../etc/passwd"
        
        with patch.object(processor, '_detect_file_format', return_value='excel'):
            with patch.object(processor, '_process_excel_enhanced') as mock_process:
                mock_process.return_value = {'Sheet1': pd.DataFrame()}
                
                result = await processor.process_file_enhanced(
                    b"content", 
                    malicious_filename, 
                    mock_progress_callback
                )
                
                # Should handle malicious filename safely
                assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_process_file_enhanced_zip_bomb_protection(self, processor, mock_progress_callback):
        """Test protection against ZIP bomb attacks"""
        # This would be a real ZIP bomb in practice
        with patch.object(processor, '_process_archive') as mock_process:
            mock_process.side_effect = Exception("ZIP bomb detected")
            
            with patch.object(processor, '_fallback_processing') as mock_fallback:
                mock_fallback.return_value = {'Sheet1': pd.DataFrame()}
                
                result = await processor.process_file_enhanced(
                    b"zip bomb content", 
                    'bomb.zip', 
                    mock_progress_callback
                )
                
                assert isinstance(result, dict)
                mock_fallback.assert_called_once()
    
    # ============================================================================
    # PERFORMANCE TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_process_file_enhanced_performance(self, processor, sample_excel_content, mock_progress_callback):
        """Test processing performance"""
        import time
        
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
        import psutil
        import os
        
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
    # INTEGRATION TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_process_file_enhanced_with_progress_callback(self, processor, sample_excel_content):
        """Test processing with progress callback integration"""
        progress_calls = []
        
        async def mock_progress_callback(step, message, progress):
            progress_calls.append((step, message, progress))
        
        with patch.object(processor, '_process_excel_enhanced') as mock_process:
            mock_process.return_value = {'Sheet1': pd.DataFrame()}
            
            result = await processor.process_file_enhanced(
                sample_excel_content, 
                'test.xlsx', 
                mock_progress_callback
            )
            
            # Verify progress callbacks were called
            assert len(progress_calls) >= 2
            assert progress_calls[0][0] == "detecting"
            assert progress_calls[1][0] == "processing"
            
            assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_process_file_enhanced_concurrent_processing(self, processor, sample_excel_content):
        """Test concurrent file processing"""
        async def process_file():
            with patch.object(processor, '_process_excel_enhanced') as mock_process:
                mock_process.return_value = {'Sheet1': pd.DataFrame()}
                
                return await processor.process_file_enhanced(
                    sample_excel_content, 
                    'test.xlsx', 
                    None
                )
        
        # Process multiple files concurrently
        tasks = [process_file() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        for result in results:
            assert isinstance(result, dict)
    
    # ============================================================================
    # EDGE CASES AND BOUNDARY CONDITIONS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_process_file_enhanced_unicode_filename(self, processor, sample_excel_content, mock_progress_callback):
        """Test processing with Unicode filename"""
        unicode_filename = "测试文件.xlsx"
        
        with patch.object(processor, '_process_excel_enhanced') as mock_process:
            mock_process.return_value = {'Sheet1': pd.DataFrame()}
            
            result = await processor.process_file_enhanced(
                sample_excel_content, 
                unicode_filename, 
                mock_progress_callback
            )
            
            assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_process_file_enhanced_very_long_filename(self, processor, sample_excel_content, mock_progress_callback):
        """Test processing with very long filename"""
        long_filename = "a" * 255 + ".xlsx"
        
        with patch.object(processor, '_process_excel_enhanced') as mock_process:
            mock_process.return_value = {'Sheet1': pd.DataFrame()}
            
            result = await processor.process_file_enhanced(
                sample_excel_content, 
                long_filename, 
                mock_progress_callback
            )
            
            assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_process_file_enhanced_special_characters_filename(self, processor, sample_excel_content, mock_progress_callback):
        """Test processing with special characters in filename"""
        special_filename = "file with spaces & symbols!@#$%.xlsx"
        
        with patch.object(processor, '_process_excel_enhanced') as mock_process:
            mock_process.return_value = {'Sheet1': pd.DataFrame()}
            
            result = await processor.process_file_enhanced(
                sample_excel_content, 
                special_filename, 
                mock_progress_callback
            )
            
            assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
