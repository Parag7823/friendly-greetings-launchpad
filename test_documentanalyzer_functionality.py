"""
Test actual DocumentAnalyzer functionality with real documents
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime

def test_documentanalyzer_processor_creation():
    """Test that DocumentAnalyzer can be created and initialized"""
    from fastapi_backend import DocumentAnalyzer
    
    # Mock OpenAI client
    mock_openai = Mock()
    
    processor = DocumentAnalyzer(mock_openai)
    
    # Check initialization
    assert processor is not None
    assert hasattr(processor, 'detect_document_type')
    assert hasattr(processor, 'analyze_document_batch')
    assert hasattr(processor, 'get_metrics')
    print("✅ DocumentAnalyzer created successfully")

@pytest.mark.asyncio
async def test_documentanalyzer_detect_document_type():
    """Test DocumentAnalyzer with real document data"""
    from fastapi_backend import DocumentAnalyzer
    
    # Mock OpenAI client
    mock_openai = Mock()
    
    processor = DocumentAnalyzer(mock_openai)
    
    # Test data - realistic document features
    test_df_hash = "test_document_hash_123"
    test_filename = "financial_report_2024.xlsx"
    
    # Mock the AI classification to avoid API calls
    with patch.object(processor, '_classify_with_ai') as mock_ai:
        mock_ai.return_value = {
            'document_type': 'Financial Statement',
            'source_platform': 'Excel',
            'confidence': 0.92
        }
        
        # Mock pattern classification
        with patch.object(processor, '_classify_by_patterns') as mock_patterns:
            mock_patterns.return_value = {
                'document_type': 'Financial Statement',
                'source_platform': 'Excel',
                'confidence': 0.85
            }
            
            # Mock OCR analysis
            with patch.object(processor, '_analyze_with_ocr') as mock_ocr:
                mock_ocr.return_value = {
                    'document_type': 'Financial Statement',
                    'source_platform': 'Excel',
                    'confidence': 0.88
                }
                
                # Test document type detection
                result = await processor.detect_document_type(test_df_hash, test_filename)
                
                # Verify result structure
                assert result is not None
                assert 'document_type' in result
                assert 'source_platform' in result
                assert 'confidence' in result
                assert 'classification_methods' in result
                assert 'accuracy_enhancement' in result
                
                print("✅ DocumentAnalyzer document type detection test passed")

@pytest.mark.asyncio
async def test_documentanalyzer_batch_processing():
    """Test DocumentAnalyzer with batch document processing"""
    from fastapi_backend import DocumentAnalyzer
    
    # Mock OpenAI client
    mock_openai = Mock()
    
    processor = DocumentAnalyzer(mock_openai)
    
    # Test batch data
    test_documents = [
        {
            'df_hash': 'doc_hash_1',
            'filename': 'invoice_2024_01.pdf'
        },
        {
            'df_hash': 'doc_hash_2', 
            'filename': 'bank_statement_2024.xlsx'
        }
    ]
    
    # Mock all the processing methods
    with patch.object(processor, '_classify_with_ai') as mock_ai, \
         patch.object(processor, '_classify_by_patterns') as mock_patterns, \
         patch.object(processor, '_analyze_with_ocr') as mock_ocr:
        
        # Setup mocks
        mock_ai.return_value = {
            'document_type': 'Invoice',
            'source_platform': 'PDF',
            'confidence': 0.9
        }
        mock_patterns.return_value = {
            'document_type': 'Invoice',
            'source_platform': 'PDF',
            'confidence': 0.85
        }
        mock_ocr.return_value = {
            'document_type': 'Invoice',
            'source_platform': 'PDF',
            'confidence': 0.88
        }
        
        # Test batch analysis
        results = await processor.analyze_document_batch(test_documents)
        
        # Verify results
        assert results is not None
        assert 'analyzed_documents' in results
        assert 'batch_metrics' in results
        assert 'processing_time' in results
        
        analyzed_documents = results['analyzed_documents']
        assert len(analyzed_documents) == 2
        
        batch_metrics = results['batch_metrics']
        assert 'total_documents' in batch_metrics
        assert 'successful_analyses' in batch_metrics
        assert batch_metrics['total_documents'] == 2
        
        print("✅ DocumentAnalyzer batch processing test passed")

def test_documentanalyzer_metrics():
    """Test DocumentAnalyzer metrics collection"""
    from fastapi_backend import DocumentAnalyzer
    
    # Mock OpenAI client
    mock_openai = Mock()
    
    processor = DocumentAnalyzer(mock_openai)
    
    # Test metrics retrieval
    metrics = processor.get_metrics()
    
    # Verify metrics structure
    assert metrics is not None
    assert isinstance(metrics, dict)
    assert 'total_analyses' in metrics
    assert 'successful_analyses' in metrics
    assert 'failed_analyses' in metrics
    assert 'average_processing_time' in metrics
    
    print("✅ DocumentAnalyzer metrics test passed")

def test_documentanalyzer_error_handling():
    """Test DocumentAnalyzer error handling with invalid data"""
    from fastapi_backend import DocumentAnalyzer
    
    # Mock OpenAI client
    mock_openai = Mock()
    
    processor = DocumentAnalyzer(mock_openai)
    
    # Test with invalid data
    invalid_df_hash = ""  # Empty hash
    invalid_filename = None  # None filename
    
    # This should handle errors gracefully
    async def test_error_handling():
        try:
            result = await processor.detect_document_type(invalid_df_hash, invalid_filename)
            # Should return a fallback result, not crash
            assert result is not None
            assert 'document_type' in result or 'error' in result
            print("✅ DocumentAnalyzer error handling test passed")
        except Exception as e:
            print(f"❌ DocumentAnalyzer error handling failed: {e}")
            raise
    
    # Run the async test
    asyncio.run(test_error_handling())

@pytest.mark.asyncio
async def test_documentanalyzer_pattern_classification():
    """Test DocumentAnalyzer pattern-based classification"""
    from fastapi_backend import DocumentAnalyzer
    
    # Mock OpenAI client
    mock_openai = Mock()
    
    processor = DocumentAnalyzer(mock_openai)
    
    # Test document features
    document_features = {
        'date_columns': ['Date', 'Transaction Date'],
        'amount_columns': ['Amount', 'Total'],
        'vendor_columns': ['Vendor', 'Merchant'],
        'has_financial_keywords': True,
        'column_count': 8
    }
    
    # Test pattern classification
    result = processor._classify_by_patterns(document_features)
    
    # Verify result structure
    assert result is not None
    assert 'document_type' in result
    assert 'source_platform' in result
    assert 'confidence' in result
    assert isinstance(result['confidence'], float)
    assert 0.0 <= result['confidence'] <= 1.0
    
    print("✅ DocumentAnalyzer pattern classification test passed")

@pytest.mark.asyncio
async def test_documentanalyzer_feature_extraction():
    """Test DocumentAnalyzer document feature extraction"""
    from fastapi_backend import DocumentAnalyzer
    
    # Mock OpenAI client
    mock_openai = Mock()
    
    processor = DocumentAnalyzer(mock_openai)
    
    # Test with sample DataFrame
    import pandas as pd
    
    # Create sample financial data
    sample_data = {
        'Date': ['2024-01-15', '2024-01-16', '2024-01-17'],
        'Amount': [1250.50, 89.99, 250.00],
        'Vendor': ['Amazon', 'Starbucks', 'Office Depot'],
        'Description': ['Office supplies', 'Coffee', 'Stationery']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Test feature extraction
    features = processor._extract_document_features(df)
    
    # Verify features structure
    assert features is not None
    assert isinstance(features, dict)
    assert 'date_columns' in features
    assert 'amount_columns' in features
    assert 'vendor_columns' in features
    assert 'has_financial_keywords' in features
    assert 'column_count' in features
    
    # Check that features were detected correctly
    assert 'Date' in features['date_columns']
    assert 'Amount' in features['amount_columns']
    assert 'Vendor' in features['vendor_columns']
    assert features['column_count'] == 4
    
    print("✅ DocumentAnalyzer feature extraction test passed")

if __name__ == "__main__":
    # Run tests
    test_documentanalyzer_processor_creation()
    asyncio.run(test_documentanalyzer_detect_document_type())
    asyncio.run(test_documentanalyzer_batch_processing())
    test_documentanalyzer_metrics()
    test_documentanalyzer_error_handling()
    asyncio.run(test_documentanalyzer_pattern_classification())
    asyncio.run(test_documentanalyzer_feature_extraction())
    print("✅ All DocumentAnalyzer functionality tests passed!")
