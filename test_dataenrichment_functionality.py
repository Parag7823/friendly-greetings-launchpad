"""
Test actual DataEnrichmentProcessor functionality with real data
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime

def test_dataenrichment_processor_creation():
    """Test that DataEnrichmentProcessor can be created and initialized"""
    from fastapi_backend import DataEnrichmentProcessor
    
    # Mock dependencies
    with patch('fastapi_backend.create_client') as mock_client:
        mock_client.return_value = Mock()
        
        # Mock OpenAI client
        mock_openai = Mock()
        
        processor = DataEnrichmentProcessor(mock_openai)
        
        # Check initialization
        assert processor is not None
        assert hasattr(processor, 'enrich_row_data')
        assert hasattr(processor, 'enrich_batch_data')
        assert hasattr(processor, 'get_metrics')
        print("✅ DataEnrichmentProcessor created successfully")

@pytest.mark.asyncio
async def test_dataenrichment_processor_single_row():
    """Test DataEnrichmentProcessor with a single row of real data"""
    from fastapi_backend import DataEnrichmentProcessor
    
    # Mock dependencies
    with patch('fastapi_backend.create_client') as mock_client:
        mock_client.return_value = Mock()
        
        # Mock OpenAI client
        mock_openai = Mock()
        
        processor = DataEnrichmentProcessor(mock_openai)
        
        # Test data - realistic financial transaction
        test_row_data = {
            'amount': 1250.50,
            'date': '2024-01-15',
            'vendor': 'AMAZON.COM',
            'description': 'Amazon purchase for office supplies',
            'currency': 'USD'
        }
        
        file_context = {
            'user_id': 'test_user_123',
            'file_name': 'test_transactions.csv',
            'file_hash': 'test_hash_123'
        }
        
        # Mock the AI classification to avoid API calls
        with patch.object(processor, '_classify_platform_and_document') as mock_classify:
            mock_classify.return_value = {
                'platform': 'Amazon',
                'document_type': 'Receipt',
                'confidence': 0.95
            }
            
            # Mock vendor standardization
            with patch.object(processor, '_standardize_vendor_with_validation') as mock_vendor:
                mock_vendor.return_value = {
                    'standardized_name': 'Amazon.com Inc.',
                    'confidence': 0.9
                }
                
                # Mock platform ID extraction
                with patch.object(processor, '_extract_platform_ids_with_validation') as mock_ids:
                    mock_ids.return_value = {
                        'primary_id': 'AMZN-ORDER-12345',
                        'secondary_ids': ['amazon-order-12345'],
                        'total_ids_found': 1
                    }
                    
                    # Test enrichment
                    result = await processor.enrich_row_data(test_row_data, file_context)
                    
                    # Verify result structure
                    assert result is not None
                    assert 'enriched_data' in result
                    assert 'accuracy_enhancement' in result
                    assert 'processing_time' in result
                    
                    enriched_data = result['enriched_data']
                    assert 'amount_usd' in enriched_data
                    assert 'vendor_standard' in enriched_data
                    assert 'platform' in enriched_data
                    assert 'platform_ids' in enriched_data
                    
                    print("✅ DataEnrichmentProcessor single row test passed")

@pytest.mark.asyncio
async def test_dataenrichment_processor_batch_processing():
    """Test DataEnrichmentProcessor with batch data"""
    from fastapi_backend import DataEnrichmentProcessor
    
    # Mock dependencies
    with patch('fastapi_backend.create_client') as mock_client:
        mock_client.return_value = Mock()
        
        # Mock OpenAI client
        mock_openai = Mock()
        
        processor = DataEnrichmentProcessor(mock_openai)
        
        # Test batch data
        test_batch_data = [
            {
                'amount': 1250.50,
                'date': '2024-01-15',
                'vendor': 'AMAZON.COM',
                'description': 'Amazon purchase',
                'currency': 'USD'
            },
            {
                'amount': 89.99,
                'date': '2024-01-16',
                'vendor': 'STARBUCKS',
                'description': 'Coffee purchase',
                'currency': 'USD'
            }
        ]
        
        file_context = {
            'user_id': 'test_user_123',
            'file_name': 'test_transactions.csv',
            'file_hash': 'test_hash_123'
        }
        
        # Mock all the processing methods
        with patch.object(processor, '_classify_platform_and_document') as mock_classify, \
             patch.object(processor, '_standardize_vendor_with_validation') as mock_vendor, \
             patch.object(processor, '_extract_platform_ids_with_validation') as mock_ids:
            
            # Setup mocks
            mock_classify.return_value = {
                'platform': 'Test Platform',
                'document_type': 'Receipt',
                'confidence': 0.9
            }
            mock_vendor.return_value = {
                'standardized_name': 'Test Vendor Inc.',
                'confidence': 0.85
            }
            mock_ids.return_value = {
                'primary_id': 'TEST-ID-123',
                'secondary_ids': [],
                'total_ids_found': 1
            }
            
            # Test batch enrichment
            results = await processor.enrich_batch_data(test_batch_data, file_context)
            
            # Verify results
            assert results is not None
            assert 'enriched_batch' in results
            assert 'batch_metrics' in results
            assert 'processing_time' in results
            
            enriched_batch = results['enriched_batch']
            assert len(enriched_batch) == 2
            
            batch_metrics = results['batch_metrics']
            assert 'total_rows' in batch_metrics
            assert 'successful_enrichments' in batch_metrics
            assert batch_metrics['total_rows'] == 2
            
            print("✅ DataEnrichmentProcessor batch processing test passed")

def test_dataenrichment_processor_metrics():
    """Test DataEnrichmentProcessor metrics collection"""
    from fastapi_backend import DataEnrichmentProcessor
    
    # Mock dependencies
    with patch('fastapi_backend.create_client') as mock_client:
        mock_client.return_value = Mock()
        
        # Mock OpenAI client
        mock_openai = Mock()
        
        processor = DataEnrichmentProcessor(mock_openai)
        
        # Test metrics retrieval
        metrics = processor.get_metrics()
        
        # Verify metrics structure
        assert metrics is not None
        assert isinstance(metrics, dict)
        assert 'total_enrichments' in metrics
        assert 'successful_enrichments' in metrics
        assert 'failed_enrichments' in metrics
        assert 'average_processing_time' in metrics
        
        print("✅ DataEnrichmentProcessor metrics test passed")

def test_dataenrichment_processor_error_handling():
    """Test DataEnrichmentProcessor error handling with invalid data"""
    from fastapi_backend import DataEnrichmentProcessor
    
    # Mock dependencies
    with patch('fastapi_backend.create_client') as mock_client:
        mock_client.return_value = Mock()
        
        # Mock OpenAI client
        mock_openai = Mock()
        
        processor = DataEnrichmentProcessor(mock_openai)
        
        # Test with invalid data
        invalid_row_data = {
            'amount': 'invalid_amount',  # Invalid amount
            'date': 'invalid_date',      # Invalid date
            'vendor': '',                # Empty vendor
        }
        
        file_context = {
            'user_id': 'test_user_123',
            'file_name': 'test.csv'
        }
        
        # This should handle errors gracefully
        async def test_error_handling():
            try:
                result = await processor.enrich_row_data(invalid_row_data, file_context)
                # Should return a fallback result, not crash
                assert result is not None
                assert 'enriched_data' in result or 'error' in result
                print("✅ DataEnrichmentProcessor error handling test passed")
            except Exception as e:
                print(f"❌ DataEnrichmentProcessor error handling failed: {e}")
                raise
        
        # Run the async test
        asyncio.run(test_error_handling())

if __name__ == "__main__":
    # Run tests
    test_dataenrichment_processor_creation()
    asyncio.run(test_dataenrichment_processor_single_row())
    asyncio.run(test_dataenrichment_processor_batch_processing())
    test_dataenrichment_processor_metrics()
    test_dataenrichment_processor_error_handling()
    print("✅ All DataEnrichmentProcessor functionality tests passed!")
