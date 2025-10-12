"""
Integration Tests for Data Enrichment Pipeline
Tests the full enrichment flow from raw data to enriched events
"""
import pytest
import pandas as pd
from unittest.mock import Mock, AsyncMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from fastapi_backend import DataEnrichmentProcessor, VendorStandardizer, PlatformIDExtractor

class TestEnrichmentPipeline:
    """Integration tests for full enrichment pipeline"""
    
    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI client"""
        mock = Mock()
        mock.chat = Mock()
        mock.chat.completions = Mock()
        mock.chat.completions.create = Mock(return_value=Mock(
            choices=[Mock(message=Mock(content='{"standard_name": "Amazon", "confidence": 0.95, "reasoning": "test"}'))]
        ))
        return mock
    
    @pytest.fixture
    def enrichment_processor(self, mock_openai):
        """Create DataEnrichmentProcessor instance"""
        return DataEnrichmentProcessor(mock_openai)
    
    @pytest.mark.asyncio
    async def test_vendor_standardization_integration(self, enrichment_processor):
        """Test vendor standardization with various inputs"""
        test_cases = [
            ("Amazon Inc.", "Amazon"),
            ("Microsoft Corporation", "Microsoft"),
            ("Google LLC", "Google"),
            ("", ""),
        ]
        
        for raw_vendor, expected_standard in test_cases:
            result = await enrichment_processor.vendor_standardizer.standardize_vendor(raw_vendor)
            assert 'vendor_standard' in result
            assert result['confidence'] > 0
    
    @pytest.mark.asyncio
    async def test_platform_id_extraction_integration(self, enrichment_processor):
        """Test platform ID extraction for different platforms"""
        # Stripe charge ID
        stripe_row = pd.Series({
            'id': 'ch_1ABC123XYZ456',
            'amount': 1000,
            'description': 'Payment'
        })
        
        result = await enrichment_processor.platform_id_extractor.extract_platform_ids(
            stripe_row.to_dict(), 'stripe', list(stripe_row.index)
        )
        
        assert 'extracted_ids' in result
        assert result['total_ids_found'] > 0
    
    @pytest.mark.asyncio
    async def test_full_enrichment_flow(self, enrichment_processor):
        """Test complete enrichment pipeline"""
        # Sample row data
        row_data = {
            'date': '2024-01-15',
            'vendor': 'Amazon Inc.',
            'amount': 1500.50,
            'description': 'Office supplies purchase',
            'currency': 'USD'
        }
        
        platform_info = {'platform': 'quickbooks', 'confidence': 0.9}
        column_names = list(row_data.keys())
        ai_classification = {
            'row_type': 'expense',
            'category': 'office_supplies',
            'subcategory': 'general',
            'confidence': 0.85
        }
        file_context = {
            'filename': 'test.xlsx',
            'user_id': 'test_user',
            'file_id': 'test_file',
            'job_id': 'test_job'
        }
        
        # Run enrichment
        result = await enrichment_processor.enrich_row_data(
            row_data, platform_info, column_names, ai_classification, file_context
        )
        
        # Verify enriched fields
        assert 'vendor_standard' in result
        assert 'amount_usd' in result
        assert 'currency' in result
        assert 'kind' in result
        assert result['amount_usd'] == 1500.50
    
    @pytest.mark.asyncio
    async def test_currency_conversion_integration(self, enrichment_processor):
        """Test currency conversion with fallback rates"""
        row_data = {
            'amount': 100,
            'currency': 'EUR',
            'date': '2024-01-15'
        }
        
        platform_info = {'platform': 'stripe', 'confidence': 0.9}
        column_names = list(row_data.keys())
        ai_classification = {'row_type': 'revenue', 'category': 'sales', 'confidence': 0.9}
        file_context = {
            'filename': 'test.csv',
            'user_id': 'test_user',
            'file_id': 'test_file',
            'job_id': 'test_job'
        }
        
        result = await enrichment_processor.enrich_row_data(
            row_data, platform_info, column_names, ai_classification, file_context
        )
        
        # Should have converted EUR to USD
        assert 'amount_usd' in result
        assert 'exchange_rate' in result
        assert result['amount_usd'] != result['amount_original']
    
    @pytest.mark.asyncio
    async def test_batch_enrichment(self, enrichment_processor):
        """Test batch enrichment of multiple rows"""
        batch_data = [
            {'vendor': 'Amazon Inc.', 'amount': 100, 'date': '2024-01-01'},
            {'vendor': 'Microsoft Corp.', 'amount': 200, 'date': '2024-01-02'},
            {'vendor': 'Google LLC', 'amount': 300, 'date': '2024-01-03'},
        ]
        
        platform_info = {'platform': 'quickbooks', 'confidence': 0.9}
        column_names = ['vendor', 'amount', 'date']
        ai_classifications = [
            {'row_type': 'expense', 'category': 'software', 'confidence': 0.9}
            for _ in batch_data
        ]
        file_context = {
            'filename': 'batch_test.xlsx',
            'user_id': 'test_user',
            'file_id': 'test_file',
            'job_id': 'test_job'
        }
        
        results = await enrichment_processor.enrich_batch_data(
            batch_data, platform_info, column_names, ai_classifications, file_context
        )
        
        assert len(results) == 3
        for result in results:
            assert 'vendor_standard' in result
            assert 'amount_usd' in result

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
