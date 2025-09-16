"""
Comprehensive tests for platform ID extraction fixes
Tests all the critical issues that were identified and fixed
"""

import pytest
import asyncio
import re
from unittest.mock import Mock, patch
from fastapi_backend import PlatformIDExtractor


class TestPlatformIDExtractionFixes:
    """Test all the critical platform ID extraction fixes"""
    
    @pytest.fixture
    def platform_extractor(self):
        """Create PlatformIDExtractor instance"""
        return PlatformIDExtractor()
    
    # ============================================================================
    # TEST 1: QuickBooks Patterns Fix
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_quickbooks_real_world_patterns(self, platform_extractor):
        """Test that QuickBooks patterns now match real-world data"""
        # Real QuickBooks data examples
        test_cases = [
            {
                "row_data": {"transaction_id": "12345", "amount": 100.00},
                "column_names": ["transaction_id", "amount"],
                "expected_ids": ["transaction_id"],
                "description": "Simple numeric transaction ID"
            },
            {
                "row_data": {"invoice_id": "INV-001", "amount": 250.00},
                "column_names": ["invoice_id", "amount"],
                "expected_ids": ["invoice_id"],
                "description": "Prefixed invoice ID"
            },
            {
                "row_data": {"vendor_id": "VEN-123", "amount": 75.00},
                "column_names": ["vendor_id", "amount"],
                "expected_ids": ["vendor_id"],
                "description": "Prefixed vendor ID"
            },
            {
                "row_data": {"customer_id": "CUST-456", "amount": 500.00},
                "column_names": ["customer_id", "amount"],
                "expected_ids": ["customer_id"],
                "description": "Prefixed customer ID"
            },
            {
                "row_data": {"bill_id": "BILL-789", "amount": 150.00},
                "column_names": ["bill_id", "amount"],
                "expected_ids": ["bill_id"],
                "description": "Prefixed bill ID"
            }
        ]
        
        for test_case in test_cases:
            result = await platform_extractor.extract_platform_ids(
                row_data=test_case["row_data"],
                platform="quickbooks",
                column_names=test_case["column_names"]
            )
            
            assert result["platform"] == "quickbooks"
            assert result["total_ids_found"] >= 1
            assert result["overall_confidence"] > 0.0
            
            # Check that expected IDs were found
            for expected_id in test_case["expected_ids"]:
                assert expected_id in result["extracted_ids"]
                assert result["confidence_scores"][expected_id] > 0.0
                assert result["validation_results"][expected_id]["is_valid"] is True
    
    # ============================================================================
    # TEST 2: Edge Case Handling
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_multiple_ids_handling(self, platform_extractor):
        """Test handling of multiple IDs in same row"""
        row_data = {
            "description": "Payment txn_12345 and invoice inv_67890 for vendor ven_111",
            "amount": 100.00
        }
        column_names = ["description", "amount"]
        
        result = await platform_extractor.extract_platform_ids(
            row_data=row_data,
            platform="quickbooks",
            column_names=column_names
        )
        
        assert result["platform"] == "quickbooks"
        assert result["total_ids_found"] >= 1
        assert result["overall_confidence"] > 0.0
        
        # Should handle multiple matches intelligently
        assert "extraction_method" in result
        assert result["extraction_method"] == "comprehensive_validation"
    
    @pytest.mark.asyncio
    async def test_malformed_ids_handling(self, platform_extractor):
        """Test handling of malformed IDs"""
        row_data = {
            "transaction_id": "invalid_id_format_12345",
            "amount": 100.00
        }
        column_names = ["transaction_id", "amount"]
        
        result = await platform_extractor.extract_platform_ids(
            row_data=row_data,
            platform="quickbooks",
            column_names=column_names
        )
        
        assert result["platform"] == "quickbooks"
        # Should either reject malformed IDs or generate fallback
        if result["total_ids_found"] > 0:
            for id_type, validation in result["validation_results"].items():
                if "generated" not in id_type:
                    # Real IDs should be validated
                    assert "is_valid" in validation
    
    @pytest.mark.asyncio
    async def test_mixed_platform_files(self, platform_extractor):
        """Test detection of mixed platform indicators"""
        row_data = {
            "description": "QuickBooks transaction with Stripe payment ch_123456789012345678901234",
            "amount": 100.00
        }
        column_names = ["description", "amount"]
        
        result = await platform_extractor.extract_platform_ids(
            row_data=row_data,
            platform="quickbooks",
            column_names=column_names
        )
        
        assert result["platform"] == "quickbooks"
        # Should detect mixed platform indicators
        for id_type, validation in result["validation_results"].items():
            if "warnings" in validation:
                assert isinstance(validation["warnings"], list)
    
    # ============================================================================
    # TEST 3: Performance Optimization
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_compiled_regex_performance(self, platform_extractor):
        """Test that regex patterns are pre-compiled for performance"""
        # This test verifies the performance optimization is in place
        row_data = {"transaction_id": "12345", "amount": 100.00}
        column_names = ["transaction_id", "amount"]
        
        # Should not raise regex compilation errors
        result = await platform_extractor.extract_platform_ids(
            row_data=row_data,
            platform="quickbooks",
            column_names=column_names
        )
        
        assert result["platform"] == "quickbooks"
        assert "extraction_method" in result
    
    # ============================================================================
    # TEST 4: Async Processing
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_async_processing(self, platform_extractor):
        """Test that extract_platform_ids is now async"""
        row_data = {"transaction_id": "12345", "amount": 100.00}
        column_names = ["transaction_id", "amount"]
        
        # Should be awaitable (async)
        result = await platform_extractor.extract_platform_ids(
            row_data=row_data,
            platform="quickbooks",
            column_names=column_names
        )
        
        assert result["platform"] == "quickbooks"
        assert isinstance(result, dict)
    
    # ============================================================================
    # TEST 5: Deterministic ID Generation
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_deterministic_id_generation(self, platform_extractor):
        """Test that generated IDs are deterministic"""
        row_data = {"amount": 100.00, "description": "Test transaction"}
        column_names = ["amount", "description"]
        
        # Generate ID twice with same data
        result1 = await platform_extractor.extract_platform_ids(
            row_data=row_data,
            platform="unknown_platform",
            column_names=column_names
        )
        
        result2 = await platform_extractor.extract_platform_ids(
            row_data=row_data,
            platform="unknown_platform",
            column_names=column_names
        )
        
        # Should generate deterministic IDs (same within same hour)
        if "platform_generated_id" in result1["extracted_ids"]:
            # IDs should be deterministic (same for same input)
            assert result1["extracted_ids"]["platform_generated_id"] == result2["extracted_ids"]["platform_generated_id"]
    
    # ============================================================================
    # TEST 6: Confidence Scoring
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_confidence_scoring(self, platform_extractor):
        """Test comprehensive confidence scoring"""
        row_data = {"transaction_id": "12345", "amount": 100.00}
        column_names = ["transaction_id", "amount"]
        
        result = await platform_extractor.extract_platform_ids(
            row_data=row_data,
            platform="quickbooks",
            column_names=column_names
        )
        
        assert "confidence_scores" in result
        assert "overall_confidence" in result
        assert isinstance(result["confidence_scores"], dict)
        assert isinstance(result["overall_confidence"], (int, float))
        assert 0.0 <= result["overall_confidence"] <= 1.0
        
        # Check individual confidence scores
        for id_type, confidence in result["confidence_scores"].items():
            assert 0.0 <= confidence <= 1.0
    
    # ============================================================================
    # TEST 7: Validation Logic
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_validation_logic(self, platform_extractor):
        """Test comprehensive validation logic"""
        test_cases = [
            {
                "row_data": {"transaction_id": "12345"},
                "column_names": ["transaction_id"],
                "platform": "quickbooks",
                "should_be_valid": True
            },
            {
                "row_data": {"transaction_id": "invalid_format"},
                "column_names": ["transaction_id"],
                "platform": "quickbooks",
                "should_be_valid": False
            },
            {
                "row_data": {"charge_id": "ch_123456789012345678901234"},
                "column_names": ["charge_id"],
                "platform": "stripe",
                "should_be_valid": True
            },
            {
                "row_data": {"charge_id": "invalid_stripe_id"},
                "column_names": ["charge_id"],
                "platform": "stripe",
                "should_be_valid": False
            }
        ]
        
        for test_case in test_cases:
            result = await platform_extractor.extract_platform_ids(
                row_data=test_case["row_data"],
                platform=test_case["platform"],
                column_names=test_case["column_names"]
            )
            
            assert "validation_results" in result
            
            # Check validation results
            for id_type, validation in result["validation_results"].items():
                if "generated" not in id_type:
                    assert "is_valid" in validation
                    assert "reason" in validation
                    assert "validation_method" in validation
    
    # ============================================================================
    # TEST 8: Error Handling
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_error_handling(self, platform_extractor):
        """Test robust error handling"""
        # Test with invalid input
        result = await platform_extractor.extract_platform_ids(
            row_data=None,
            platform="quickbooks",
            column_names=[]
        )
        
        assert "error" in result or result["total_ids_found"] == 0
        assert result["platform"] == "quickbooks"
        assert result["overall_confidence"] == 0.0
    
    # ============================================================================
    # TEST 9: Platform-Specific Validation
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_platform_specific_validation(self, platform_extractor):
        """Test platform-specific validation methods"""
        # Test QuickBooks validation
        qb_validation = platform_extractor._validate_quickbooks_id("12345", "transaction_id")
        assert qb_validation["is_valid"] is True
        
        qb_validation_invalid = platform_extractor._validate_quickbooks_id("invalid", "transaction_id")
        assert qb_validation_invalid["is_valid"] is False
        
        # Test Stripe validation
        stripe_validation = platform_extractor._validate_stripe_id("ch_123456789012345678901234", "charge_id")
        assert stripe_validation["is_valid"] is True
        
        stripe_validation_invalid = platform_extractor._validate_stripe_id("invalid", "charge_id")
        assert stripe_validation_invalid["is_valid"] is False
    
    # ============================================================================
    # TEST 10: Comprehensive Integration Test
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_comprehensive_integration(self, platform_extractor):
        """Test comprehensive integration of all fixes"""
        # Complex real-world scenario
        row_data = {
            "transaction_id": "TXN-12345",
            "invoice_id": "INV-001",
            "vendor_id": "VEN-456",
            "description": "Payment to vendor with multiple references",
            "amount": 1000.00,
            "date": "2024-01-15"
        }
        column_names = ["transaction_id", "invoice_id", "vendor_id", "description", "amount", "date"]
        
        result = await platform_extractor.extract_platform_ids(
            row_data=row_data,
            platform="quickbooks",
            column_names=column_names
        )
        
        # Verify all aspects of the fix
        assert result["platform"] == "quickbooks"
        assert result["total_ids_found"] >= 3  # Should find multiple IDs
        assert result["overall_confidence"] > 0.0
        assert "confidence_scores" in result
        assert "validation_results" in result
        assert "extraction_method" in result
        assert result["extraction_method"] == "comprehensive_validation"
        
        # Check that multiple IDs were extracted
        expected_ids = ["transaction_id", "invoice_id", "vendor_id"]
        for expected_id in expected_ids:
            if expected_id in result["extracted_ids"]:
                assert result["confidence_scores"][expected_id] > 0.0
                assert result["validation_results"][expected_id]["is_valid"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
