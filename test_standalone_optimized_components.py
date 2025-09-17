#!/usr/bin/env python3
"""
Standalone test suite for optimized components without dependencies on fastapi_backend.py
Tests UniversalExtractorsOptimized, UniversalPlatformDetectorOptimized, 
UniversalDocumentClassifierOptimized, and EntityResolverOptimized
"""

import asyncio
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from typing import Dict, Any, List
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock OpenAI client for testing
class MockOpenAIClient:
    def __init__(self):
        self.api_key = "test-key"
    
    @property
    def chat(self):
        return MockChat()
    
    @property
    def completions(self):
        return MockCompletions()

class MockChat:
    async def create(self, **kwargs):
        return MockResponse()
    
    @property
    def completions(self):
        return MockCompletions()

class MockCompletions:
    async def create(self, **kwargs):
        return MockResponse()

class MockResponse:
    def __init__(self):
        self.choices = [MockChoice()]

class MockChoice:
    def __init__(self):
        self.message = MockMessage()

class MockMessage:
    def __init__(self):
        self.content = json.dumps({
            "platform": "Test Platform",
            "confidence": 0.95,
            "reasoning": "Test classification"
        })

# Mock cache client
class MockCacheClient:
    def __init__(self):
        self.cache = {}
    
    async def get(self, key: str):
        return self.cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        self.cache[key] = value
        return True
    
    async def delete(self, key: str):
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    async def clear(self):
        self.cache.clear()
        return True

# Test data
SAMPLE_CSV_DATA = b"""Date,Amount,Description,Vendor
2024-01-15,150.00,Grocery Shopping,Whole Foods
2024-01-16,75.50,Gas Station,Shell
2024-01-17,200.00,Restaurant,McDonald's"""

SAMPLE_JSON_DATA = b"""{
    "transactions": [
        {"date": "2024-01-15", "amount": 150.00, "description": "Grocery Shopping", "vendor": "Whole Foods"},
        {"date": "2024-01-16", "amount": 75.50, "description": "Gas Station", "vendor": "Shell"}
    ]
}"""

SAMPLE_ROW_DATA = {
    "date": "2024-01-15",
    "amount": 150.00,
    "description": "Grocery Shopping",
    "vendor": "Whole Foods"
}

async def test_universal_platform_detector():
    """Test UniversalPlatformDetectorOptimized"""
    logger.info("🧪 Testing UniversalPlatformDetectorOptimized...")
    
    try:
        # Import with mock OpenAI client
        os.environ['OPENAI_API_KEY'] = 'test-key'
        from universal_platform_detector_optimized import UniversalPlatformDetectorOptimized
        
        # Initialize with mock client
        mock_openai = MockOpenAIClient()
        detector = UniversalPlatformDetectorOptimized(openai_client=mock_openai)
        
        # Test platform detection
        payload = {"file_content": SAMPLE_CSV_DATA, "filename": "transactions.csv"}
        result = await detector.detect_platform_universal(
            payload=payload,
            filename="transactions.csv",
            user_id="test-user"
        )
        
        assert result is not None
        assert 'platform' in result
        assert 'confidence' in result
        
        logger.info(f"✅ Platform detection result: {result}")
        return True
        
    except Exception as e:
        logger.error(f"❌ UniversalPlatformDetectorOptimized test failed: {e}")
        return False

async def test_universal_document_classifier():
    """Test UniversalDocumentClassifierOptimized"""
    logger.info("🧪 Testing UniversalDocumentClassifierOptimized...")
    
    try:
        from universal_document_classifier_optimized import UniversalDocumentClassifierOptimized
        
        # Initialize with mock client
        mock_openai = MockOpenAIClient()
        classifier = UniversalDocumentClassifierOptimized(openai_client=mock_openai)
        
        # Test document classification
        payload = {"file_content": SAMPLE_CSV_DATA, "filename": "transactions.csv"}
        result = await classifier.classify_document_universal(
            payload=payload,
            filename="transactions.csv",
            user_id="test-user"
        )
        
        assert result is not None
        assert 'document_type' in result
        assert 'confidence' in result
        
        logger.info(f"✅ Document classification result: {result}")
        return True
        
    except Exception as e:
        logger.error(f"❌ UniversalDocumentClassifierOptimized test failed: {e}")
        return False

async def test_entity_resolver():
    """Test EntityResolverOptimized"""
    logger.info("🧪 Testing EntityResolverOptimized...")
    
    try:
        from entity_resolver_optimized import EntityResolverOptimized
        
        # Initialize
        resolver = EntityResolverOptimized()
        
        # Test entity resolution
        entities = {
            "vendor": ["Whole Foods", "WHOLE FOODS MARKET", "Shell"]
        }
        
        result = await resolver.resolve_entities_batch(
            entities=entities,
            platform="csv",
            user_id="test-user",
            row_data=SAMPLE_ROW_DATA,
            column_names=["date", "amount", "description", "vendor"],
            source_file="transactions.csv",
            row_id="row_1"
        )
        
        assert result is not None
        # Entity resolver returns empty results in test mode, which is expected
        assert isinstance(result, dict)
        
        logger.info(f"✅ Entity resolution result: {len(result.get('resolved_entities', []))} entities resolved")
        return True
        
    except Exception as e:
        logger.error(f"❌ EntityResolverOptimized test failed: {e}")
        return False

async def test_universal_extractors():
    """Test UniversalExtractorsOptimized with fallback field detection"""
    logger.info("🧪 Testing UniversalExtractorsOptimized...")
    
    try:
        # Mock the field detector availability
        import universal_extractors_optimized
        universal_extractors_optimized.FIELD_DETECTOR_AVAILABLE = False
        
        from universal_extractors_optimized import UniversalExtractorsOptimized
        
        # Initialize with mock cache
        mock_cache = MockCacheClient()
        extractor = UniversalExtractorsOptimized(cache_client=mock_cache)
        
        # Test data extraction
        result = await extractor.extract_data_universal(
            file_content=SAMPLE_CSV_DATA,
            filename="transactions.csv",
            user_id="test-user"
        )
        
        assert result is not None
        assert 'extracted_data' in result
        assert 'metadata' in result
        
        logger.info(f"✅ Data extraction result: {len(result['extracted_data'])} rows extracted")
        return True
        
    except Exception as e:
        logger.error(f"❌ UniversalExtractorsOptimized test failed: {e}")
        return False

async def test_integration_workflow():
    """Test integration between all optimized components"""
    logger.info("🧪 Testing integration workflow...")
    
    try:
        # Set up environment
        os.environ['OPENAI_API_KEY'] = 'test-key'
        
        # Import components
        from universal_platform_detector_optimized import UniversalPlatformDetectorOptimized
        from universal_document_classifier_optimized import UniversalDocumentClassifierOptimized
        from entity_resolver_optimized import EntityResolverOptimized
        
        import universal_extractors_optimized
        universal_extractors_optimized.FIELD_DETECTOR_AVAILABLE = False
        from universal_extractors_optimized import UniversalExtractorsOptimized
        
        # Initialize components
        mock_openai = MockOpenAIClient()
        mock_cache = MockCacheClient()
        
        detector = UniversalPlatformDetectorOptimized(openai_client=mock_openai)
        classifier = UniversalDocumentClassifierOptimized(openai_client=mock_openai)
        resolver = EntityResolverOptimized()
        extractor = UniversalExtractorsOptimized(cache_client=mock_cache)
        
        # Simulate full workflow
        user_id = "test-user"
        file_content = SAMPLE_CSV_DATA
        filename = "transactions.csv"
        
        # Step 1: Extract data
        extraction_result = await extractor.extract_data_universal(
            file_content=file_content,
            filename=filename,
            user_id=user_id
        )
        
        # Step 2: Detect platform
        platform_payload = {"file_content": file_content, "filename": filename}
        platform_result = await detector.detect_platform_universal(
            payload=platform_payload,
            filename=filename,
            user_id=user_id
        )
        
        # Step 3: Classify document
        classification_payload = {"file_content": file_content, "filename": filename}
        classification_result = await classifier.classify_document_universal(
            payload=classification_payload,
            filename=filename,
            user_id=user_id
        )
        
        # Step 4: Resolve entities (if we have extracted data)
        if extraction_result.get('extracted_data'):
            entities = {
                "vendor": []
            }
            data = extraction_result['extracted_data']
            if isinstance(data, list):
                for row in data[:3]:  # First 3 rows
                    if isinstance(row, dict) and 'vendor' in row:
                        entities["vendor"].append(row['vendor'])
            
            if entities and entities.get("vendor"):
                resolution_result = await resolver.resolve_entities_batch(
                    entities=entities,
                    platform=platform_result.get('platform', 'unknown'),
                    user_id=user_id,
                    row_data=SAMPLE_ROW_DATA,
                    column_names=["date", "amount", "description", "vendor"],
                    source_file=filename,
                    row_id="row_1"
                )
                
                logger.info(f"✅ Integration workflow completed successfully")
                logger.info(f"   - Extracted {len(extraction_result['extracted_data'])} rows")
                logger.info(f"   - Detected platform: {platform_result.get('platform')}")
                logger.info(f"   - Classified document: {classification_result.get('document_type')}")
                logger.info(f"   - Resolved {len(resolution_result.get('resolved_entities', []))} entities")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration workflow test failed: {e}")
        return False

async def run_performance_test():
    """Test performance with multiple files"""
    logger.info("🧪 Running performance test...")
    
    try:
        os.environ['OPENAI_API_KEY'] = 'test-key'
        
        from universal_platform_detector_optimized import UniversalPlatformDetectorOptimized
        
        mock_openai = MockOpenAIClient()
        detector = UniversalPlatformDetectorOptimized(openai_client=mock_openai)
        
        # Test with multiple files
        test_files = [
            (SAMPLE_CSV_DATA, "transactions.csv"),
            (SAMPLE_JSON_DATA, "transactions.json"),
            (b"Date,Amount\n2024-01-01,100.00", "simple.csv")
        ]
        
        start_time = time.time()
        results = []
        
        for file_content, filename in test_files:
            payload = {"file_content": file_content, "filename": filename}
            result = await detector.detect_platform_universal(
                payload=payload,
                filename=filename,
                user_id="test-user"
            )
            results.append(result)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info(f"✅ Performance test completed in {processing_time:.2f} seconds")
        logger.info(f"   - Processed {len(test_files)} files")
        logger.info(f"   - Average time per file: {processing_time/len(test_files):.2f} seconds")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        return False

async def main():
    """Run all tests"""
    logger.info("🚀 Starting optimized components test suite...")
    
    test_results = []
    
    # Individual component tests
    test_results.append(await test_universal_platform_detector())
    test_results.append(await test_universal_document_classifier())
    test_results.append(await test_entity_resolver())
    test_results.append(await test_universal_extractors())
    
    # Integration tests
    test_results.append(await test_integration_workflow())
    test_results.append(await run_performance_test())
    
    # Summary
    passed = sum(test_results)
    total = len(test_results)
    
    logger.info(f"\n📊 Test Results Summary:")
    logger.info(f"   - Passed: {passed}/{total}")
    logger.info(f"   - Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        logger.info("🎉 All tests passed! Optimized components are working correctly.")
        return True
    else:
        logger.error(f"❌ {total-passed} tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)