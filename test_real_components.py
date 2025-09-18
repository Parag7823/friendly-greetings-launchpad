#!/usr/bin/env python3
"""
Real Component Testing - Actual Testing of Production Components
==============================================================

This script tests the actual components from fastapi_backend.py to verify
they work correctly with real data and configurations.

Author: Principal Engineer - Quality, Testing & Resilience
"""

import asyncio
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Set environment variables for testing
os.environ['OPENAI_API_KEY'] = 'test-key-for-testing'
os.environ['SUPABASE_URL'] = 'https://test.supabase.co'
os.environ['SUPABASE_SERVICE_ROLE_KEY'] = 'test-key-for-testing'
os.environ['DISABLE_SECURITY_VALIDATION'] = 'true'

try:
    from fastapi_backend import (
        DataEnrichmentProcessor,
        DocumentAnalyzer,
        ExcelProcessor,
        UniversalFieldDetector,
        UniversalPlatformDetector,
        UniversalDocumentClassifier,
        UniversalExtractors,
        EntityResolver
    )
    print("✅ Successfully imported all components from fastapi_backend.py")
except Exception as e:
    print(f"❌ Failed to import components: {e}")
    sys.exit(1)


async def test_data_enrichment_processor():
    """Test DataEnrichmentProcessor with real data"""
    print("\n🧪 Testing DataEnrichmentProcessor...")
    
    try:
        processor = DataEnrichmentProcessor()
        
        # Test with sample financial data
        sample_data = [
            {
                'transaction_id': 'TXN_001',
                'date': '2024-01-15',
                'vendor': 'Apple Inc',
                'amount': 1250.00,
                'description': 'Software purchase'
            },
            {
                'transaction_id': 'TXN_002', 
                'date': '2024-01-16',
                'vendor': 'Microsoft Corp',
                'amount': 850.00,
                'description': 'Cloud services'
            }
        ]
        
        result = await processor.enrich_data(
            data=sample_data,
            user_id='test_user',
            enrichment_types=['vendor_standardization', 'category_classification']
        )
        
        print(f"✅ DataEnrichmentProcessor: Success")
        print(f"   - Input records: {len(sample_data)}")
        print(f"   - Output records: {len(result.get('enriched_data', []))}")
        print(f"   - Status: {result.get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ DataEnrichmentProcessor: Failed - {e}")
        return False


async def test_excel_processor():
    """Test ExcelProcessor with real Excel data"""
    print("\n🧪 Testing ExcelProcessor...")
    
    try:
        processor = ExcelProcessor()
        
        # Create a simple Excel file for testing
        import pandas as pd
        import io
        
        # Create sample data
        df = pd.DataFrame({
            'Date': ['2024-01-15', '2024-01-16', '2024-01-17'],
            'Vendor': ['Apple Inc', 'Microsoft Corp', 'Google LLC'],
            'Amount': [1250.00, 850.00, 2100.00],
            'Description': ['Software', 'Cloud Services', 'Advertising']
        })
        
        # Convert to Excel bytes
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_content = excel_buffer.getvalue()
        
        result = await processor.stream_xlsx_processing(
            file_content=excel_content,
            filename='test.xlsx',
            user_id='test_user'
        )
        
        print(f"✅ ExcelProcessor: Success")
        print(f"   - Sheets processed: {len(result.get('sheets', []))}")
        print(f"   - Status: {result.get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ ExcelProcessor: Failed - {e}")
        return False


async def test_universal_field_detector():
    """Test UniversalFieldDetector with real data"""
    print("\n🧪 Testing UniversalFieldDetector...")
    
    try:
        detector = UniversalFieldDetector()
        
        # Test with sample data
        sample_data = {
            'amount': 1250.00,
            'date': '2024-01-15',
            'vendor': 'Apple Inc',
            'description': 'Software purchase'
        }
        
        result = await detector.detect_field_types_universal(
            data=sample_data,
            filename='test.csv',
            user_id='test_user'
        )
        
        print(f"✅ UniversalFieldDetector: Success")
        print(f"   - Fields detected: {len(result.get('detected_fields', {}))}")
        print(f"   - Status: {result.get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ UniversalFieldDetector: Failed - {e}")
        return False


async def test_universal_platform_detector():
    """Test UniversalPlatformDetector with real data"""
    print("\n🧪 Testing UniversalPlatformDetector...")
    
    try:
        detector = UniversalPlatformDetector()
        
        # Create sample CSV content
        csv_content = b"Date,Vendor,Amount\n2024-01-15,Apple Inc,1250.00\n2024-01-16,Microsoft Corp,850.00"
        
        result = await detector.detect_platform_universal(
            payload={'file_content': csv_content, 'filename': 'test.csv'},
            filename='test.csv',
            user_id='test_user'
        )
        
        print(f"✅ UniversalPlatformDetector: Success")
        print(f"   - Platform detected: {result.get('platform', 'unknown')}")
        print(f"   - Confidence: {result.get('confidence', 0)}")
        print(f"   - Status: {result.get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ UniversalPlatformDetector: Failed - {e}")
        return False


async def test_universal_document_classifier():
    """Test UniversalDocumentClassifier with real data"""
    print("\n🧪 Testing UniversalDocumentClassifier...")
    
    try:
        classifier = UniversalDocumentClassifier()
        
        # Create sample content
        content = b"INVOICE\nInvoice Number: INV-001\nDate: 2024-01-15\nVendor: Apple Inc\nAmount: $1250.00"
        
        result = await classifier.classify_document_universal(
            payload={'file_content': content, 'filename': 'invoice.txt'},
            filename='invoice.txt',
            user_id='test_user'
        )
        
        print(f"✅ UniversalDocumentClassifier: Success")
        print(f"   - Document type: {result.get('document_type', 'unknown')}")
        print(f"   - Confidence: {result.get('confidence', 0)}")
        print(f"   - Status: {result.get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ UniversalDocumentClassifier: Failed - {e}")
        return False


async def test_universal_extractors():
    """Test UniversalExtractors with real data"""
    print("\n🧪 Testing UniversalExtractors...")
    
    try:
        extractor = UniversalExtractors()
        
        # Create sample CSV content
        csv_content = b"Date,Vendor,Amount\n2024-01-15,Apple Inc,1250.00\n2024-01-16,Microsoft Corp,850.00"
        
        result = await extractor.extract_data_universal(
            file_content=csv_content,
            filename='test.csv',
            user_id='test_user'
        )
        
        print(f"✅ UniversalExtractors: Success")
        print(f"   - Data extracted: {len(result.get('extracted_data', []))}")
        print(f"   - Format detected: {result.get('file_format', 'unknown')}")
        print(f"   - Status: {result.get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ UniversalExtractors: Failed - {e}")
        return False


async def test_entity_resolver():
    """Test EntityResolver with real data"""
    print("\n🧪 Testing EntityResolver...")
    
    try:
        resolver = EntityResolver()
        
        # Test with sample entities
        entities = {
            'vendor': ['Apple Inc', 'Apple Incorporated', 'APPLE INC']
        }
        
        result = await resolver.resolve_entities_batch(
            entities=entities,
            platform='test_platform',
            user_id='test_user',
            row_data={'vendor': 'Apple Inc'},
            column_names=['vendor'],
            source_file='test.csv',
            row_id='row_1'
        )
        
        print(f"✅ EntityResolver: Success")
        print(f"   - Entities resolved: {len(result.get('resolved_entities', []))}")
        print(f"   - Conflicts: {len(result.get('conflicts', []))}")
        print(f"   - Status: {result.get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ EntityResolver: Failed - {e}")
        return False


async def test_document_analyzer():
    """Test DocumentAnalyzer with real data"""
    print("\n🧪 Testing DocumentAnalyzer...")
    
    try:
        analyzer = DocumentAnalyzer()
        
        # Create sample content
        content = b"This is a sample document with financial data. Amount: $1250.00, Date: 2024-01-15"
        
        result = await analyzer.analyze_document(
            content=content,
            filename='test.txt',
            user_id='test_user'
        )
        
        print(f"✅ DocumentAnalyzer: Success")
        print(f"   - Analysis completed: {result.get('analysis', {}).get('status', 'unknown')}")
        print(f"   - Status: {result.get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ DocumentAnalyzer: Failed - {e}")
        return False


async def main():
    """Run all component tests"""
    print("🚀 Starting Real Component Testing")
    print("=" * 60)
    
    # List of test functions
    tests = [
        ('DataEnrichmentProcessor', test_data_enrichment_processor),
        ('ExcelProcessor', test_excel_processor),
        ('UniversalFieldDetector', test_universal_field_detector),
        ('UniversalPlatformDetector', test_universal_platform_detector),
        ('UniversalDocumentClassifier', test_universal_document_classifier),
        ('UniversalExtractors', test_universal_extractors),
        ('EntityResolver', test_entity_resolver),
        ('DocumentAnalyzer', test_document_analyzer)
    ]
    
    results = {}
    total_tests = len(tests)
    passed_tests = 0
    
    for component_name, test_func in tests:
        try:
            result = await test_func()
            results[component_name] = result
            if result:
                passed_tests += 1
        except Exception as e:
            print(f"❌ {component_name}: Critical error - {e}")
            results[component_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 REAL COMPONENT TEST SUMMARY")
    print("=" * 60)
    
    for component_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{component_name}: {status}")
    
    success_rate = (passed_tests / total_tests) * 100
    print(f"\nOverall Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")
    
    if success_rate >= 80:
        print("🎉 MOST COMPONENTS WORKING - System is functional!")
    elif success_rate >= 50:
        print("⚠️  SOME COMPONENTS FAILING - Review and fix issues")
    else:
        print("❌ MANY COMPONENTS FAILING - Major issues need attention")
    
    return success_rate >= 80


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
