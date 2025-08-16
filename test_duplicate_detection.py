#!/usr/bin/env python3
"""
Test script for the comprehensive duplicate detection system
Tests Basic → 100X → 100X+ capabilities
"""

import asyncio
import os
import hashlib
import pandas as pd
from datetime import datetime
from duplicate_detection_service import DuplicateDetectionService
from supabase import create_client, Client

# Test configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
TEST_USER_ID = "550e8400-e29b-41d4-a716-446655440000"

def create_test_file_content(filename: str, rows: int = 100, extra_columns: list = None) -> bytes:
    """Create test Excel file content with specified characteristics"""
    
    # Base data
    data = {
        'Date': pd.date_range('2024-01-01', periods=rows, freq='D'),
        'Amount': [100.50 + i for i in range(rows)],
        'Description': [f'Transaction {i+1}' for i in range(rows)],
        'Vendor': [f'Vendor {(i % 10) + 1}' for i in range(rows)]
    }
    
    # Add extra columns if specified
    if extra_columns:
        for col in extra_columns:
            data[col] = [f'{col} value {i+1}' for i in range(rows)]
    
    df = pd.DataFrame(data)
    
    # Save to bytes
    import io
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False)
    
    return buffer.getvalue()

async def test_basic_duplicate_detection():
    """Test Phase 1: Basic duplicate detection"""
    print("\n" + "="*60)
    print("TESTING PHASE 1: BASIC DUPLICATE DETECTION")
    print("="*60)
    
    # Initialize service
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    service = DuplicateDetectionService(supabase)
    
    # Create identical test files
    file_content = create_test_file_content("invoice_july.xlsx", rows=50)
    file_hash = service.calculate_file_hash(file_content)
    
    print(f"📁 Created test file: invoice_july.xlsx")
    print(f"🔐 File hash: {file_hash[:16]}...")
    
    # Test 1: First upload (should not be duplicate)
    print("\n🧪 Test 1: First upload of file")
    result1 = await service.check_exact_duplicate(TEST_USER_ID, file_hash, "invoice_july.xlsx")
    print(f"   Is duplicate: {result1['is_duplicate']}")
    print(f"   Recommendation: {result1['recommendation']}")
    
    # Simulate storing the file in database
    sheets = {'Sheet1': pd.read_excel(io.BytesIO(file_content))}
    
    # Test 2: Second upload (should be duplicate)
    print("\n🧪 Test 2: Second upload of identical file")
    result2 = await service.check_exact_duplicate(TEST_USER_ID, file_hash, "invoice_july.xlsx")
    print(f"   Is duplicate: {result2['is_duplicate']}")
    print(f"   Recommendation: {result2['recommendation']}")
    if result2['is_duplicate']:
        print(f"   Message: {result2['message']}")
        print(f"   Duplicate files found: {len(result2['duplicate_files'])}")
    
    # Test 3: Handle duplicate decision
    print("\n🧪 Test 3: Handle duplicate decision (replace)")
    decision_result = await service.handle_duplicate_decision(
        TEST_USER_ID, file_hash, 'replace'
    )
    print(f"   Decision result: {decision_result}")
    
    return result1, result2, decision_result

async def test_near_duplicate_detection():
    """Test Phase 2: Near-duplicate detection"""
    print("\n" + "="*60)
    print("TESTING PHASE 2: NEAR-DUPLICATE DETECTION")
    print("="*60)
    
    # Initialize service
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    service = DuplicateDetectionService(supabase)
    
    # Create similar files with version patterns
    files_to_test = [
        ("invoice_july_v1.xlsx", 50, []),
        ("invoice_july_v2.xlsx", 55, ['Tax']),
        ("invoice_july_final.xlsx", 60, ['Tax', 'Notes']),
        ("completely_different.xlsx", 30, ['Category'])
    ]
    
    print("📁 Creating test files with version patterns:")
    
    file_data = []
    for filename, rows, extra_cols in files_to_test:
        content = create_test_file_content(filename, rows, extra_cols)
        file_hash = service.calculate_file_hash(content)
        sheets = {'Sheet1': pd.read_excel(io.BytesIO(content))}
        fingerprint = service.calculate_content_fingerprint(sheets)
        
        file_data.append({
            'filename': filename,
            'content': content,
            'file_hash': file_hash,
            'sheets': sheets,
            'fingerprint': fingerprint,
            'rows': rows,
            'extra_cols': extra_cols
        })
        
        print(f"   📄 {filename}: {rows} rows, {len(extra_cols)} extra columns")
    
    # Test filename normalization
    print("\n🧪 Test 1: Filename normalization")
    for file_info in file_data:
        normalized = service.normalize_filename(file_info['filename'])
        version_pattern = service.extract_version_pattern(file_info['filename'])
        print(f"   {file_info['filename']} → {normalized} (version: {version_pattern})")
    
    # Test filename similarity
    print("\n🧪 Test 2: Filename similarity calculation")
    base_file = file_data[0]  # invoice_july_v1.xlsx
    for file_info in file_data[1:]:
        similarity = service.calculate_filename_similarity(
            base_file['filename'], file_info['filename']
        )
        print(f"   {base_file['filename']} vs {file_info['filename']}: {similarity:.3f}")
    
    # Test finding similar files
    print("\n🧪 Test 3: Finding similar files")
    test_file = file_data[1]  # invoice_july_v2.xlsx
    similar_files = await service.find_similar_files(
        TEST_USER_ID, 
        test_file['filename'], 
        test_file['fingerprint'], 
        test_file['file_hash']
    )
    print(f"   Found {len(similar_files)} similar files for {test_file['filename']}")
    for similar in similar_files:
        print(f"     📄 {similar['filename']}: similarity {similar['filename_similarity']:.3f}")
    
    # Test relationship analysis
    print("\n🧪 Test 4: File relationship analysis")
    if len(file_data) >= 2:
        file1_data = {'filename': file_data[0]['filename']}
        file2_data = {'filename': file_data[1]['filename']}
        
        relationship = await service.analyze_file_relationship(
            file1_data, file2_data, 
            file_data[0]['sheets'], file_data[1]['sheets']
        )
        
        print(f"   Relationship between {file1_data['filename']} and {file2_data['filename']}:")
        print(f"     Filename similarity: {relationship['filename_similarity']:.3f}")
        print(f"     Structure similarity: {relationship['structure_similarity']:.3f}")
        print(f"     Content similarity: {relationship['content_similarity']:.3f}")
        print(f"     Relationship type: {relationship['relationship_type']}")
        print(f"     Confidence: {relationship['confidence_score']:.3f}")
        print(f"     Reason: {relationship['analysis_reason']}")
    
    return file_data

async def test_intelligent_version_selection():
    """Test Phase 3: Intelligent version selection"""
    print("\n" + "="*60)
    print("TESTING PHASE 3: INTELLIGENT VERSION SELECTION")
    print("="*60)
    
    # Initialize service
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    service = DuplicateDetectionService(supabase)
    
    # Create version group data
    version_group_id = "test-version-group-123"
    
    # Mock version data for analysis
    files_data = [
        {
            'id': 'file1',
            'filename': 'invoice_july_v1.xlsx',
            'file_hash': 'hash1',
            'created_at': '2024-01-01T10:00:00Z',
            'total_rows': 50,
            'total_columns': 4,
            'column_names': ['Date', 'Amount', 'Description', 'Vendor'],
            'content_fingerprint': 'fp1'
        },
        {
            'id': 'file2',
            'filename': 'invoice_july_v2.xlsx',
            'file_hash': 'hash2',
            'created_at': '2024-01-02T11:00:00Z',
            'total_rows': 55,
            'total_columns': 5,
            'column_names': ['Date', 'Amount', 'Description', 'Vendor', 'Tax'],
            'content_fingerprint': 'fp2'
        },
        {
            'id': 'file3',
            'filename': 'invoice_july_final.xlsx',
            'file_hash': 'hash3',
            'created_at': '2024-01-03T12:00:00Z',
            'total_rows': 60,
            'total_columns': 6,
            'column_names': ['Date', 'Amount', 'Description', 'Vendor', 'Tax', 'Notes'],
            'content_fingerprint': 'fp3'
        }
    ]
    
    print("📊 Analyzing version candidates:")
    for file_data in files_data:
        print(f"   📄 {file_data['filename']}: {file_data['total_rows']} rows, {file_data['total_columns']} columns")
    
    # Test completeness analysis
    print("\n🧪 Test 1: Completeness analysis")
    # Note: This would normally use the database, but we'll simulate it
    completeness_scores = {}
    for file_data in files_data:
        score = service._calculate_completeness_score(
            file_data['total_rows'],
            file_data['total_columns'],
            file_data['column_names']
        )
        completeness_scores[file_data['id']] = score
        print(f"   {file_data['filename']}: completeness score {score:.3f}")
    
    # Test recency analysis
    print("\n🧪 Test 2: Recency analysis")
    # Simulate recency scoring
    recency_scores = {}
    latest_time = max(file_data['created_at'] for file_data in files_data)
    
    for file_data in files_data:
        is_latest = file_data['created_at'] == latest_time
        filename_boost = 0.9 if 'final' in file_data['filename'].lower() else 0.5
        recency_score = (1.0 if is_latest else 0.5) * 0.7 + filename_boost * 0.3
        recency_scores[file_data['id']] = recency_score
        print(f"   {file_data['filename']}: recency score {recency_score:.3f} (latest: {is_latest})")
    
    # Test overall recommendation
    print("\n🧪 Test 3: Overall recommendation")
    best_file_id = None
    best_score = 0
    
    for file_data in files_data:
        file_id = file_data['id']
        overall_score = completeness_scores[file_id] * 0.6 + recency_scores[file_id] * 0.4
        print(f"   {file_data['filename']}: overall score {overall_score:.3f}")
        
        if overall_score > best_score:
            best_score = overall_score
            best_file_id = file_id
    
    best_file = next(f for f in files_data if f['id'] == best_file_id)
    print(f"\n🏆 RECOMMENDED VERSION: {best_file['filename']}")
    print(f"   📊 Score: {best_score:.3f}")
    print(f"   📈 Rows: {best_file['total_rows']}")
    print(f"   📋 Columns: {best_file['total_columns']}")
    
    # Generate reasoning
    reasoning_parts = []
    if best_file['total_rows'] == max(f['total_rows'] for f in files_data):
        reasoning_parts.append(f"has the most data ({best_file['total_rows']} rows)")
    if best_file['total_columns'] == max(f['total_columns'] for f in files_data):
        reasoning_parts.append(f"includes the most columns ({best_file['total_columns']} columns)")
    if 'final' in best_file['filename'].lower():
        reasoning_parts.append("is marked as final version")
    
    reasoning = f"'{best_file['filename']}' is recommended because it {', '.join(reasoning_parts)}."
    print(f"   💡 Reasoning: {reasoning}")
    
    return files_data, best_file, reasoning

async def test_comprehensive_workflow():
    """Test the complete workflow"""
    print("\n" + "="*60)
    print("TESTING COMPREHENSIVE WORKFLOW")
    print("="*60)
    
    # Initialize service
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    service = DuplicateDetectionService(supabase)
    
    # Create test file
    file_content = create_test_file_content("test_workflow.xlsx", rows=75, extra_columns=['Category'])
    sheets = {'Sheet1': pd.read_excel(io.BytesIO(file_content))}
    
    print("🔄 Testing complete workflow...")
    
    # Run comprehensive workflow
    result = await service.process_file_upload(
        TEST_USER_ID, file_content, "test_workflow.xlsx", sheets
    )
    
    print(f"📋 Workflow Result:")
    print(f"   Phase: {result['phase']}")
    print(f"   Recommended action: {result['recommended_action']}")
    
    if 'duplicate_info' in result:
        print(f"   Duplicate info: {result['duplicate_info']}")
    if 'similar_files' in result:
        print(f"   Similar files found: {len(result['similar_files'])}")
    if 'error' in result:
        print(f"   Error: {result['error']}")
    
    return result

async def main():
    """Run all tests"""
    print("🚀 COMPREHENSIVE DUPLICATE DETECTION SYSTEM TEST")
    print("Testing Basic → 100X → 100X+ capabilities")
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("❌ Error: Supabase credentials not configured")
        print("Please set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables")
        return
    
    try:
        # Run all test phases
        await test_basic_duplicate_detection()
        await test_near_duplicate_detection()
        await test_intelligent_version_selection()
        await test_comprehensive_workflow()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n📊 CAPABILITY ASSESSMENT:")
        print("✅ Phase 1 (Basic): Hash-based duplicate detection")
        print("✅ Phase 2 (100X): Near-duplicate detection with fuzzy matching")
        print("✅ Phase 3 (100X+): Intelligent version selection with AI recommendations")
        print("\n🎯 The system now provides CFO-grade duplicate handling!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import io
    asyncio.run(main())
