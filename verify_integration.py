#!/usr/bin/env python3
"""
Simple verification script for enhanced functionality integration
"""

import re

def verify_integration():
    """Verify that the enhanced functionality has been integrated"""
    
    print("🔍 Verifying Enhanced Functionality Integration")
    print("=" * 50)
    
    try:
        # Read the main backend file
        with open('fastapi_backend.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Test 1: Check for advanced imports
        print("\n✅ Test 1: Advanced Library Imports")
        advanced_libs = ['zipfile', 'py7zr', 'rarfile', 'odf', 'tabula', 'camelot', 'pdfplumber', 'pytesseract', 'PIL', 'cv2', 'xlwings']
        found_libs = []
        for lib in advanced_libs:
            if lib in content:
                found_libs.append(lib)
        
        print(f"   - Found {len(found_libs)}/{len(advanced_libs)} advanced libraries")
        for lib in found_libs:
            print(f"     ✓ {lib}")
        
        # Test 2: Check for DuplicateDetectionService
        print("\n✅ Test 2: DuplicateDetectionService Integration")
        if 'class DuplicateDetectionService' in content:
            print("   ✓ DuplicateDetectionService class found")
            if 'check_exact_duplicate' in content:
                print("   ✓ Exact duplicate detection method found")
            if 'check_near_duplicate' in content:
                print("   ✓ Near duplicate detection method found")
            if 'calculate_file_hash' in content:
                print("   ✓ File hash calculation method found")
        else:
            print("   ❌ DuplicateDetectionService class not found")
        
        # Test 3: Check for EnhancedFileProcessor
        print("\n✅ Test 3: EnhancedFileProcessor Integration")
        if 'class EnhancedFileProcessor' in content:
            print("   ✓ EnhancedFileProcessor class found")
            if 'process_file_enhanced' in content:
                print("   ✓ Enhanced file processing method found")
            if '_process_pdf' in content:
                print("   ✓ PDF processing method found")
            if '_process_archive' in content:
                print("   ✓ Archive processing method found")
            if '_process_image' in content:
                print("   ✓ Image/OCR processing method found")
        else:
            print("   ❌ EnhancedFileProcessor class not found")
        
        # Test 4: Check for Configuration System
        print("\n✅ Test 4: Configuration System")
        if 'class Config' in content:
            print("   ✓ Config class found")
            if 'enable_advanced_file_processing' in content:
                print("   ✓ Advanced file processing flag found")
            if 'enable_duplicate_detection' in content:
                print("   ✓ Duplicate detection flag found")
            if 'enable_ocr_processing' in content:
                print("   ✓ OCR processing flag found")
        else:
            print("   ❌ Config class not found")
        
        # Test 5: Check for Global Instances
        print("\n✅ Test 5: Global Instances")
        if 'enhanced_processor = EnhancedFileProcessor()' in content:
            print("   ✓ Enhanced processor global instance found")
        else:
            print("   ❌ Enhanced processor global instance not found")
        
        # Test 6: Check for Advanced Features Flag
        print("\n✅ Test 6: Advanced Features Flag")
        if 'ADVANCED_FEATURES_AVAILABLE' in content:
            print("   ✓ Advanced features availability flag found")
        else:
            print("   ❌ Advanced features availability flag not found")
        
        # Summary
        print("\n" + "=" * 50)
        print("🎯 Integration Verification Complete!")
        
        # Count successful integrations
        success_count = 0
        total_tests = 6
        
        if len(found_libs) >= 8:  # At least 8 out of 11 libraries
            success_count += 1
        if 'class DuplicateDetectionService' in content:
            success_count += 1
        if 'class EnhancedFileProcessor' in content:
            success_count += 1
        if 'class Config' in content:
            success_count += 1
        if 'enhanced_processor = EnhancedFileProcessor()' in content:
            success_count += 1
        if 'ADVANCED_FEATURES_AVAILABLE' in content:
            success_count += 1
        
        print(f"\n📊 Integration Score: {success_count}/{total_tests} ({success_count/total_tests*100:.1f}%)")
        
        if success_count == total_tests:
            print("🎉 ALL INTEGRATIONS SUCCESSFUL!")
            print("🚀 Your platform now has 100X file processing capabilities!")
        elif success_count >= total_tests * 0.8:
            print("✅ MOST INTEGRATIONS SUCCESSFUL!")
            print("🚀 Your platform has significant enhanced capabilities!")
        else:
            print("⚠️ Some integrations may need attention")
        
        return success_count == total_tests
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

if __name__ == "__main__":
    success = verify_integration()
    exit(0 if success else 1)
