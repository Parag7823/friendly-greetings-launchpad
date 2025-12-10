import sys
import os
import asyncio

# Add current directory to path
sys.path.append(os.getcwd())

async def verify():
    print("Verifying imports...")
    errors = False
    
    try:
        print("Importing UniversalDocumentClassifierOptimized...")
        from data_ingestion_normalization.universal_document_classifier_optimized import UniversalDocumentClassifierOptimized
        print("✅ UniversalDocumentClassifierOptimized imported successfully")
    except Exception as e:
        print(f"❌ UniversalDocumentClassifierOptimized failed: {e}")
        errors = True

    try:
        print("Importing UniversalPlatformDetectorOptimized...")
        from data_ingestion_normalization.universal_platform_detector_optimized import UniversalPlatformDetectorOptimized
        print("✅ UniversalPlatformDetectorOptimized imported successfully")
    except Exception as e:
        print(f"❌ UniversalPlatformDetectorOptimized failed: {e}")
        errors = True

    try:
        print("Importing ProductionDuplicateDetectionService...")
        from duplicate_detection_fraud.production_duplicate_detection_service import ProductionDuplicateDetectionService
        print("✅ ProductionDuplicateDetectionService imported successfully")
    except Exception as e:
        print(f"❌ ProductionDuplicateDetectionService failed: {e}")
        errors = True
        
    if not errors:
        print("\n🎉 All modules verified successfully!")
    else:
        print("\n⚠️ Verification failed with errors.")

if __name__ == "__main__":
    try:
        asyncio.run(verify())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Script crash: {e}")
