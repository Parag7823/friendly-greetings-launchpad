#!/usr/bin/env python3
"""
Verification script to check if all critical methods exist in ExcelProcessor.
Runs at container startup to catch Docker cache issues immediately.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_excel_processor_methods():
    """Verify all critical methods exist in ExcelProcessor class"""
    try:
        # Import the class
        from fastapi_backend_v2 import ExcelProcessor
        
        # Critical methods that must exist
        critical_methods = [
            '_normalize_entity_type',
            '_store_entity_matches',
            '_store_platform_patterns',
            '_store_discovered_platforms',
            '_extract_entities_from_events',
            '_resolve_entities',
            '_learn_platform_patterns',
            '_discover_new_platforms',
            'process_file'
        ]
        
        # Check which methods exist
        processor = ExcelProcessor()
        existing_methods = [m for m in dir(processor) if not m.startswith('__')]
        missing_methods = [m for m in critical_methods if not hasattr(processor, m)]
        
        logger.info(f"=" * 80)
        logger.info(f"🔍 EXCEL PROCESSOR METHOD VERIFICATION")
        logger.info(f"=" * 80)
        logger.info(f"Total methods in ExcelProcessor: {len(existing_methods)}")
        logger.info(f"Critical methods required: {len(critical_methods)}")
        logger.info(f"Critical methods found: {len(critical_methods) - len(missing_methods)}")
        
        if missing_methods:
            logger.error(f"❌ MISSING METHODS: {missing_methods}")
            logger.error(f"❌ This indicates Docker is using OLD CACHED CODE!")
            logger.error(f"❌ Expected ~50+ methods, found only {len(existing_methods)}")
            logger.error(f"=" * 80)
            return False
        else:
            logger.info(f"✅ All {len(critical_methods)} critical methods found!")
            logger.info(f"✅ Container is using LATEST CODE")
            logger.info(f"=" * 80)
            return True
            
    except Exception as e:
        logger.error(f"❌ Verification failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = verify_excel_processor_methods()
    sys.exit(0 if success else 1)
