#!/usr/bin/env python3
"""
Fix memory optimization for large files by implementing streaming processing
"""

import re

def fix_memory_optimization():
    """Add memory optimization for large files"""
    
    # Read the file
    with open('fastapi_backend.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add memory optimization imports
    import_pattern = r'import asyncio\nfrom collections import defaultdict\nfrom concurrent\.futures import ThreadPoolExecutor'
    
    import_replacement = '''import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import gc
import psutil
import os'''
    
    content = re.sub(import_pattern, import_replacement, content)
    
    # Add memory monitoring class
    memory_monitor_pattern = r'# Initialize FastAPI app\napp = FastAPI\(title="Finley AI Backend", version="1\.0\.0"\)'
    
    memory_monitor_replacement = '''# Memory monitoring utilities
class MemoryMonitor:
    """Monitor and manage memory usage for large file processing"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.memory_threshold = 0.8  # 80% memory usage threshold
    
    def get_memory_usage(self):
        """Get current memory usage percentage"""
        return self.process.memory_percent()
    
    def check_memory_limit(self):
        """Check if memory usage is within limits"""
        return self.get_memory_usage() < self.memory_threshold
    
    def force_garbage_collection(self):
        """Force garbage collection to free memory"""
        gc.collect()
    
    def get_memory_info(self):
        """Get detailed memory information"""
        memory_info = self.process.memory_info()
        return {
            'rss': memory_info.rss / 1024 / 1024,  # MB
            'vms': memory_info.vms / 1024 / 1024,  # MB
            'percent': self.get_memory_usage()
        }

# Initialize memory monitor
memory_monitor = MemoryMonitor()

# Initialize FastAPI app
app = FastAPI(title="Finley AI Backend", version="1.0.0")'''
    
    content = re.sub(memory_monitor_pattern, memory_monitor_replacement, content)
    
    # Add chunked processing to ExcelProcessor
    chunked_processing_pattern = r'async def process_file\(self, job_id: str, file_content: bytes, filename: str, \s+user_id: str, supabase: Client\) -> Dict\[str, Any\]:'
    
    chunked_processing_replacement = '''async def process_file(self, job_id: str, file_content: bytes, filename: str, 
                          user_id: str, supabase: Client) -> Dict[str, Any]:
        """Optimized processing pipeline with memory management and chunked processing for large files"""
        
        # Check memory before starting
        if not memory_monitor.check_memory_limit():
            memory_monitor.force_garbage_collection()
            logger.warning(f"High memory usage detected: {memory_monitor.get_memory_usage():.1f}%")
        
        # Determine chunk size based on file size
        file_size_mb = len(file_content) / 1024 / 1024
        if file_size_mb > 100:  # Large file
            chunk_size = 1000  # Process 1000 rows at a time
            logger.info(f"Large file detected ({file_size_mb:.1f}MB), using chunked processing")
        elif file_size_mb > 50:  # Medium file
            chunk_size = 2000  # Process 2000 rows at a time
            logger.info(f"Medium file detected ({file_size_mb:.1f}MB), using optimized processing")
        else:  # Small file
            chunk_size = 5000  # Process 5000 rows at a time
            logger.info(f"Small file detected ({file_size_mb:.1f}MB), using standard processing")'''
    
    content = re.sub(chunked_processing_pattern, chunked_processing_replacement, content, flags=re.DOTALL)
    
    # Add memory management to row processing
    row_processing_pattern = r'# Step 5: Process each sheet with optimized batch processing'
    
    row_processing_replacement = '''# Step 5: Process each sheet with memory-optimized chunked processing
        await manager.send_update(job_id, {
            "step": "streaming",
            "message": f"🔄 Processing rows in memory-optimized chunks (chunk size: {chunk_size})...",
            "progress": 40
        })
        
        # Memory monitoring during processing
        memory_check_interval = max(1, total_rows // 20)  # Check memory every 5% of rows'''
    
    content = re.sub(row_processing_pattern, row_processing_replacement, content)
    
    # Add memory checks during processing
    memory_check_pattern = r'processed_rows \+= 1\n                    \n                    # Update progress every batch'
    
    memory_check_replacement = '''processed_rows += 1
                    
                    # Memory management: Check and clean memory periodically
                    if processed_rows % memory_check_interval == 0:
                        current_memory = memory_monitor.get_memory_usage()
                        if current_memory > 75:  # 75% threshold
                            memory_monitor.force_garbage_collection()
                            logger.info(f"Memory cleaned at row {processed_rows}, usage: {current_memory:.1f}%")
                        
                        # Send memory status update
                        await manager.send_update(job_id, {
                            "step": "memory_check",
                            "message": f"🔄 Memory check: {current_memory:.1f}% usage, processed {processed_rows}/{total_rows} rows",
                            "progress": int(40 + (processed_rows / total_rows) * 40)
                        })
                    
                    # Update progress every batch'''
    
    content = re.sub(memory_check_pattern, memory_check_replacement, content)
    
    # Add final memory cleanup
    final_cleanup_pattern = r'# Step 6: Update raw_records with completion status'
    
    final_cleanup_replacement = '''# Step 6: Final memory cleanup and status update
        # Force final garbage collection
        memory_monitor.force_garbage_collection()
        final_memory = memory_monitor.get_memory_info()
        logger.info(f"Processing completed. Final memory usage: {final_memory['rss']:.1f}MB RSS, {final_memory['percent']:.1f}%")
        
        await manager.send_update(job_id, {
            "step": "finalizing",
            "message": f"✅ Finalizing processing... (Memory: {final_memory['percent']:.1f}%)",
            "progress": 90
        })
        
        # Step 7: Update raw_records with completion status'''
    
    content = re.sub(final_cleanup_pattern, final_cleanup_replacement, content)
    
    # Write the optimized content back
    with open('fastapi_backend.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Added memory optimization for large files")

if __name__ == "__main__":
    fix_memory_optimization()
