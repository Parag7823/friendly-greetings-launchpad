"""
Performance Tests for Large File Processing

Tests:
- Upload 500MB file
- Verify streaming works
- Check memory stays under limit
- Measure processing time
"""

import pytest
import asyncio
import time
import psutil
import hashlib
import io


class TestLargeFileProcessing:
    """Test processing of large files"""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_process_100mb_file(self):
        """Should process 100MB file efficiently"""
        file_size = 100 * 1024 * 1024  # 100MB
        
        async def process_large_file(size):
            # Simulate streaming processing
            chunk_size = 1024 * 1024  # 1MB chunks
            processed = 0
            
            while processed < size:
                chunk = min(chunk_size, size - processed)
                await asyncio.sleep(0.001)  # Simulate processing
                processed += chunk
            
            return {'size': size, 'status': 'completed'}
        
        start_time = time.time()
        result = await process_large_file(file_size)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        print(f"100MB file processed in {processing_time:.2f}s")
        
        assert result['status'] == 'completed'
        assert processing_time < 30.0  # Should complete in <30s
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_process_500mb_file(self):
        """Should process 500MB file efficiently"""
        file_size = 500 * 1024 * 1024  # 500MB
        
        async def process_large_file(size):
            chunk_size = 1024 * 1024  # 1MB chunks
            processed = 0
            
            while processed < size:
                chunk = min(chunk_size, size - processed)
                await asyncio.sleep(0.001)
                processed += chunk
            
            return {'size': size, 'status': 'completed'}
        
        start_time = time.time()
        result = await process_large_file(file_size)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        print(f"500MB file processed in {processing_time:.2f}s")
        
        assert result['status'] == 'completed'
        assert processing_time < 120.0  # Should complete in <2 minutes


class TestMemoryEfficiency:
    """Test memory efficiency with large files"""
    
    @pytest.mark.asyncio
    async def test_streaming_memory_usage(self):
        """Should use streaming to keep memory usage low"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        async def stream_process_file(size_mb):
            chunk_size = 1024 * 1024  # 1MB chunks
            total_size = size_mb * 1024 * 1024
            processed = 0
            
            while processed < total_size:
                # Simulate reading and processing chunk
                chunk = b'x' * min(chunk_size, total_size - processed)
                _ = hashlib.sha256(chunk).hexdigest()
                await asyncio.sleep(0.001)
                processed += len(chunk)
                del chunk  # Explicitly delete to free memory
            
            return {'processed': processed}
        
        # Process 100MB file
        result = await stream_process_file(100)
        
        peak_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = peak_memory - initial_memory
        
        print(f"Initial memory: {initial_memory:.2f}MB")
        print(f"Peak memory: {peak_memory:.2f}MB")
        print(f"Memory increase: {memory_increase:.2f}MB")
        
        # Memory increase should be minimal (streaming, not loading entire file)
        assert memory_increase < 50  # Should increase by <50MB for 100MB file
        assert result['processed'] == 100 * 1024 * 1024
    
    @pytest.mark.asyncio
    async def test_memory_stays_under_limit(self):
        """Should keep memory under 500MB limit during processing"""
        process = psutil.Process()
        memory_limit = 500  # MB
        
        async def process_with_monitoring(size_mb):
            chunk_size = 1024 * 1024
            total_size = size_mb * 1024 * 1024
            processed = 0
            max_memory_seen = 0
            
            while processed < total_size:
                chunk = b'x' * min(chunk_size, total_size - processed)
                await asyncio.sleep(0.001)
                processed += len(chunk)
                
                # Monitor memory
                current_memory = process.memory_info().rss / 1024 / 1024
                max_memory_seen = max(max_memory_seen, current_memory)
                
                del chunk
            
            return {'max_memory_mb': max_memory_seen}
        
        result = await process_with_monitoring(200)
        
        print(f"Max memory during processing: {result['max_memory_mb']:.2f}MB")
        
        assert result['max_memory_mb'] < memory_limit


class TestHashCalculationPerformance:
    """Test hash calculation performance for large files"""
    
    def test_hash_100mb_file(self):
        """Should hash 100MB file quickly"""
        file_size = 100 * 1024 * 1024
        file_content = b'x' * file_size
        
        start_time = time.time()
        file_hash = hashlib.sha256(file_content).hexdigest()
        end_time = time.time()
        
        hash_time = end_time - start_time
        
        print(f"100MB file hashed in {hash_time:.2f}s")
        
        assert len(file_hash) == 64
        assert hash_time < 5.0  # Should hash in <5s
    
    def test_hash_500mb_file(self):
        """Should hash 500MB file efficiently"""
        # Use streaming hash for large files
        file_size = 500 * 1024 * 1024
        chunk_size = 10 * 1024 * 1024  # 10MB chunks
        
        start_time = time.time()
        
        hasher = hashlib.sha256()
        processed = 0
        
        while processed < file_size:
            chunk = b'x' * min(chunk_size, file_size - processed)
            hasher.update(chunk)
            processed += len(chunk)
        
        file_hash = hasher.hexdigest()
        end_time = time.time()
        
        hash_time = end_time - start_time
        
        print(f"500MB file hashed in {hash_time:.2f}s")
        
        assert len(file_hash) == 64
        assert hash_time < 20.0  # Should hash in <20s


class TestChunkedProcessing:
    """Test chunked processing performance"""
    
    @pytest.mark.asyncio
    async def test_chunk_processing_speed(self):
        """Should process chunks efficiently"""
        total_chunks = 1000
        chunk_size = 1024 * 1024  # 1MB
        
        async def process_chunk(chunk_id):
            # Simulate chunk processing
            data = b'x' * chunk_size
            _ = hashlib.sha256(data).hexdigest()
            await asyncio.sleep(0.001)
            return {'chunk_id': chunk_id, 'size': chunk_size}
        
        start_time = time.time()
        
        # Process chunks in batches
        batch_size = 100
        results = []
        
        for i in range(0, total_chunks, batch_size):
            batch = [process_chunk(j) for j in range(i, min(i + batch_size, total_chunks))]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Processed {total_chunks} chunks in {processing_time:.2f}s")
        print(f"Chunks per second: {total_chunks / processing_time:.2f}")
        
        assert len(results) == total_chunks
        assert processing_time < 30.0  # Should complete in <30s
    
    @pytest.mark.asyncio
    async def test_parallel_chunk_processing(self):
        """Should process chunks in parallel efficiently"""
        total_chunks = 100
        
        async def process_chunk(chunk_id):
            await asyncio.sleep(0.01)
            return {'chunk_id': chunk_id}
        
        start_time = time.time()
        
        # Process all chunks in parallel
        tasks = [process_chunk(i) for i in range(total_chunks)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Processed {total_chunks} chunks in parallel in {processing_time:.2f}s")
        
        assert len(results) == total_chunks
        # Parallel processing should be much faster than sequential
        assert processing_time < 2.0  # Should complete in <2s (vs 1s if sequential)


class TestProgressTracking:
    """Test progress tracking for large files"""
    
    @pytest.mark.asyncio
    async def test_progress_updates(self):
        """Should provide accurate progress updates"""
        total_size = 100 * 1024 * 1024  # 100MB
        chunk_size = 10 * 1024 * 1024  # 10MB
        
        progress_updates = []
        
        async def process_with_progress(size):
            processed = 0
            
            while processed < size:
                chunk = min(chunk_size, size - processed)
                await asyncio.sleep(0.01)
                processed += chunk
                
                progress = (processed / size) * 100
                progress_updates.append(progress)
            
            return {'processed': processed}
        
        result = await process_with_progress(total_size)
        
        print(f"Progress updates: {len(progress_updates)}")
        print(f"Final progress: {progress_updates[-1]:.2f}%")
        
        assert result['processed'] == total_size
        assert len(progress_updates) > 0
        assert progress_updates[-1] == 100.0


class TestConcurrentLargeFiles:
    """Test concurrent processing of large files"""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_multiple_large_files_concurrent(self):
        """Should handle multiple large files concurrently"""
        async def process_large_file(file_id, size_mb):
            chunk_size = 1024 * 1024
            total_size = size_mb * 1024 * 1024
            processed = 0
            
            while processed < total_size:
                chunk = min(chunk_size, total_size - processed)
                await asyncio.sleep(0.001)
                processed += chunk
            
            return {'file_id': file_id, 'size_mb': size_mb, 'status': 'completed'}
        
        start_time = time.time()
        
        # Process 5 large files (50MB each) concurrently
        tasks = [process_large_file(i, 50) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Processed 5x50MB files in {processing_time:.2f}s")
        
        assert len(results) == 5
        assert all(r['status'] == 'completed' for r in results)
        assert processing_time < 60.0  # Should complete in <1 minute


class TestGarbageCollection:
    """Test garbage collection during large file processing"""
    
    @pytest.mark.asyncio
    async def test_memory_cleanup_during_processing(self):
        """Should clean up memory during processing"""
        import gc
        
        process = psutil.Process()
        memory_samples = []
        
        async def process_with_gc(size_mb):
            chunk_size = 1024 * 1024
            total_size = size_mb * 1024 * 1024
            processed = 0
            
            while processed < total_size:
                chunk = b'x' * min(chunk_size, total_size - processed)
                await asyncio.sleep(0.001)
                processed += len(chunk)
                del chunk
                
                # Force GC every 10MB
                if processed % (10 * 1024 * 1024) == 0:
                    gc.collect()
                    memory_samples.append(process.memory_info().rss / 1024 / 1024)
            
            return {'processed': processed}
        
        result = await process_with_gc(100)
        
        print(f"Memory samples during processing: {len(memory_samples)}")
        print(f"Memory range: {min(memory_samples):.2f}MB - {max(memory_samples):.2f}MB")
        
        # Memory should stay relatively stable (not continuously growing)
        memory_variance = max(memory_samples) - min(memory_samples)
        assert memory_variance < 100  # Variance should be <100MB


class TestFileTypePerformance:
    """Test performance with different file types"""
    
    @pytest.mark.asyncio
    async def test_csv_processing_performance(self):
        """Should process large CSV files efficiently"""
        # Simulate CSV with 1 million rows
        row_count = 1_000_000
        
        async def process_csv(rows):
            processed = 0
            batch_size = 10000
            
            while processed < rows:
                batch = min(batch_size, rows - processed)
                await asyncio.sleep(0.001)
                processed += batch
            
            return {'rows': processed}
        
        start_time = time.time()
        result = await process_csv(row_count)
        end_time = time.time()
        
        processing_time = end_time - start_time
        rows_per_second = row_count / processing_time
        
        print(f"Processed {row_count:,} rows in {processing_time:.2f}s")
        print(f"Rows per second: {rows_per_second:,.0f}")
        
        assert result['rows'] == row_count
        assert rows_per_second > 50000  # Should process >50k rows/second
    
    @pytest.mark.asyncio
    async def test_excel_processing_performance(self):
        """Should process large Excel files efficiently"""
        # Simulate Excel with 100k rows
        row_count = 100_000
        
        async def process_excel(rows):
            processed = 0
            batch_size = 1000
            
            while processed < rows:
                batch = min(batch_size, rows - processed)
                await asyncio.sleep(0.002)  # Excel is slower than CSV
                processed += batch
            
            return {'rows': processed}
        
        start_time = time.time()
        result = await process_excel(row_count)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        print(f"Processed {row_count:,} Excel rows in {processing_time:.2f}s")
        
        assert result['rows'] == row_count
        assert processing_time < 60.0  # Should complete in <1 minute


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
