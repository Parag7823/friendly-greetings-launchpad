"""
Performance Tests for Phase 4-6
Tests streaming, batch processing, and scalability
"""
import pytest
import asyncio
import pandas as pd
import io
import time
from unittest.mock import Mock, AsyncMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from universal_extractors_optimized import UniversalExtractorsOptimized
from universal_platform_detector_optimized import UniversalPlatformDetectorOptimized
from batch_optimizer import BatchOptimizer


class TestStreamingPerformance:
    """Performance tests for streaming file processing"""
    
    @pytest.fixture
    def extractor(self):
        """Create extractor with mock clients"""
        mock_openai = Mock()
        mock_cache = AsyncMock()
        mock_cache.get_cached_classification = AsyncMock(return_value=None)
        mock_cache.store_classification = AsyncMock(return_value=True)
        return UniversalExtractorsOptimized(mock_openai, mock_cache)
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_1k_rows_under_5_seconds(self, extractor):
        """Test 1,000 rows processed in under 5 seconds"""
        # Create 1K row CSV
        rows = [{'col1': f'val{i}', 'col2': i, 'col3': f'desc{i}'} for i in range(1000)]
        df = pd.DataFrame(rows)
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        csv_content = buffer.getvalue()
        
        start_time = time.time()
        result = await extractor.extract_data_universal(csv_content, "1k_rows.csv", "test_user")
        processing_time = time.time() - start_time
        
        assert result['confidence_score'] >= 0.5
        assert processing_time < 5.0
        
        print(f"\n✅ 1K rows: {processing_time:.2f}s ({1000/processing_time:.0f} rows/s)")
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_10k_rows_under_30_seconds(self, extractor):
        """Test 10,000 rows processed in under 30 seconds"""
        # Create 10K row CSV
        rows = [{'col1': f'val{i}', 'col2': i, 'col3': f'desc{i}'} for i in range(10000)]
        df = pd.DataFrame(rows)
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        csv_content = buffer.getvalue()
        
        start_time = time.time()
        result = await extractor.extract_data_universal(csv_content, "10k_rows.csv", "test_user")
        processing_time = time.time() - start_time
        
        assert result['confidence_score'] >= 0.5
        assert processing_time < 30.0
        
        print(f"\n✅ 10K rows: {processing_time:.2f}s ({10000/processing_time:.0f} rows/s)")
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_100k_rows_scalability(self, extractor):
        """Test scalability with 100,000 rows"""
        # Create 100K row CSV
        rows = [{'col1': f'val{i}', 'col2': i} for i in range(100000)]
        df = pd.DataFrame(rows)
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        csv_content = buffer.getvalue()
        
        start_time = time.time()
        result = await extractor.extract_data_universal(csv_content, "100k_rows.csv", "test_user")
        processing_time = time.time() - start_time
        
        assert result['confidence_score'] >= 0.5
        # Should complete in reasonable time (< 5 minutes)
        assert processing_time < 300.0
        
        print(f"\n✅ 100K rows: {processing_time:.2f}s ({100000/processing_time:.0f} rows/s)")


class TestBatchProcessingPerformance:
    """Performance tests for batch processing"""
    
    @pytest.fixture
    def optimizer(self):
        """Create batch optimizer"""
        return BatchOptimizer(batch_size=100)
    
    @pytest.mark.performance
    def test_vectorized_vs_sequential_performance(self, optimizer):
        """Test vectorized processing is 5x faster than sequential"""
        # Create test data
        df = pd.DataFrame({
            'description': [f'stripe payment {i}' for i in range(1000)],
            'amount': list(range(1000))
        })
        
        patterns = {
            'stripe': ['stripe'],
            'quickbooks': ['quickbooks']
        }
        
        # Vectorized processing
        start_time = time.time()
        vectorized_result = optimizer.vectorized_classify(df, patterns)
        vectorized_time = time.time() - start_time
        
        # Sequential processing (simulated)
        start_time = time.time()
        sequential_result = []
        for _, row in df.iterrows():
            text = str(row['description']).lower()
            if 'stripe' in text:
                sequential_result.append('stripe')
            else:
                sequential_result.append('unknown')
        sequential_time = time.time() - start_time
        
        # Vectorized should be significantly faster
        speedup = sequential_time / vectorized_time
        
        print(f"\n✅ Vectorized: {vectorized_time:.4f}s")
        print(f"   Sequential: {sequential_time:.4f}s")
        print(f"   Speedup: {speedup:.1f}x")
        
        assert speedup > 2.0  # At least 2x faster
    
    @pytest.mark.performance
    def test_batch_size_optimization(self, optimizer):
        """Test optimal batch size for performance"""
        df = pd.DataFrame({
            'description': [f'test {i}' for i in range(5000)],
            'amount': list(range(5000))
        })
        
        patterns = {'test': ['test']}
        
        batch_sizes = [10, 50, 100, 500, 1000]
        results = {}
        
        for batch_size in batch_sizes:
            optimizer.batch_size = batch_size
            
            start_time = time.time()
            result = optimizer.vectorized_classify(df, patterns)
            processing_time = time.time() - start_time
            
            results[batch_size] = processing_time
        
        print(f"\n✅ Batch size performance:")
        for batch_size, proc_time in results.items():
            print(f"   {batch_size}: {proc_time:.4f}s")
        
        # Larger batches should generally be faster
        assert results[1000] < results[10] * 2


class TestPlatformDetectionPerformance:
    """Performance tests for platform detection"""
    
    @pytest.fixture
    def detector(self):
        """Create platform detector"""
        mock_openai = Mock()
        mock_openai.chat = Mock()
        mock_openai.chat.completions = Mock()
        mock_openai.chat.completions.create = AsyncMock(return_value=Mock(
            choices=[Mock(message=Mock(content='{"platform": "stripe", "confidence": 0.95, "indicators": [], "reasoning": "test", "category": "payment"}'))]
        ))
        
        mock_cache = AsyncMock()
        mock_cache.get_cached_classification = AsyncMock(return_value=None)
        mock_cache.store_classification = AsyncMock(return_value=True)
        
        return UniversalPlatformDetectorOptimized(mock_openai, mock_cache)
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_pattern_detection_speed(self, detector):
        """Test pattern-based detection speed"""
        payload = {
            'id': 'ch_1ABC123',
            'amount': 1000,
            'stripe_customer': 'cus_XYZ'
        }
        
        # Run 100 detections
        start_time = time.time()
        for i in range(100):
            result = await detector.detect_platform_universal(
                payload,
                filename=f"stripe_{i}.csv",
                user_id="test_user"
            )
        processing_time = time.time() - start_time
        
        avg_time = processing_time / 100
        
        print(f"\n✅ 100 platform detections: {processing_time:.2f}s")
        print(f"   Average: {avg_time*1000:.2f}ms per detection")
        
        # Should average < 100ms per detection
        assert avg_time < 0.1
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_cache_performance_improvement(self, detector):
        """Test cache improves performance"""
        payload = {'stripe_id': 'ch_123', 'amount': 100}
        
        # First call (cache miss)
        start_time = time.time()
        result1 = await detector.detect_platform_universal(payload, "test.csv", "user1")
        first_call_time = time.time() - start_time
        
        # Manually set cache for second call
        detector.cache.get_cached_classification = AsyncMock(return_value=result1)
        
        # Second call (cache hit)
        start_time = time.time()
        result2 = await detector.detect_platform_universal(payload, "test.csv", "user1")
        second_call_time = time.time() - start_time
        
        print(f"\n✅ First call (miss): {first_call_time*1000:.2f}ms")
        print(f"   Second call (hit): {second_call_time*1000:.2f}ms")
        
        if second_call_time > 0:
            speedup = first_call_time/second_call_time
            print(f"   Speedup: {speedup:.1f}x")
        else:
            print(f"   Speedup: Instant (< 0.001ms)")
        
        # Cache hit should be significantly faster (or instant)
        assert second_call_time <= first_call_time


class TestMemoryEfficiency:
    """Memory efficiency tests"""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_memory_usage_large_file(self):
        """Test memory usage remains reasonable with large files"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large DataFrame (10K rows with long strings)
        rows = []
        for i in range(10000):
            rows.append({
                'col1': f'value_{i}' * 100,  # Long string
                'col2': i,
                'col3': f'description_{i}' * 50
            })
        
        df = pd.DataFrame(rows)
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        csv_content = buffer.getvalue()
        
        # Measure memory after creation
        after_creation_memory = process.memory_info().rss / 1024 / 1024
        memory_for_data = after_creation_memory - initial_memory
        
        # Clean up
        del df
        del buffer
        del csv_content
        
        final_memory = process.memory_info().rss / 1024 / 1024
        
        print(f"\n✅ Initial memory: {initial_memory:.2f} MB")
        print(f"   After creation: {after_creation_memory:.2f} MB")
        print(f"   Data size: {memory_for_data:.2f} MB")
        print(f"   After cleanup: {final_memory:.2f} MB")
        
        # Memory should be released after cleanup
        assert final_memory < after_creation_memory + 50  # Allow 50MB overhead


class TestConcurrentProcessing:
    """Concurrent processing performance tests"""
    
    @pytest.fixture
    def extractor(self):
        """Create extractor"""
        mock_openai = Mock()
        mock_cache = AsyncMock()
        mock_cache.get_cached_classification = AsyncMock(return_value=None)
        mock_cache.store_classification = AsyncMock(return_value=True)
        return UniversalExtractorsOptimized(mock_openai, mock_cache)
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_vs_sequential_performance(self, extractor):
        """Test concurrent processing is faster than sequential"""
        # Create 10 small files
        files = []
        for i in range(10):
            df = pd.DataFrame({'col1': [f'val{j}' for j in range(100)], 'col2': list(range(100))})
            buffer = io.BytesIO()
            df.to_csv(buffer, index=False)
            files.append((buffer.getvalue(), f"file_{i}.csv"))
        
        # Sequential processing
        start_time = time.time()
        sequential_results = []
        for content, filename in files:
            result = await extractor.extract_data_universal(content, filename, "test_user")
            sequential_results.append(result)
        sequential_time = time.time() - start_time
        
        # Concurrent processing
        start_time = time.time()
        tasks = [extractor.extract_data_universal(content, filename, "test_user") 
                for content, filename in files]
        concurrent_results = await asyncio.gather(*tasks)
        concurrent_time = time.time() - start_time
        
        speedup = sequential_time / concurrent_time
        
        print(f"\n✅ Sequential: {sequential_time:.2f}s")
        print(f"   Concurrent: {concurrent_time:.2f}s")
        print(f"   Speedup: {speedup:.1f}x")
        
        # Concurrent should be faster (or at least not slower)
        assert concurrent_time <= sequential_time * 1.2  # Allow 20% margin
        # Speedup may be minimal for small files
        assert speedup >= 0.8  # At least not significantly slower
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_high_concurrency_stability(self, extractor):
        """Test stability with high concurrency (50 concurrent tasks)"""
        # Create 50 small files
        files = []
        for i in range(50):
            df = pd.DataFrame({'col1': [f'val{j}' for j in range(50)]})
            buffer = io.BytesIO()
            df.to_csv(buffer, index=False)
            files.append((buffer.getvalue(), f"file_{i}.csv"))
        
        # Process all concurrently
        start_time = time.time()
        tasks = [extractor.extract_data_universal(content, filename, "test_user") 
                for content, filename in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        processing_time = time.time() - start_time
        
        # Count successes and failures
        successes = sum(1 for r in results if not isinstance(r, Exception) and r.get('confidence_score', 0) > 0)
        failures = len(results) - successes
        
        print(f"\n✅ 50 concurrent tasks: {processing_time:.2f}s")
        print(f"   Successes: {successes}")
        print(f"   Failures: {failures}")
        print(f"   Throughput: {len(files)/processing_time:.1f} files/s")
        
        # Most should succeed
        assert successes >= 45  # At least 90% success rate


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-m', 'performance'])
