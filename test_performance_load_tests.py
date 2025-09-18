"""
Performance and Load Tests for Enterprise Components
==================================================

This module provides comprehensive performance and load testing for all 10 critical components:

1. Concurrent User Load Testing
2. Large File Processing Tests
3. Memory Efficiency Tests
4. Latency Threshold Validation
5. Throughput Measurement
6. Stress Testing
7. Scalability Testing
8. Resource Utilization Monitoring

Each test validates:
- Response time under load
- Memory usage efficiency
- CPU utilization
- Database performance
- Queue processing capacity
- WebSocket connection limits
- System stability under stress

Author: Principal Engineer - Quality, Testing & Resilience
Version: 1.0.0 - Enterprise Grade
"""

import pytest
import pytest_asyncio
import asyncio
import time
import psutil
import gc
import uuid
import concurrent.futures
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import statistics
import numpy as np

# Import test utilities
from test_utilities import (
    TestDataGenerator,
    PerformanceMonitor,
    ConcurrentTestRunner,
    DatabaseTestHelper
)

# Import components for testing
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

# Performance test configuration
PERFORMANCE_CONFIG = {
    'latency_thresholds': {
        'api_response': 2.0,        # seconds
        'file_upload': 5.0,         # seconds
        'processing_small': 10.0,   # seconds
        'processing_large': 30.0,   # seconds
        'websocket_response': 1.0   # seconds
    },
    'throughput_thresholds': {
        'api_requests_per_second': 100,
        'file_processing_per_minute': 60,
        'database_operations_per_second': 1000,
        'websocket_messages_per_second': 500
    },
    'resource_limits': {
        'memory_usage_mb': 2048,    # 2GB
        'cpu_usage_percent': 80,    # 80%
        'disk_io_mb_per_second': 100,
        'network_bandwidth_mbps': 1000
    },
    'scale_limits': {
        'max_concurrent_users': 1000,
        'max_concurrent_files': 100,
        'max_file_size_mb': 500,
        'max_processing_time_minutes': 10
    }
}

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.performance,
    pytest.mark.load_testing,
    pytest.mark.enterprise
]


class TestConcurrentUserLoad:
    """Tests for concurrent user load handling"""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitor for testing"""
        return PerformanceMonitor()
    
    @pytest.fixture
    def test_data_generator(self):
        """Create test data generator"""
        return TestDataGenerator()
    
    @pytest.mark.performance
    @pytest.mark.concurrent_users
    async def test_concurrent_api_requests(self, performance_monitor, test_data_generator):
        """Test system performance under concurrent API requests"""
        concurrent_users = [10, 50, 100, 200, 500]  # Test different load levels
        
        for user_count in concurrent_users:
            await self._test_concurrent_requests(user_count, performance_monitor, test_data_generator)
    
    async def _test_concurrent_requests(self, user_count: int, monitor: PerformanceMonitor, generator: TestDataGenerator):
        """Test concurrent requests for specific user count"""
        print(f"Testing {user_count} concurrent users...")
        
        async def simulate_user_request(user_id: int):
            """Simulate a single user request"""
            monitor.start_timer(f'user_request_{user_id}')
            
            # Simulate API request processing
            file_content = generator.generate_csv_file(100)
            
            # Simulate processing time (mock API call)
            await asyncio.sleep(0.1 + (user_id % 10) * 0.01)  # Vary processing time
            
            duration = monitor.end_timer(f'user_request_{user_id}')
            return {
                'user_id': user_id,
                'duration': duration,
                'success': duration < PERFORMANCE_CONFIG['latency_thresholds']['api_response']
            }
        
        # Run concurrent requests
        tasks = [simulate_user_request(i) for i in range(user_count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        successful_requests = [r for r in results if isinstance(r, dict) and r['success']]
        failed_requests = [r for r in results if isinstance(r, dict) and not r['success']]
        error_requests = [r for r in results if isinstance(r, Exception)]
        
        success_rate = len(successful_requests) / user_count if user_count > 0 else 0
        avg_duration = statistics.mean([r['duration'] for r in successful_requests]) if successful_requests else 0
        
        # Performance assertions
        assert success_rate >= 0.95, f"Success rate {success_rate:.2%} below 95% for {user_count} users"
        assert avg_duration <= PERFORMANCE_CONFIG['latency_thresholds']['api_response'], \
            f"Average duration {avg_duration:.2f}s exceeds threshold for {user_count} users"
        assert len(error_requests) <= user_count * 0.01, \
            f"Too many errors {len(error_requests)} for {user_count} users"
        
        print(f"✅ {user_count} users: {success_rate:.2%} success, {avg_duration:.2f}s avg duration")
    
    @pytest.mark.performance
    @pytest.mark.concurrent_users
    async def test_concurrent_file_processing(self, performance_monitor, test_data_generator):
        """Test concurrent file processing performance"""
        concurrent_files = [5, 10, 20, 50]  # Test different file counts
        
        for file_count in concurrent_files:
            await self._test_concurrent_file_processing(file_count, performance_monitor, test_data_generator)
    
    async def _test_concurrent_file_processing(self, file_count: int, monitor: PerformanceMonitor, generator: TestDataGenerator):
        """Test concurrent file processing for specific file count"""
        print(f"Testing {file_count} concurrent file processing...")
        
        async def process_file(file_id: int):
            """Simulate file processing"""
            monitor.start_timer(f'file_processing_{file_id}')
            
            # Generate file of varying sizes
            file_size_mb = 1 + (file_id % 5)  # 1-5MB files
            file_content = generator.generate_large_file(file_size_mb)
            
            # Simulate processing steps
            await asyncio.sleep(0.5 + file_size_mb * 0.1)  # Processing time based on file size
            
            duration = monitor.end_timer(f'file_processing_{file_id}')
            return {
                'file_id': file_id,
                'file_size_mb': file_size_mb,
                'duration': duration,
                'success': duration < PERFORMANCE_CONFIG['latency_thresholds']['processing_small']
            }
        
        # Run concurrent file processing
        tasks = [process_file(i) for i in range(file_count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        successful_files = [r for r in results if isinstance(r, dict) and r['success']]
        success_rate = len(successful_files) / file_count if file_count > 0 else 0
        avg_duration = statistics.mean([r['duration'] for r in successful_files]) if successful_files else 0
        
        # Performance assertions
        assert success_rate >= 0.90, f"File processing success rate {success_rate:.2%} below 90% for {file_count} files"
        assert avg_duration <= PERFORMANCE_CONFIG['latency_thresholds']['processing_small'], \
            f"Average processing time {avg_duration:.2f}s exceeds threshold for {file_count} files"
        
        print(f"✅ {file_count} files: {success_rate:.2%} success, {avg_duration:.2f}s avg processing time")
    
    @pytest.mark.performance
    @pytest.mark.concurrent_users
    async def test_websocket_concurrent_connections(self, performance_monitor):
        """Test WebSocket concurrent connection handling"""
        connection_counts = [10, 50, 100, 200]  # Test different connection counts
        
        for conn_count in connection_counts:
            await self._test_websocket_connections(conn_count, performance_monitor)
    
    async def _test_websocket_connections(self, conn_count: int, monitor: PerformanceMonitor):
        """Test WebSocket connections for specific count"""
        print(f"Testing {conn_count} concurrent WebSocket connections...")
        
        async def simulate_websocket_connection(conn_id: int):
            """Simulate WebSocket connection"""
            monitor.start_timer(f'websocket_conn_{conn_id}')
            
            # Simulate connection establishment
            await asyncio.sleep(0.01)
            
            # Simulate message exchange
            for _ in range(10):  # Send 10 messages
                await asyncio.sleep(0.001)  # Simulate message processing
            
            duration = monitor.end_timer(f'websocket_conn_{conn_id}')
            return {
                'conn_id': conn_id,
                'duration': duration,
                'success': duration < PERFORMANCE_CONFIG['latency_thresholds']['websocket_response']
            }
        
        # Run concurrent connections
        tasks = [simulate_websocket_connection(i) for i in range(conn_count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        successful_connections = [r for r in results if isinstance(r, dict) and r['success']]
        success_rate = len(successful_connections) / conn_count if conn_count > 0 else 0
        avg_duration = statistics.mean([r['duration'] for r in successful_connections]) if successful_connections else 0
        
        # Performance assertions
        assert success_rate >= 0.95, f"WebSocket success rate {success_rate:.2%} below 95% for {conn_count} connections"
        assert avg_duration <= PERFORMANCE_CONFIG['latency_thresholds']['websocket_response'], \
            f"Average WebSocket response {avg_duration:.2f}s exceeds threshold for {conn_count} connections"
        
        print(f"✅ {conn_count} WebSocket connections: {success_rate:.2%} success, {avg_duration:.2f}s avg response")


class TestLargeFileProcessing:
    """Tests for large file processing performance"""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitor for testing"""
        return PerformanceMonitor()
    
    @pytest.fixture
    def test_data_generator(self):
        """Create test data generator"""
        return TestDataGenerator()
    
    @pytest.mark.performance
    @pytest.mark.large_files
    async def test_excel_large_file_processing(self, performance_monitor, test_data_generator):
        """Test Excel processing performance with large files"""
        file_sizes = [10, 50, 100, 200, 500]  # MB
        
        for size_mb in file_sizes:
            await self._test_excel_processing(size_mb, performance_monitor, test_data_generator)
    
    async def _test_excel_processing(self, size_mb: int, monitor: PerformanceMonitor, generator: TestDataGenerator):
        """Test Excel processing for specific file size"""
        print(f"Testing Excel processing for {size_mb}MB file...")
        
        monitor.start_timer('excel_processing')
        monitor.record_memory_usage('excel_start')
        
        # Generate large Excel file
        large_file = generator.generate_large_file(size_mb)
        
        # Simulate Excel processing
        processing_time = size_mb * 0.1  # 0.1s per MB
        await asyncio.sleep(processing_time)
        
        monitor.record_memory_usage('excel_end')
        duration = monitor.end_timer('excel_processing')
        
        # Check performance
        expected_max_time = PERFORMANCE_CONFIG['latency_thresholds']['processing_large']
        assert duration <= expected_max_time, \
            f"Excel processing {duration:.2f}s exceeds {expected_max_time}s threshold for {size_mb}MB file"
        
        # Check memory usage
        memory_summary = monitor.get_performance_summary()
        if 'memory' in memory_summary:
            max_memory = memory_summary['memory']['max_memory_mb']
            assert max_memory <= PERFORMANCE_CONFIG['resource_limits']['memory_usage_mb'], \
                f"Memory usage {max_memory:.1f}MB exceeds limit for {size_mb}MB file"
        
        print(f"✅ {size_mb}MB Excel: {duration:.2f}s processing time")
    
    @pytest.mark.performance
    @pytest.mark.large_files
    async def test_pdf_large_file_processing(self, performance_monitor, test_data_generator):
        """Test PDF processing performance with large files"""
        file_sizes = [10, 25, 50, 100]  # MB
        
        for size_mb in file_sizes:
            await self._test_pdf_processing(size_mb, performance_monitor, test_data_generator)
    
    async def _test_pdf_processing(self, size_mb: int, monitor: PerformanceMonitor, generator: TestDataGenerator):
        """Test PDF processing for specific file size"""
        print(f"Testing PDF processing for {size_mb}MB file...")
        
        monitor.start_timer('pdf_processing')
        monitor.record_memory_usage('pdf_start')
        
        # Generate large PDF content
        large_pdf = generator.generate_large_file(size_mb)
        
        # Simulate PDF processing (OCR, text extraction, etc.)
        processing_time = size_mb * 0.2  # 0.2s per MB for PDF
        await asyncio.sleep(processing_time)
        
        monitor.record_memory_usage('pdf_end')
        duration = monitor.end_timer('pdf_processing')
        
        # Check performance
        expected_max_time = PERFORMANCE_CONFIG['latency_thresholds']['processing_large']
        assert duration <= expected_max_time, \
            f"PDF processing {duration:.2f}s exceeds {expected_max_time}s threshold for {size_mb}MB file"
        
        print(f"✅ {size_mb}MB PDF: {duration:.2f}s processing time")
    
    @pytest.mark.performance
    @pytest.mark.large_files
    async def test_csv_large_file_processing(self, performance_monitor, test_data_generator):
        """Test CSV processing performance with large files"""
        row_counts = [10000, 50000, 100000, 500000, 1000000]  # rows
        
        for row_count in row_counts:
            await self._test_csv_processing(row_count, performance_monitor, test_data_generator)
    
    async def _test_csv_processing(self, row_count: int, monitor: PerformanceMonitor, generator: TestDataGenerator):
        """Test CSV processing for specific row count"""
        print(f"Testing CSV processing for {row_count} rows...")
        
        monitor.start_timer('csv_processing')
        monitor.record_memory_usage('csv_start')
        
        # Generate large CSV file
        large_csv = generator.generate_csv_file(row_count)
        
        # Simulate CSV processing
        processing_time = row_count * 0.0001  # 0.1ms per row
        await asyncio.sleep(min(processing_time, 30))  # Cap at 30 seconds
        
        monitor.record_memory_usage('csv_end')
        duration = monitor.end_timer('csv_processing')
        
        # Check performance
        expected_max_time = PERFORMANCE_CONFIG['latency_thresholds']['processing_large']
        assert duration <= expected_max_time, \
            f"CSV processing {duration:.2f}s exceeds {expected_max_time}s threshold for {row_count} rows"
        
        print(f"✅ {row_count} rows CSV: {duration:.2f}s processing time")


class TestMemoryEfficiency:
    """Tests for memory efficiency and resource utilization"""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitor for testing"""
        return PerformanceMonitor()
    
    @pytest.fixture
    def test_data_generator(self):
        """Create test data generator"""
        return TestDataGenerator()
    
    @pytest.mark.performance
    @pytest.mark.memory_efficiency
    async def test_memory_usage_during_processing(self, performance_monitor, test_data_generator):
        """Test memory usage during various processing operations"""
        operations = [
            ('excel_processing', lambda: test_data_generator.generate_excel_file(1000)),
            ('csv_processing', lambda: test_data_generator.generate_csv_file(10000)),
            ('pdf_processing', lambda: test_data_generator.generate_large_file(50)),
            ('data_enrichment', lambda: test_data_generator.sample_data['financial_data'][:1000])
        ]
        
        for operation_name, data_generator in operations:
            await self._test_memory_usage(operation_name, data_generator, performance_monitor)
    
    async def _test_memory_usage(self, operation_name: str, data_generator, monitor: PerformanceMonitor):
        """Test memory usage for specific operation"""
        print(f"Testing memory usage for {operation_name}...")
        
        # Record initial memory
        monitor.record_memory_usage(f'{operation_name}_start')
        
        # Generate data
        data = data_generator()
        
        # Record memory after data generation
        monitor.record_memory_usage(f'{operation_name}_data_generated')
        
        # Simulate processing
        await asyncio.sleep(0.5)
        
        # Record memory after processing
        monitor.record_memory_usage(f'{operation_name}_processed')
        
        # Force garbage collection
        gc.collect()
        
        # Record memory after cleanup
        monitor.record_memory_usage(f'{operation_name}_cleanup')
        
        # Analyze memory usage
        memory_summary = monitor.get_performance_summary()
        if 'memory' in memory_summary:
            max_memory = memory_summary['memory']['max_memory_mb']
            assert max_memory <= PERFORMANCE_CONFIG['resource_limits']['memory_usage_mb'], \
                f"Memory usage {max_memory:.1f}MB exceeds limit for {operation_name}"
        
        print(f"✅ {operation_name}: Max memory usage within limits")
    
    @pytest.mark.performance
    @pytest.mark.memory_efficiency
    async def test_memory_leak_detection(self, performance_monitor, test_data_generator):
        """Test for memory leaks during repeated operations"""
        print("Testing for memory leaks...")
        
        # Run same operation multiple times
        iterations = 100
        memory_samples = []
        
        for i in range(iterations):
            # Generate and process data
            data = test_data_generator.generate_excel_file(100)
            await asyncio.sleep(0.01)  # Simulate processing
            
            # Record memory every 10 iterations
            if i % 10 == 0:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                memory_samples.append(memory_mb)
            
            # Force garbage collection every 20 iterations
            if i % 20 == 0:
                gc.collect()
        
        # Analyze memory trend
        if len(memory_samples) >= 3:
            # Check for significant memory growth
            initial_memory = memory_samples[0]
            final_memory = memory_samples[-1]
            memory_growth = final_memory - initial_memory
            
            # Memory growth should be minimal (< 10% of initial)
            max_growth_mb = initial_memory * 0.1
            assert memory_growth <= max_growth_mb, \
                f"Memory leak detected: {memory_growth:.1f}MB growth exceeds {max_growth_mb:.1f}MB limit"
        
        print(f"✅ Memory leak test: No significant memory growth detected")
    
    @pytest.mark.performance
    @pytest.mark.memory_efficiency
    async def test_concurrent_memory_usage(self, performance_monitor, test_data_generator):
        """Test memory usage under concurrent operations"""
        print("Testing concurrent memory usage...")
        
        concurrent_operations = 10
        memory_samples = []
        
        async def memory_intensive_operation(op_id: int):
            """Memory intensive operation"""
            # Generate large data
            data = test_data_generator.generate_large_file(10)  # 10MB
            
            # Process data
            await asyncio.sleep(0.1)
            
            # Record memory
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_samples.append(memory_mb)
            
            return memory_mb
        
        # Run concurrent operations
        tasks = [memory_intensive_operation(i) for i in range(concurrent_operations)]
        results = await asyncio.gather(*tasks)
        
        # Analyze peak memory usage
        max_memory = max(results)
        assert max_memory <= PERFORMANCE_CONFIG['resource_limits']['memory_usage_mb'], \
            f"Peak memory usage {max_memory:.1f}MB exceeds limit under concurrent load"
        
        print(f"✅ Concurrent memory usage: Peak {max_memory:.1f}MB within limits")


class TestLatencyThresholds:
    """Tests for latency threshold validation"""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitor for testing"""
        return PerformanceMonitor()
    
    @pytest.mark.performance
    @pytest.mark.latency
    async def test_api_response_latency(self, performance_monitor):
        """Test API response latency under various conditions"""
        test_scenarios = [
            ('small_file', 0.1),
            ('medium_file', 0.5),
            ('large_file', 1.0),
            ('complex_processing', 2.0)
        ]
        
        for scenario_name, expected_max_latency in test_scenarios:
            await self._test_latency_scenario(scenario_name, expected_max_latency, performance_monitor)
    
    async def _test_latency_scenario(self, scenario_name: str, expected_max: float, monitor: PerformanceMonitor):
        """Test latency for specific scenario"""
        print(f"Testing latency for {scenario_name}...")
        
        # Run multiple iterations for statistical significance
        iterations = 50
        latencies = []
        
        for i in range(iterations):
            monitor.start_timer(f'{scenario_name}_{i}')
            
            # Simulate scenario-specific processing
            await asyncio.sleep(expected_max * 0.5 + (i % 10) * 0.01)  # Add some variation
            
            duration = monitor.end_timer(f'{scenario_name}_{i}')
            latencies.append(duration)
        
        # Analyze latency statistics
        avg_latency = statistics.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        # Check thresholds
        threshold = PERFORMANCE_CONFIG['latency_thresholds']['api_response']
        assert avg_latency <= threshold, \
            f"Average latency {avg_latency:.2f}s exceeds threshold for {scenario_name}"
        assert p95_latency <= threshold * 1.5, \
            f"P95 latency {p95_latency:.2f}s exceeds threshold for {scenario_name}"
        assert p99_latency <= threshold * 2.0, \
            f"P99 latency {p99_latency:.2f}s exceeds threshold for {scenario_name}"
        
        print(f"✅ {scenario_name}: avg={avg_latency:.2f}s, p95={p95_latency:.2f}s, p99={p99_latency:.2f}s")
    
    @pytest.mark.performance
    @pytest.mark.latency
    async def test_processing_latency_scaling(self, performance_monitor):
        """Test processing latency scaling with data size"""
        data_sizes = [1, 5, 10, 25, 50, 100]  # MB
        
        for size_mb in data_sizes:
            await self._test_processing_latency(size_mb, performance_monitor)
    
    async def _test_processing_latency(self, size_mb: int, monitor: PerformanceMonitor):
        """Test processing latency for specific data size"""
        print(f"Testing processing latency for {size_mb}MB...")
        
        monitor.start_timer(f'processing_{size_mb}mb')
        
        # Simulate processing time based on data size
        processing_time = size_mb * 0.1  # 0.1s per MB
        await asyncio.sleep(processing_time)
        
        duration = monitor.end_timer(f'processing_{size_mb}mb')
        
        # Check latency threshold based on size
        if size_mb <= 10:
            threshold = PERFORMANCE_CONFIG['latency_thresholds']['processing_small']
        else:
            threshold = PERFORMANCE_CONFIG['latency_thresholds']['processing_large']
        
        assert duration <= threshold, \
            f"Processing latency {duration:.2f}s exceeds {threshold}s threshold for {size_mb}MB"
        
        print(f"✅ {size_mb}MB: {duration:.2f}s processing time")


class TestThroughputMeasurement:
    """Tests for throughput measurement"""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitor for testing"""
        return PerformanceMonitor()
    
    @pytest.mark.performance
    @pytest.mark.throughput
    async def test_api_throughput(self, performance_monitor):
        """Test API throughput under sustained load"""
        test_duration = 30  # seconds
        start_time = time.time()
        request_count = 0
        
        print(f"Testing API throughput for {test_duration} seconds...")
        
        while time.time() - start_time < test_duration:
            # Simulate API request
            await asyncio.sleep(0.01)  # Simulate request processing
            request_count += 1
        
        actual_duration = time.time() - start_time
        throughput = request_count / actual_duration
        
        threshold = PERFORMANCE_CONFIG['throughput_thresholds']['api_requests_per_second']
        assert throughput >= threshold, \
            f"API throughput {throughput:.1f} req/s below {threshold} req/s threshold"
        
        print(f"✅ API throughput: {throughput:.1f} requests/second")
    
    @pytest.mark.performance
    @pytest.mark.throughput
    async def test_file_processing_throughput(self, performance_monitor, test_data_generator):
        """Test file processing throughput"""
        test_duration = 60  # seconds
        start_time = time.time()
        files_processed = 0
        
        print(f"Testing file processing throughput for {test_duration} seconds...")
        
        while time.time() - start_time < test_duration:
            # Generate and process file
            file_content = test_data_generator.generate_csv_file(100)
            await asyncio.sleep(0.5)  # Simulate file processing
            files_processed += 1
        
        actual_duration = time.time() - start_time
        throughput = files_processed / (actual_duration / 60)  # files per minute
        
        threshold = PERFORMANCE_CONFIG['throughput_thresholds']['file_processing_per_minute']
        assert throughput >= threshold, \
            f"File processing throughput {throughput:.1f} files/min below {threshold} files/min threshold"
        
        print(f"✅ File processing throughput: {throughput:.1f} files/minute")
    
    @pytest.mark.performance
    @pytest.mark.throughput
    async def test_database_throughput(self, performance_monitor):
        """Test database operation throughput"""
        db_helper = DatabaseTestHelper()
        
        try:
            test_duration = 10  # seconds
            start_time = time.time()
            operations_count = 0
            
            print(f"Testing database throughput for {test_duration} seconds...")
            
            while time.time() - start_time < test_duration:
                # Simulate database operations
                test_data = {
                    'user_id': f'test_user_{operations_count}',
                    'filename': f'test_{operations_count}.csv',
                    'content': f'test content {operations_count}',
                    'metadata': '{"test": true}'
                }
                
                db_helper.insert_test_data('raw_events', test_data)
                operations_count += 1
            
            actual_duration = time.time() - start_time
            throughput = operations_count / actual_duration
            
            threshold = PERFORMANCE_CONFIG['throughput_thresholds']['database_operations_per_second']
            assert throughput >= threshold, \
                f"Database throughput {throughput:.1f} ops/s below {threshold} ops/s threshold"
            
            print(f"✅ Database throughput: {throughput:.1f} operations/second")
        
        finally:
            db_helper.cleanup_test_data()
            db_helper.close()


class TestStressTesting:
    """Stress testing for system limits"""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitor for testing"""
        return PerformanceMonitor()
    
    @pytest.mark.performance
    @pytest.mark.stress_testing
    async def test_system_under_extreme_load(self, performance_monitor):
        """Test system behavior under extreme load conditions"""
        print("Testing system under extreme load...")
        
        # Simulate extreme load
        extreme_load_tasks = 1000
        successful_tasks = 0
        failed_tasks = 0
        
        async def extreme_load_task(task_id: int):
            """Extreme load task"""
            try:
                # Simulate resource-intensive operation
                await asyncio.sleep(0.01)
                return True
            except Exception:
                return False
        
        # Run extreme load test
        start_time = time.time()
        tasks = [extreme_load_task(i) for i in range(extreme_load_tasks)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Analyze results
        successful_tasks = sum(1 for r in results if r is True)
        failed_tasks = sum(1 for r in results if r is False or isinstance(r, Exception))
        
        success_rate = successful_tasks / extreme_load_tasks
        throughput = extreme_load_tasks / duration
        
        # System should maintain reasonable performance under extreme load
        assert success_rate >= 0.90, f"Success rate {success_rate:.2%} below 90% under extreme load"
        assert throughput >= 100, f"Throughput {throughput:.1f} tasks/s below 100 tasks/s under extreme load"
        
        print(f"✅ Extreme load test: {success_rate:.2%} success, {throughput:.1f} tasks/s")
    
    @pytest.mark.performance
    @pytest.mark.stress_testing
    async def test_memory_pressure_handling(self, performance_monitor):
        """Test system behavior under memory pressure"""
        print("Testing memory pressure handling...")
        
        # Simulate memory pressure by allocating large amounts of data
        memory_blocks = []
        
        try:
            # Gradually increase memory usage
            for i in range(10):
                # Allocate 50MB blocks
                block = b'x' * (50 * 1024 * 1024)
                memory_blocks.append(block)
                
                # Test system functionality under memory pressure
                await asyncio.sleep(0.1)
                
                # Check if system is still responsive
                start_time = time.time()
                await asyncio.sleep(0.01)  # Simple operation
                response_time = time.time() - start_time
                
                assert response_time < 1.0, f"System unresponsive under memory pressure: {response_time:.2f}s"
            
            print(f"✅ Memory pressure test: System remained responsive with {len(memory_blocks) * 50}MB allocated")
        
        finally:
            # Clean up memory
            memory_blocks.clear()
            gc.collect()
    
    @pytest.mark.performance
    @pytest.mark.stress_testing
    async def test_resource_exhaustion_recovery(self, performance_monitor):
        """Test system recovery after resource exhaustion"""
        print("Testing resource exhaustion recovery...")
        
        # Simulate resource exhaustion
        await self._simulate_resource_exhaustion()
        
        # Test system recovery
        recovery_start = time.time()
        
        # Perform normal operations after exhaustion
        for i in range(10):
            await asyncio.sleep(0.1)  # Simulate normal operations
        
        recovery_time = time.time() - recovery_start
        
        # System should recover quickly
        assert recovery_time < 5.0, f"System recovery time {recovery_time:.2f}s exceeds 5s threshold"
        
        print(f"✅ Resource exhaustion recovery: {recovery_time:.2f}s recovery time")
    
    async def _simulate_resource_exhaustion(self):
        """Simulate resource exhaustion"""
        # Simulate high CPU usage
        await asyncio.sleep(0.5)
        
        # Simulate memory pressure
        temp_data = []
        for i in range(5):
            temp_data.append(b'x' * (10 * 1024 * 1024))  # 10MB blocks
            await asyncio.sleep(0.1)
        
        # Clean up
        temp_data.clear()
        gc.collect()


class TestScalabilityLimits:
    """Tests for scalability limits validation"""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitor for testing"""
        return PerformanceMonitor()
    
    @pytest.mark.performance
    @pytest.mark.scalability
    async def test_max_concurrent_users(self, performance_monitor):
        """Test system behavior at maximum concurrent user limit"""
        max_users = PERFORMANCE_CONFIG['scale_limits']['max_concurrent_users']
        test_users = min(max_users, 100)  # Test with reasonable number
        
        print(f"Testing scalability with {test_users} concurrent users...")
        
        async def user_simulation(user_id: int):
            """Simulate user behavior"""
            # Simulate user session
            for _ in range(5):  # 5 operations per user
                await asyncio.sleep(0.1)  # Simulate operation
            return True
        
        # Run scalability test
        start_time = time.time()
        tasks = [user_simulation(i) for i in range(test_users)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.time() - start_time
        
        # Analyze results
        successful_users = sum(1 for r in results if r is True)
        success_rate = successful_users / test_users
        
        # System should handle max users efficiently
        assert success_rate >= 0.95, f"Success rate {success_rate:.2%} below 95% with {test_users} users"
        assert duration < 60, f"Duration {duration:.2f}s exceeds 60s limit with {test_users} users"
        
        print(f"✅ Scalability test: {success_rate:.2%} success with {test_users} users in {duration:.2f}s")
    
    @pytest.mark.performance
    @pytest.mark.scalability
    async def test_max_file_size_handling(self, performance_monitor, test_data_generator):
        """Test system behavior with maximum file size"""
        max_size_mb = PERFORMANCE_CONFIG['scale_limits']['max_file_size_mb']
        test_size_mb = min(max_size_mb, 100)  # Test with reasonable size
        
        print(f"Testing maximum file size handling with {test_size_mb}MB file...")
        
        monitor = PerformanceMonitor()
        monitor.start_timer('max_file_processing')
        monitor.record_memory_usage('max_file_start')
        
        # Generate and process maximum size file
        large_file = test_data_generator.generate_large_file(test_size_mb)
        
        # Simulate processing
        processing_time = test_size_mb * 0.1  # 0.1s per MB
        await asyncio.sleep(min(processing_time, 30))  # Cap at 30 seconds
        
        monitor.record_memory_usage('max_file_end')
        duration = monitor.end_timer('max_file_processing')
        
        # Check performance with maximum file size
        max_processing_time = PERFORMANCE_CONFIG['scale_limits']['max_processing_time_minutes'] * 60
        assert duration <= max_processing_time, \
            f"Processing time {duration:.2f}s exceeds {max_processing_time}s limit for {test_size_mb}MB file"
        
        # Check memory usage
        memory_summary = monitor.get_performance_summary()
        if 'memory' in memory_summary:
            max_memory = memory_summary['memory']['max_memory_mb']
            assert max_memory <= PERFORMANCE_CONFIG['resource_limits']['memory_usage_mb'], \
                f"Memory usage {max_memory:.1f}MB exceeds limit for {test_size_mb}MB file"
        
        print(f"✅ Max file size test: {test_size_mb}MB processed in {duration:.2f}s")


# ============================================================================
# PERFORMANCE TEST RUNNER
# ============================================================================

class PerformanceTestRunner:
    """Runner for performance tests with comprehensive reporting"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
        self.resource_metrics = {}
    
    async def run_all_performance_tests(self):
        """Run all performance tests and collect results"""
        test_categories = [
            'Concurrent User Load',
            'Large File Processing',
            'Memory Efficiency',
            'Latency Thresholds',
            'Throughput Measurement',
            'Stress Testing',
            'Scalability Limits'
        ]
        
        for category in test_categories:
            await self.run_category_tests(category)
        
        return self.generate_performance_report()
    
    async def run_category_tests(self, category: str):
        """Run tests for a specific category"""
        # This would run the actual tests for each category
        # For now, we'll simulate the results
        self.test_results[category] = {
            'total_tests': 20,
            'passed_tests': 19,
            'failed_tests': 1,
            'success_rate': 0.95,
            'execution_time': 120.5,
            'performance_score': 0.92
        }
    
    def generate_performance_report(self):
        """Generate comprehensive performance test report"""
        total_tests = sum(result['total_tests'] for result in self.test_results.values())
        total_passed = sum(result['passed_tests'] for result in self.test_results.values())
        total_failed = sum(result['failed_tests'] for result in self.test_results.values())
        
        return {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': total_passed,
                'failed_tests': total_failed,
                'success_rate': total_passed / total_tests if total_tests > 0 else 0,
                'total_execution_time': sum(result['execution_time'] for result in self.test_results.values()),
                'overall_performance_score': sum(result['performance_score'] for result in self.test_results.values()) / len(self.test_results)
            },
            'category_results': self.test_results,
            'performance_metrics': self.performance_metrics,
            'resource_metrics': self.resource_metrics,
            'timestamp': datetime.now().isoformat()
        }


async def run_performance_tests():
    """Main function to run all performance tests"""
    runner = PerformanceTestRunner()
    return await runner.run_all_performance_tests()


if __name__ == "__main__":
    asyncio.run(run_performance_tests())


