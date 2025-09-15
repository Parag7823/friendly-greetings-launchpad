"""
Test suite for the observability system.
Tests structured logging, metrics collection, and monitoring capabilities.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import Dict, Any

# Test the observability system
class TestObservabilitySystem:
    """Test suite for the observability system"""
    
    @pytest.fixture
    def structured_logger(self):
        from observability_system import StructuredLogger
        return StructuredLogger("test_app")
    
    @pytest.fixture
    def metrics_collector(self):
        from observability_system import MetricsCollector
        return MetricsCollector()
    
    @pytest.fixture
    def performance_monitor(self, metrics_collector, structured_logger):
        from observability_system import PerformanceMonitor
        return PerformanceMonitor(metrics_collector, structured_logger)
    
    @pytest.fixture
    def health_checker(self, structured_logger):
        from observability_system import HealthChecker
        return HealthChecker(structured_logger)
    
    @pytest.fixture
    def observability_system(self):
        from observability_system import ObservabilitySystem
        return ObservabilitySystem("test_app")
    
    def test_structured_logger_basic_logging(self, structured_logger):
        """Test basic structured logging functionality"""
        # Test info logging
        with patch.object(structured_logger.logger, 'info') as mock_info:
            structured_logger.info("Test message", {"key": "value"})
            mock_info.assert_called_once()
            
            # Check that the logged message contains JSON
            call_args = mock_info.call_args[0][0]
            assert "Test message" in call_args
            assert "key" in call_args
            assert "value" in call_args
    
    def test_structured_logger_context_management(self, structured_logger):
        """Test context management in structured logger"""
        # Test request context
        structured_logger.set_request_context(
            user_id="test_user",
            request_id="test_request",
            operation="test_operation"
        )
        
        with patch.object(structured_logger.logger, 'info') as mock_info:
            structured_logger.info("Test message")
            call_args = mock_info.call_args[0][0]
            assert "test_user" in call_args
            assert "test_request" in call_args
            assert "test_operation" in call_args
        
        # Test context stack
        structured_logger.push_context({"stack_key": "stack_value"})
        
        with patch.object(structured_logger.logger, 'info') as mock_info:
            structured_logger.info("Test message")
            call_args = mock_info.call_args[0][0]
            assert "stack_key" in call_args
            assert "stack_value" in call_args
        
        # Test context manager
        with structured_logger.with_context({"context_key": "context_value"}):
            with patch.object(structured_logger.logger, 'info') as mock_info:
                structured_logger.info("Test message")
                call_args = mock_info.call_args[0][0]
                assert "context_key" in call_args
                assert "context_value" in call_args
    
    def test_structured_logger_error_logging(self, structured_logger):
        """Test error logging with exceptions"""
        test_error = ValueError("Test error")
        
        with patch.object(structured_logger.logger, 'error') as mock_error:
            structured_logger.error("Error message", error=test_error)
            mock_error.assert_called_once()
            
            call_args = mock_error.call_args[0][0]
            assert "Error message" in call_args
            assert "Test error" in call_args
    
    def test_metrics_collector_counters(self, metrics_collector):
        """Test counter metrics collection"""
        # Test increment counter
        metrics_collector.increment_counter("test_counter", 5.0)
        assert metrics_collector.get_counter("test_counter") == 5.0
        
        # Test increment again
        metrics_collector.increment_counter("test_counter", 3.0)
        assert metrics_collector.get_counter("test_counter") == 8.0
        
        # Test with labels
        metrics_collector.increment_counter("labeled_counter", 2.0, {"label1": "value1"})
        assert metrics_collector.get_counter("labeled_counter") == 2.0
    
    def test_metrics_collector_gauges(self, metrics_collector):
        """Test gauge metrics collection"""
        # Test set gauge
        metrics_collector.set_gauge("test_gauge", 42.0)
        assert metrics_collector.get_gauge("test_gauge") == 42.0
        
        # Test update gauge
        metrics_collector.set_gauge("test_gauge", 100.0)
        assert metrics_collector.get_gauge("test_gauge") == 100.0
        
        # Test with labels
        metrics_collector.set_gauge("labeled_gauge", 50.0, {"label1": "value1"})
        assert metrics_collector.get_gauge("labeled_gauge") == 50.0
    
    def test_metrics_collector_histograms(self, metrics_collector):
        """Test histogram metrics collection"""
        # Add some values
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in values:
            metrics_collector.observe_histogram("test_histogram", value)
        
        # Get statistics
        stats = metrics_collector.get_histogram_stats("test_histogram")
        
        assert stats['count'] == 5
        assert stats['min'] == 1.0
        assert stats['max'] == 5.0
        assert stats['mean'] == 3.0
        assert stats['p50'] == 3.0
        assert stats['p95'] == 5.0
        assert stats['p99'] == 5.0
    
    def test_metrics_collector_timers(self, metrics_collector):
        """Test timer metrics collection"""
        # Add some durations
        durations = [0.1, 0.2, 0.3, 0.4, 0.5]
        for duration in durations:
            metrics_collector.record_timer("test_timer", duration)
        
        # Get statistics
        stats = metrics_collector.get_timer_stats("test_timer")
        
        assert stats['count'] == 5
        assert stats['min'] == 0.1
        assert stats['max'] == 0.5
        assert stats['mean'] == 0.3
        assert stats['p50'] == 0.3
        assert stats['p95'] == 0.5
        assert stats['p99'] == 0.5
    
    def test_metrics_collector_all_metrics(self, metrics_collector):
        """Test getting all metrics"""
        # Add some metrics
        metrics_collector.increment_counter("test_counter", 10.0)
        metrics_collector.set_gauge("test_gauge", 25.0)
        metrics_collector.observe_histogram("test_histogram", 1.0)
        metrics_collector.record_timer("test_timer", 0.5)
        
        # Get all metrics
        all_metrics = metrics_collector.get_all_metrics()
        
        assert 'counters' in all_metrics
        assert 'gauges' in all_metrics
        assert 'histograms' in all_metrics
        assert 'timers' in all_metrics
        
        assert all_metrics['counters']['test_counter'] == 10.0
        assert all_metrics['gauges']['test_gauge'] == 25.0
        assert 'test_histogram' in all_metrics['histograms']
        assert 'test_timer' in all_metrics['timers']
    
    def test_performance_monitor_operation_tracking(self, performance_monitor):
        """Test operation tracking in performance monitor"""
        # Start operation
        operation_id = performance_monitor.start_operation("test_operation")
        assert operation_id is not None
        assert operation_id in performance_monitor.active_operations
        
        # End operation
        performance_monitor.end_operation(operation_id, "test_operation", success=True)
        assert operation_id not in performance_monitor.active_operations
    
    def test_performance_monitor_function_decorator(self, performance_monitor):
        """Test function monitoring decorator"""
        @performance_monitor.monitor_function("decorated_function")
        def test_function():
            time.sleep(0.01)  # Small delay
            return "success"
        
        # Call function
        result = test_function()
        assert result == "success"
        
        # Check that metrics were recorded
        assert performance_monitor.metrics.get_counter("decorated_function_count") > 0
        assert performance_monitor.metrics.get_counter("decorated_function_success") > 0
    
    def test_performance_monitor_async_function_decorator(self, performance_monitor):
        """Test async function monitoring decorator"""
        @performance_monitor.monitor_async_function("async_decorated_function")
        async def test_async_function():
            await asyncio.sleep(0.01)  # Small delay
            return "async_success"
        
        # Call async function
        result = asyncio.run(test_async_function())
        assert result == "async_success"
        
        # Check that metrics were recorded
        assert performance_monitor.metrics.get_counter("async_decorated_function_count") > 0
        assert performance_monitor.metrics.get_counter("async_decorated_function_success") > 0
    
    def test_performance_monitor_error_handling(self, performance_monitor):
        """Test error handling in performance monitor"""
        @performance_monitor.monitor_function("error_function")
        def test_error_function():
            raise ValueError("Test error")
        
        # Call function that raises error
        with pytest.raises(ValueError):
            test_error_function()
        
        # Check that error metrics were recorded
        assert performance_monitor.metrics.get_counter("error_function_count") > 0
        assert performance_monitor.metrics.get_counter("error_function_error") > 0
        assert performance_monitor.metrics.get_counter("error_function_success") == 0
    
    def test_health_checker_registration(self, health_checker):
        """Test health check registration"""
        def test_health_check():
            return True
        
        # Register health check
        health_checker.register_health_check("test_check", test_health_check)
        assert "test_check" in health_checker.health_checks
    
    @pytest.mark.asyncio
    async def test_health_checker_execution(self, health_checker):
        """Test health check execution"""
        def healthy_check():
            return True
        
        def unhealthy_check():
            return False
        
        def error_check():
            raise Exception("Check failed")
        
        # Register health checks
        health_checker.register_health_check("healthy", healthy_check)
        health_checker.register_health_check("unhealthy", unhealthy_check)
        health_checker.register_health_check("error", error_check)
        
        # Run health checks
        results = await health_checker.run_health_checks()
        
        assert results['overall_healthy'] == False  # At least one check failed
        assert 'checks' in results
        assert 'healthy' in results['checks']
        assert 'unhealthy' in results['checks']
        assert 'error' in results['checks']
        
        assert results['checks']['healthy']['healthy'] == True
        assert results['checks']['unhealthy']['healthy'] == False
        assert results['checks']['error']['healthy'] == False
    
    @pytest.mark.asyncio
    async def test_health_checker_async_checks(self, health_checker):
        """Test async health check execution"""
        async def async_healthy_check():
            await asyncio.sleep(0.01)
            return True
        
        # Register async health check
        health_checker.register_health_check("async_healthy", async_healthy_check)
        
        # Run health checks
        results = await health_checker.run_health_checks()
        
        assert results['overall_healthy'] == True
        assert results['checks']['async_healthy']['healthy'] == True
    
    def test_health_checker_status(self, health_checker):
        """Test health status retrieval"""
        # Initially no status
        status = health_checker.get_health_status()
        assert status == {}
        
        # Register and run a check
        health_checker.register_health_check("test_check", lambda: True)
        asyncio.run(health_checker.run_health_checks())
        
        # Now should have status
        status = health_checker.get_health_status()
        assert 'test_check' in status
    
    @pytest.mark.asyncio
    async def test_observability_system_lifecycle(self, observability_system):
        """Test observability system lifecycle"""
        # Start system
        await observability_system.start()
        assert observability_system.running == True
        
        # Get system status
        status = observability_system.get_system_status()
        assert status['name'] == "test_app"
        assert status['running'] == True
        assert 'metrics' in status
        assert 'health' in status
        
        # Stop system
        await observability_system.stop()
        assert observability_system.running == False
    
    def test_observability_system_operation_logging(self, observability_system):
        """Test operation logging in observability system"""
        # Log operation start
        observability_system.log_operation_start("test_operation", {
            'user_id': 'test_user',
            'operation_id': 'test_op_123'
        })
        
        # Check that counter was incremented
        assert observability_system.metrics.get_counter("test_operation_started") > 0
        
        # Log operation end
        observability_system.log_operation_end("test_operation", True, 1.5, {
            'user_id': 'test_user',
            'operation_id': 'test_op_123'
        })
        
        # Check that counters were incremented
        assert observability_system.metrics.get_counter("test_operation_completed") > 0
        assert observability_system.metrics.get_counter("test_operation_success") > 0
        
        # Check that timer was recorded
        timer_stats = observability_system.metrics.get_timer_stats("test_operation_duration")
        assert timer_stats['count'] > 0
        assert timer_stats['mean'] == 1.5
    
    def test_observability_system_health_check_registration(self, observability_system):
        """Test health check registration in observability system"""
        def test_health_check():
            return True
        
        # Register health check
        observability_system.register_health_check("test_check", test_health_check)
        
        # Check that it was registered
        assert "test_check" in observability_system.health_checker.health_checks
    
    def test_observability_system_error_logging(self, observability_system):
        """Test error logging in observability system"""
        # Log operation end with error
        test_error = ValueError("Test error")
        observability_system.log_operation_end("test_operation", False, 0.5, {
            'user_id': 'test_user'
        }, test_error)
        
        # Check that error counter was incremented
        assert observability_system.metrics.get_counter("test_operation_error") > 0
        assert observability_system.metrics.get_counter("test_operation_success") == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
