"""
Production-Grade Observability System
Provides structured logging, metrics collection, and monitoring capabilities.
"""

import orjson as json  # LIBRARY REPLACEMENT: orjson for 3-5x faster JSON parsing
import time
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import defaultdict, deque
import uuid

logger = logging.getLogger(__name__)

class LogLevel(Enum):
    """Log level enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class MetricType(Enum):
    """Metric type enumeration"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: datetime
    level: LogLevel
    message: str
    context: Dict[str, Any]
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    operation: Optional[str] = None
    duration: Optional[float] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.value,
            'message': self.message,
            'context': self.context,
            'user_id': self.user_id,
            'request_id': self.request_id,
            'operation': self.operation,
            'duration': self.duration,
            'error': self.error
        }

@dataclass
class MetricEntry:
    """Metric entry"""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    labels: Dict[str, str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'name': self.name,
            'value': self.value,
            'type': self.metric_type.value,
            'labels': self.labels,
            'timestamp': self.timestamp.isoformat()
        }

class StructuredLogger:
    """
    Production-grade structured logger with context management.
    """
    
    def __init__(self, name: str = "app"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.context_stack: List[Dict[str, Any]] = []
        self.request_context: Dict[str, Any] = {}
        
        # Configure logging if not already configured
        if not self.logger.handlers:
            self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _create_log_entry(self, level: LogLevel, message: str, 
                         context: Dict[str, Any] = None,
                         error: Exception = None) -> LogEntry:
        """Create structured log entry"""
        # Merge context from stack and current context
        merged_context = {}
        for ctx in self.context_stack:
            merged_context.update(ctx)
        if context:
            merged_context.update(context)
        
        # Add request context
        merged_context.update(self.request_context)
        
        return LogEntry(
            timestamp=datetime.utcnow(),
            level=level,
            message=message,
            context=merged_context,
            user_id=self.request_context.get('user_id'),
            request_id=self.request_context.get('request_id'),
            operation=self.request_context.get('operation'),
            error=str(error) if error else None
        )
    
    def _log(self, level: LogLevel, message: str, context: Dict[str, Any] = None,
             error: Exception = None):
        """Internal logging method"""
        log_entry = self._create_log_entry(level, message, context, error)
        
        # Log to standard logger
        log_message = json.dumps(log_entry.to_dict(), default=str)
        getattr(self.logger, level.value.lower())(log_message)
    
    def debug(self, message: str, context: Dict[str, Any] = None):
        """Log debug message"""
        self._log(LogLevel.DEBUG, message, context)
    
    def info(self, message: str, context: Dict[str, Any] = None):
        """Log info message"""
        self._log(LogLevel.INFO, message, context)
    
    def warning(self, message: str, context: Dict[str, Any] = None):
        """Log warning message"""
        self._log(LogLevel.WARNING, message, context)
    
    def error(self, message: str, context: Dict[str, Any] = None, error: Exception = None):
        """Log error message"""
        self._log(LogLevel.ERROR, message, context, error)
    
    def critical(self, message: str, context: Dict[str, Any] = None, error: Exception = None):
        """Log critical message"""
        self._log(LogLevel.CRITICAL, message, context, error)
    
    def set_request_context(self, user_id: str = None, request_id: str = None,
                           operation: str = None, **kwargs):
        """Set request context for logging"""
        self.request_context = {
            'user_id': user_id,
            'request_id': request_id or str(uuid.uuid4()),
            'operation': operation,
            **kwargs
        }
    
    def clear_request_context(self):
        """Clear request context"""
        self.request_context = {}
    
    def push_context(self, context: Dict[str, Any]):
        """Push context onto stack"""
        self.context_stack.append(context)
    
    def pop_context(self) -> Dict[str, Any]:
        """Pop context from stack"""
        return self.context_stack.pop() if self.context_stack else {}
    
    def with_context(self, context: Dict[str, Any]):
        """Context manager for temporary context"""
        return ContextManager(self, context)

class ContextManager:
    """Context manager for temporary logging context"""
    
    def __init__(self, logger: StructuredLogger, context: Dict[str, Any]):
        self.logger = logger
        self.context = context
    
    def __enter__(self):
        self.logger.push_context(self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.pop_context()

class MetricsCollector:
    """
    Production-grade metrics collector with aggregation and export capabilities.
    """
    
    def __init__(self):
        self.metrics: Dict[str, List[MetricEntry]] = defaultdict(list)
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
        
        # Configuration
        self.max_histogram_samples = 1000
        self.max_timer_samples = 1000
        self.retention_period = timedelta(hours=24)
    
    def increment_counter(self, name: str, value: float = 1.0, 
                         labels: Dict[str, str] = None):
        """Increment counter metric"""
        with self.lock:
            self.counters[name] += value
            self._add_metric(name, value, MetricType.COUNTER, labels or {})
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set gauge metric"""
        with self.lock:
            self.gauges[name] = value
            self._add_metric(name, value, MetricType.GAUGE, labels or {})
    
    def observe_histogram(self, name: str, value: float, 
                         labels: Dict[str, str] = None):
        """Observe histogram metric"""
        with self.lock:
            if len(self.histograms[name]) >= self.max_histogram_samples:
                self.histograms[name] = self.histograms[name][-self.max_histogram_samples//2:]
            self.histograms[name].append(value)
            self._add_metric(name, value, MetricType.HISTOGRAM, labels or {})
    
    def record_timer(self, name: str, duration: float, 
                    labels: Dict[str, str] = None):
        """Record timer metric"""
        with self.lock:
            if len(self.timers[name]) >= self.max_timer_samples:
                self.timers[name] = self.timers[name][-self.max_timer_samples//2:]
            self.timers[name].append(duration)
            self._add_metric(name, duration, MetricType.TIMER, labels or {})
    
    def _add_metric(self, name: str, value: float, metric_type: MetricType,
                   labels: Dict[str, str]):
        """Add metric entry"""
        metric_entry = MetricEntry(
            name=name,
            value=value,
            metric_type=metric_type,
            labels=labels,
            timestamp=datetime.utcnow()
        )
        self.metrics[name].append(metric_entry)
    
    def get_counter(self, name: str) -> float:
        """Get counter value"""
        with self.lock:
            return self.counters.get(name, 0.0)
    
    def get_gauge(self, name: str) -> float:
        """Get gauge value"""
        with self.lock:
            return self.gauges.get(name, 0.0)
    
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get histogram statistics"""
        with self.lock:
            values = self.histograms.get(name, [])
            if not values:
                return {}
            
            return {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / len(values),
                'p50': self._percentile(values, 50),
                'p95': self._percentile(values, 95),
                'p99': self._percentile(values, 99)
            }
    
    def get_timer_stats(self, name: str) -> Dict[str, float]:
        """Get timer statistics"""
        with self.lock:
            values = self.timers.get(name, [])
            if not values:
                return {}
            
            return {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / len(values),
                'p50': self._percentile(values, 50),
                'p95': self._percentile(values, 95),
                'p99': self._percentile(values, 99)
            }
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int((percentile / 100.0) * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics summary"""
        with self.lock:
            return {
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'histograms': {name: self.get_histogram_stats(name) 
                              for name in self.histograms.keys()},
                'timers': {name: self.get_timer_stats(name) 
                          for name in self.timers.keys()}
            }
    
    def cleanup_old_metrics(self):
        """Clean up old metrics"""
        cutoff_time = datetime.utcnow() - self.retention_period
        
        with self.lock:
            for name in list(self.metrics.keys()):
                self.metrics[name] = [
                    metric for metric in self.metrics[name]
                    if metric.timestamp > cutoff_time
                ]
                
                if not self.metrics[name]:
                    del self.metrics[name]

class PerformanceMonitor:
    """
    Performance monitoring with automatic instrumentation.
    """
    
    def __init__(self, metrics_collector: MetricsCollector, logger: StructuredLogger):
        self.metrics = metrics_collector
        self.logger = logger
        self.active_operations: Dict[str, float] = {}
        self.operation_lock = threading.Lock()
    
    def start_operation(self, operation_name: str) -> str:
        """Start monitoring an operation"""
        operation_id = str(uuid.uuid4())
        start_time = time.time()
        
        with self.operation_lock:
            self.active_operations[operation_id] = start_time
        
        self.logger.debug(f"Started operation: {operation_name}", {
            'operation_id': operation_id,
            'operation_name': operation_name
        })
        
        return operation_id
    
    def end_operation(self, operation_id: str, operation_name: str,
                     success: bool = True, error: Exception = None):
        """End monitoring an operation"""
        end_time = time.time()
        
        with self.operation_lock:
            start_time = self.active_operations.pop(operation_id, end_time)
        
        duration = end_time - start_time
        
        # Record metrics
        self.metrics.record_timer(f"{operation_name}_duration", duration)
        self.metrics.increment_counter(f"{operation_name}_count")
        
        if success:
            self.metrics.increment_counter(f"{operation_name}_success")
        else:
            self.metrics.increment_counter(f"{operation_name}_error")
        
        # Log operation completion
        log_level = LogLevel.INFO if success else LogLevel.ERROR
        self.logger._log(log_level, f"Completed operation: {operation_name}", {
            'operation_id': operation_id,
            'duration': duration,
            'success': success
        }, error)
    
    def monitor_function(self, operation_name: str = None):
        """Decorator to monitor function execution"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                operation_id = self.start_operation(op_name)
                
                try:
                    result = func(*args, **kwargs)
                    self.end_operation(operation_id, op_name, success=True)
                    return result
                except Exception as e:
                    self.end_operation(operation_id, op_name, success=False, error=e)
                    raise
            
            return wrapper
        return decorator
    
    def monitor_async_function(self, operation_name: str = None):
        """Decorator to monitor async function execution"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                operation_id = self.start_operation(op_name)
                
                try:
                    result = await func(*args, **kwargs)
                    self.end_operation(operation_id, op_name, success=True)
                    return result
                except Exception as e:
                    self.end_operation(operation_id, op_name, success=False, error=e)
                    raise
            
            return wrapper
        return decorator

class HealthChecker:
    """
    Health check system for monitoring system components.
    """
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.health_checks: Dict[str, Callable] = {}
        self.health_status: Dict[str, Dict[str, Any]] = {}
    
    def register_health_check(self, name: str, check_function: Callable):
        """Register a health check function"""
        self.health_checks[name] = check_function
        self.logger.info(f"Registered health check: {name}")
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {}
        overall_healthy = True
        
        for name, check_function in self.health_checks.items():
            try:
                start_time = time.time()
                
                if asyncio.iscoroutinefunction(check_function):
                    result = await check_function()
                else:
                    result = check_function()
                
                duration = time.time() - start_time
                
                # Standardize health check result
                if isinstance(result, dict):
                    healthy = result.get('healthy', True)
                    details = result
                else:
                    healthy = bool(result)
                    details = {'result': result}
                
                results[name] = {
                    'healthy': healthy,
                    'duration': duration,
                    'details': details,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                if not healthy:
                    overall_healthy = False
                
                self.logger.debug(f"Health check {name}: {'PASS' if healthy else 'FAIL'}", {
                    'health_check': name,
                    'healthy': healthy,
                    'duration': duration
                })
                
            except Exception as e:
                results[name] = {
                    'healthy': False,
                    'duration': 0.0,
                    'details': {'error': str(e)},
                    'timestamp': datetime.utcnow().isoformat()
                }
                overall_healthy = False
                
                self.logger.error(f"Health check {name} failed", {
                    'health_check': name,
                    'error': str(e)
                }, e)
        
        self.health_status = results
        
        return {
            'overall_healthy': overall_healthy,
            'checks': results,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        return self.health_status

class ObservabilitySystem:
    """
    Main observability system that coordinates logging, metrics, and monitoring.
    """
    
    def __init__(self, name: str = "app"):
        self.name = name
        self.logger = StructuredLogger(name)
        self.metrics = MetricsCollector()
        self.performance_monitor = PerformanceMonitor(self.metrics, self.logger)
        self.health_checker = HealthChecker(self.logger)
        
        # Background tasks
        self.cleanup_task = None
        self.health_check_task = None
        self.running = False
    
    async def start(self):
        """Start the observability system"""
        if self.running:
            return
        
        self.running = True
        
        # Start background tasks
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        self.logger.info("Observability system started")
    
    async def stop(self):
        """Stop the observability system"""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel background tasks
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.health_check_task:
            self.health_check_task.cancel()
        
        self.logger.info("Observability system stopped")
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while self.running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                self.metrics.cleanup_old_metrics()
                self.logger.debug("Cleaned up old metrics")
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Cleanup loop error", error=e)
    
    async def _health_check_loop(self):
        """Background health check loop"""
        while self.running:
            try:
                await asyncio.sleep(60)  # Run every minute
                await self.health_checker.run_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Health check loop error", error=e)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'name': self.name,
            'running': self.running,
            'metrics': self.metrics.get_all_metrics(),
            'health': self.health_checker.get_health_status(),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def register_health_check(self, name: str, check_function: Callable):
        """Register a health check"""
        self.health_checker.register_health_check(name, check_function)
    
    def log_operation_start(self, operation: str, context: Dict[str, Any] = None):
        """Log operation start"""
        self.logger.info(f"Starting operation: {operation}", context)
        self.metrics.increment_counter(f"{operation}_started")
    
    def log_operation_end(self, operation: str, success: bool = True,
                         duration: float = None, context: Dict[str, Any] = None,
                         error: Exception = None):
        """Log operation end"""
        level = LogLevel.INFO if success else LogLevel.ERROR
        self.logger._log(level, f"Completed operation: {operation}", context, error)
        
        self.metrics.increment_counter(f"{operation}_completed")
        if success:
            self.metrics.increment_counter(f"{operation}_success")
        else:
            self.metrics.increment_counter(f"{operation}_error")
        
        if duration is not None:
            self.metrics.record_timer(f"{operation}_duration", duration)

# Global observability system instance
_global_observability_system: Optional[ObservabilitySystem] = None

def get_global_observability_system() -> ObservabilitySystem:
    """Get or create global observability system"""
    global _global_observability_system
    
    if _global_observability_system is None:
        _global_observability_system = ObservabilitySystem()
    
    return _global_observability_system
