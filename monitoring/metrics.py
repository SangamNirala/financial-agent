"""System monitoring and metrics collection."""

import time
import psutil
import asyncio
from typing import Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import structlog

logger = structlog.get_logger(__name__)

# Prometheus metrics
AGENT_TASK_COUNTER = Counter(
    'agent_tasks_total',
    'Total number of agent tasks executed',
    ['agent_name', 'status']
)

AGENT_TASK_DURATION = Histogram(
    'agent_task_duration_seconds',
    'Time spent executing agent tasks',
    ['agent_name']
)

WORKFLOW_COUNTER = Counter(
    'workflows_total',
    'Total number of workflows executed',
    ['status']
)

WORKFLOW_DURATION = Histogram(
    'workflow_duration_seconds',
    'Time spent executing workflows'
)

SYSTEM_MEMORY_USAGE = Gauge(
    'system_memory_usage_bytes',
    'System memory usage in bytes'
)

SYSTEM_CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage'
)

@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_usage: float
    memory_usage: float
    memory_available: float
    disk_usage: float
    agent_count: int
    active_workflows: int
    timestamp: datetime

class MetricsCollector:
    """Collects and manages system metrics."""
    
    def __init__(self, collection_interval: int = 30):
        self.collection_interval = collection_interval
        self.metrics_history: List[SystemMetrics] = []
        self._running = False
    
    async def start_collection(self):
        """Start metrics collection."""
        self._running = True
        logger.info("Starting metrics collection")
        
        while self._running:
            try:
                metrics = await self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 1000 entries
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                # Update Prometheus metrics
                SYSTEM_CPU_USAGE.set(metrics.cpu_usage)
                SYSTEM_MEMORY_USAGE.set(metrics.memory_usage)
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error("Error collecting metrics", error=str(e))
                await asyncio.sleep(self.collection_interval)
    
    def stop_collection(self):
        """Stop metrics collection."""
        self._running = False
        logger.info("Stopping metrics collection")
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.used
        memory_available = memory.available
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage = (disk.used / disk.total) * 100
        
        return SystemMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            memory_available=memory_available,
            disk_usage=disk_usage,
            agent_count=0,  # To be set by orchestrator
            active_workflows=0,  # To be set by orchestrator
            timestamp=datetime.utcnow()
        )
    
    def get_recent_metrics(self, minutes: int = 60) -> List[SystemMetrics]:
        """Get metrics from the last N minutes."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        return [
            metric for metric in self.metrics_history
            if metric.timestamp >= cutoff_time
        ]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent_metrics = self.get_recent_metrics(60)
        
        if not recent_metrics:
            return {"status": "no_recent_data"}
        
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_disk = sum(m.disk_usage for m in recent_metrics) / len(recent_metrics)
        
        return {
            "status": "healthy",
            "avg_cpu_usage": avg_cpu,
            "avg_memory_usage_gb": avg_memory / (1024**3),
            "avg_disk_usage": avg_disk,
            "metrics_count": len(recent_metrics),
            "collection_period_minutes": 60
        }

def record_agent_task(agent_name: str, status: str, duration: float):
    """Record agent task metrics."""
    AGENT_TASK_COUNTER.labels(agent_name=agent_name, status=status).inc()
    AGENT_TASK_DURATION.labels(agent_name=agent_name).observe(duration)

def record_workflow(status: str, duration: float):
    """Record workflow metrics."""
    WORKFLOW_COUNTER.labels(status=status).inc()
    WORKFLOW_DURATION.observe(duration)

def start_metrics_server(port: int = 8001):
    """Start Prometheus metrics server."""
    start_http_server(port)
    logger.info("Metrics server started", port=port)
