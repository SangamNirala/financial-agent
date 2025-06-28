"""Simplified base agent without Redis dependencies."""

import asyncio
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class AgentStatus(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class AgentResult:
    agent_id: str
    task_id: str
    status: AgentStatus
    result: Optional[Dict[str, Any]] = None
    explanation: Optional[str] = None
    confidence: Optional[float] = None
    execution_time: Optional[float] = None
    error: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class SimpleBaseAgent(ABC):
    """Simplified base agent class."""
    
    def __init__(self, agent_id: Optional[str] = None, name: Optional[str] = None):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = name or self.__class__.__name__
        self.status = AgentStatus.IDLE
        self.created_at = datetime.utcnow()
        self.memory_manager = None  # Will be set by orchestrator
        
        logger.info(f"Agent initialized: {self.name}")
    
    @abstractmethod
    async def execute_task(self, task) -> AgentResult:
        """Execute a specific task."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities."""
        pass
    
    async def process_request(self, request: Dict[str, Any]) -> AgentResult:
        """Process a request and return result."""
        task_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        try:
            self.status = AgentStatus.EXECUTING
            logger.info(f"Processing request: {self.name} - {task_id}")
            
            # Create simple task object
            task = type('Task', (), {
                'task_id': task_id,
                'objective': request.get('objective', ''),
                'parameters': request.get('parameters', {})
            })()
            
            result = await self.execute_task(task)
            result.execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Store result if memory manager available
            if self.memory_manager:
                await self.memory_manager.store_agent_result(result)
            
            logger.info(f"Task completed: {self.name} - {task_id}")
            return result
            
        except Exception as e:
            error_result = AgentResult(
                agent_id=self.agent_id,
                task_id=task_id,
                status=AgentStatus.ERROR,
                error=str(e),
                execution_time=(datetime.utcnow() - start_time).total_seconds()
            )
            
            logger.error(f"Task failed: {self.name} - {task_id} - {e}")
            return error_result
        
        finally:
            self.status = AgentStatus.IDLE
