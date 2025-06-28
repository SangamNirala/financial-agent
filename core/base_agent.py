"""Base agent class for the multi-agent system."""

import asyncio
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

import structlog
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage

from config.settings import settings
from core.memory_manager import MemoryManager
from core.explainability import ExplanationGenerator

logger = structlog.get_logger(__name__)

class AgentStatus(str, Enum):
    """Agent execution status."""
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class AgentResult:
    """Agent execution result."""
    agent_id: str
    task_id: str
    status: AgentStatus
    result: Optional[Dict[str, Any]] = None
    explanation: Optional[str] = None
    confidence: Optional[float] = None
    execution_time: Optional[float] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class AgentTask:
    """Agent task definition."""
    task_id: str
    agent_type: str
    objective: str
    parameters: Dict[str, Any]
    priority: int = 1
    timeout: int = 300
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

class BaseAgent(ABC):
    """Base class for all AI agents in the system."""
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        memory_manager: Optional[MemoryManager] = None,
        explanation_generator: Optional[ExplanationGenerator] = None
    ):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = name or self.__class__.__name__
        self.description = description or "Base AI Agent"
        self.status = AgentStatus.IDLE
        self.created_at = datetime.utcnow()
        
        # Core components
        self.memory_manager = memory_manager or MemoryManager()
        self.explanation_generator = explanation_generator or ExplanationGenerator()
        
        # Execution tracking
        self.current_task: Optional[AgentTask] = None
        self.task_history: List[AgentResult] = []
        
        # Agent-specific memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Logger
        self.logger = structlog.get_logger(__name__).bind(
            agent_id=self.agent_id,
            agent_name=self.name
        )
        
        self.logger.info("Agent initialized", agent_type=type(self).__name__)
    
    @abstractmethod
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Execute a specific task. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities."""
        pass
    
    async def process_request(self, request: Dict[str, Any]) -> AgentResult:
        """Process a request and return result with explanation."""
        task_id = str(uuid.uuid4())
        task = AgentTask(
            task_id=task_id,
            agent_type=self.name,
            objective=request.get("objective", ""),
            parameters=request.get("parameters", {}),
            priority=request.get("priority", 1),
            timeout=request.get("timeout", settings.agent_timeout)
        )
        
        return await self.execute_task(task)
    
    async def _execute_with_monitoring(self, task: AgentTask) -> AgentResult:
        """Execute task with monitoring and error handling."""
        start_time = datetime.utcnow()
        self.current_task = task
        self.status = AgentStatus.THINKING
        
        try:
            self.logger.info("Starting task execution", task_id=task.task_id)
            
            # Execute the actual task
            self.status = AgentStatus.EXECUTING
            result = await asyncio.wait_for(
                self.execute_task(task),
                timeout=task.timeout
            )
            
            # Generate explanation
            if result.result:
                explanation = await self.explanation_generator.generate_explanation(
                    agent_name=self.name,
                    task=task,
                    result=result.result
                )
                result.explanation = explanation
            
            # Update status and save to memory
            result.execution_time = (datetime.utcnow() - start_time).total_seconds()
            result.status = AgentStatus.COMPLETED
            
            await self._save_result_to_memory(result)
            
            self.logger.info(
                "Task completed successfully",
                task_id=task.task_id,
                execution_time=result.execution_time
            )
            
            return result
            
        except asyncio.TimeoutError:
            error_msg = f"Task timed out after {task.timeout} seconds"
            self.logger.error("Task timeout", task_id=task.task_id, timeout=task.timeout)
            return AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                status=AgentStatus.ERROR,
                error=error_msg,
                execution_time=(datetime.utcnow() - start_time).total_seconds()
            )
            
        except Exception as e:
            error_msg = f"Task execution failed: {str(e)}"
            self.logger.error(
                "Task execution error",
                task_id=task.task_id,
                error=str(e),
                exc_info=True
            )
            return AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                status=AgentStatus.ERROR,
                error=error_msg,
                execution_time=(datetime.utcnow() - start_time).total_seconds()
            )
        
        finally:
            self.status = AgentStatus.IDLE
            self.current_task = None
    
    async def _save_result_to_memory(self, result: AgentResult):
        """Save execution result to memory."""
        try:
            await self.memory_manager.store_agent_result(result)
            self.task_history.append(result)
            
            # Update agent memory
            self.memory.chat_memory.add_message(HumanMessage(
                content=f"Task: {result.task_id}"
            ))
            self.memory.chat_memory.add_message(AIMessage(
                content=f"Result: {result.result}"
            ))
            
        except Exception as e:
            self.logger.error("Failed to save result to memory", error=str(e))
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "status": self.status.value,
            "current_task": self.current_task.task_id if self.current_task else None,
            "task_count": len(self.task_history),
            "uptime": (datetime.utcnow() - self.created_at).total_seconds(),
            "capabilities": self.get_capabilities()
        }
    
    async def get_task_history(self, limit: int = 10) -> List[AgentResult]:
        """Get recent task history."""
        return self.task_history[-limit:]
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id[:8]}, status={self.status.value})"
