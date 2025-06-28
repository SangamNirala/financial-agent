"""Central orchestrator for managing multi-agent workflows."""

import asyncio
import uuid
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

import structlog
from semantic_kernel import Kernel
from semantic_kernel.planning import BasicPlanner
from langchain.schema import BaseMessage

from config.settings import settings
from core.base_agent import BaseAgent, AgentTask, AgentResult, AgentStatus
from core.memory_manager import MemoryManager
from core.explainability import ExplanationGenerator

logger = structlog.get_logger(__name__)

class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class WorkflowStep:
    """Individual step in a workflow."""
    step_id: str
    agent_type: str
    objective: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    timeout: int = 300
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class Workflow:
    """Workflow definition."""
    workflow_id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: Dict[str, Any] = field(default_factory=dict)

class AgentOrchestrator:
    """Central orchestrator for managing multi-agent workflows."""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.workflows: Dict[str, Workflow] = {}
        self.memory_manager = MemoryManager()
        self.explanation_generator = ExplanationGenerator()
        
        # Semantic Kernel setup
        self.kernel = Kernel()
        self.planner = BasicPlanner()
        
        # Workflow execution tracking
        self.active_workflows: Set[str] = set()
        self.workflow_results: Dict[str, List[AgentResult]] = {}
        
        self.logger = structlog.get_logger(__name__)
        self.logger.info("Orchestrator initialized")
    
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator."""
        self.agents[agent.name] = agent
        self.logger.info(
            "Agent registered",
            agent_name=agent.name,
            agent_id=agent.agent_id,
            capabilities=agent.get_capabilities()
        )
    
    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """Get an agent by name."""
        return self.agents.get(agent_name)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents and their status."""
        return [agent.get_status() for agent in self.agents.values()]
    
    async def create_workflow(
        self,
        name: str,
        description: str,
        steps: List[Dict[str, Any]]
    ) -> str:
        """Create a new workflow."""
        workflow_id = str(uuid.uuid4())
        
        workflow_steps = []
        for step_data in steps:
            step = WorkflowStep(
                step_id=str(uuid.uuid4()),
                agent_type=step_data["agent_type"],
                objective=step_data["objective"],
                parameters=step_data.get("parameters", {}),
                dependencies=step_data.get("dependencies", []),
                timeout=step_data.get("timeout", 300),
                max_retries=step_data.get("max_retries", 3)
            )
            workflow_steps.append(step)
        
        workflow = Workflow(
            workflow_id=workflow_id,
            name=name,
            description=description,
            steps=workflow_steps
        )
        
        self.workflows[workflow_id] = workflow
        self.workflow_results[workflow_id] = []
        
        self.logger.info(
            "Workflow created",
            workflow_id=workflow_id,
            name=name,
            steps_count=len(workflow_steps)
        )
        
        return workflow_id
    
    async def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Execute a workflow."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        
        if workflow_id in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} is already running")
        
        self.active_workflows.add(workflow_id)
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.utcnow()
        
        try:
            self.logger.info("Starting workflow execution", workflow_id=workflow_id)
            
            # Execute steps based on dependencies
            executed_steps = set()
            step_results = {}
            
            while len(executed_steps) < len(workflow.steps):
                # Find steps ready for execution
                ready_steps = []
                for step in workflow.steps:
                    if (step.step_id not in executed_steps and 
                        all(dep in executed_steps for dep in step.dependencies)):
                        ready_steps.append(step)
                
                if not ready_steps:
                    raise RuntimeError("Circular dependency detected or no steps ready")
                
                # Execute ready steps in parallel
                tasks = []
                for step in ready_steps:
                    task = self._execute_workflow_step(workflow_id, step, step_results)
                    tasks.append(task)
                
                # Wait for all parallel steps to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for i, result in enumerate(results):
                    step = ready_steps[i]
                    if isinstance(result, Exception):
                        self.logger.error(
                            "Step execution failed",
                            workflow_id=workflow_id,
                            step_id=step.step_id,
                            error=str(result)
                        )
                        
                        # Retry logic
                        if step.retry_count < step.max_retries:
                            step.retry_count += 1
                            self.logger.info(
                                "Retrying step",
                                workflow_id=workflow_id,
                                step_id=step.step_id,
                                retry_count=step.retry_count
                            )
                            continue
                        else:
                            workflow.status = WorkflowStatus.FAILED
                            raise result
                    else:
                        step_results[step.step_id] = result
                        executed_steps.add(step.step_id)
            
            # Workflow completed successfully
            workflow.status = WorkflowStatus.COMPLETED
            workflow.completed_at = datetime.utcnow()
            workflow.results = step_results
            
            # Generate workflow explanation
            explanation = await self.explanation_generator.generate_workflow_explanation(
                workflow, step_results
            )
            
            execution_time = (workflow.completed_at - workflow.started_at).total_seconds()
            
            self.logger.info(
                "Workflow completed successfully",
                workflow_id=workflow_id,
                execution_time=execution_time
            )
            
            return {
                "workflow_id": workflow_id,
                "status": workflow.status.value,
                "execution_time": execution_time,
                "results": step_results,
                "explanation": explanation
            }
            
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            workflow.completed_at = datetime.utcnow()
            
            self.logger.error(
                "Workflow execution failed",
                workflow_id=workflow_id,
                error=str(e),
                exc_info=True
            )
            
            return {
                "workflow_id": workflow_id,
                "status": workflow.status.value,
                "error": str(e),
                "results": step_results if 'step_results' in locals() else {}
            }
        
        finally:
            self.active_workflows.discard(workflow_id)
    
    async def _execute_workflow_step(
        self,
        workflow_id: str,
        step: WorkflowStep,
        previous_results: Dict[str, Any]
    ) -> AgentResult:
        """Execute a single workflow step."""
        agent = self.get_agent(step.agent_type)
        if not agent:
            raise ValueError(f"Agent {step.agent_type} not found")
        
        # Prepare task with context from previous steps
        enriched_parameters = step.parameters.copy()
        enriched_parameters["previous_results"] = previous_results
        enriched_parameters["workflow_id"] = workflow_id
        
        task = AgentTask(
            task_id=step.step_id,
            agent_type=step.agent_type,
            objective=step.objective,
            parameters=enriched_parameters,
            timeout=step.timeout
        )
        
        self.logger.info(
            "Executing workflow step",
            workflow_id=workflow_id,
            step_id=step.step_id,
            agent_type=step.agent_type
        )
        
        result = await agent._execute_with_monitoring(task)
        self.workflow_results[workflow_id].append(result)
        
        return result
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow execution status."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        results = self.workflow_results.get(workflow_id, [])
        
        return {
            "workflow_id": workflow_id,
            "name": workflow.name,
            "status": workflow.status.value,
            "created_at": workflow.created_at.isoformat(),
            "started_at": workflow.started_at.isoformat() if workflow.started_at else None,
            "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None,
            "steps_total": len(workflow.steps),
            "steps_completed": len([r for r in results if r.status == AgentStatus.COMPLETED]),
            "steps_failed": len([r for r in results if r.status == AgentStatus.ERROR]),
            "results": workflow.results
        }
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        if workflow_id not in self.workflows:
            return False
        
        if workflow_id in self.active_workflows:
            workflow = self.workflows[workflow_id]
            workflow.status = WorkflowStatus.CANCELLED
            workflow.completed_at = datetime.utcnow()
            self.active_workflows.discard(workflow_id)
            
            self.logger.info("Workflow cancelled", workflow_id=workflow_id)
            return True
        
        return False
    
    async def create_financial_risk_workflow(
        self,
        portfolio_data: Dict[str, Any],
        risk_types: List[str]
    ) -> str:
        """Create a predefined financial risk analysis workflow."""
        steps = [
            {
                "agent_type": "DataAgent",
                "objective": "Fetch and validate market data",
                "parameters": {
                    "portfolio": portfolio_data,
                    "lookback_period": 252
                }
            },
            {
                "agent_type": "RiskAgent",
                "objective": "Calculate risk metrics",
                "parameters": {
                    "risk_types": risk_types,
                    "confidence_level": settings.confidence_level
                },
                "dependencies": ["DataAgent"]
            },
            {
                "agent_type": "ForecastAgent",
                "objective": "Generate risk forecasts",
                "parameters": {
                    "forecast_horizon": 30,
                    "scenarios": ["base", "stress", "optimistic"]
                },
                "dependencies": ["DataAgent", "RiskAgent"]
            },
            {
                "agent_type": "ReportGeneratorAgent",
                "objective": "Generate comprehensive risk report",
                "parameters": {
                    "report_type": "comprehensive",
                    "include_charts": True
                },
                "dependencies": ["DataAgent", "RiskAgent", "ForecastAgent"]
            }
        ]
        
        return await self.create_workflow(
            name="Financial Risk Analysis",
            description="Comprehensive financial risk analysis workflow",
            steps=steps
        )
