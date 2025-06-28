"""Main application entry point for Financial Risk AI System."""

import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import structlog
from datetime import datetime
from typing import Dict, Any, List

# Core imports
from core.orchestrator import AgentOrchestrator
from agents.data_agent import DataAgent
from agents.risk_agent import RiskAgent
from agents.forecast_agent import ForecastAgent
from agents.report_generator_agent import ReportGeneratorAgent
from config.settings import settings
from config.logging_config import setup_logging

# Setup logging
setup_logging()
logger = structlog.get_logger(__name__)

# Global orchestrator instance
orchestrator: AgentOrchestrator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global orchestrator
    
    # Startup
    logger.info("Starting Financial Risk AI System")
    
    # Initialize orchestrator and agents
    orchestrator = AgentOrchestrator()
    
    # Register agents
    data_agent = DataAgent()
    risk_agent = RiskAgent()
    forecast_agent = ForecastAgent()
    report_agent = ReportGeneratorAgent()
    
    orchestrator.register_agent(data_agent)
    orchestrator.register_agent(risk_agent)
    orchestrator.register_agent(forecast_agent)
    orchestrator.register_agent(report_agent)
    
    logger.info("All agents registered successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Financial Risk AI System")

# Create FastAPI app
app = FastAPI(
    title="Financial Risk AI System",
    description="Scalable Multi-Agent AI System for Financial Risk Management",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Financial Risk AI System",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "agents": len(orchestrator.agents) if orchestrator else 0
    }

@app.get("/agents")
async def list_agents():
    """List all registered agents and their status."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return {
        "agents": orchestrator.list_agents(),
        "total_agents": len(orchestrator.agents)
    }

@app.post("/workflow/risk-analysis")
async def create_risk_analysis_workflow(
    portfolio_data: Dict[str, Any],
    risk_types: List[str] = ["var", "monte_carlo", "stress_test"],
    background_tasks: BackgroundTasks = None
):
    """Create and execute a financial risk analysis workflow."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Create workflow
        workflow_id = await orchestrator.create_financial_risk_workflow(
            portfolio_data=portfolio_data,
            risk_types=risk_types
        )
        
        logger.info("Risk analysis workflow created", workflow_id=workflow_id)
        
        # Execute workflow in background if requested
        if background_tasks:
            background_tasks.add_task(orchestrator.execute_workflow, workflow_id)
            return {
                "workflow_id": workflow_id,
                "status": "queued",
                "message": "Workflow queued for execution"
            }
        else:
            # Execute synchronously
            result = await orchestrator.execute_workflow(workflow_id)
            return result
            
    except Exception as e:
        logger.error("Failed to create risk analysis workflow", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/workflow/custom")
async def create_custom_workflow(
    workflow_data: Dict[str, Any],
    background_tasks: BackgroundTasks = None
):
    """Create a custom workflow with specified steps."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        workflow_id = await orchestrator.create_workflow(
            name=workflow_data.get("name", "Custom Workflow"),
            description=workflow_data.get("description", "Custom financial analysis workflow"),
            steps=workflow_data.get("steps", [])
        )
        
        logger.info("Custom workflow created", workflow_id=workflow_id)
        
        if background_tasks:
            background_tasks.add_task(orchestrator.execute_workflow, workflow_id)
            return {
                "workflow_id": workflow_id,
                "status": "queued"
            }
        else:
            result = await orchestrator.execute_workflow(workflow_id)
            return result
            
    except Exception as e:
        logger.error("Failed to create custom workflow", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workflow/{workflow_id}/status")
async def get_workflow_status(workflow_id: str):
    """Get status of a specific workflow."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        status = await orchestrator.get_workflow_status(workflow_id)
        return status
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Failed to get workflow status", workflow_id=workflow_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/workflow/{workflow_id}/cancel")
async def cancel_workflow(workflow_id: str):
    """Cancel a running workflow."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        success = await orchestrator.cancel_workflow(workflow_id)
        if success:
            return {"message": "Workflow cancelled successfully"}
        else:
            raise HTTPException(status_code=400, detail="Workflow not found or not running")
    except Exception as e:
        logger.error("Failed to cancel workflow", workflow_id=workflow_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/{agent_name}/task")
async def execute_agent_task(
    agent_name: str,
    task_data: Dict[str, Any]
):
    """Execute a task on a specific agent."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    agent = orchestrator.get_agent(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
    
    try:
        result = await agent.process_request(task_data)
        return {
            "agent_name": agent_name,
            "task_id": result.task_id,
            "status": result.status.value,
            "result": result.result,
            "confidence": result.confidence,
            "execution_time": result.execution_time,
            "explanation": result.explanation
        }
    except Exception as e:
        logger.error("Agent task execution failed", agent_name=agent_name, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_system_metrics():
    """Get system performance metrics."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return {
        "system_status": "operational",
        "agents_count": len(orchestrator.agents),
        "active_workflows": len(orchestrator.active_workflows),
        "total_workflows": len(orchestrator.workflows),
        "uptime": datetime.utcnow().isoformat(),
        "agent_details": orchestrator.list_agents()
    }

# Example portfolio data for testing
EXAMPLE_PORTFOLIO = {
    "instruments": [
        {"symbol": "AAPL", "weight": 0.3},
        {"symbol": "GOOGL", "weight": 0.25},
        {"symbol": "MSFT", "weight": 0.25},
        {"symbol": "TSLA", "weight": 0.2}
    ],
    "portfolio_value": 1000000,
    "currency": "USD"
}

@app.get("/example/portfolio")
async def get_example_portfolio():
    """Get example portfolio for testing."""
    return EXAMPLE_PORTFOLIO

@app.post("/example/risk-analysis")
async def run_example_risk_analysis(background_tasks: BackgroundTasks = None):
    """Run example risk analysis with predefined portfolio."""
    return await create_risk_analysis_workflow(
        portfolio_data=EXAMPLE_PORTFOLIO,
        background_tasks=background_tasks
    )

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.metrics_port,
        log_level=settings.log_level.lower(),
        reload=settings.debug
    )
