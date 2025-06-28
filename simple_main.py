"""Simplified main application without Docker and Redis."""

import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

# Create data directory
Path("data").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/financial_ai.log'),
        logging.StreamHandler()
    ]
)

# Import simplified components
from core.simple_memory_manager import SimpleMemoryManager
from config.simple_settings import settings

# Import agents (we'll modify these to not require Redis)
from agents.data_agent import DataAgent
from agents.risk_agent import RiskAgent
from agents.forecast_agent import ForecastAgent
from agents.report_generator_agent import ReportGeneratorAgent

class SimpleOrchestrator:
    """Simplified orchestrator without Redis dependencies."""
    
    def __init__(self):
        self.agents = {}
        self.workflows = {}
        self.active_workflows = set()
        self.memory_manager = SimpleMemoryManager()
        
    def register_agent(self, agent):
        """Register an agent."""
        # Update agent to use simple memory manager
        agent.memory_manager = self.memory_manager
        self.agents[agent.name] = agent
        logging.info(f"Agent registered: {agent.name}")
    
    def get_agent(self, agent_name: str):
        """Get agent by name."""
        return self.agents.get(agent_name)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all agents."""
        return [
            {
                "name": agent.name,
                "status": "idle",
                "capabilities": agent.get_capabilities()
            }
            for agent in self.agents.values()
        ]
    
    async def create_simple_workflow(
        self, 
        portfolio_data: Dict[str, Any],
        risk_types: List[str] = ["var"]
    ) -> str:
        """Create a simple workflow."""
        import uuid
        workflow_id = str(uuid.uuid4())
        
        workflow = {
            "workflow_id": workflow_id,
            "portfolio_data": portfolio_data,
            "risk_types": risk_types,
            "status": "created",
            "created_at": datetime.utcnow().isoformat()
        }
        
        self.workflows[workflow_id] = workflow
        return workflow_id
    
    async def execute_simple_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Execute a simple workflow."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        workflow["status"] = "running"
        
        try:
            results = {}
            
            # Step 1: Data Collection
            data_agent = self.get_agent("DataAgent")
            if data_agent:
                data_result = await data_agent.process_request({
                    "objective": "Fetch market data",
                    "parameters": {
                        "portfolio": workflow["portfolio_data"],
                        "lookback_period": 60  # Reduced for speed
                    }
                })
                results["data"] = data_result.result if data_result.result else {}
            
            # Step 2: Risk Analysis
            risk_agent = self.get_agent("RiskAgent")
            if risk_agent and "var" in workflow["risk_types"]:
                risk_result = await risk_agent.process_request({
                    "objective": "Calculate VaR",
                    "parameters": {
                        "previous_results": results.get("data", {}),
                        "confidence_level": 0.95
                    }
                })
                results["risk"] = risk_result.result if risk_result.result else {}
            
            # Step 3: Simple Report
            report_agent = self.get_agent("ReportGeneratorAgent")
            if report_agent:
                report_result = await report_agent.process_request({
                    "objective": "Generate simple report",
                    "parameters": {
                        "previous_results": results,
                        "include_charts": False  # Disable charts for simplicity
                    }
                })
                results["report"] = report_result.result if report_result.result else {}
            
            workflow["status"] = "completed"
            workflow["results"] = results
            workflow["completed_at"] = datetime.utcnow().isoformat()
            
            return {
                "workflow_id": workflow_id,
                "status": "completed",
                "results": results
            }
            
        except Exception as e:
            workflow["status"] = "failed"
            workflow["error"] = str(e)
            logging.error(f"Workflow {workflow_id} failed: {e}")
            
            return {
                "workflow_id": workflow_id,
                "status": "failed",
                "error": str(e)
            }

# Global orchestrator
orchestrator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    global orchestrator
    
    # Startup
    logging.info("Starting Simple Financial Risk AI System")
    
    orchestrator = SimpleOrchestrator()
    
    # Register agents with simplified memory
    data_agent = DataAgent()
    risk_agent = RiskAgent()
    forecast_agent = ForecastAgent()
    report_agent = ReportGeneratorAgent()
    
    orchestrator.register_agent(data_agent)
    orchestrator.register_agent(risk_agent)
    orchestrator.register_agent(forecast_agent)
    orchestrator.register_agent(report_agent)
    
    logging.info("System started successfully")
    
    yield
    
    # Shutdown
    logging.info("Shutting down system")

# Create FastAPI app
app = FastAPI(
    title="Simple Financial Risk AI System",
    description="Lightweight Financial Risk Analysis without Docker/Redis",
    version="1.0.0-simple",
    lifespan=lifespan
)

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
        "message": "Simple Financial Risk AI System",
        "version": "1.0.0-simple",
        "timestamp": datetime.utcnow().isoformat(),
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "agents": len(orchestrator.agents) if orchestrator else 0
    }

@app.get("/agents")
async def list_agents():
    """List all agents."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return {
        "agents": orchestrator.list_agents(),
        "total_agents": len(orchestrator.agents)
    }

@app.post("/analyze")
async def analyze_portfolio(portfolio_data: Dict[str, Any]):
    """Simple portfolio analysis endpoint."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Create and execute workflow
        workflow_id = await orchestrator.create_simple_workflow(
            portfolio_data=portfolio_data,
            risk_types=["var"]
        )
        
        result = await orchestrator.execute_simple_workflow(workflow_id)
        return result
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/example")
async def run_example():
    """Run example analysis."""
    example_portfolio = {
        "instruments": [
            {"symbol": "AAPL", "weight": 0.4},
            {"symbol": "MSFT", "weight": 0.3},
            {"symbol": "GOOGL", "weight": 0.3}
        ],
        "portfolio_value": 1000000,
        "currency": "USD"
    }
    
    return await analyze_portfolio(example_portfolio)

if __name__ == "__main__":
    print("🚀 Starting Simple Financial Risk AI System...")
    print(f"📊 API will be available at: http://{settings.host}:{settings.port}")
    print(f"📚 API docs at: http://{settings.host}:{settings.port}/docs")
    print("🔧 No Docker or Redis required!")
    
    uvicorn.run(
        "simple_main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=settings.debug
    )
