"""Integration tests for the Financial Risk AI System."""

import pytest
import asyncio
from datetime import datetime
import numpy as np
import pandas as pd

from core.orchestrator import AgentOrchestrator
from agents.data_agent import DataAgent
from agents.risk_agent import RiskAgent
from agents.forecast_agent import ForecastAgent
from agents.report_generator_agent import ReportGeneratorAgent

class TestSystemIntegration:
    """Test suite for system integration."""
    
    @pytest.fixture
    async def orchestrator(self):
        """Create orchestrator with all agents."""
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
        
        return orchestrator
    
    @pytest.fixture
    def sample_portfolio(self):
        """Sample portfolio data for testing."""
        return {
            "instruments": [
                {"symbol": "AAPL", "weight": 0.4},
                {"symbol": "GOOGL", "weight": 0.3},
                {"symbol": "MSFT", "weight": 0.3}
            ],
            "portfolio_value": 1000000,
            "currency": "USD"
        }
    
    @pytest.mark.asyncio
    async def test_agent_registration(self, orchestrator):
        """Test agent registration."""
        agents = orchestrator.list_agents()
        assert len(agents) == 4
        
        agent_names = [agent["name"] for agent in agents]
        expected_agents = ["DataAgent", "RiskAgent", "ForecastAgent", "ReportGeneratorAgent"]
        
        for expected_agent in expected_agents:
            assert expected_agent in agent_names
    
    @pytest.mark.asyncio
    async def test_data_agent_execution(self, orchestrator, sample_portfolio):
        """Test data agent execution."""
        data_agent = orchestrator.get_agent("DataAgent")
        
        task_data = {
            "objective": "Fetch and validate market data",
            "parameters": {
                "portfolio": sample_portfolio,
                "lookback_period": 30
            }
        }
        
        result = await data_agent.process_request(task_data)
        
        assert result.status.value == "completed"
        assert result.result is not None
        assert "portfolio_data" in result.result
        assert result.confidence > 0
    
    @pytest.mark.asyncio
    async def test_workflow_creation(self, orchestrator, sample_portfolio):
        """Test workflow creation and execution."""
        workflow_id = await orchestrator.create_financial_risk_workflow(
            portfolio_data=sample_portfolio,
            risk_types=["var", "monte_carlo"]
        )
        
        assert workflow_id is not None
        assert workflow_id in orchestrator.workflows
        
        # Check workflow status
        status = await orchestrator.get_workflow_status(workflow_id)
        assert status["workflow_id"] == workflow_id
        assert "steps_total" in status
    
    @pytest.mark.asyncio
    async def test_full_workflow_execution(self, orchestrator, sample_portfolio):
        """Test complete workflow execution."""
        workflow_id = await orchestrator.create_financial_risk_workflow(
            portfolio_data=sample_portfolio,
            risk_types=["var"]
        )
        
        # Execute workflow
        result = await orchestrator.execute_workflow(workflow_id)
        
        assert result["status"] == "completed"
        assert "results" in result
        assert "execution_time" in result
        assert result["execution_time"] > 0
    
    @pytest.mark.asyncio
    async def test_agent_error_handling(self, orchestrator):
        """Test agent error handling."""
        risk_agent = orchestrator.get_agent("RiskAgent")
        
        # Test with invalid data
        task_data = {
            "objective": "Calculate VaR",
            "parameters": {
                "invalid_data": "test"
            }
        }
        
        result = await risk_agent.process_request(task_data)
        
        assert result.status.value == "error"
        assert result.error is not None
    
    @pytest.mark.asyncio
    async def test_workflow_cancellation(self, orchestrator, sample_portfolio):
        """Test workflow cancellation."""
        workflow_id = await orchestrator.create_financial_risk_workflow(
            portfolio_data=sample_portfolio,
            risk_types=["var", "monte_carlo", "stress_test"]
        )
        
        # Start workflow execution in background
        task = asyncio.create_task(orchestrator.execute_workflow(workflow_id))
        
        # Cancel workflow
        await asyncio.sleep(0.1)  # Let it start
        success = await orchestrator.cancel_workflow(workflow_id)
        
        assert success is True
        
        # Check if task was cancelled
        try:
            await task
        except asyncio.CancelledError:
            pass  # Expected
    
    def test_risk_model_calculations(self):
        """Test risk model calculations."""
        from models.var_models import VaRCalculator
        
        # Create sample data
        returns_data = {
            "AAPL": np.random.normal(0.001, 0.02, 252),
            "GOOGL": np.random.normal(0.0008, 0.025, 252),
            "MSFT": np.random.normal(0.0012, 0.018, 252)
        }
        
        weights = {"AAPL": 0.4, "GOOGL": 0.3, "MSFT": 0.3}
        
        var_calculator = VaRCalculator()
        
        # Test portfolio returns calculation
        portfolio_returns = asyncio.run(
            var_calculator._calculate_portfolio_returns(returns_data, weights)
        )
        
        assert len(portfolio_returns) == 252
        assert not np.isnan(portfolio_returns).any()
    
    def test_data_validation(self):
        """Test data validation functionality."""
        from data.validators import DataValidator
        
        # Create sample data
        data = pd.DataFrame({
            'Open': [100, 101, 99, 102, 98],
            'High': [102, 103, 101, 104, 100],
            'Low': [99, 100, 98, 101, 97],
            'Close': [101, 99, 102, 98, 99],
            'Volume': [1000000, 1200000, 800000, 1500000, 900000]
        })
        
        validator = DataValidator()
        
        # Test data quality assessment
        quality = validator.assess_data_quality(data)
        
        assert "completeness" in quality
        assert "accuracy" in quality
        assert "consistency" in quality
        assert all(0 <= score <= 1 for score in quality.values())

if __name__ == "__main__":
    pytest.main([__file__])
