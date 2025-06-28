"""Risk Agent for financial risk calculations and analysis."""

import asyncio
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta

from core.base_agent import BaseAgent, AgentTask, AgentResult, AgentStatus
from models.risk_models import RiskModelFactory
from models.monte_carlo import MonteCarloSimulator
from models.var_models import VaRCalculator
from models.stress_testing import StressTester
from integrations.huggingface_client import HuggingFaceClient

from core.simple_base_agent import SimpleBaseAgent, AgentResult, AgentStatus

class RiskAgent(BaseAgent):
    """Agent responsible for financial risk calculations and analysis."""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="RiskAgent",
            description="Financial risk calculation and analysis agent",
            **kwargs
        )
        self.risk_model_factory = RiskModelFactory()
        self.monte_carlo = MonteCarloSimulator()
        self.var_calculator = VaRCalculator()
        self.stress_tester = StressTester()
        self.hf_client = HuggingFaceClient()
    
    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities."""
        return [
            "value_at_risk_calculation",
            "monte_carlo_simulation",
            "stress_testing",
            "portfolio_optimization",
            "risk_attribution",
            "correlation_analysis",
            "volatility_modeling",
            "scenario_analysis"
        ]
    
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Execute risk calculation task."""
        try:
            objective = task.objective.lower()
            parameters = task.parameters
            
            if "var" in objective or "value at risk" in objective:
                result_data = await self._calculate_var(parameters)
            elif "monte carlo" in objective:
                result_data = await self._run_monte_carlo_simulation(parameters)
            elif "stress" in objective:
                result_data = await self._perform_stress_testing(parameters)
            elif "correlation" in objective:
                result_data = await self._analyze_correlations(parameters)
            elif "volatility" in objective:
                result_data = await self._model_volatility(parameters)
            else:
                # Default: comprehensive risk analysis
                result_data = await self._comprehensive_risk_analysis(parameters)
            
            return AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                status=AgentStatus.COMPLETED,
                result=result_data,
                confidence=self._calculate_risk_confidence(result_data)
            )
            
        except Exception as e:
            self.logger.error("Risk agent task failed", error=str(e), exc_info=True)
            return AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                status=AgentStatus.ERROR,
                error=str(e)
            )
    
    async def _calculate_var(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Value at Risk using multiple methods."""
        portfolio_data = parameters.get("previous_results", {}).get("portfolio_data", {})
        confidence_level = parameters.get("confidence_level", 0.95)
        holding_period = parameters.get("holding_period", 1)
        
        if not portfolio_data:
            raise ValueError("No portfolio data available for VaR calculation")
        
        # Prepare returns data
        returns_data = {}
        portfolio_weights = {}
        
        for symbol, data in portfolio_data.items():
            returns = data.get("returns", [])
            if returns:
                returns_data[symbol] = np.array(returns)
                # Assume equal weights if not specified
                portfolio_weights[symbol] = 1.0 / len(portfolio_data)
        
        if not returns_data:
            raise ValueError("No returns data available for VaR calculation")
        
        # Calculate VaR using different methods
        var_results = {}
        
        # Parametric VaR
        parametric_var = await self.var_calculator.parametric_var(
            returns_data, portfolio_weights, confidence_level, holding_period
        )
        var_results["parametric"] = parametric_var
        
        # Historical VaR
        historical_var = await self.var_calculator.historical_var(
            returns_data, portfolio_weights, confidence_level, holding_period
        )
        var_results["historical"] = historical_var
        
        # Monte Carlo VaR
        mc_var = await self.var_calculator.monte_carlo_var(
            returns_data, portfolio_weights, confidence_level, holding_period
        )
        var_results["monte_carlo"] = mc_var
        
        # Calculate Expected Shortfall (CVaR)
        expected_shortfall = await self.var_calculator.expected_shortfall(
            returns_data, portfolio_weights, confidence_level
        )
        
        # Risk decomposition
        risk_attribution = await self._calculate_risk_attribution(
            returns_data, portfolio_weights
        )
        
        return {
            "var_estimates": var_results,
            "expected_shortfall": expected_shortfall,
            "risk_attribution": risk_attribution,
            "confidence_level": confidence_level,
            "holding_period": holding_period,
            "calculation_timestamp": datetime.utcnow().isoformat(),
            "portfolio_volatility": self._calculate_portfolio_volatility(returns_data, portfolio_weights)
        }
    
    async def _run_monte_carlo_simulation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run Monte Carlo simulation for portfolio risk."""
        portfolio_data = parameters.get("previous_results", {}).get("portfolio_data", {})
        num_simulations = parameters.get("num_simulations", 10000)
        time_horizon = parameters.get("time_horizon", 252)  # Trading days
        
        if not portfolio_data:
            raise ValueError("No portfolio data available for Monte Carlo simulation")
        
        # Prepare data for simulation
        returns_data = {}
        for symbol, data in portfolio_data.items():
            returns = data.get("returns", [])
            if returns:
                returns_data[symbol] = np.array(returns)
        
        # Run Monte Carlo simulation
        simulation_results = await self.monte_carlo.simulate_portfolio_paths(
            returns_data=returns_data,
            num_simulations=num_simulations,
            time_horizon=time_horizon
        )
        
        # Analyze simulation results
        portfolio_returns = simulation_results["portfolio_returns"]
        
        # Calculate risk metrics
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        risk_metrics = {
            f"percentile_{p}": np.percentile(portfolio_returns, p)
            for p in percentiles
        }
        
        # Probability of loss
        prob_loss = np.mean(portfolio_returns < 0)
        
        # Maximum drawdown distribution
        drawdowns = simulation_results.get("max_drawdowns", [])
        
        return {
            "simulation_results": {
                "num_simulations": num_simulations,
                "time_horizon": time_horizon,
                "portfolio_return_distribution": risk_metrics,
                "probability_of_loss": prob_loss,
                "expected_return": np.mean(portfolio_returns),
                "volatility": np.std(portfolio_returns),
                "skewness": stats.skew(portfolio_returns),
                "kurtosis": stats.kurtosis(portfolio_returns)
            },
            "drawdown_analysis": {
                "average_max_drawdown": np.mean(drawdowns) if drawdowns else 0,
                "worst_case_drawdown": np.min(drawdowns) if drawdowns else 0,
                "drawdown_95_percentile": np.percentile(drawdowns, 5) if drawdowns else 0
            },
            "simulation_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _perform_stress_testing(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform stress testing under various scenarios."""
        portfolio_data = parameters.get("previous_results", {}).get("portfolio_data", {})
        stress_scenarios = parameters.get("stress_scenarios", ["market_crash", "interest_rate_shock", "currency_crisis"])
        
        if not portfolio_data:
            raise ValueError("No portfolio data available for stress testing")
        
        stress_results = {}
        
        for scenario in stress_scenarios:
            try:
                scenario_result = await self.stress_tester.run_stress_scenario(
                    portfolio_data=portfolio_data,
                    scenario_name=scenario
                )
                stress_results[scenario] = scenario_result
                
                self.logger.info(f"Completed stress test: {scenario}")
                
            except Exception as e:
                self.logger.error(f"Stress test failed for {scenario}", error=str(e))
                stress_results[scenario] = {"error": str(e)}
        
        # Calculate aggregate stress impact
        aggregate_impact = await self._calculate_aggregate_stress_impact(stress_results)
        
        # Generate stress test narrative using LLM
        stress_narrative = await self._generate_stress_narrative(stress_results)
        
        return {
            "stress_test_results": stress_results,
            "aggregate_impact": aggregate_impact,
            "stress_narrative": stress_narrative,
            "test_timestamp": datetime.utcnow().isoformat(),
            "scenarios_tested": len(stress_results)
        }
    
    async def _analyze_correlations(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze correlation structure of portfolio assets."""
        portfolio_data = parameters.get("previous_results", {}).get("portfolio_data", {})
        
        if not portfolio_data:
            raise ValueError("No portfolio data available for correlation analysis")
        
        # Prepare returns matrix
        returns_df = pd.DataFrame()
        for symbol, data in portfolio_data.items():
            returns = data.get("returns", [])
            if returns:
                returns_df[symbol] = returns[:min(len(r) for r in [data.get("returns", []) for data in portfolio_data.values()])]
        
        if returns_df.empty:
            raise ValueError("No returns data available for correlation analysis")
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        # Identify high correlations
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # High correlation threshold
                    high_correlations.append({
                        "asset1": correlation_matrix.columns[i],
                        "asset2": correlation_matrix.columns[j],
                        "correlation": corr_value
                    })
        
        # Rolling correlation analysis
        window = min(63, len(returns_df) // 2)  # Quarter or half of data
        rolling_correlations = {}
        
        if len(returns_df) > window:
            for col1 in returns_df.columns:
                for col2 in returns_df.columns:
                    if col1 != col2:
                        rolling_corr = returns_df[col1].rolling(window).corr(returns_df[col2])
                        rolling_correlations[f"{col1}_{col2}"] = {
                            "current": rolling_corr.iloc[-1],
                            "average": rolling_corr.mean(),
                            "volatility": rolling_corr.std()
                        }
        
        return {
            "correlation_matrix": correlation_matrix.to_dict(),
            "high_correlations": high_correlations,
            "rolling_correlations": rolling_correlations,
            "portfolio_diversification_ratio": self._calculate_diversification_ratio(correlation_matrix),
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _model_volatility(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Model volatility using various approaches."""
        portfolio_data = parameters.get("previous_results", {}).get("portfolio_data", {})
        
        volatility_models = {}
        
        for symbol, data in portfolio_data.items():
            returns = data.get("returns", [])
            if not returns:
                continue
            
            returns_array = np.array(returns)
            
            # Historical volatility
            hist_vol = np.std(returns_array) * np.sqrt(252)
            
            # EWMA volatility
            ewma_vol = self._calculate_ewma_volatility(returns_array)
            
            # Rolling volatility
            rolling_vol = self._calculate_rolling_volatility(returns_array)
            
            volatility_models[symbol] = {
                "historical_volatility": hist_vol,
                "ewma_volatility": ewma_vol,
                "rolling_volatility": rolling_vol,
                "current_volatility": data.get("volatility", hist_vol)
            }
        
        return {
            "volatility_models": volatility_models,
            "modeling_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _comprehensive_risk_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive risk analysis."""
        # Run all risk analyses
        var_result = await self._calculate_var(parameters)
        mc_result = await self._run_monte_carlo_simulation(parameters)
        stress_result = await self._perform_stress_testing(parameters)
        correlation_result = await self._analyze_correlations(parameters)
        volatility_result = await self._model_volatility(parameters)
        
        # Generate overall risk assessment
        risk_assessment = await self._generate_risk_assessment(
            var_result, mc_result, stress_result, correlation_result, volatility_result
        )
        
        return {
            "var_analysis": var_result,
            "monte_carlo_analysis": mc_result,
            "stress_testing": stress_result,
            "correlation_analysis": correlation_result,
            "volatility_analysis": volatility_result,
            "overall_risk_assessment": risk_assessment,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _calculate_risk_attribution(
        self, 
        returns_data: Dict[str, np.ndarray], 
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate risk attribution for portfolio components."""
        attribution = {}
        
        # Calculate portfolio returns
        portfolio_returns = np.zeros(len(next(iter(returns_data.values()))))
        for symbol, returns in returns_data.items():
            portfolio_returns += weights[symbol] * returns
        
        portfolio_var = np.var(portfolio_returns)
        
        # Individual asset contribution to portfolio risk
        for symbol, returns in returns_data.items():
            covariance = np.cov(returns, portfolio_returns)[0, 1]
            risk_contribution = weights[symbol] * covariance / portfolio_var
            attribution[symbol] = {
                "weight": weights[symbol],
                "risk_contribution": risk_contribution,
                "risk_contribution_percent": risk_contribution * 100
            }
        
        return attribution
    
    def _calculate_portfolio_volatility(
        self, 
        returns_data: Dict[str, np.ndarray], 
        weights: Dict[str, float]
    ) -> float:
        """Calculate portfolio volatility."""
        portfolio_returns = np.zeros(len(next(iter(returns_data.values()))))
        for symbol, returns in returns_data.items():
            portfolio_returns += weights[symbol] * returns
        
        return np.std(portfolio_returns) * np.sqrt(252)
    
    def _calculate_diversification_ratio(self, correlation_matrix: pd.DataFrame) -> float:
        """Calculate portfolio diversification ratio."""
        n = len(correlation_matrix)
        if n <= 1:
            return 1.0
        
        avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
        return (1 + (n - 1) * avg_correlation) / n
    
    def _calculate_ewma_volatility(self, returns: np.ndarray, lambda_param: float = 0.94) -> float:
        """Calculate EWMA volatility."""
        if len(returns) < 2:
            return 0.0
        
        weights = np.array([(1 - lambda_param) * (lambda_param ** i) for i in range(len(returns))])
        weights = weights / weights.sum()
        
        weighted_variance = np.sum(weights * returns**2)
        return np.sqrt(weighted_variance * 252)
    
    def _calculate_rolling_volatility(self, returns: np.ndarray, window: int = 30) -> Dict[str, float]:
        """Calculate rolling volatility statistics."""
        if len(returns) < window:
            return {"current": 0.0, "average": 0.0, "max": 0.0, "min": 0.0}
        
        rolling_vol = []
        for i in range(window, len(returns)):
            vol = np.std(returns[i-window:i]) * np.sqrt(252)
            rolling_vol.append(vol)
        
        return {
            "current": rolling_vol[-1] if rolling_vol else 0.0,
            "average": np.mean(rolling_vol) if rolling_vol else 0.0,
            "max": np.max(rolling_vol) if rolling_vol else 0.0,
            "min": np.min(rolling_vol) if rolling_vol else 0.0
        }
    
    async def _calculate_aggregate_stress_impact(self, stress_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate aggregate impact across stress scenarios."""
        impacts = []
        scenario_names = []
        
        for scenario, result in stress_results.items():
            if "error" not in result:
                impact = result.get("portfolio_impact", {}).get("total_loss", 0)
                impacts.append(abs(impact))
                scenario_names.append(scenario)
        
        if not impacts:
            return {"error": "No valid stress test results"}
        
        return {
            "worst_case_scenario": scenario_names[np.argmax(impacts)],
            "worst_case_loss": max(impacts),
            "average_loss": np.mean(impacts),
            "scenarios_analyzed": len(impacts)
        }
    
    async def _generate_stress_narrative(self, stress_results: Dict[str, Any]) -> str:
        """Generate narrative explanation of stress test results using LLM."""
        try:
            # Prepare summary for LLM
            summary_text = "Stress Test Results Summary:\n"
            for scenario, result in stress_results.items():
                if "error" not in result:
                    impact = result.get("portfolio_impact", {})
                    summary_text += f"\n{scenario}: Portfolio loss of {impact.get('total_loss', 0):.2%}"
            
            # Generate narrative using Hugging Face
            narrative = await self.hf_client.generate_text(
                prompt=f"Provide a professional analysis of these financial stress test results:\n{summary_text}",
                model="microsoft/DialoGPT-large",
                max_length=200
            )
            
            return narrative.get("generated_text", "Stress test completed with multiple scenarios analyzed.")
            
        except Exception as e:
            self.logger.error("Failed to generate stress narrative", error=str(e))
            return "Stress testing completed. Detailed analysis available in results."
    
    async def _generate_risk_assessment(
        self,
        var_result: Dict[str, Any],
        mc_result: Dict[str, Any],
        stress_result: Dict[str, Any],
        correlation_result: Dict[str, Any],
        volatility_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate overall risk assessment."""
        # Risk level determination
        var_95 = var_result.get("var_estimates", {}).get("parametric", {}).get("var_95", 0)
        portfolio_vol = var_result.get("portfolio_volatility", 0)
        
        # Risk categorization
        if portfolio_vol > 0.25:  # >25% annual volatility
            risk_level = "High"
        elif portfolio_vol > 0.15:  # >15% annual volatility
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Key risk factors
        high_correlations = correlation_result.get("high_correlations", [])
        concentration_risk = len(high_correlations) > 3
        
        stress_impact = stress_result.get("aggregate_impact", {})
        worst_case_loss = stress_impact.get("worst_case_loss", 0)
        
        return {
            "overall_risk_level": risk_level,
            "portfolio_volatility": portfolio_vol,
            "var_95_1day": var_95,
            "concentration_risk": concentration_risk,
            "worst_case_stress_loss": worst_case_loss,
            "key_risk_factors": [
                "High correlation" if concentration_risk else "Diversified portfolio",
                f"Volatility: {portfolio_vol:.1%}",
                f"Worst stress scenario: {stress_impact.get('worst_case_scenario', 'N/A')}"
            ],
            "assessment_timestamp": datetime.utcnow().isoformat()
        }
    
    def _calculate_risk_confidence(self, result_data: Dict[str, Any]) -> float:
        """Calculate confidence score for risk analysis."""
        confidence_factors = []
        
        # Data quality factor
        if "var_analysis" in result_data:
            var_data = result_data["var_analysis"]
            if var_data.get("var_estimates"):
                confidence_factors.append(0.9)  # High confidence in VaR
        
        # Monte Carlo convergence
        if "monte_carlo_analysis" in result_data:
            mc_data = result_data["monte_carlo_analysis"]
            num_sims = mc_data.get("simulation_results", {}).get("num_simulations", 0)
            if num_sims >= 10000:
                confidence_factors.append(0.95)
            elif num_sims >= 1000:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.6)
        
        # Stress test completeness
        if "stress_testing" in result_data:
            stress_data = result_data["stress_testing"]
            scenarios_completed = stress_data.get("scenarios_tested", 0)
            confidence_factors.append(min(scenarios_completed / 3, 1.0))
        
        return np.mean(confidence_factors) if confidence_factors else 0.7
