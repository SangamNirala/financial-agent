"""Explainability module for generating explanations of agent decisions."""

from typing import Dict, Any, List
from datetime import datetime

from core.base_agent import AgentTask
from integrations.huggingface_client import HuggingFaceClient

class ExplanationGenerator:
    """Generates explanations for agent decisions and workflows."""
    
    def __init__(self):
        self.hf_client = HuggingFaceClient()
    
    async def generate_explanation(
        self,
        agent_name: str,
        task: AgentTask,
        result: Dict[str, Any]
    ) -> str:
        """Generate explanation for agent task execution."""
        try:
            # Create explanation prompt
            prompt = self._create_explanation_prompt(agent_name, task, result)
            
            # Generate explanation using LLM
            response = await self.hf_client.generate_text(
                prompt=prompt,
                model="microsoft/DialoGPT-large",
                max_length=150
            )
            
            explanation = response.get("generated_text", "")
            
            # Fallback to rule-based explanation
            if not explanation or len(explanation) < 50:
                explanation = self._generate_rule_based_explanation(agent_name, task, result)
            
            return explanation
            
        except Exception as e:
            return self._generate_rule_based_explanation(agent_name, task, result)
    
    def _create_explanation_prompt(
        self,
        agent_name: str,
        task: AgentTask,
        result: Dict[str, Any]
    ) -> str:
        """Create prompt for LLM explanation generation."""
        return f"""
        Explain the following financial AI agent analysis in simple terms:
        
        Agent: {agent_name}
        Task: {task.objective}
        
        Key Results:
        {self._summarize_results(result)}
        
        Provide a clear, professional explanation of what was analyzed and the key findings.
        """
    
    def _summarize_results(self, result: Dict[str, Any]) -> str:
        """Summarize results for explanation prompt."""
        summary_parts = []
        
        # Extract key metrics based on result structure
        if "var_estimates" in result:
            var_data = result["var_estimates"]
            if "parametric" in var_data and "var_95" in var_data["parametric"]:
                var_95 = var_data["parametric"]["var_95"]
                summary_parts.append(f"Portfolio VaR (95%): {var_95:.2%}")
        
        if "portfolio_volatility" in result:
            vol = result["portfolio_volatility"]
            summary_parts.append(f"Portfolio Volatility: {vol:.1%}")
        
        if "simulation_results" in result:
            sim_data = result["simulation_results"]
            prob_loss = sim_data.get("probability_of_loss", 0)
            summary_parts.append(f"Probability of Loss: {prob_loss:.1%}")
        
        if "price_forecasts" in result:
            forecasts = result["price_forecasts"]
            assets_forecasted = len(forecasts)
            summary_parts.append(f"Price forecasts generated for {assets_forecasted} assets")
        
        if "portfolio_data" in result:
            portfolio_data = result["portfolio_data"]
            assets_analyzed = len(portfolio_data)
            summary_parts.append(f"Data collected for {assets_analyzed} portfolio assets")
        
        return ". ".join(summary_parts) if summary_parts else "Analysis completed successfully"
    
    def _generate_rule_based_explanation(
        self,
        agent_name: str,
        task: AgentTask,
        result: Dict[str, Any]
    ) -> str:
        """Generate rule-based explanation as fallback."""
        explanations = {
            "DataAgent": self._explain_data_agent_result,
            "RiskAgent": self._explain_risk_agent_result,
            "ForecastAgent": self._explain_forecast_agent_result,
            "ReportGeneratorAgent": self._explain_report_agent_result
        }
        
        explain_func = explanations.get(agent_name, self._explain_generic_result)
        return explain_func(task, result)
    
    def _explain_data_agent_result(self, task: AgentTask, result: Dict[str, Any]) -> str:
        """Explain data agent results."""
        portfolio_data = result.get("portfolio_data", {})
        market_indices = result.get("market_indices", {})
        sentiment_analysis = result.get("sentiment_analysis", {})
        
        explanation = f"Data collection completed for {len(portfolio_data)} portfolio assets. "
        
        if market_indices:
            explanation += f"Market data retrieved for {len(market_indices)} indices. "
        
        if sentiment_analysis:
            explanation += f"Sentiment analysis performed on {len(sentiment_analysis)} assets. "
        
        data_quality = result.get("data_coverage", {}).get("overall", 0)
        explanation += f"Overall data quality score: {data_quality:.1%}."
        
        return explanation
    
    def _explain_risk_agent_result(self, task: AgentTask, result: Dict[str, Any]) -> str:
        """Explain risk agent results."""
        explanation = "Risk analysis completed using multiple methodologies. "
        
        if "var_estimates" in result:
            var_data = result["var_estimates"]
            methods = list(var_data.keys())
            explanation += f"VaR calculated using {len(methods)} methods: {', '.join(methods)}. "
            
            if "parametric" in var_data:
                var_95 = var_data["parametric"].get("var_95", 0)
                explanation += f"95% VaR estimated at {abs(var_95):.2%} of portfolio value. "
        
        if "simulation_results" in result:
            sim_data = result["simulation_results"]
            num_sims = sim_data.get("num_simulations", 0)
            prob_loss = sim_data.get("probability_of_loss", 0)
            explanation += f"Monte Carlo simulation with {num_sims:,} iterations shows {prob_loss:.1%} probability of loss. "
        
        if "stress_test_results" in result:
            stress_results = result["stress_test_results"]
            scenarios = len(stress_results)
            explanation += f"Stress testing completed for {scenarios} adverse scenarios."
        
        return explanation
    
    def _explain_forecast_agent_result(self, task: AgentTask, result: Dict[str, Any]) -> str:
        """Explain forecast agent results."""
        explanation = "Financial forecasting analysis completed. "
        
        if "price_forecasts" in result:
            forecasts = result["price_forecasts"]
            horizon = result.get("forecast_horizon", 30)
            explanation += f"Price forecasts generated for {len(forecasts)} assets over {horizon} days using ensemble modeling. "
        
        if "volatility_forecasts" in result:
            vol_forecasts = result["volatility_forecasts"]
            explanation += f"Volatility forecasts computed for {len(vol_forecasts)} assets. "
        
        if "scenarios" in result:
            scenarios = result["scenarios"]
            explanation += f"Scenario analysis performed for {len(scenarios)} market conditions. "
        
        if "trend_analysis" in result:
            trend_data = result["trend_analysis"]
            portfolio_trends = trend_data.get("portfolio_trends", {})
            explanation += f"Trend analysis completed for {len(portfolio_trends)} portfolio assets."
        
        return explanation
    
    def _explain_report_agent_result(self, task: AgentTask, result: Dict[str, Any]) -> str:
        """Explain report generator results."""
        explanation = "Comprehensive risk report generated successfully. "
        
        sections = []
        if "executive_summary" in result:
            sections.append("executive summary")
        if "risk_assessment" in result:
            sections.append("risk assessment")
        if "portfolio_analysis" in result:
            sections.append("portfolio analysis")
        if "forecast_analysis" in result:
            sections.append("forecast analysis")
        if "recommendations" in result:
            sections.append("recommendations")
        
        if sections:
            explanation += f"Report includes: {', '.join(sections)}. "
        
        if "charts" in result:
            charts = result["charts"]
            chart_count = len([v for v in charts.values() if v and v != ""])
            explanation += f"{chart_count} visualizations generated."
        
        return explanation
    
    def _explain_generic_result(self, task: AgentTask, result: Dict[str, Any]) -> str:
        """Generic explanation for unknown agent types."""
        return f"Task '{task.objective}' completed successfully with comprehensive analysis results."
    
    async def generate_workflow_explanation(
        self,
        workflow: Any,
        step_results: Dict[str, Any]
    ) -> str:
        """Generate explanation for entire workflow execution."""
        try:
            # Create workflow summary
            summary = f"Financial risk analysis workflow '{workflow.name}' completed with {len(step_results)} steps. "
            
            # Analyze step outcomes
            successful_steps = len([r for r in step_results.values() if r.status.value == "completed"])
            failed_steps = len([r for r in step_results.values() if r.status.value == "error"])
            
            summary += f"{successful_steps} steps completed successfully"
            if failed_steps > 0:
                summary += f", {failed_steps} steps encountered errors"
            summary += ". "
            
            # Extract key insights from results
            insights = self._extract_workflow_insights(step_results)
            if insights:
                summary += "Key findings: " + ". ".join(insights) + "."
            
            # Generate detailed explanation using LLM
            prompt = f"""
            Provide a professional summary of this financial risk analysis workflow:
            
            {summary}
            
            Focus on the business value and key risk insights discovered.
            """
            
            response = await self.hf_client.generate_text(
                prompt=prompt,
                model="microsoft/DialoGPT-large",
                max_length=200
            )
            
            detailed_explanation = response.get("generated_text", summary)
            return detailed_explanation if len(detailed_explanation) > 50 else summary
            
        except Exception as e:
            return f"Workflow {workflow.name} completed with {len(step_results)} analytical steps."
    
    def _extract_workflow_insights(self, step_results: Dict[str, Any]) -> List[str]:
        """Extract key insights from workflow step results."""
        insights = []
        
        for step_id, result in step_results.items():
            if result.status.value != "completed" or not result.result:
                continue
            
            result_data = result.result
            
            # Extract VaR insights
            if "var_estimates" in result_data:
                var_data = result_data["var_estimates"].get("parametric", {})
                if "var_95" in var_data:
                    var_95 = abs(var_data["var_95"])
                    if var_95 > 0.1:  # >10% VaR
                        insights.append(f"High portfolio risk detected (VaR: {var_95:.1%})")
                    elif var_95 < 0.02:  # <2% VaR
                        insights.append(f"Low portfolio risk profile (VaR: {var_95:.1%})")
            
            # Extract stress test insights
            if "aggregate_impact" in result_data:
                worst_case = abs(result_data["aggregate_impact"].get("worst_case_loss", 0))
                if worst_case > 0.2:  # >20% stress loss
                    scenario = result_data["aggregate_impact"].get("worst_case_scenario", "stress")
                    insights.append(f"Significant stress scenario risk identified ({scenario}: {worst_case:.1%})")
            
            # Extract correlation insights
            if "high_correlations" in result_data:
                high_corr_count = len(result_data["high_correlations"])
                if high_corr_count > 3:
                    insights.append(f"Portfolio concentration risk detected ({high_corr_count} high correlations)")
            
            # Extract forecast insights
            if "integrated_summary" in result_data:
                forecast_summary = result_data["integrated_summary"].get("forecast_summary", {})
                expected_return = forecast_summary.get("average_expected_return", 0)
                if expected_return > 0.1:
                    insights.append(f"Positive return outlook ({expected_return:.1%})")
                elif expected_return < -0.05:
                    insights.append(f"Negative return outlook ({expected_return:.1%})")
        
        return insights[:3]  # Return top 3 insights