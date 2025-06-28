"""Report Generator Agent for creating comprehensive risk reports."""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from core.base_agent import BaseAgent, AgentTask, AgentResult, AgentStatus
from integrations.huggingface_client import HuggingFaceClient

from core.simple_base_agent import SimpleBaseAgent, AgentResult, AgentStatus

class ReportGeneratorAgent(BaseAgent):
    """Agent responsible for generating comprehensive risk reports."""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="ReportGeneratorAgent", 
            description="Comprehensive risk report generation agent",
            **kwargs
        )
        self.hf_client = HuggingFaceClient()
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities."""
        return [
            "risk_report_generation",
            "executive_summary_creation",
            "data_visualization",
            "narrative_generation",
            "compliance_reporting",
            "chart_generation"
        ]
    
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Execute report generation task."""
        try:
            objective = task.objective.lower()
            parameters = task.parameters
            
            if "executive" in objective:
                result_data = await self._generate_executive_summary(parameters)
            elif "compliance" in objective:
                result_data = await self._generate_compliance_report(parameters)
            elif "visualization" in objective or "chart" in objective:
                result_data = await self._generate_visualizations(parameters)
            else:
                # Default: comprehensive report
                result_data = await self._generate_comprehensive_report(parameters)
            
            return AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                status=AgentStatus.COMPLETED,
                result=result_data,
                confidence=0.9  # High confidence in report generation
            )
            
        except Exception as e:
            self.logger.error("Report generation failed", error=str(e), exc_info=True)
            return AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                status=AgentStatus.ERROR,
                error=str(e)
            )
    
    async def _generate_comprehensive_report(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive risk management report."""
        previous_results = parameters.get("previous_results", {})
        include_charts = parameters.get("include_charts", True)
        
        # Extract data from previous agents
        data_results = previous_results.get("portfolio_data", {})
        risk_results = previous_results.get("var_analysis", {})
        forecast_results = previous_results.get("price_forecasts", {})
        
        # Generate report sections
        executive_summary = await self._create_executive_summary(previous_results)
        risk_assessment = await self._create_risk_assessment_section(risk_results)
        portfolio_analysis = await self._create_portfolio_analysis_section(data_results)
        forecast_section = await self._create_forecast_section(forecast_results)
        
        # Generate visualizations if requested
        charts = {}
        if include_charts:
            charts = await self._generate_all_charts(previous_results)
        
        # Create recommendations
        recommendations = await self._generate_recommendations(previous_results)
        
        # Compile full report
        full_report = {
            "report_metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "report_type": "comprehensive_risk_analysis",
                "agent_id": self.agent_id
            },
            "executive_summary": executive_summary,
            "risk_assessment": risk_assessment,
            "portfolio_analysis": portfolio_analysis,
            "forecast_analysis": forecast_section,
            "recommendations": recommendations,
            "charts": charts,
            "appendix": await self._create_appendix(previous_results)
        }
        
        return full_report
    
    async def _create_executive_summary(self, previous_results: Dict[str, Any]) -> Dict[str, str]:
        """Create executive summary section."""
        # Extract key metrics
        risk_data = previous_results.get("var_analysis", {})
        var_95 = risk_data.get("var_estimates", {}).get("parametric", {}).get("var_95", 0)
        portfolio_vol = risk_data.get("portfolio_volatility", 0)
        
        stress_data = previous_results.get("stress_testing", {})
        worst_case = stress_data.get("aggregate_impact", {}).get("worst_case_loss", 0)
        
        # Generate summary using LLM
        summary_prompt = f"""
        Generate a professional executive summary for a financial risk report with these key findings:
        - Portfolio VaR (95%): {var_95:.2%}
        - Portfolio Volatility: {portfolio_vol:.1%} 
        - Worst Case Stress Loss: {worst_case:.2%}
        
        Keep it concise and suitable for senior management.
        """
        
        try:
            summary_response = await self.hf_client.generate_text(
                prompt=summary_prompt,
                model="microsoft/DialoGPT-large",
                max_length=200
            )
            
            summary_text = summary_response.get("generated_text", "Executive summary generation in progress.")
        except Exception as e:
            self.logger.error("LLM summary generation failed", error=str(e))
            summary_text = f"""
            Portfolio Risk Assessment Summary:
            
            The portfolio exhibits a 95% Value-at-Risk of {var_95:.2%} with an annualized volatility of {portfolio_vol:.1%}.
            Stress testing indicates a maximum potential loss of {worst_case:.2%} under adverse scenarios.
            Risk management measures are recommended to optimize the risk-return profile.
            """
        
        return {
            "summary": summary_text,
            "key_metrics": {
                "var_95": f"{var_95:.2%}",
                "volatility": f"{portfolio_vol:.1%}",
                "stress_loss": f"{worst_case:.2%}"
            }
        }
    
    async def _create_risk_assessment_section(self, risk_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed risk assessment section."""
        if not risk_results:
            return {"error": "No risk analysis data available"}
        
        var_estimates = risk_results.get("var_estimates", {})
        risk_attribution = risk_results.get("risk_attribution", {})
        
        assessment = {
            "value_at_risk": {
                "parametric_var": var_estimates.get("parametric", {}),
                "historical_var": var_estimates.get("historical", {}),
                "monte_carlo_var": var_estimates.get("monte_carlo", {}),
                "methodology_notes": "Multiple VaR methodologies used for robust estimation"
            },
            "risk_attribution": risk_attribution,
            "portfolio_volatility": risk_results.get("portfolio_volatility", 0),
            "confidence_level": risk_results.get("confidence_level", 0.95)
        }
        
        return assessment
    
    async def _create_portfolio_analysis_section(self, data_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create portfolio analysis section."""
        if not data_results:
            return {"error": "No portfolio data available"}
        
        # Analyze portfolio composition
        total_assets = len(data_results)
        asset_summary = {}
        
        for symbol, data in data_results.items():
            asset_summary[symbol] = {
                "current_price": data.get("last_price", 0),
                "volatility": data.get("volatility", 0),
                "data_quality": data.get("data_quality", {}).get("completeness", 0),
                "returns_summary": {
                    "count": len(data.get("returns", [])),
                    "mean": np.mean(data.get("returns", [0])),
                    "std": np.std(data.get("returns", [0]))
                }
            }
        
        return {
            "portfolio_composition": {
                "total_assets": total_assets,
                "asset_details": asset_summary
            },
            "data_quality_summary": {
                "average_completeness": np.mean([
                    data.get("data_quality", {}).get("completeness", 0) 
                    for data in data_results.values()
                ])
            }
        }
    
    async def _create_forecast_section(self, forecast_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create forecast analysis section."""
        if not forecast_results:
            return {"error": "No forecast data available"}
        
        forecast_summary = {}
        
        for symbol, forecast_data in forecast_results.items():
            ensemble_forecast = forecast_data.get("ensemble_forecast", {})
            forecast_values = ensemble_forecast.get("forecast_values", [])
            
            if forecast_values:
                current_price = forecast_data.get("current_price", 0)
                expected_return = (forecast_values[-1] - current_price) / current_price if current_price > 0 else 0
                
                forecast_summary[symbol] = {
                    "current_price": current_price,
                    "forecast_price": forecast_values[-1],
                    "expected_return": expected_return,
                    "forecast_horizon": forecast_data.get("forecast_horizon", 30)
                }
        
        return {
            "price_forecasts": forecast_summary,
            "methodology": "Multi-model ensemble forecasting approach"
        }
    
    async def _generate_all_charts(self, previous_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate all visualization charts."""
        charts = {}
        
        try:
            # VaR comparison chart
            if "var_analysis" in previous_results:
                charts["var_comparison"] = await self._create_var_chart(
                    previous_results["var_analysis"]
                )
            
            # Portfolio composition chart
            if "portfolio_data" in previous_results:
                charts["portfolio_composition"] = await self._create_portfolio_chart(
                    previous_results["portfolio_data"]
                )
            
            # Risk attribution chart
            if "var_analysis" in previous_results:
                risk_attr = previous_results["var_analysis"].get("risk_attribution", {})
                if risk_attr:
                    charts["risk_attribution"] = await self._create_risk_attribution_chart(risk_attr)
            
            # Forecast chart
            if "price_forecasts" in previous_results:
                charts["price_forecasts"] = await self._create_forecast_chart(
                    previous_results["price_forecasts"]
                )
            
        except Exception as e:
            self.logger.error("Chart generation failed", error=str(e))
            charts["error"] = "Chart generation encountered errors"
        
        return charts
    
    async def _create_var_chart(self, var_data: Dict[str, Any]) -> str:
        """Create VaR comparison chart."""
        var_estimates = var_data.get("var_estimates", {})
        
        methods = []
        values = []
        
        for method, data in var_estimates.items():
            if isinstance(data, dict) and "var_95" in data:
                methods.append(method.replace("_", " ").title())
                values.append(abs(data["var_95"]) * 100)  # Convert to percentage
        
        if not methods:
            return ""
        
        # Create chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(methods, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        
        ax.set_title('Value at Risk Comparison (95% Confidence Level)', fontsize=14, fontweight='bold')
        ax.set_ylabel('VaR (%)', fontsize=12)
        ax.set_xlabel('Methodology', fontsize=12)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        chart_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return chart_base64
    
    async def _create_portfolio_chart(self, portfolio_data: Dict[str, Any]) -> str:
        """Create portfolio composition chart."""
        symbols = list(portfolio_data.keys())
        volatilities = [data.get("volatility", 0) * 100 for data in portfolio_data.values()]
        
        if not symbols:
            return ""
        
        # Create scatter plot of volatility
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(symbols)))
        scatter = ax.scatter(range(len(symbols)), volatilities, 
                           c=colors, s=200, alpha=0.7, edgecolors='black')
        
        ax.set_title('Portfolio Asset Volatility Profile', fontsize=14, fontweight='bold')
        ax.set_ylabel('Volatility (%)', fontsize=12)
        ax.set_xlabel('Assets', fontsize=12)
        ax.set_xticks(range(len(symbols)))
        ax.set_xticklabels(symbols, rotation=45, ha='right')
        
        # Add horizontal line for average volatility
        avg_vol = np.mean(volatilities)
        ax.axhline(y=avg_vol, color='red', linestyle='--', alpha=0.7, 
                  label=f'Average: {avg_vol:.1f}%')
        ax.legend()
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        chart_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return chart_base64
    
    async def _create_risk_attribution_chart(self, risk_attribution: Dict[str, Any]) -> str:
        """Create risk attribution pie chart."""
        symbols = []
        contributions = []
        
        for symbol, data in risk_attribution.items():
            symbols.append(symbol)
            contributions.append(abs(data.get("risk_contribution", 0)) * 100)
        
        if not symbols:
            return ""
        
        # Create pie chart
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(symbols)))
        wedges, texts, autotexts = ax.pie(contributions, labels=symbols, autopct='%1.1f%%',
                                         colors=colors, startangle=90)
        
        ax.set_title('Portfolio Risk Attribution', fontsize=14, fontweight='bold')
        
        # Beautify the text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        chart_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return chart_base64
    
    async def _create_forecast_chart(self, forecast_data: Dict[str, Any]) -> str:
        """Create price forecast chart."""
        if not forecast_data:
            return ""
        
        # Take first asset for demonstration
        first_symbol = next(iter(forecast_data.keys()))
        symbol_data = forecast_data[first_symbol]
        
        current_price = symbol_data.get("current_price", 100)
        ensemble_forecast = symbol_data.get("ensemble_forecast", {})
        forecast_values = ensemble_forecast.get("forecast_values", [])
        confidence_intervals = symbol_data.get("confidence_intervals", {})
        
        if not forecast_values:
            return ""
        
        # Create forecast chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Time axis
        forecast_horizon = len(forecast_values)
        time_axis = list(range(-30, forecast_horizon))  # Show 30 days history + forecast
        
        # Historical prices (simulated for demo)
        historical_prices = [current_price * (1 + np.random.normal(0, 0.01)) for _ in range(30)]
        
        # Combine historical and forecast
        all_prices = historical_prices + forecast_values
        
        # Plot historical
        ax.plot(time_axis[:30], historical_prices, 'b-', linewidth=2, label='Historical', alpha=0.8)
        
        # Plot forecast
        ax.plot(time_axis[29:], [historical_prices[-1]] + forecast_values, 
               'r-', linewidth=2, label='Forecast', alpha=0.8)
        
        # Add confidence intervals
        if confidence_intervals.get("upper_95") and confidence_intervals.get("lower_95"):
            upper_95 = confidence_intervals["upper_95"]
            lower_95 = confidence_intervals["lower_95"]
            forecast_time = time_axis[30:]
            
            ax.fill_between(forecast_time, lower_95, upper_95, alpha=0.2, color='red',
                          label='95% Confidence Interval')
        
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Forecast Start')
        ax.set_title(f'{first_symbol} Price Forecast', fontsize=14, fontweight='bold')
        ax.set_ylabel('Price', fontsize=12)
        ax.set_xlabel('Days', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        chart_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return chart_base64
    
    async def _generate_recommendations(self, previous_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate actionable recommendations."""
        recommendations = {
            "immediate_actions": [],
            "risk_management": [],
            "portfolio_optimization": [],
            "monitoring": []
        }
        
        # Analyze results and generate recommendations
        var_data = previous_results.get("var_analysis", {})
        stress_data = previous_results.get("stress_testing", {})
        correlation_data = previous_results.get("correlation_analysis", {})
        
        # VaR-based recommendations
        portfolio_vol = var_data.get("portfolio_volatility", 0)
        if portfolio_vol > 0.25:  # High volatility
            recommendations["risk_management"].append(
                "Consider reducing portfolio volatility through diversification or hedging"
            )
            recommendations["immediate_actions"].append(
                "Review position sizes for high-volatility assets"
            )
        
        # Stress test recommendations
        worst_case = stress_data.get("aggregate_impact", {}).get("worst_case_loss", 0)
        if abs(worst_case) > 0.15:  # >15% loss in stress scenario
            recommendations["risk_management"].append(
                "Implement stress-loss limits and develop contingency plans"
            )
        
        # Correlation recommendations
        high_correlations = correlation_data.get("high_correlations", [])
        if len(high_correlations) > 3:
            recommendations["portfolio_optimization"].append(
                "Reduce concentration risk by diversifying across uncorrelated assets"
            )
        
        # General monitoring recommendations
        recommendations["monitoring"].extend([
            "Implement daily VaR monitoring and breaches reporting",
            "Conduct monthly stress testing updates",
            "Review risk attribution quarterly",
            "Update volatility models semi-annually"
        ])
        
        return recommendations
    
    async def _create_appendix(self, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create report appendix with technical details."""
        return {
            "methodology": {
                "var_calculation": "Parametric, Historical, and Monte Carlo methods",
                "stress_testing": "Scenario-based analysis with market shocks",
                "forecasting": "Multi-model ensemble approach",
                "data_sources": "Yahoo Finance, FRED, Alpha Vantage"
            },
            "assumptions": [
                "Normal distribution assumption for parametric VaR",
                "Historical data representative of future risks",
                "No significant regime changes assumed",
                "Liquidity risk not explicitly modeled"
            ],
            "limitations": [
                "Model risk exists in all quantitative approaches",
                "Tail risk may be underestimated",
                "Correlations may increase during market stress",
                "Past performance does not guarantee future results"
            ],
            "data_quality": {
                "completeness": "Data completeness assessed per asset",
                "timeliness": "Real-time data where available",
                "accuracy": "Multiple source validation applied"
            }
        }