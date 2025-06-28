"""Forecast Agent for financial scenario forecasting and prediction."""

import asyncio
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from core.base_agent import BaseAgent, AgentTask, AgentResult, AgentStatus
from integrations.huggingface_client import HuggingFaceClient

from core.simple_base_agent import SimpleBaseAgent, AgentResult, AgentStatus

class ForecastAgent(BaseAgent):
    """Agent responsible for financial forecasting and scenario analysis."""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="ForecastAgent",
            description="Financial forecasting and scenario analysis agent",
            **kwargs
        )
        self.hf_client = HuggingFaceClient()
        self.scaler = StandardScaler()
    
    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities."""
        return [
            "price_forecasting",
            "volatility_forecasting",
            "scenario_generation",
            "trend_analysis",
            "risk_scenario_modeling",
            "economic_forecasting",
            "correlation_forecasting"
        ]
    
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Execute forecasting task."""
        try:
            objective = task.objective.lower()
            parameters = task.parameters
            
            if "price" in objective:
                result_data = await self._forecast_prices(parameters)
            elif "volatility" in objective:
                result_data = await self._forecast_volatility(parameters)
            elif "scenario" in objective:
                result_data = await self._generate_scenarios(parameters)
            elif "trend" in objective:
                result_data = await self._analyze_trends(parameters)
            else:
                # Default: comprehensive forecasting
                result_data = await self._comprehensive_forecast(parameters)
            
            return AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                status=AgentStatus.COMPLETED,
                result=result_data,
                confidence=self._calculate_forecast_confidence(result_data)
            )
            
        except Exception as e:
            self.logger.error("Forecast agent task failed", error=str(e), exc_info=True)
            return AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                status=AgentStatus.ERROR,
                error=str(e)
            )
    
    async def _forecast_prices(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast asset prices using multiple models."""
        portfolio_data = parameters.get("previous_results", {}).get("portfolio_data", {})
        forecast_horizon = parameters.get("forecast_horizon", 30)
        
        if not portfolio_data:
            raise ValueError("No portfolio data available for price forecasting")
        
        price_forecasts = {}
        
        for symbol, data in portfolio_data.items():
            try:
                prices = data.get("prices", [])
                if not prices or len(prices) < 20:  # Need minimum data
                    continue
                
                # Prepare price series
                price_series = pd.DataFrame(prices)
                if 'Close' not in price_series.columns:
                    continue
                
                close_prices = price_series['Close'].values
                
                # Multiple forecasting approaches
                forecasts = {}
                
                # 1. Linear trend extrapolation
                linear_forecast = await self._linear_trend_forecast(close_prices, forecast_horizon)
                forecasts["linear_trend"] = linear_forecast
                
                # 2. Random Forest model
                rf_forecast = await self._random_forest_forecast(close_prices, forecast_horizon)
                forecasts["random_forest"] = rf_forecast
                
                # 3. ARIMA-like simple model
                arima_forecast = await self._simple_arima_forecast(close_prices, forecast_horizon)
                forecasts["arima"] = arima_forecast
                
                # 4. Ensemble forecast
                ensemble_forecast = await self._ensemble_forecast(forecasts)
                
                price_forecasts[symbol] = {
                    "current_price": float(close_prices[-1]),
                    "forecasts": forecasts,
                    "ensemble_forecast": ensemble_forecast,
                    "forecast_horizon": forecast_horizon,
                    "confidence_intervals": await self._calculate_prediction_intervals(
                        close_prices, ensemble_forecast
                    )
                }
                
            except Exception as e:
                self.logger.error(f"Price forecast failed for {symbol}", error=str(e))
                continue
        
        return {
            "price_forecasts": price_forecasts,
            "forecast_horizon": forecast_horizon,
            "forecast_timestamp": datetime.utcnow().isoformat(),
            "methodology": "Multi-model ensemble approach"
        }
    
    async def _forecast_volatility(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast volatility for portfolio assets."""
        portfolio_data = parameters.get("previous_results", {}).get("portfolio_data", {})
        forecast_horizon = parameters.get("forecast_horizon", 30)
        
        volatility_forecasts = {}
        
        for symbol, data in portfolio_data.items():
            try:
                returns = data.get("returns", [])
                if not returns or len(returns) < 30:
                    continue
                
                returns_array = np.array(returns)
                
                # GARCH-like volatility forecasting
                vol_forecast = await self._garch_volatility_forecast(returns_array, forecast_horizon)
                
                # EWMA volatility forecast
                ewma_forecast = await self._ewma_volatility_forecast(returns_array, forecast_horizon)
                
                # Historical volatility projection
                hist_vol = np.std(returns_array) * np.sqrt(252)
                
                volatility_forecasts[symbol] = {
                    "current_volatility": data.get("volatility", hist_vol),
                    "garch_forecast": vol_forecast,
                    "ewma_forecast": ewma_forecast,
                    "historical_volatility": hist_vol,
                    "forecast_horizon": forecast_horizon
                }
                
            except Exception as e:
                self.logger.error(f"Volatility forecast failed for {symbol}", error=str(e))
                continue
        
        return {
            "volatility_forecasts": volatility_forecasts,
            "forecast_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _generate_scenarios(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate multiple scenarios for risk analysis."""
        forecast_horizon = parameters.get("forecast_horizon", 30)
        scenarios = parameters.get("scenarios", ["base", "stress", "optimistic"])
        portfolio_data = parameters.get("previous_results", {}).get("portfolio_data", {})
        
        scenario_results = {}
        
        for scenario_name in scenarios:
            try:
                scenario_params = await self._get_scenario_parameters(scenario_name)
                scenario_forecast = await self._generate_scenario_forecast(
                    portfolio_data, scenario_params, forecast_horizon
                )
                
                scenario_results[scenario_name] = scenario_forecast
                
            except Exception as e:
                self.logger.error(f"Scenario generation failed for {scenario_name}", error=str(e))
                continue
        
        # Generate scenario narrative
        scenario_narrative = await self._generate_scenario_narrative(scenario_results)
        
        return {
            "scenarios": scenario_results,
            "scenario_narrative": scenario_narrative,
            "forecast_horizon": forecast_horizon,
            "generation_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _analyze_trends(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market and portfolio trends."""
        portfolio_data = parameters.get("previous_results", {}).get("portfolio_data", {})
        market_data = parameters.get("previous_results", {}).get("market_indices", {})
        
        trend_analysis = {}
        
        # Portfolio asset trends
        for symbol, data in portfolio_data.items():
            try:
                prices = data.get("prices", [])
                if not prices:
                    continue
                
                price_series = pd.DataFrame(prices)
                if 'Close' not in price_series.columns:
                    continue
                
                close_prices = price_series['Close'].values
                trend_metrics = await self._calculate_trend_metrics(close_prices)
                trend_analysis[symbol] = trend_metrics
                
            except Exception as e:
                self.logger.error(f"Trend analysis failed for {symbol}", error=str(e))
                continue
        
        # Market trend analysis
        market_trends = {}
        for index, data in market_data.items():
            try:
                current_level = data.get("current_level", 0)
                trend = data.get("trend", "neutral")
                
                market_trends[index] = {
                    "current_trend": trend,
                    "current_level": current_level,
                    "trend_strength": self._assess_trend_strength(data)
                }
                
            except Exception as e:
                self.logger.error(f"Market trend analysis failed for {index}", error=str(e))
                continue
        
        return {
            "portfolio_trends": trend_analysis,
            "market_trends": market_trends,
            "trend_summary": await self._summarize_trends(trend_analysis, market_trends),
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _comprehensive_forecast(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive forecasting analysis."""
        # Run all forecasting components
        price_forecast = await self._forecast_prices(parameters)
        volatility_forecast = await self._forecast_volatility(parameters)
        scenario_forecast = await self._generate_scenarios(parameters)
        trend_analysis = await self._analyze_trends(parameters)
        
        # Generate integrated forecast summary
        integrated_summary = await self._generate_integrated_forecast_summary(
            price_forecast, volatility_forecast, scenario_forecast, trend_analysis
        )
        
        return {
            "price_forecasts": price_forecast,
            "volatility_forecasts": volatility_forecast,
            "scenario_analysis": scenario_forecast,
            "trend_analysis": trend_analysis,
            "integrated_summary": integrated_summary,
            "forecast_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _linear_trend_forecast(self, prices: np.ndarray, horizon: int) -> Dict[str, Any]:
        """Generate linear trend forecast."""
        x = np.arange(len(prices)).reshape(-1, 1)
        y = prices
        
        model = LinearRegression()
        model.fit(x, y)
        
        # Forecast future values
        future_x = np.arange(len(prices), len(prices) + horizon).reshape(-1, 1)
        forecast = model.predict(future_x)
        
        return {
            "forecast_values": forecast.tolist(),
            "trend_slope": float(model.coef_[0]),
            "r_squared": float(model.score(x, y))
        }
    
    async def _random_forest_forecast(self, prices: np.ndarray, horizon: int) -> Dict[str, Any]:
        """Generate Random Forest forecast."""
        # Create features (lagged prices)
        lag_periods = [1, 2, 3, 5, 10]
        features = []
        targets = []
        
        max_lag = max(lag_periods)
        for i in range(max_lag, len(prices)):
            feature_row = [prices[i - lag] for lag in lag_periods]
            features.append(feature_row)
            targets.append(prices[i])
        
        if len(features) < 10:  # Need minimum samples
            return {"forecast_values": [prices[-1]] * horizon, "feature_importance": {}}
        
        X = np.array(features)
        y = np.array(targets)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Generate forecasts
        forecast_values = []
        current_prices = prices[-max_lag:].tolist()
        
        for _ in range(horizon):
            feature_row = [current_prices[-lag] for lag in lag_periods]
            next_price = model.predict([feature_row])[0]
            forecast_values.append(next_price)
            current_prices.append(next_price)
        
        return {
            "forecast_values": forecast_values,
            "feature_importance": dict(zip(
                [f"lag_{lag}" for lag in lag_periods],
                model.feature_importances_.tolist()
            ))
        }
    
    async def _simple_arima_forecast(self, prices: np.ndarray, horizon: int) -> Dict[str, Any]:
        """Generate simple ARIMA-like forecast."""
        # Simple AR(1) model
        returns = np.diff(np.log(prices))
        if len(returns) < 5:
            return {"forecast_values": [prices[-1]] * horizon}
        
        # Estimate AR(1) coefficient
        ar_coef = np.corrcoef(returns[:-1], returns[1:])[0, 1]
        mean_return = np.mean(returns)
        
        # Generate forecast
        forecast_values = []
        last_price = prices[-1]
        last_return = returns[-1]
        
        for _ in range(horizon):
            next_return = ar_coef * last_return + (1 - ar_coef) * mean_return
            next_price = last_price * np.exp(next_return)
            forecast_values.append(next_price)
            last_price = next_price
            last_return = next_return
        
        return {
            "forecast_values": forecast_values,
            "ar_coefficient": float(ar_coef),
            "mean_return": float(mean_return)
        }
    
    async def _ensemble_forecast(self, forecasts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Combine multiple forecasts into ensemble."""
        # Equal weighting for simplicity
        weights = {method: 1.0 / len(forecasts) for method in forecasts.keys()}
        
        # Get forecast values
        all_forecasts = []
        for method, forecast_data in forecasts.items():
            forecast_values = forecast_data.get("forecast_values", [])
            if forecast_values:
                all_forecasts.append(np.array(forecast_values))
        
        if not all_forecasts:
            return {"forecast_values": [], "weights": weights}
        
        # Ensure all forecasts have same length
        min_length = min(len(f) for f in all_forecasts)
        trimmed_forecasts = [f[:min_length] for f in all_forecasts]
        
        # Calculate weighted average
        ensemble_forecast = np.average(trimmed_forecasts, axis=0, weights=list(weights.values()))
        
        return {
            "forecast_values": ensemble_forecast.tolist(),
            "weights": weights,
            "forecast_variance": np.var(trimmed_forecasts, axis=0).tolist()
        }
    
    async def _calculate_prediction_intervals(
        self, 
        historical_prices: np.ndarray, 
        forecast_data: Dict[str, Any]
    ) -> Dict[str, List[float]]:
        """Calculate prediction intervals for forecasts."""
        forecast_values = forecast_data.get("forecast_values", [])
        if not forecast_values:
            return {"upper_95": [], "lower_95": [], "upper_68": [], "lower_68": []}
        
        # Calculate historical volatility
        returns = np.diff(np.log(historical_prices))
        volatility = np.std(returns)
        
        # Calculate prediction intervals (assuming normal distribution)
        forecast_array = np.array(forecast_values)
        
        # Increasing uncertainty over time
        time_adjustment = np.sqrt(np.arange(1, len(forecast_values) + 1))
        
        intervals = {}
        for confidence, z_score in [("95", 1.96), ("68", 1.0)]:
            upper_bound = forecast_array * np.exp(z_score * volatility * time_adjustment)
            lower_bound = forecast_array * np.exp(-z_score * volatility * time_adjustment)
            
            intervals[f"upper_{confidence}"] = upper_bound.tolist()
            intervals[f"lower_{confidence}"] = lower_bound.tolist()
        
        return intervals
    
    async def _garch_volatility_forecast(self, returns: np.ndarray, horizon: int) -> Dict[str, Any]:
        """Simple GARCH-like volatility forecast."""
        # Simple EWMA model for volatility
        lambda_param = 0.94
        
        # Calculate squared returns
        squared_returns = returns ** 2
        
        # EWMA variance
        weights = np.array([(1 - lambda_param) * (lambda_param ** i) for i in range(len(squared_returns))])
        weights = weights / weights.sum()
        
        current_variance = np.sum(weights * squared_returns[::-1])
        
        # Project forward (mean-reverting to long-term average)
        long_term_var = np.var(returns)
        forecast_vars = []
        
        for t in range(horizon):
            if t == 0:
                forecast_var = current_variance
            else:
                # Mean reversion
                forecast_var = long_term_var + (current_variance - long_term_var) * (lambda_param ** t)
            forecast_vars.append(forecast_var)
        
        forecast_vols = [np.sqrt(var * 252) for var in forecast_vars]  # Annualized
        
        return {
            "forecast_volatilities": forecast_vols,
            "current_volatility": np.sqrt(current_variance * 252),
            "long_term_volatility": np.sqrt(long_term_var * 252)
        }
    
    async def _ewma_volatility_forecast(self, returns: np.ndarray, horizon: int) -> Dict[str, Any]:
        """EWMA volatility forecast."""
        # Similar to GARCH but simpler persistence
        current_vol = np.std(returns[-30:]) * np.sqrt(252) if len(returns) >= 30 else np.std(returns) * np.sqrt(252)
        long_term_vol = np.std(returns) * np.sqrt(252)
        
        # Simple exponential decay to long-term
        decay_rate = 0.95
        forecast_vols = []
        
        for t in range(horizon):
            forecast_vol = long_term_vol + (current_vol - long_term_vol) * (decay_rate ** t)
            forecast_vols.append(forecast_vol)
        
        return {
            "forecast_volatilities": forecast_vols,
            "decay_rate": decay_rate
        }
    
    async def _get_scenario_parameters(self, scenario_name: str) -> Dict[str, Any]:
        """Get parameters for different scenarios."""
        scenario_params = {
            "base": {
                "growth_rate": 0.0,
                "volatility_multiplier": 1.0,
                "correlation_adjustment": 0.0
            },
            "stress": {
                "growth_rate": -0.20,
                "volatility_multiplier": 2.0,
                "correlation_adjustment": 0.3
            },
            "optimistic": {
                "growth_rate": 0.15,
                "volatility_multiplier": 0.8,
                "correlation_adjustment": -0.1
            }
        }
        
        return scenario_params.get(scenario_name, scenario_params["base"])
    
    async def _generate_scenario_forecast(
        self, 
        portfolio_data: Dict[str, Any], 
        scenario_params: Dict[str, Any], 
        horizon: int
    ) -> Dict[str, Any]:
        """Generate forecast for specific scenario."""
        scenario_forecasts = {}
        
        growth_rate = scenario_params.get("growth_rate", 0.0)
        vol_multiplier = scenario_params.get("volatility_multiplier", 1.0)
        
        for symbol, data in portfolio_data.items():
            current_price = data.get("last_price", 100)
            volatility = data.get("volatility", 0.2) * vol_multiplier
            
            # Generate scenario path
            dt = 1 / 252  # Daily time step
            prices = [current_price]
            
            for t in range(horizon):
                drift = growth_rate * dt
                shock = volatility * np.sqrt(dt) * np.random.normal()
                next_price = prices[-1] * np.exp(drift + shock)
                prices.append(next_price)
            
            scenario_forecasts[symbol] = {
                "price_path": prices[1:],  # Exclude initial price
                "final_price": prices[-1],
                "total_return": (prices[-1] - current_price) / current_price,
                "scenario_volatility": volatility
            }
        
        return scenario_forecasts
    
    async def _calculate_trend_metrics(self, prices: np.ndarray) -> Dict[str, Any]:
        """Calculate trend metrics for price series."""
        if len(prices) < 20:
            return {"trend": "insufficient_data"}
        
        # Moving averages
        ma_20 = np.mean(prices[-20:])
        ma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else ma_20
        
        current_price = prices[-1]
        
        # Trend determination
        short_trend = "up" if current_price > ma_20 else "down"
        long_trend = "up" if ma_20 > ma_50 else "down"
        
        # Trend strength
        price_momentum = (current_price - prices[-min(20, len(prices))]) / prices[-min(20, len(prices))]
        
        return {
            "short_term_trend": short_trend,
            "long_term_trend": long_trend,
            "price_momentum": float(price_momentum),
            "ma_20": float(ma_20),
            "ma_50": float(ma_50),
            "current_price": float(current_price)
        }
    
    def _assess_trend_strength(self, market_data: Dict[str, Any]) -> str:
        """Assess trend strength from market data."""
        daily_change = market_data.get("daily_change", 0)
        volatility = market_data.get("volatility", 0)
        
        if abs(daily_change) > volatility * 2:
            return "strong"
        elif abs(daily_change) > volatility:
            return "moderate"
        else:
            return "weak"
    
    async def _summarize_trends(
        self, 
        portfolio_trends: Dict[str, Any], 
        market_trends: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Summarize trend analysis."""
        # Portfolio trend summary
        up_trends = sum(1 for trend in portfolio_trends.values() 
                       if trend.get("short_term_trend") == "up")
        total_assets = len(portfolio_trends)
        
        # Market trend summary
        positive_market_trends = sum(1 for trend in market_trends.values() 
                                   if trend.get("current_trend") == "up")
        total_indices = len(market_trends)
        
        return {
            "portfolio_trend_bias": "bullish" if up_trends > total_assets / 2 else "bearish",
            "portfolio_trend_ratio": up_trends / total_assets if total_assets > 0 else 0,
            "market_trend_bias": "bullish" if positive_market_trends > total_indices / 2 else "bearish",
            "market_trend_ratio": positive_market_trends / total_indices if total_indices > 0 else 0,
            "overall_trend_sentiment": "positive" if (up_trends / max(total_assets, 1) + 
                                                    positive_market_trends / max(total_indices, 1)) / 2 > 0.5 else "negative"
        }
    
    async def _generate_scenario_narrative(self, scenario_results: Dict[str, Any]) -> str:
        """Generate narrative for scenario analysis."""
        try:
            # Prepare scenario summary
            summary_text = "Scenario Analysis Results:\n"
            for scenario, results in scenario_results.items():
                if results:
                    avg_return = np.mean([data.get("total_return", 0) for data in results.values()])
                    summary_text += f"\n{scenario.title()} scenario: Average portfolio return {avg_return:.2%}"
            
            # Generate narrative using LLM
            narrative = await self.hf_client.generate_text(
                prompt=f"Provide a professional analysis of these financial scenarios:\n{summary_text}",
                model="microsoft/DialoGPT-large",
                max_length=150
            )
            
            return narrative.get("generated_text", "Scenario analysis completed with multiple outcomes analyzed.")
            
        except Exception as e:
            self.logger.error("Failed to generate scenario narrative", error=str(e))
            return "Multiple scenarios analyzed showing varying portfolio outcomes."
    
    async def _generate_integrated_forecast_summary(
        self,
        price_forecast: Dict[str, Any],
        volatility_forecast: Dict[str, Any],
        scenario_forecast: Dict[str, Any],
        trend_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate integrated forecast summary."""
        # Extract key metrics
        price_data = price_forecast.get("price_forecasts", {})
        vol_data = volatility_forecast.get("volatility_forecasts", {})
        trend_data = trend_analysis.get("trend_summary", {})
        
        # Calculate aggregate metrics
        avg_expected_return = 0
        avg_volatility = 0
        asset_count = 0
        
        for symbol in price_data.keys():
            if symbol in price_data and symbol in vol_data:
                ensemble_forecast = price_data[symbol].get("ensemble_forecast", {})
                forecast_values = ensemble_forecast.get("forecast_values", [])
                
                if forecast_values:
                    current_price = price_data[symbol].get("current_price", 0)
                    expected_return = (forecast_values[-1] - current_price) / current_price
                    avg_expected_return += expected_return
                    asset_count += 1
                
                current_vol = vol_data[symbol].get("current_volatility", 0)
                avg_volatility += current_vol
        
        if asset_count > 0:
            avg_expected_return /= asset_count
            avg_volatility /= asset_count
        
        return {
            "forecast_summary": {
                "average_expected_return": avg_expected_return,
                "average_volatility": avg_volatility,
                "portfolio_trend_bias": trend_data.get("portfolio_trend_bias", "neutral"),
                "market_trend_bias": trend_data.get("market_trend_bias", "neutral"),
                "forecast_confidence": "medium"
            },
            "key_insights": [
                f"Expected portfolio return: {avg_expected_return:.2%}",
                f"Average volatility: {avg_volatility:.1%}",
                f"Trend bias: {trend_data.get('portfolio_trend_bias', 'neutral')}"
            ],
            "summary_timestamp": datetime.utcnow().isoformat()
        }
    
    def _calculate_forecast_confidence(self, result_data: Dict[str, Any]) -> float:
        """Calculate confidence score for forecasts."""
        confidence_factors = []
        
        # Price forecast confidence
        if "price_forecasts" in result_data:
            price_data = result_data["price_forecasts"]
            for symbol_data in price_data.values():
                forecasts = symbol_data.get("forecasts", {})
                if len(forecasts) >= 3:  # Multiple models used
                    confidence_factors.append(0.8)
                elif len(forecasts) >= 2:
                    confidence_factors.append(0.7)
                else:
                    confidence_factors.append(0.5)
        
        # Volatility forecast confidence
        if "volatility_forecasts" in result_data:
            vol_data = result_data["volatility_forecasts"]
            if len(vol_data) > 0:
                confidence_factors.append(0.75)
        
        # Scenario analysis confidence
        if "scenario_analysis" in result_data:
            scenarios = result_data["scenario_analysis"].get("scenarios", {})
            scenario_count = len(scenarios)
            confidence_factors.append(min(scenario_count / 3, 1.0))
        
        # Trend analysis confidence
        if "trend_analysis" in result_data:
            portfolio_trends = result_data["trend_analysis"].get("portfolio_trends", {})
            if len(portfolio_trends) > 0:
                confidence_factors.append(0.7)
        
        return np.mean(confidence_factors) if confidence_factors else 0.6