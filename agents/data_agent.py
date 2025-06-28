"""Data Agent for financial data collection and preprocessing."""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from core.base_agent import BaseAgent, AgentTask, AgentResult, AgentStatus
from data.sources import DataSourceManager
from data.validators import DataValidator
from integrations.huggingface_client import HuggingFaceClient

from core.simple_base_agent import SimpleBaseAgent, AgentResult, AgentStatus

class DataAgent(BaseAgent):
    """Agent responsible for financial data collection and preprocessing."""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="DataAgent",
            description="Financial data collection and preprocessing agent",
            **kwargs
        )
        self.data_source_manager = DataSourceManager()
        self.data_validator = DataValidator()
        self.hf_client = HuggingFaceClient()
    
    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities."""
        return [
            "market_data_collection",
            "economic_indicators",
            "news_sentiment_analysis",
            "data_validation",
            "data_preprocessing",
            "time_series_analysis"
        ]
    
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Execute data collection and preprocessing task."""
        try:
            objective = task.objective.lower()
            parameters = task.parameters
            
            if "market data" in objective or "fetch" in objective:
                result_data = await self._fetch_market_data(parameters)
            elif "sentiment" in objective:
                result_data = await self._analyze_news_sentiment(parameters)
            elif "validate" in objective:
                result_data = await self._validate_data(parameters)
            elif "preprocess" in objective:
                result_data = await self._preprocess_data(parameters)
            else:
                # Default: comprehensive data collection
                result_data = await self._comprehensive_data_collection(parameters)
            
            return AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                status=AgentStatus.COMPLETED,
                result=result_data,
                confidence=self._calculate_data_confidence(result_data)
            )
            
        except Exception as e:
            self.logger.error("Data agent task failed", error=str(e), exc_info=True)
            return AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                status=AgentStatus.ERROR,
                error=str(e)
            )
    
    async def _fetch_market_data(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch market data for portfolio instruments."""
        portfolio = parameters.get("portfolio", {})
        lookback_period = parameters.get("lookback_period", 252)
        
        instruments = portfolio.get("instruments", [])
        if not instruments:
            raise ValueError("No instruments specified in portfolio")
        
        # Fetch data for each instrument
        market_data = {}
        price_data = {}
        
        for instrument in instruments:
            symbol = instrument.get("symbol")
            if not symbol:
                continue
                
            try:
                # Fetch historical price data
                data = await self.data_source_manager.get_historical_data(
                    symbol=symbol,
                    period=f"{lookback_period}d"
                )
                
                if data is not None and not data.empty:
                    price_data[symbol] = {
                        "prices": data.to_dict('records'),
                        "returns": data['Close'].pct_change().dropna().tolist(),
                        "volatility": data['Close'].pct_change().std() * np.sqrt(252),
                        "last_price": float(data['Close'].iloc[-1]),
                        "data_quality": self.data_validator.assess_data_quality(data)
                    }
                    
                    self.logger.info(f"Fetched data for {symbol}", records=len(data))
                else:
                    self.logger.warning(f"No data available for {symbol}")
                    
            except Exception as e:
                self.logger.error(f"Failed to fetch data for {symbol}", error=str(e))
                continue
        
        # Fetch market indices and economic indicators
        market_indices = await self._fetch_market_indices()
        economic_indicators = await self._fetch_economic_indicators()
        
        return {
            "portfolio_data": price_data,
            "market_indices": market_indices,
            "economic_indicators": economic_indicators,
            "collection_timestamp": datetime.utcnow().isoformat(),
            "data_coverage": self._calculate_data_coverage(price_data)
        }
    
    async def _fetch_market_indices(self) -> Dict[str, Any]:
        """Fetch major market indices data."""
        indices = ["^GSPC", "^DJI", "^IXIC", "^VIX"]  # S&P 500, Dow, NASDAQ, VIX
        indices_data = {}
        
        for index in indices:
            try:
                data = await self.data_source_manager.get_historical_data(index, "252d")
                if data is not None and not data.empty:
                    indices_data[index] = {
                        "current_level": float(data['Close'].iloc[-1]),
                        "daily_change": float(data['Close'].iloc[-1] - data['Close'].iloc[-2]),
                        "volatility": data['Close'].pct_change().std() * np.sqrt(252),
                        "trend": "up" if data['Close'].iloc[-1] > data['Close'].iloc[-20] else "down"
                    }
            except Exception as e:
                self.logger.error(f"Failed to fetch {index}", error=str(e))
        
        return indices_data
    
    async def _fetch_economic_indicators(self) -> Dict[str, Any]:
        """Fetch key economic indicators."""
        indicators = {
            "GDP": "GDP",
            "UNRATE": "Unemployment Rate",
            "CPIAUCSL": "CPI",
            "FEDFUNDS": "Federal Funds Rate"
        }
        
        econ_data = {}
        for code, name in indicators.items():
            try:
                data = await self.data_source_manager.get_fred_data(code)
                if data is not None:
                    econ_data[name] = {
                        "current_value": float(data.iloc[-1]),
                        "previous_value": float(data.iloc[-2]) if len(data) > 1 else None,
                        "change": float(data.iloc[-1] - data.iloc[-2]) if len(data) > 1 else None,
                        "last_updated": data.index[-1].isoformat()
                    }
            except Exception as e:
                self.logger.error(f"Failed to fetch {code}", error=str(e))
        
        return econ_data
    
    async def _analyze_news_sentiment(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market sentiment from news."""
        symbols = parameters.get("symbols", [])
        lookback_days = parameters.get("lookback_days", 7)
        
        sentiment_results = {}
        
        for symbol in symbols:
            try:
                # Fetch recent news
                news_data = await self.data_source_manager.get_news_data(
                    symbol=symbol,
                    days=lookback_days
                )
                
                if news_data:
                    # Analyze sentiment using Hugging Face
                    sentiments = []
                    for article in news_data:
                        sentiment = await self.hf_client.analyze_sentiment(
                            text=article.get("title", "") + " " + article.get("summary", ""),
                            model="ProsusAI/finbert"
                        )
                        sentiments.append(sentiment)
                    
                    # Aggregate sentiment scores
                    sentiment_results[symbol] = {
                        "average_sentiment": np.mean([s["score"] for s in sentiments]),
                        "sentiment_trend": self._calculate_sentiment_trend(sentiments),
                        "article_count": len(sentiments),
                        "positive_ratio": len([s for s in sentiments if s["label"] == "positive"]) / len(sentiments),
                        "articles": news_data[:5]  # Sample articles
                    }
                    
            except Exception as e:
                self.logger.error(f"Sentiment analysis failed for {symbol}", error=str(e))
        
        return {
            "sentiment_analysis": sentiment_results,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "lookback_period": lookback_days
        }
    
    async def _validate_data(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data quality and completeness."""
        data_source = parameters.get("data_source")
        validation_rules = parameters.get("validation_rules", {})
        
        validation_results = await self.data_validator.validate_dataset(
            data_source, validation_rules
        )
        
        return {
            "validation_results": validation_results,
            "data_quality_score": validation_results.get("overall_score", 0),
            "validation_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _preprocess_data(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess and clean financial data."""
        raw_data = parameters.get("raw_data")
        preprocessing_steps = parameters.get("steps", ["clean", "normalize", "feature_engineering"])
        
        processed_data = raw_data.copy()
        
        if "clean" in preprocessing_steps:
            processed_data = await self._clean_data(processed_data)
        
        if "normalize" in preprocessing_steps:
            processed_data = await self._normalize_data(processed_data)
        
        if "feature_engineering" in preprocessing_steps:
            processed_data = await self._engineer_features(processed_data)
        
        return {
            "processed_data": processed_data,
            "preprocessing_steps": preprocessing_steps,
            "processing_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _comprehensive_data_collection(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive data collection for risk analysis."""
        portfolio = parameters.get("portfolio", {})
        
        # Collect market data
        market_data_result = await self._fetch_market_data(parameters)
        
        # Analyze sentiment
        symbols = [inst.get("symbol") for inst in portfolio.get("instruments", [])]
        sentiment_result = await self._analyze_news_sentiment({"symbols": symbols})
        
        # Combine results
        return {
            **market_data_result,
            **sentiment_result,
            "data_collection_complete": True,
            "collection_summary": {
                "instruments_collected": len(market_data_result.get("portfolio_data", {})),
                "sentiment_analyzed": len(sentiment_result.get("sentiment_analysis", {})),
                "data_quality": "high" if self._assess_overall_quality(market_data_result) > 0.8 else "medium"
            }
        }
    
    def _calculate_data_confidence(self, result_data: Dict[str, Any]) -> float:
        """Calculate confidence score for collected data."""
        if not result_data:
            return 0.0
        
        confidence_factors = []
        
        # Data completeness
        portfolio_data = result_data.get("portfolio_data", {})
        if portfolio_data:
            completeness = len([v for v in portfolio_data.values() if v.get("data_quality", {}).get("completeness", 0) > 0.8])
            confidence_factors.append(completeness / len(portfolio_data) if portfolio_data else 0)
        
        # Market data availability
        market_indices = result_data.get("market_indices", {})
        if market_indices:
            confidence_factors.append(len(market_indices) / 4)  # Expected 4 indices
        
        # Economic indicators availability
        econ_indicators = result_data.get("economic_indicators", {})
        if econ_indicators:
            confidence_factors.append(len(econ_indicators) / 4)  # Expected 4 indicators
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _calculate_data_coverage(self, price_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate data coverage metrics."""
        if not price_data:
            return {"overall": 0.0}
        
        coverage_metrics = {}
        total_completeness = 0
        
        for symbol, data in price_data.items():
            quality = data.get("data_quality", {})
            completeness = quality.get("completeness", 0)
            coverage_metrics[symbol] = completeness
            total_completeness += completeness
        
        coverage_metrics["overall"] = total_completeness / len(price_data)
        return coverage_metrics
    
    def _calculate_sentiment_trend(self, sentiments: List[Dict[str, Any]]) -> str:
        """Calculate sentiment trend over time."""
        if len(sentiments) < 2:
            return "neutral"
        
        scores = [s["score"] for s in sentiments]
        recent_avg = np.mean(scores[-3:]) if len(scores) >= 3 else scores[-1]
        earlier_avg = np.mean(scores[:-3]) if len(scores) >= 6 else np.mean(scores[:-1])
        
        if recent_avg > earlier_avg + 0.1:
            return "improving"
        elif recent_avg < earlier_avg - 0.1:
            return "deteriorating"
        else:
            return "stable"
    
    def _assess_overall_quality(self, market_data_result: Dict[str, Any]) -> float:
        """Assess overall data quality."""
        portfolio_data = market_data_result.get("portfolio_data", {})
        if not portfolio_data:
            return 0.0
        
        quality_scores = []
        for data in portfolio_data.values():
            quality = data.get("data_quality", {})
            score = (
                quality.get("completeness", 0) * 0.4 +
                quality.get("accuracy", 0) * 0.3 +
                quality.get("timeliness", 0) * 0.3
            )
            quality_scores.append(score)
        
        return np.mean(quality_scores) if quality_scores else 0.0
    
    async def _clean_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and handle missing data."""
        # Implementation for data cleaning
        return data
    
    async def _normalize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize data for analysis."""
        # Implementation for data normalization
        return data
    
    async def _engineer_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Engineer features for risk analysis."""
        # Implementation for feature engineering
        return data

    