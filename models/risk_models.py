"""Risk models implementation for financial analysis."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod
from scipy import stats
from datetime import datetime
import asyncio

class RiskModel(ABC):
    """Abstract base class for risk models."""
    
    @abstractmethod
    async def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk metrics."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return model name."""
        pass

class VaRModel(RiskModel):
    """Value at Risk model implementation."""
    
    def __init__(self, confidence_level: float = 0.95, holding_period: int = 1):
        self.confidence_level = confidence_level
        self.holding_period = holding_period
    
    def get_model_name(self) -> str:
        return f"VaR_{int(self.confidence_level*100)}_{self.holding_period}d"
    
    async def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate VaR using multiple methods."""
        returns_data = data.get("returns_data", {})
        weights = data.get("weights", {})
        
        if not returns_data or not weights:
            raise ValueError("Returns data and weights required for VaR calculation")
        
        results = {}
        
        # Parametric VaR
        results["parametric"] = await self._parametric_var(returns_data, weights)
        
        # Historical VaR
        results["historical"] = await self._historical_var(returns_data, weights)
        
        # Modified VaR (Cornish-Fisher)
        results["modified"] = await self._modified_var(returns_data, weights)
        
        return results
    
    async def _parametric_var(self, returns_data: Dict[str, np.ndarray], weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate parametric VaR assuming normal distribution."""
        # Calculate portfolio returns
        portfolio_returns = await self._calculate_portfolio_returns(returns_data, weights)
        
        # Portfolio statistics
        mean_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)
        
        # Adjust for holding period
        holding_period_mean = mean_return * self.holding_period
        holding_period_std = std_return * np.sqrt(self.holding_period)
        
        # Calculate VaR
        z_score = stats.norm.ppf(1 - self.confidence_level)
        var = -(holding_period_mean + z_score * holding_period_std)
        
        return {
            f"var_{int(self.confidence_level*100)}": var,
            "mean_return": mean_return,
            "volatility": std_return,
            "method": "parametric_normal"
        }
    
    async def _historical_var(self, returns_data: Dict[str, np.ndarray], weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate historical VaR using empirical distribution."""
        portfolio_returns = await self._calculate_portfolio_returns(returns_data, weights)
        
        # Adjust for holding period if needed
        if self.holding_period > 1:
            # Simple scaling approach (could be improved with overlapping returns)
            portfolio_returns = portfolio_returns * np.sqrt(self.holding_period)
        
        # Calculate VaR as percentile
        var = -np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)
        
        # Additional statistics
        mean_return = np.mean(portfolio_returns)
        
        return {
            f"var_{int(self.confidence_level*100)}": var,
            "mean_return": mean_return,
            "method": "historical_simulation"
        }
    
    async def _modified_var(self, returns_data: Dict[str, np.ndarray], weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate modified VaR using Cornish-Fisher expansion."""
        portfolio_returns = await self._calculate_portfolio_returns(returns_data, weights)
        
        # Calculate moments
        mean_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)
        skewness = stats.skew(portfolio_returns)
        kurtosis = stats.kurtosis(portfolio_returns)
        
        # Cornish-Fisher adjustment
        z = stats.norm.ppf(1 - self.confidence_level)
        z_cf = (z + 
                (z**2 - 1) * skewness / 6 + 
                (z**3 - 3*z) * kurtosis / 24 - 
                (2*z**3 - 5*z) * skewness**2 / 36)
        
        # Adjust for holding period
        holding_period_mean = mean_return * self.holding_period
        holding_period_std = std_return * np.sqrt(self.holding_period)
        
        var = -(holding_period_mean + z_cf * holding_period_std)
        
        return {
            f"var_{int(self.confidence_level*100)}": var,
            "mean_return": mean_return,
            "volatility": std_return,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "method": "cornish_fisher"
        }
    
    async def _calculate_portfolio_returns(self, returns_data: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
        """Calculate portfolio returns from individual asset returns."""
        # Ensure all return series have the same length
        min_length = min(len(returns) for returns in returns_data.values())
        
        portfolio_returns = np.zeros(min_length)
        for symbol, returns in returns_data.items():
            weight = weights.get(symbol, 0)
            portfolio_returns += weight * returns[:min_length]
        
        return portfolio_returns

class ExpectedShortfallModel(RiskModel):
    """Expected Shortfall (Conditional VaR) model."""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
    
    def get_model_name(self) -> str:
        return f"ES_{int(self.confidence_level*100)}"
    
    async def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Expected Shortfall."""
        returns_data = data.get("returns_data", {})
        weights = data.get("weights", {})
        
        # Calculate portfolio returns
        var_model = VaRModel(self.confidence_level)
        portfolio_returns = await var_model._calculate_portfolio_returns(returns_data, weights)
        
        # Calculate VaR threshold
        var_threshold = -np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)
        
        # Calculate Expected Shortfall (mean of returns beyond VaR)
        tail_returns = portfolio_returns[portfolio_returns <= -var_threshold]
        expected_shortfall = -np.mean(tail_returns) if len(tail_returns) > 0 else var_threshold
        
        return {
            f"es_{int(self.confidence_level*100)}": expected_shortfall,
            f"var_{int(self.confidence_level*100)}": var_threshold,
            "tail_observations": len(tail_returns),
            "method": "historical_es"
        }

class RiskModelFactory:
    """Factory for creating risk models."""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> RiskModel:
        """Create risk model instance."""
        models = {
            "var": VaRModel,
            "expected_shortfall": ExpectedShortfallModel,
            "cvar": ExpectedShortfallModel  # Alias
        }
        
        model_class = models.get(model_type.lower())
        if not model_class:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model_class(**kwargs)
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available model types."""
        return ["var", "expected_shortfall", "cvar"]
