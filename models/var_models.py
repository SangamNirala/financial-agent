"""Value at Risk calculation models."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
import asyncio

class VaRCalculator:
    """Comprehensive VaR calculation using multiple methodologies."""
    
    def __init__(self):
        pass
    
    async def parametric_var(
        self,
        returns_data: Dict[str, np.ndarray],
        weights: Dict[str, float],
        confidence_level: float = 0.95,
        holding_period: int = 1
    ) -> Dict[str, float]:
        """Calculate parametric VaR assuming normal distribution."""
        
        portfolio_returns = await self._calculate_portfolio_returns(returns_data, weights)
        
        # Portfolio statistics
        mean_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)
        
        # Adjust for holding period
        holding_period_mean = mean_return * holding_period
        holding_period_std = std_return * np.sqrt(holding_period)
        
        # Calculate VaR for different confidence levels
        confidence_levels = [0.90, 0.95, 0.99]
        var_results = {}
        
        for cl in confidence_levels:
            z_score = stats.norm.ppf(1 - cl)
            var = -(holding_period_mean + z_score * holding_period_std)
            var_results[f"var_{int(cl*100)}"] = var
        
        var_results.update({
            "mean_return": mean_return,
            "volatility": std_return,
            "holding_period": holding_period,
            "method": "parametric"
        })
        
        return var_results
    
    

    async def historical_var(
        self,
        returns_data: Dict[str, np.ndarray],
        weights: Dict[str, float],
        confidence_level: float = 0.95,
        holding_period: int = 1
    ) -> Dict[str, float]:
        """Calculate historical VaR using empirical distribution."""
        
        portfolio_returns = await self._calculate_portfolio_returns(returns_data, weights)
        
        # Adjust for holding period using overlapping returns if possible
        if holding_period > 1 and len(portfolio_returns) >= holding_period:
            # Calculate overlapping holding period returns
            hp_returns = []
            for i in range(len(portfolio_returns) - holding_period + 1):
                hp_return = np.prod(1 + portfolio_returns[i:i+holding_period]) - 1
                hp_returns.append(hp_return)
            portfolio_returns = np.array(hp_returns)
        elif holding_period > 1:
            # Simple scaling if insufficient data
            portfolio_returns = portfolio_returns * np.sqrt(holding_period)
        
        # Calculate VaR for different confidence levels
        confidence_levels = [0.90, 0.95, 0.99]
        var_results = {}
        
        for cl in confidence_levels:
            var = -np.percentile(portfolio_returns, (1 - cl) * 100)
            var_results[f"var_{int(cl*100)}"] = var
        
        var_results.update({
            "observations": len(portfolio_returns),
            "holding_period": holding_period,
            "method": "historical"
        })
        
        return var_results
    
    async def monte_carlo_var(
        self,
        returns_data: Dict[str, np.ndarray],
        weights: Dict[str, float],
        confidence_level: float = 0.95,
        holding_period: int = 1,
        num_simulations: int = 10000
    ) -> Dict[str, float]:
        """Calculate Monte Carlo VaR."""
        
        # Calculate statistics for simulation
        symbols = list(returns_data.keys())
        returns_matrix = np.column_stack([returns_data[symbol] for symbol in symbols])
        
        mean_returns = np.mean(returns_matrix, axis=0)
        cov_matrix = np.cov(returns_matrix.T)
        weight_vector = np.array([weights.get(symbol, 0) for symbol in symbols])
        
        # Generate random scenarios
        simulated_returns = np.random.multivariate_normal(
            mean_returns, cov_matrix, num_simulations
        )
        
        # Calculate portfolio returns for each simulation
        portfolio_returns = np.dot(simulated_returns, weight_vector)
        
        # Adjust for holding period
        if holding_period > 1:
            portfolio_returns = portfolio_returns * np.sqrt(holding_period)
        
        # Calculate VaR for different confidence levels
        confidence_levels = [0.90, 0.95, 0.99]
        var_results = {}
        
        for cl in confidence_levels:
            var = -np.percentile(portfolio_returns, (1 - cl) * 100)
            var_results[f"var_{int(cl*100)}"] = var
        
        var_results.update({
            "num_simulations": num_simulations,
            "holding_period": holding_period,
            "method": "monte_carlo"
        })
        
        return var_results
    
    async def expected_shortfall(
        self,
        returns_data: Dict[str, np.ndarray],
        weights: Dict[str, float],
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """Calculate Expected Shortfall (Conditional VaR)."""
        
        portfolio_returns = await self._calculate_portfolio_returns(returns_data, weights)
        
        # Calculate VaR threshold
        var_threshold = -np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        
        # Calculate Expected Shortfall
        tail_returns = portfolio_returns[portfolio_returns <= -var_threshold]
        
        confidence_levels = [0.90, 0.95, 0.99]
        es_results = {}
        
        for cl in confidence_levels:
            var_thresh = -np.percentile(portfolio_returns, (1 - cl) * 100)
            tail_rets = portfolio_returns[portfolio_returns <= -var_thresh]
            
            if len(tail_rets) > 0:
                es = -np.mean(tail_rets)
            else:
                es = var_thresh
            
            es_results[f"es_{int(cl*100)}"] = es
            es_results[f"var_{int(cl*100)}"] = var_thresh
        
        es_results.update({
            "tail_observations": len(tail_returns),
            "method": "historical_es"
        })
        
        return es_results
    
    async def _calculate_portfolio_returns(
        self, 
        returns_data: Dict[str, np.ndarray], 
        weights: Dict[str, float]
    ) -> np.ndarray:
        """Calculate portfolio returns from individual asset returns."""
        
        # Ensure all return series have the same length
        min_length = min(len(returns) for returns in returns_data.values())
        
        portfolio_returns = np.zeros(min_length)
        for symbol, returns in returns_data.items():
            weight = weights.get(symbol, 0)
            portfolio_returns += weight * returns[:min_length]
        
        return portfolio_returns