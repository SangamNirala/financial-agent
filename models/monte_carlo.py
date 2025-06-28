"""Monte Carlo simulation for portfolio risk analysis."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from scipy.linalg import cholesky
from datetime import datetime
import asyncio

class MonteCarloSimulator:
    """Monte Carlo simulation for portfolio risk analysis."""
    
    def __init__(self, random_seed: Optional[int] = None):
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    async def simulate_portfolio_paths(
        self,
        returns_data: Dict[str, np.ndarray],
        num_simulations: int = 10000,
        time_horizon: int = 252,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Simulate portfolio price paths using Monte Carlo."""
        
        # Prepare data
        symbols = list(returns_data.keys())
        returns_matrix = np.column_stack([returns_data[symbol] for symbol in symbols])
        
        # Equal weights if not provided
        if weights is None:
            weights = {symbol: 1.0 / len(symbols) for symbol in symbols}
        
        weight_vector = np.array([weights.get(symbol, 0) for symbol in symbols])
        
        # Calculate statistics
        mean_returns = np.mean(returns_matrix, axis=0)
        cov_matrix = np.cov(returns_matrix.T)
        
        # Generate correlated random returns
        simulated_returns = await self._generate_correlated_returns(
            mean_returns, cov_matrix, num_simulations, time_horizon
        )
        
        # Calculate portfolio returns for each simulation
        portfolio_returns = np.dot(simulated_returns, weight_vector)
        
        # Calculate cumulative returns and prices
        cumulative_returns = np.cumprod(1 + portfolio_returns, axis=1)
        final_returns = cumulative_returns[:, -1] - 1
        
        # Calculate maximum drawdowns
        max_drawdowns = await self._calculate_max_drawdowns(cumulative_returns)
        
        return {
            "portfolio_returns": final_returns,
            "portfolio_paths": cumulative_returns,
            "individual_asset_returns": simulated_returns,
            "max_drawdowns": max_drawdowns,
            "simulation_parameters": {
                "num_simulations": num_simulations,
                "time_horizon": time_horizon,
                "random_seed": self.random_seed
            }
        }
    
    async def _generate_correlated_returns(
        self,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        num_simulations: int,
        time_horizon: int
    ) -> np.ndarray:
        """Generate correlated random returns using Cholesky decomposition."""
        
        num_assets = len(mean_returns)
        
        # Ensure covariance matrix is positive definite
        try:
            chol_matrix = cholesky(cov_matrix, lower=True)
        except np.linalg.LinAlgError:
            # If Cholesky fails, use eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            eigenvalues = np.maximum(eigenvalues, 1e-8)  # Ensure positive
            cov_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            chol_matrix = cholesky(cov_matrix, lower=True)
        
        # Generate independent normal random variables
        independent_randoms = np.random.normal(
            size=(num_simulations, time_horizon, num_assets)
        )
        
        # Apply correlation structure
        correlated_randoms = np.zeros_like(independent_randoms)
        for i in range(num_simulations):
            for t in range(time_horizon):
                correlated_randoms[i, t, :] = chol_matrix @ independent_randoms[i, t, :]
        
        # Convert to returns (assuming daily frequency)
        dt = 1.0 / 252  # Daily time step
        simulated_returns = np.zeros_like(correlated_randoms)
        
        for i in range(num_simulations):
            for t in range(time_horizon):
                # Geometric Brownian Motion
                drift = (mean_returns - 0.5 * np.diag(cov_matrix)) * dt
                diffusion = correlated_randoms[i, t, :] * np.sqrt(dt)
                simulated_returns[i, t, :] = drift + diffusion
        
        return simulated_returns
    
    async def _calculate_max_drawdowns(self, cumulative_returns: np.ndarray) -> np.ndarray:
        """Calculate maximum drawdown for each simulation path."""
        max_drawdowns = np.zeros(cumulative_returns.shape[0])
        
        for i in range(cumulative_returns.shape[0]):
            path = cumulative_returns[i, :]
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(path)
            
            # Calculate drawdowns
            drawdowns = (path - running_max) / running_max
            
            # Store maximum drawdown (most negative)
            max_drawdowns[i] = np.min(drawdowns)
        
        return max_drawdowns
    
    async def simulate_var_confidence_intervals(
        self,
        returns_data: Dict[str, np.ndarray],
        confidence_level: float = 0.95,
        num_bootstrap: int = 1000,
        sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Simulate confidence intervals for VaR estimates using bootstrap."""
        
        # Prepare portfolio returns
        symbols = list(returns_data.keys())
        returns_matrix = np.column_stack([returns_data[symbol] for symbol in symbols])
        
        # Equal weights assumption
        weights = np.ones(len(symbols)) / len(symbols)
        portfolio_returns = np.dot(returns_matrix, weights)
        
        if sample_size is None:
            sample_size = len(portfolio_returns)
        
        # Bootstrap VaR estimates
        var_estimates = []
        
        for _ in range(num_bootstrap):
            # Bootstrap sample
            bootstrap_indices = np.random.choice(
                len(portfolio_returns), size=sample_size, replace=True
            )
            bootstrap_returns = portfolio_returns[bootstrap_indices]
            
            # Calculate VaR for bootstrap sample
            var_estimate = -np.percentile(bootstrap_returns, (1 - confidence_level) * 100)
            var_estimates.append(var_estimate)
        
        var_estimates = np.array(var_estimates)
        
        return {
            "var_estimates": var_estimates,
            "mean_var": np.mean(var_estimates),
            "var_std": np.std(var_estimates),
            "confidence_intervals": {
                "95%": [np.percentile(var_estimates, 2.5), np.percentile(var_estimates, 97.5)],
                "90%": [np.percentile(var_estimates, 5), np.percentile(var_estimates, 95)],
                "68%": [np.percentile(var_estimates, 16), np.percentile(var_estimates, 84)]
            },
            "bootstrap_parameters": {
                "num_bootstrap": num_bootstrap,
                "sample_size": sample_size,
                "confidence_level": confidence_level
            }
        }
    
    async def simulate_stress_scenarios(
        self,
        returns_data: Dict[str, np.ndarray],
        stress_parameters: Dict[str, Any],
        num_simulations: int = 1000
    ) -> Dict[str, Any]:
        """Simulate portfolio performance under stress scenarios."""
        
        symbols = list(returns_data.keys())
        returns_matrix = np.column_stack([returns_data[symbol] for symbol in symbols])
        
        # Base statistics
        mean_returns = np.mean(returns_matrix, axis=0)
        cov_matrix = np.cov(returns_matrix.T)
        
        stress_results = {}
        
        for scenario_name, params in stress_parameters.items():
            # Modify parameters for stress scenario
            stressed_means = mean_returns.copy()
            stressed_cov = cov_matrix.copy()
            
            # Apply market shock
            if "market_shock" in params:
                shock = params["market_shock"]
                stressed_means += shock
            
            # Apply volatility scaling
            if "volatility_multiplier" in params:
                multiplier = params["volatility_multiplier"]
                stressed_cov *= multiplier
            
            # Apply correlation shock
            if "correlation_shock" in params:
                corr_shock = params["correlation_shock"]
                # Increase correlations during stress
                stressed_corr = np.corrcoef(returns_matrix.T)
                stressed_corr = np.minimum(stressed_corr + corr_shock, 0.99)
                
                # Convert back to covariance matrix
                vol_vector = np.sqrt(np.diag(stressed_cov))
                stressed_cov = np.outer(vol_vector, vol_vector) * stressed_corr
            
            # Run simulation
            simulated_returns = await self._generate_correlated_returns(
                stressed_means, stressed_cov, num_simulations, 1  # Single period
            )
            
            # Calculate portfolio returns (equal weights)
            weights = np.ones(len(symbols)) / len(symbols)
            portfolio_stress_returns = np.dot(simulated_returns[:, 0, :], weights)
            
            stress_results[scenario_name] = {
                "simulated_returns": portfolio_stress_returns,
                "mean_return": np.mean(portfolio_stress_returns),
                "var_95": -np.percentile(portfolio_stress_returns, 5),
                "var_99": -np.percentile(portfolio_stress_returns, 1),
                "expected_shortfall_95": -np.mean(portfolio_stress_returns[portfolio_stress_returns <= np.percentile(portfolio_stress_returns, 5)]),
                "worst_case": np.min(portfolio_stress_returns),
                "probability_large_loss": np.mean(portfolio_stress_returns < -0.1)  # >10% loss
            }
        
        return stress_results
