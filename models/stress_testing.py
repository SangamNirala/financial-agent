"""Stress testing models for portfolio risk analysis."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from config.settings import RISK_PARAMETERS

class StressTester:
    """Comprehensive stress testing framework."""
    
    def __init__(self):
        self.stress_scenarios = RISK_PARAMETERS["stress_test"]
    
    async def run_stress_scenario(
        self,
        portfolio_data: Dict[str, Any],
        scenario_name: str
    ) -> Dict[str, Any]:
        """Run a specific stress scenario."""
        
        scenario_params = self.stress_scenarios.get(scenario_name)
        if not scenario_params:
            raise ValueError(f"Unknown stress scenario: {scenario_name}")
        
        # Apply stress scenario based on type
        if scenario_name == "market_crash":
            return await self._market_crash_scenario(portfolio_data, scenario_params)
        elif scenario_name == "interest_rate_shock":
            return await self._interest_rate_scenario(portfolio_data, scenario_params)
        elif scenario_name == "currency_crisis":
            return await self._currency_crisis_scenario(portfolio_data, scenario_params)
        else:
            return await self._generic_stress_scenario(portfolio_data, scenario_params)
    
    async def _market_crash_scenario(
        self, 
        portfolio_data: Dict[str, Any], 
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate market crash scenario."""
        
        equity_shock = params.get("equity_shock", -0.30)
        credit_spread = params.get("credit_spread", 0.005)
        
        asset_impacts = {}
        total_portfolio_value = 0
        total_loss = 0
        
        for symbol, data in portfolio_data.items():
            current_price = data.get("last_price", 100)
            volatility = data.get("volatility", 0.2)
            
            # Apply equity shock (more severe for high-vol assets)
            vol_multiplier = 1 + volatility  # Higher vol = more sensitive
            asset_shock = equity_shock * vol_multiplier
            
            # Calculate impact
            shocked_price = current_price * (1 + asset_shock)
            asset_loss = (shocked_price - current_price) / current_price
            
            asset_impacts[symbol] = {
                "current_price": current_price,
                "shocked_price": shocked_price,
                "price_change": asset_shock,
                "loss_amount": asset_loss * current_price,
                "loss_percentage": asset_loss
            }
            
            # Assume equal portfolio weights for simplicity
            weight = 1.0 / len(portfolio_data)
            total_portfolio_value += weight * current_price
            total_loss += weight * asset_loss * current_price
        
        portfolio_loss_pct = total_loss / total_portfolio_value if total_portfolio_value > 0 else 0
        
        return {
            "scenario_name": "market_crash",
            "scenario_parameters": params,
            "asset_impacts": asset_impacts,
            "portfolio_impact": {
                "total_loss": portfolio_loss_pct,
                "total_loss_amount": total_loss,
                "portfolio_value": total_portfolio_value
            },
            "stress_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _interest_rate_scenario(
        self, 
        portfolio_data: Dict[str, Any], 
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate interest rate shock scenario."""
        
        rate_increase = params.get("rate_increase", 0.02)  # 200 bps increase
        
        asset_impacts = {}
        total_portfolio_value = 0
        total_loss = 0
        
        for symbol, data in portfolio_data.items():
            current_price = data.get("last_price", 100)
            
            # Interest rate sensitivity (simplified duration model)
            # Assume equity duration of ~3, bonds higher
            duration = 3.0  # Simplified assumption
            
            # Price impact from interest rate change
            price_impact = -duration * rate_increase
            shocked_price = current_price * (1 + price_impact)
            asset_loss = (shocked_price - current_price) / current_price
            
            asset_impacts[symbol] = {
                "current_price": current_price,
                "shocked_price": shocked_price,
                "price_change": price_impact,
                "loss_amount": asset_loss * current_price,
                "loss_percentage": asset_loss,
                "duration": duration
            }
            
            weight = 1.0 / len(portfolio_data)
            total_portfolio_value += weight * current_price
            total_loss += weight * asset_loss * current_price
        
        portfolio_loss_pct = total_loss / total_portfolio_value if total_portfolio_value > 0 else 0
        
        return {
            "scenario_name": "interest_rate_shock",
            "scenario_parameters": params,
            "asset_impacts": asset_impacts,
            "portfolio_impact": {
                "total_loss": portfolio_loss_pct,
                "total_loss_amount": total_loss,
                "portfolio_value": total_portfolio_value
            },
            "stress_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _currency_crisis_scenario(
        self, 
        portfolio_data: Dict[str, Any], 
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate currency crisis scenario."""
        
        fx_shock = params.get("fx_shock", 0.15)  # 15% currency devaluation
        
        asset_impacts = {}
        total_portfolio_value = 0
        total_loss = 0
        
        for symbol, data in portfolio_data.items():
            current_price = data.get("last_price", 100)
            
            # Assume some foreign exchange exposure
            fx_exposure = 0.3  # 30% FX exposure assumption
            
            # Price impact from currency shock
            price_impact = -fx_exposure * fx_shock
            shocked_price = current_price * (1 + price_impact)
            asset_loss = (shocked_price - current_price) / current_price
            
            asset_impacts[symbol] = {
                "current_price": current_price,
                "shocked_price": shocked_price,
                "price_change": price_impact,
                "loss_amount": asset_loss * current_price,
                "loss_percentage": asset_loss,
                "fx_exposure": fx_exposure
            }
            
            weight = 1.0 / len(portfolio_data)
            total_portfolio_value += weight * current_price
            total_loss += weight * asset_loss * current_price
        
        portfolio_loss_pct = total_loss / total_portfolio_value if total_portfolio_value > 0 else 0
        
        return {
            "scenario_name": "currency_crisis",
            "scenario_parameters": params,
            "asset_impacts": asset_impacts,
            "portfolio_impact": {
                "total_loss": portfolio_loss_pct,
                "total_loss_amount": total_loss,
                "portfolio_value": total_portfolio_value
            },
            "stress_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _generic_stress_scenario(
        self, 
        portfolio_data: Dict[str, Any], 
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generic stress scenario framework."""
        
        # Extract common stress parameters
        market_shock = params.get("market_shock", 0)
        volatility_multiplier = params.get("volatility_multiplier", 1.0)
        correlation_increase = params.get("correlation_increase", 0)
        
        asset_impacts = {}
        total_portfolio_value = 0
        total_loss = 0
        
        for symbol, data in portfolio_data.items():
            current_price = data.get("last_price", 100)
            volatility = data.get("volatility", 0.2)
            
            # Apply market shock with volatility scaling
            total_shock = market_shock * (1 + volatility * volatility_multiplier)
            shocked_price = current_price * (1 + total_shock)
            asset_loss = (shocked_price - current_price) / current_price
            
            asset_impacts[symbol] = {
                "current_price": current_price,
                "shocked_price": shocked_price,
                "price_change": total_shock,
                "loss_amount": asset_loss * current_price,
                "loss_percentage": asset_loss
            }
            
            weight = 1.0 / len(portfolio_data)
            total_portfolio_value += weight * current_price
            total_loss += weight * asset_loss * current_price
        
        portfolio_loss_pct = total_loss / total_portfolio_value if total_portfolio_value > 0 else 0
        
        return {
            "scenario_name": "generic_stress",
            "scenario_parameters": params,
            "asset_impacts": asset_impacts,
            "portfolio_impact": {
                "total_loss": portfolio_loss_pct,
                "total_loss_amount": total_loss,
                "portfolio_value": total_portfolio_value
            },
            "stress_timestamp": datetime.utcnow().isoformat()
        }
    
    async def run_reverse_stress_test(
        self,
        portfolio_data: Dict[str, Any],
        target_loss: float = 0.10
    ) -> Dict[str, Any]:
        """Run reverse stress test to find scenarios that cause target loss."""
        
        scenarios_found = []
        
        # Test various shock magnitudes
        shock_ranges = {
            "market_shock": np.arange(-0.50, -0.05, 0.05),
            "volatility_multiplier": np.arange(1.5, 4.0, 0.5),
            "interest_rate_shock": np.arange(0.01, 0.05, 0.005)
        }
        
        for shock_type, shock_values in shock_ranges.items():
            for shock_value in shock_values:
                # Create scenario parameters
                if shock_type == "market_shock":
                    params = {"equity_shock": shock_value}
                    scenario_result = await self._market_crash_scenario(portfolio_data, params)
                elif shock_type == "interest_rate_shock":
                    params = {"rate_increase": shock_value}
                    scenario_result = await self._interest_rate_scenario(portfolio_data, params)
                else:
                    continue
                
                # Check if scenario produces target loss
                actual_loss = abs(scenario_result["portfolio_impact"]["total_loss"])
                if abs(actual_loss - target_loss) < 0.01:  # Within 1% tolerance
                    scenarios_found.append({
                        "shock_type": shock_type,
                        "shock_value": shock_value,
                        "actual_loss": actual_loss,
                        "scenario_result": scenario_result
                    })
        
        return {
            "target_loss": target_loss,
            "scenarios_found": scenarios_found,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }