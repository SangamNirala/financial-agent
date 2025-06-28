"""Data validation utilities for financial data quality assessment."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio

class DataValidator:
    """Validates and assesses quality of financial data."""
    
    def __init__(self):
        self.quality_thresholds = {
            "completeness": 0.8,  # 80% data availability
            "accuracy": 0.95,     # 95% accuracy threshold
            "timeliness": 24,     # 24 hours for timeliness
            "consistency": 0.9    # 90% consistency
        }
    
    async def validate_dataset(
        self,
        data: pd.DataFrame,
        validation_rules: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Validate complete dataset."""
        if data is None or data.empty:
            return {
                "overall_score": 0.0,
                "completeness": 0.0,
                "accuracy": 0.0,
                "timeliness": 0.0,
                "consistency": 0.0,
                "errors": ["Dataset is empty or None"]
            }
        
        validation_results = {}
        errors = []
        
        # Completeness check
        completeness_score = await self._check_completeness(data)
        validation_results["completeness"] = completeness_score
        
        # Accuracy check
        accuracy_score = await self._check_accuracy(data)
        validation_results["accuracy"] = accuracy_score
        
        # Timeliness check
        timeliness_score = await self._check_timeliness(data)
        validation_results["timeliness"] = timeliness_score
        
        # Consistency check
        consistency_score = await self._check_consistency(data)
        validation_results["consistency"] = consistency_score
        
        # Overall score (weighted average)
        weights = {"completeness": 0.3, "accuracy": 0.3, "timeliness": 0.2, "consistency": 0.2}
        overall_score = sum(validation_results[metric] * weight for metric, weight in weights.items())
        validation_results["overall_score"] = overall_score
        
        # Quality assessment
        quality_level = self._assess_quality_level(overall_score)
        validation_results["quality_level"] = quality_level
        
        validation_results["errors"] = errors
        validation_results["validation_timestamp"] = datetime.utcnow().isoformat()
        
        return validation_results
    
    async def _check_completeness(self, data: pd.DataFrame) -> float:
        """Check data completeness (missing values)."""
        if data.empty:
            return 0.0
        
        total_cells = data.size
        missing_cells = data.isnull().sum().sum()
        
        completeness = (total_cells - missing_cells) / total_cells
        return float(completeness)
    
    async def _check_accuracy(self, data: pd.DataFrame) -> float:
        """Check data accuracy using various heuristics."""
        accuracy_checks = []
        
        # Check for negative prices (if price columns exist)
        price_columns = [col for col in data.columns if any(term in col.lower() for term in ['price', 'close', 'open', 'high', 'low'])]
        if price_columns:
            for col in price_columns:
                if col in data.columns:
                    negative_prices = (data[col] < 0).sum()
                    accuracy_checks.append(1.0 - (negative_prices / len(data)))
        
        # Check for extreme outliers (beyond 5 standard deviations)
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if len(data[col].dropna()) > 10:  # Need sufficient data
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outliers = (z_scores > 5).sum()
                accuracy_checks.append(1.0 - (outliers / len(data)))
        
        # Check for impossible values (e.g., volume < 0)
        if 'Volume' in data.columns:
            negative_volume = (data['Volume'] < 0).sum()
            accuracy_checks.append(1.0 - (negative_volume / len(data)))
        
        # Check High >= Low for OHLC data
        if all(col in data.columns for col in ['High', 'Low']):
            invalid_hl = (data['High'] < data['Low']).sum()
            accuracy_checks.append(1.0 - (invalid_hl / len(data)))
        
        return float(np.mean(accuracy_checks)) if accuracy_checks else 1.0
    
    async def _check_timeliness(self, data: pd.DataFrame) -> float:
        """Check data timeliness (how recent the data is)."""
        if data.empty or not hasattr(data.index, 'max'):
            return 0.5  # Default score for non-time series data
        
        try:
            # Try to get the latest timestamp
            if isinstance(data.index, pd.DatetimeIndex):
                latest_date = data.index.max()
            elif 'Date' in data.columns:
                latest_date = pd.to_datetime(data['Date']).max()
            else:
                return 0.8  # Default good score for non-temporal data
            
            # Calculate hours since latest data
            now = datetime.now()
            if latest_date.tzinfo is None:
                latest_date = latest_date.replace(tzinfo=None)
            if now.tzinfo is not None:
                now = now.replace(tzinfo=None)
                
            hours_old = (now - latest_date).total_seconds() / 3600
            
            # Score based on freshness (exponential decay)
            timeliness_score = np.exp(-hours_old / 24)  # Decay over 24 hours
            return float(max(timeliness_score, 0.1))  # Minimum score of 0.1
            
        except Exception:
            return 0.5  # Default score if timeliness can't be assessed
    
    async def _check_consistency(self, data: pd.DataFrame) -> float:
        """Check data consistency across different metrics."""
        consistency_checks = []
        
        # Check for consistent data types
        for col in data.columns:
            try:
                # Check if numeric columns have consistent numeric types
                if data[col].dtype in [np.float64, np.int64]:
                    # Check for mixed types (strings in numeric columns)
                    non_numeric = pd.to_numeric(data[col], errors='coerce').isnull().sum()
                    original_null = data[col].isnull().sum()
                    consistency_checks.append(1.0 - ((non_numeric - original_null) / len(data)))
            except:
                continue
        
        # Check for reasonable price relationships in OHLC data
        if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            # High should be >= Open, Close
            high_consistent = ((data['High'] >= data['Open']) & (data['High'] >= data['Close'])).sum()
            consistency_checks.append(high_consistent / len(data))
            
            # Low should be <= Open, Close
            low_consistent = ((data['Low'] <= data['Open']) & (data['Low'] <= data['Close'])).sum()
            consistency_checks.append(low_consistent / len(data))
        
        # Check for reasonable volume patterns (no extreme jumps without cause)
        if 'Volume' in data.columns and len(data) > 10:
            volume_series = data['Volume'].replace(0, np.nan).dropna()
            if len(volume_series) > 5:
                volume_changes = volume_series.pct_change().dropna()
                extreme_changes = (np.abs(volume_changes) > 10).sum()  # >1000% change
                consistency_checks.append(1.0 - (extreme_changes / len(volume_changes)))
        
        return float(np.mean(consistency_checks)) if consistency_checks else 1.0
    
    def _assess_quality_level(self, overall_score: float) -> str:
        """Assess overall quality level based on score."""
        if overall_score >= 0.9:
            return "excellent"
        elif overall_score >= 0.8:
            return "good"
        elif overall_score >= 0.7:
            return "fair"
        elif overall_score >= 0.6:
            return "poor"
        else:
            return "very_poor"
    
    async def validate_portfolio_data(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate portfolio-specific data structure."""
        validation_results = {}
        
        for symbol, data in portfolio_data.items():
            if isinstance(data, dict):
                # Check required fields
                required_fields = ['last_price', 'returns', 'volatility']
                missing_fields = [field for field in required_fields if field not in data]
                
                symbol_quality = {
                    "missing_fields": missing_fields,
                    "has_required_data": len(missing_fields) == 0,
                    "data_points": len(data.get('returns', [])),
                    "price_validity": data.get('last_price', 0) > 0,
                    "volatility_reasonable": 0 < data.get('volatility', 0) < 2.0  # <200% annual vol
                }
                
                # Calculate symbol-specific quality score
                quality_factors = [
                    symbol_quality["has_required_data"],
                    symbol_quality["data_points"] > 50,  # Sufficient data points
                    symbol_quality["price_validity"],
                    symbol_quality["volatility_reasonable"]
                ]
                
                symbol_quality["quality_score"] = sum(quality_factors) / len(quality_factors)
                validation_results[symbol] = symbol_quality
        
        # Overall portfolio data quality
        if validation_results:
            overall_quality = np.mean([result["quality_score"] for result in validation_results.values()])
            validation_results["portfolio_overall_quality"] = overall_quality
        else:
            validation_results["portfolio_overall_quality"] = 0.0
        
        return validation_results
    
    def assess_data_quality(self, data: pd.DataFrame) -> Dict[str, float]:
        """Quick data quality assessment for individual datasets."""
        if data is None or data.empty:
            return {"completeness": 0.0, "accuracy": 0.0, "consistency": 0.0}
        
        # Completeness
        completeness = 1.0 - (data.isnull().sum().sum() / data.size)
        
        # Basic accuracy (no negative prices, reasonable ranges)
        accuracy = 1.0
        if 'Close' in data.columns:
            negative_prices = (data['Close'] < 0).sum()
            accuracy *= (1.0 - negative_prices / len(data))
        
        # Basic consistency
        consistency = 1.0
        if all(col in data.columns for col in ['High', 'Low']):
            invalid_hl = (data['High'] < data['Low']).sum()
            consistency *= (1.0 - invalid_hl / len(data))
        
        return {
            "completeness": float(completeness),
            "accuracy": float(accuracy),
            "consistency": float(consistency)
        }
