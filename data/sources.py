"""Data source management for financial data collection."""

import asyncio
import aiohttp
from typing import Dict, Any, List, Optional
import pandas as pd
import yfinance as yf
from fredapi import Fred
import numpy as np
from datetime import datetime, timedelta

from config.settings import settings, DATA_SOURCES

class DataSourceManager:
    """Manages financial data from multiple sources."""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.fred_client = Fred(api_key=settings.fred_api_key) if settings.fred_api_key else None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_historical_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Get historical price data from Yahoo Finance."""
        try:
            # Use yfinance for historical data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                return None
            
            # Ensure required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                print(f"Missing columns for {symbol}: {missing_columns}")
                return None
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    async def get_real_time_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time price data."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                "symbol": symbol,
                "current_price": info.get("currentPrice", info.get("regularMarketPrice")),
                "previous_close": info.get("previousClose"),
                "open": info.get("open"),
                "day_high": info.get("dayHigh"),
                "day_low": info.get("dayLow"),
                "volume": info.get("volume"),
                "market_cap": info.get("marketCap"),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            print(f"Error fetching real-time data for {symbol}: {e}")
            return None
    
    async def get_fred_data(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.Series]:
        """Get economic data from FRED."""
        if not self.fred_client:
            print("FRED API key not configured")
            return None
        
        try:
            # Set default date range if not provided
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            data = self.fred_client.get_series(
                series_id,
                start=start_date,
                end=end_date
            )
            
            return data
            
        except Exception as e:
            print(f"Error fetching FRED data for {series_id}: {e}")
            return None
    
    async def get_news_data(
        self,
        symbol: str,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """Get news data for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            # Filter recent news
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_news = []
            
            for article in news[:10]:  # Limit to recent articles
                try:
                    # Convert timestamp if available
                    pub_date = datetime.fromtimestamp(article.get('providerPublishTime', 0))
                    if pub_date >= cutoff_date:
                        recent_news.append({
                            "title": article.get("title", ""),
                            "summary": article.get("summary", ""),
                            "publisher": article.get("publisher", ""),
                            "publish_time": pub_date.isoformat(),
                            "url": article.get("link", "")
                        })
                except:
                    continue
            
            return recent_news
            
        except Exception as e:
            print(f"Error fetching news for {symbol}: {e}")
            return []
    
    async def get_market_indices(self) -> Dict[str, Any]:
        """Get major market indices data."""
        indices = {
            "S&P 500": "^GSPC",
            "Dow Jones": "^DJI", 
            "NASDAQ": "^IXIC",
            "VIX": "^VIX",
            "Russell 2000": "^RUT"
        }
        
        indices_data = {}
        
        for name, symbol in indices.items():
            data = await self.get_historical_data(symbol, period="5d")
            if data is not None and not data.empty:
                current_price = float(data['Close'].iloc[-1])
                previous_price = float(data['Close'].iloc[-2]) if len(data) > 1 else current_price
                
                indices_data[name] = {
                    "symbol": symbol,
                    "current_price": current_price,
                    "previous_close": previous_price,
                    "change": current_price - previous_price,
                    "change_percent": (current_price - previous_price) / previous_price if previous_price != 0 else 0,
                    "volatility": float(data['Close'].pct_change().std() * np.sqrt(252))
                }
        
        return indices_data
    
    async def get_crypto_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get cryptocurrency data."""
        crypto_data = {}
        
        for symbol in symbols:
            # Add -USD suffix for crypto symbols
            crypto_symbol = f"{symbol}-USD" if not symbol.endswith("-USD") else symbol
            
            data = await self.get_historical_data(crypto_symbol, period="30d")
            if data is not None and not data.empty:
                current_price = float(data['Close'].iloc[-1])
                
                crypto_data[symbol] = {
                    "symbol": crypto_symbol,
                    "current_price": current_price,
                    "volatility": float(data['Close'].pct_change().std() * np.sqrt(365)),
                    "returns_30d": float((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]),
                    "volume_24h": float(data['Volume'].iloc[-1])
                }
        
        return crypto_data
    
    async def validate_symbol(self, symbol: str) -> bool:
        """Validate if a symbol exists and has data."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Check if symbol has basic information
            return bool(info.get("symbol") or info.get("shortName"))
            
        except:
            return False
    
    async def get_batch_data(
        self,
        symbols: List[str],
        period: str = "1y"
    ) -> Dict[str, pd.DataFrame]:
        """Get data for multiple symbols efficiently."""
        data_dict = {}
        
        # Process in batches to avoid rate limits
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i + batch_size]
            
            tasks = []
            for symbol in batch_symbols:
                task = self.get_historical_data(symbol, period)
                tasks.append(task)
            
            # Wait for batch to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Store results
            for symbol, result in zip(batch_symbols, batch_results):
                if isinstance(result, pd.DataFrame) and not result.empty:
                    data_dict[symbol] = result
                elif isinstance(result, Exception):
                    print(f"Error fetching {symbol}: {result}")
            
            # Rate limiting
            await asyncio.sleep(0.1)
        
        return data_dict