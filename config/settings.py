"""Configuration settings for the Financial Risk AI System."""

import os
from typing import Dict, Any, Optional
from pydantic import BaseSettings, Field
from enum import Enum

class Environment(str, Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class RiskModelType(str, Enum):
    """Risk model types."""
    MONTE_CARLO = "monte_carlo"
    VAR_PARAMETRIC = "var_parametric"
    VAR_HISTORICAL = "var_historical"
    VAR_SIMULATION = "var_simulation"
    STRESS_TEST = "stress_test"

class Settings(BaseSettings):
    """Application settings."""
    
    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=True)
    
    # API Keys
    huggingface_api_key: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    alpha_vantage_api_key: Optional[str] = Field(default=None, env="ALPHA_VANTAGE_API_KEY")
    fred_api_key: Optional[str] = Field(default=None, env="FRED_API_KEY")
    
    # Database & Storage
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    chroma_persist_directory: str = Field(default="./chroma_db", env="CHROMA_PERSIST_DIR")
    
    # Model Configuration
    default_llm_model: str = Field(default="microsoft/DialoGPT-large")
    financial_model: str = Field(default="ProsusAI/finbert")
    max_tokens: int = Field(default=1000)
    temperature: float = Field(default=0.1)
    
    # Risk Parameters
    confidence_level: float = Field(default=0.95)
    var_window: int = Field(default=252)  # Trading days
    monte_carlo_simulations: int = Field(default=10000)
    stress_test_scenarios: int = Field(default=100)
    
    # Agent Configuration
    max_agent_iterations: int = Field(default=10)
    agent_timeout: int = Field(default=300)  # seconds
    
    # Monitoring
    enable_tracing: bool = Field(default=True)
    metrics_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Model configurations
HUGGINGFACE_MODELS = {
    "sentiment": "ProsusAI/finbert",
    "classification": "nlptown/bert-base-multilingual-uncased-sentiment",
    "summarization": "facebook/bart-large-cnn",
    "qa": "deepset/roberta-base-squad2",
    "reasoning": "microsoft/DialoGPT-large"
}

# Financial data sources configuration
DATA_SOURCES = {
    "yahoo_finance": {
        "base_url": "https://query1.finance.yahoo.com",
        "rate_limit": 2000  # requests per hour
    },
    "alpha_vantage": {
        "base_url": "https://www.alphavantage.co/query",
        "rate_limit": 500  # requests per day for free tier
    },
    "fred": {
        "base_url": "https://api.stlouisfed.org/fred/series",
        "rate_limit": 120  # requests per minute
    }
}

# Risk model parameters
RISK_PARAMETERS = {
    "var": {
        "confidence_levels": [0.95, 0.99],
        "holding_periods": [1, 10, 22],  # days
        "window_sizes": [126, 252, 504]  # trading days
    },
    "monte_carlo": {
        "default_simulations": 10000,
        "max_simulations": 100000,
        "random_seed": 42
    },
    "stress_test": {
        "market_crash": {"equity_shock": -0.30, "credit_spread": 0.005},
        "interest_rate_shock": {"rate_increase": 0.02},
        "currency_crisis": {"fx_shock": 0.15}
    }
}
