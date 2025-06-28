"""Simplified settings without Redis dependencies."""

import os
from typing import Dict, Any, Optional
from enum import Enum

class Environment(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"

class SimpleSettings:
    """Simplified application settings."""
    
    def __init__(self):
        # Environment
        self.environment = Environment.DEVELOPMENT
        self.debug = True
        
        # API Keys (optional - system works without them)
        self.huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY") 
        self.alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.fred_api_key = os.getenv("FRED_API_KEY")
        
        # Model Configuration
        self.default_llm_model = "microsoft/DialoGPT-small"  # Smaller model
        self.financial_model = "ProsusAI/finbert"
        self.max_tokens = 150  # Reduced for faster processing
        self.temperature = 0.1
        
        # Risk Parameters
        self.confidence_level = 0.95
        self.var_window = 252
        self.monte_carlo_simulations = 1000  # Reduced for speed
        self.stress_test_scenarios = 10  # Reduced for speed
        
        # Agent Configuration
        self.max_agent_iterations = 5  # Reduced
        self.agent_timeout = 120  # Reduced
        
        # Storage
        self.data_directory = "data"
        self.memory_db_path = "data/memory.db"
        
        # Server
        self.host = "127.0.0.1"
        self.port = 8000
        self.log_level = "INFO"

# Global settings instance
settings = SimpleSettings()

# Simplified model configurations
HUGGINGFACE_MODELS = {
    "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",  # Smaller model
    "summarization": "sshleifer/distilbart-cnn-12-6",  # Smaller model
    "reasoning": "microsoft/DialoGPT-small"  # Smaller model
}

# Risk model parameters
RISK_PARAMETERS = {
    "var": {
        "confidence_levels": [0.95, 0.99],
        "holding_periods": [1, 10],
        "window_sizes": [126, 252]
    },
    "monte_carlo": {
        "default_simulations": 1000,  # Reduced
        "max_simulations": 5000,  # Reduced
        "random_seed": 42
    },
    "stress_test": {
        "market_crash": {"equity_shock": -0.30, "credit_spread": 0.005},
        "interest_rate_shock": {"rate_increase": 0.02},
        "currency_crisis": {"fx_shock": 0.15}
    }
}
