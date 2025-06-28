"""Semantic Kernel integration for advanced AI orchestration."""

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.core_skills import TextSkill, TimeSkill
from typing import Dict, Any, List
import asyncio

from config.settings import settings

class SemanticKernelSetup:
    """Setup Semantic Kernel for financial AI operations."""
    
    def __init__(self):
        self.kernel = sk.Kernel()
        self._setup_ai_services()
        self._register_core_skills()
        self._register_financial_skills()
    
    def _setup_ai_services(self):
        """Setup AI services for the kernel."""
        if settings.openai_api_key:
            self.kernel.add_chat_service(
                "openai_chat",
                OpenAIChatCompletion(
                    "gpt-3.5-turbo",
                    settings.openai_api_key
                )
            )
    
    def _register_core_skills(self):
        """Register core Semantic Kernel skills."""
        # Register built-in skills
        self.kernel.import_skill(TextSkill(), "text")
        self.kernel.import_skill(TimeSkill(), "time")
    
    def _register_financial_skills(self):
        """Register custom financial analysis skills."""
        # Risk Analysis Skill
        risk_skill = self.kernel.create_semantic_function(
            function_definition="""
            Analyze the financial risk metrics provided and generate insights:
            
            Input: {{$input}}
            
            Provide a professional analysis focusing on:
            1. Risk level assessment
            2. Key risk factors
            3. Recommendations for risk mitigation
            4. Market outlook implications
            
            Analysis:
            """,
            skill_name="RiskAnalysisSkill",
            description="Analyzes financial risk metrics and provides insights"
        )
        
        # Portfolio Optimization Skill
        portfolio_skill = self.kernel.create_semantic_function(
            function_definition="""
            Analyze portfolio composition and suggest optimizations:
            
            Portfolio Data: {{$input}}
            
            Provide recommendations for:
            1. Asset allocation optimization
            2. Diversification improvements
            3. Risk-adjusted return enhancement
            4. Rebalancing strategies
            
            Recommendations:
            """,
            skill_name="PortfolioOptimizationSkill",
            description="Optimizes portfolio allocation and composition"
        )
        
        # Market Sentiment Skill
        sentiment_skill = self.kernel.create_semantic_function(
            function_definition="""
            Analyze market sentiment from the provided data:
            
            Market Data: {{$input}}
            
            Assess:
            1. Overall market sentiment (bullish/bearish/neutral)
            2. Key sentiment drivers
            3. Sentiment impact on portfolio
            4. Timing implications
            
            Sentiment Analysis:
            """,
            skill_name="MarketSentimentSkill",
            description="Analyzes market sentiment and its implications"
        )
        
        self.kernel.import_skill(risk_skill, "risk")
        self.kernel.import_skill(portfolio_skill, "portfolio")
        self.kernel.import_skill(sentiment_skill, "sentiment")
    
    async def run_skill(self, skill_name: str, function_name: str, input_data: str) -> str:
        """Run a specific skill function."""
        try:
            function = self.kernel.skills.get_function(skill_name, function_name)
            result = await self.kernel.run_async(function, input_str=input_data)
            return str(result)
        except Exception as e:
            return f"Error running skill {skill_name}.{function_name}: {str(e)}"
    
    async def create_plan(self, goal: str) -> Any:
        """Create an execution plan for a complex goal."""
        try:
            planner = self.kernel.import_skill("Microsoft.SemanticKernel.Planning.BasicPlanner")
            plan = await planner.create_plan_async(goal)
            return plan
        except Exception as e:
            print(f"Error creating plan: {e}")
            return None
    
    def get_available_skills(self) -> Dict[str, List[str]]:
        """Get list of available skills and their functions."""
        skills_info = {}
        for skill_name, skill in self.kernel.skills._skills.items():
            functions = list(skill._functions.keys())
            skills_info[skill_name] = functions
        return skills_info
