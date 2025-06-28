"""Simplified memory manager using SQLite and in-memory storage."""

import sqlite3
import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from core.base_agent import AgentResult

class SimpleMemoryManager:
    """Simplified memory manager without Redis or ChromaDB."""
    
    def __init__(self, db_path: str = "data/memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        # In-memory caches
        self._agent_results_cache = {}
        self._workflow_cache = {}
        
        # Initialize SQLite database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_results (
                id TEXT PRIMARY KEY,
                agent_id TEXT,
                task_id TEXT,
                status TEXT,
                result TEXT,
                confidence REAL,
                execution_time REAL,
                explanation TEXT,
                error TEXT,
                timestamp TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS workflows (
                workflow_id TEXT PRIMARY KEY,
                workflow_data TEXT,
                timestamp TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def store_agent_result(self, result: AgentResult) -> None:
        """Store agent execution result."""
        try:
            # Store in memory cache
            cache_key = f"{result.agent_id}:{result.task_id}"
            self._agent_results_cache[cache_key] = result
            
            # Store in SQLite
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO agent_results 
                (id, agent_id, task_id, status, result, confidence, execution_time, explanation, error, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                cache_key,
                result.agent_id,
                result.task_id,
                result.status.value,
                json.dumps(result.result, default=str) if result.result else None,
                result.confidence,
                result.execution_time,
                result.explanation,
                result.error,
                result.timestamp.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Failed to store agent result: {e}")
    
    async def get_agent_result(self, agent_id: str, task_id: str) -> Optional[AgentResult]:
        """Retrieve agent result by ID."""
        cache_key = f"{agent_id}:{task_id}"
        
        # Check memory cache first
        if cache_key in self._agent_results_cache:
            return self._agent_results_cache[cache_key]
        
        # Check SQLite
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM agent_results WHERE id = ?
            ''', (cache_key,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return AgentResult(
                    agent_id=row[1],
                    task_id=row[2],
                    status=row[3],
                    result=json.loads(row[4]) if row[4] else None,
                    confidence=row[5],
                    execution_time=row[6],
                    explanation=row[7],
                    error=row[8],
                    timestamp=datetime.fromisoformat(row[9])
                )
        except Exception as e:
            print(f"Failed to retrieve agent result: {e}")
        
        return None
    
    async def search_similar_results(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Simple text search in explanations."""
        results = []
        for result in self._agent_results_cache.values():
            if result.explanation and query.lower() in result.explanation.lower():
                results.append({
                    "explanation": result.explanation,
                    "metadata": {
                        "agent_id": result.agent_id,
                        "task_id": result.task_id,
                        "timestamp": result.timestamp.isoformat()
                    }
                })
                if len(results) >= limit:
                    break
        return results
    
    async def store_workflow_result(self, workflow_id: str, workflow_data: Dict[str, Any]) -> None:
        """Store workflow execution result."""
        try:
            # Store in memory
            self._workflow_cache[workflow_id] = workflow_data
            
            # Store in SQLite
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO workflows (workflow_id, workflow_data, timestamp)
                VALUES (?, ?, ?)
            ''', (
                workflow_id,
                json.dumps(workflow_data, default=str),
                datetime.utcnow().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Failed to store workflow result: {e}")
    
    async def get_recent_agent_results(self, agent_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent results for a specific agent."""
        results = []
        for key, result in self._agent_results_cache.items():
            if result.agent_id == agent_id:
                results.append({
                    "agent_id": result.agent_id,
                    "task_id": result.task_id,
                    "status": result.status.value,
                    "timestamp": result.timestamp.isoformat(),
                    "confidence": result.confidence
                })
        
        # Sort by timestamp and limit
        results.sort(key=lambda x: x["timestamp"], reverse=True)
        return results[:limit]
