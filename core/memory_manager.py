"""Memory management for the multi-agent system."""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import redis
import chromadb
from chromadb.config import Settings

from core.base_agent import AgentResult
from config.settings import settings

class MemoryManager:
    """Manages memory storage and retrieval for the multi-agent system."""
    
    def __init__(self):
        # Redis for fast access memory
        self.redis_client = redis.from_url(settings.redis_url)
        
        # ChromaDB for vector storage
        self.chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory,
            settings=Settings(allow_reset=True)
        )
        
        # Collections
        self.agent_results_collection = self._get_or_create_collection("agent_results")
        self.workflows_collection = self._get_or_create_collection("workflows")
        
    def _get_or_create_collection(self, name: str):
        """Get or create a ChromaDB collection."""
        try:
            return self.chroma_client.get_collection(name)
        except:
            return self.chroma_client.create_collection(name)
    
    async def store_agent_result(self, result: AgentResult) -> None:
        """Store agent execution result."""
        try:
            # Store in Redis for fast access
            redis_key = f"agent_result:{result.agent_id}:{result.task_id}"
            result_data = {
                "agent_id": result.agent_id,
                "task_id": result.task_id,
                "status": result.status.value,
                "result": result.result,
                "confidence": result.confidence,
                "execution_time": result.execution_time,
                "timestamp": result.timestamp.isoformat(),
                "explanation": result.explanation,
                "error": result.error
            }
            
            await asyncio.to_thread(
                self.redis_client.setex,
                redis_key,
                timedelta(hours=24).total_seconds(),
                json.dumps(result_data, default=str)
            )
            
            # Store in ChromaDB for semantic search
            if result.explanation:
                self.agent_results_collection.add(
                    documents=[result.explanation],
                    metadatas=[result_data],
                    ids=[f"{result.agent_id}_{result.task_id}"]
                )
                
        except Exception as e:
            print(f"Failed to store agent result: {e}")
    
    async def get_agent_result(self, agent_id: str, task_id: str) -> Optional[AgentResult]:
        """Retrieve agent result by ID."""
        try:
            redis_key = f"agent_result:{agent_id}:{task_id}"
            data = await asyncio.to_thread(self.redis_client.get, redis_key)
            
            if data:
                result_data = json.loads(data)
                return AgentResult(
                    agent_id=result_data["agent_id"],
                    task_id=result_data["task_id"],
                    status=result_data["status"],
                    result=result_data.get("result"),
                    confidence=result_data.get("confidence"),
                    execution_time=result_data.get("execution_time"),
                    explanation=result_data.get("explanation"),
                    error=result_data.get("error"),
                    timestamp=datetime.fromisoformat(result_data["timestamp"])
                )
        except Exception as e:
            print(f"Failed to retrieve agent result: {e}")
        
        return None
    
    async def search_similar_results(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar agent results using semantic search."""
        try:
            results = self.agent_results_collection.query(
                query_texts=[query],
                n_results=limit
            )
            
            return [{
                "document": doc,
                "metadata": meta,
                "distance": dist
            } for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0], 
                results["distances"][0]
            )]
            
        except Exception as e:
            print(f"Failed to search similar results: {e}")
            return []
    
    async def store_workflow_result(self, workflow_id: str, workflow_data: Dict[str, Any]) -> None:
        """Store workflow execution result."""
        try:
            redis_key = f"workflow:{workflow_id}"
            await asyncio.to_thread(
                self.redis_client.setex,
                redis_key,
                timedelta(days=7).total_seconds(),
                json.dumps(workflow_data, default=str)
            )
            
        except Exception as e:
            print(f"Failed to store workflow result: {e}")
    
    async def get_recent_agent_results(self, agent_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent results for a specific agent."""
        try:
            pattern = f"agent_result:{agent_id}:*"
            keys = await asyncio.to_thread(self.redis_client.keys, pattern)
            
            results = []
            for key in keys[-limit:]:  # Get most recent
                data = await asyncio.to_thread(self.redis_client.get, key)
                if data:
                    results.append(json.loads(data))
            
            return sorted(results, key=lambda x: x["timestamp"], reverse=True)
            
        except Exception as e:
            print(f"Failed to get recent agent results: {e}")
            return []