"""
ContextOrchestrator: Routes queries between KnowledgeBaseManager and MemoryManager.
Combines results, applies reranking, and sends prompt to LLMClient.
"""
from typing import List, Dict, Any
import asyncio

class ContextOrchestrator:
    def __init__(self, llm_client, knowledge_manager, memory_manager, reranker):
        self.llm_client = llm_client
        self.knowledge_manager = knowledge_manager
        self.memory_manager = memory_manager
        self.reranker = reranker

    async def route_query(self, user_query: str, session_id: str) -> Dict[str, Any]:
        # Step 1: Query transformation (LLM quick rewrite)
        rewritten = await self.llm_client.generate(f"Rewrite the following query for search: {user_query}")
        query = rewritten.get("choices", [{}])[0].get("message", {}).get("content", user_query)

        # Step 2: Retrieve from KnowledgeBase and Memory
        kb_results = await self.knowledge_manager.search(query)
        mem_results = await self.memory_manager.search(query, session_id)

        # Step 3: Combine and rerank
        combined = kb_results + mem_results
        reranked = await self.reranker.rerank(query, combined)

        # Step 4: Send to LLMClient for final answer
        context = "\n".join([r["content"] for r in reranked])
        response = await self.llm_client.generate(f"Context:\n{context}\n\nUser: {user_query}")
        return response
