"""
FlashRank Reranker (CPU-only): Async interface for reranking search results.
"""
from typing import List, Dict, Any
import asyncio

class FlashRankReranker:
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        # For CPU-only, use sentence-transformers or fastembed
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name, device="cpu")
        except ImportError:
            self.model = None

    async def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.model or not docs:
            return docs[:top_k]
        pairs = [[query, d["content"]] for d in docs]
        # Run in thread to avoid blocking event loop
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(None, self.model.predict, pairs)
        scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in scored[:top_k]]
