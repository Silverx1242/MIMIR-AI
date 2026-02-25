"""
FastEmbedWrapper: Lightweight CPU-only embedding for ingestion/search.
"""
from typing import List

class FastEmbedWrapper:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        try:
            import fastembed
            self.model = fastembed.EmbeddingModel(model_name=model_name, device="cpu")
        except ImportError:
            self.model = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self.model:
            raise RuntimeError("FastEmbed not available. Please install fastembed.")
        return list(self.model.embed(texts))
