"""
KnowledgeBaseManager: Handles static document ingestion, deduplication, and search.
Uses ChromaDB (CPU-only) and metadata filtering.
"""
from typing import List, Dict, Any
import hashlib
from pathlib import Path
import chromadb
from chromadb.config import Settings

class KnowledgeBaseManager:
    def __init__(self, db_path: str, collection_name: str = "knowledge_base"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def _file_hash(self, file_path: Path) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    def ingest_document(self, file_path: Path, metadata: Dict[str, Any]) -> None:
        file_hash = self._file_hash(file_path)
        # Check for duplicates
        existing = self.collection.get(where={"file_hash": file_hash})
        if existing["ids"]:
            return  # Already ingested
        # Read content
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        self.collection.add(
            documents=[content],
            metadatas=[{**metadata, "file_hash": file_hash, "source_type": "static_knowledge"}],
            ids=[file_hash]
        )

    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # Hybrid search: dense + sparse (BM25)
        # For now, only dense (ChromaDB) for simplicity
        results = self.collection.query(query_texts=[query], n_results=top_k)
        documents = results.get("documents")
        metadatas = results.get("metadatas")
        if not documents or not metadatas or documents[0] is None or metadatas[0] is None:
            return []
        return [{"content": doc, **meta} for doc, meta in zip(documents[0], metadatas[0])]
