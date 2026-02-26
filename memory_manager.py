"""
MemoryManager: Episodic memory for chat/session context.
Stores and retrieves chat history, supports summarization and metadata filtering.
"""
from typing import List, Dict, Any
import hashlib
from pathlib import Path
import chromadb
from chromadb.config import Settings

class MemoryManager:
    def __init__(self, db_path: str, collection_name: str = "episodic_memory"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.documents = []  # Lista de documentos en memoria
        self.counter = 1     # Para IDs únicos

    def add_document(self, text):
        doc = {
            "doc_id": f"doc_{self.counter}",
            "filename": "Sin nombre",
            "source": "manual",
            "num_chunks": 1,
            "content": text
        }
        self.documents.append(doc)
        self.counter += 1
        return doc["doc_id"]

    def list_documents(self):
        return self.documents

    def clear(self):
        self.documents.clear()
        self.counter = 1
        return True

    async def add_message(self, session_id: str, message: str, metadata: Dict[str, Any]) -> None:
        msg_hash = hashlib.sha256((session_id + message).encode("utf-8")).hexdigest()
        self.collection.add(
            documents=[message],
            metadatas=[{**metadata, "session_id": session_id, "source_type": "episodic_chat"}],
            ids=[msg_hash]
        )

    async def search(self, query: str, session_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # Filter by session_id and source_type
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"session_id": session_id, "source_type": "episodic_chat"}
        )
        if not results.get("documents") or not results.get("metadatas"):
            return []
        return [{"content": doc, **meta} for doc, meta in zip(results["documents"][0], results["metadatas"][0])] # type: ignore

    async def summarize_history(self, llm_client, session_id: str) -> None:
        # Get all messages for session
        results = self.collection.get(where={"session_id": session_id, "source_type": "episodic_chat"})
        documents = results.get("documents")
        if not documents:
            return  # No messages to summarize
        messages = [doc for doc in documents]
        if len(messages) < 10:
            return  # Not enough to summarize
        # Summarize oldest messages
        to_summarize = messages[:-5]
        summary = await llm_client.generate(f"Summarize the following chat history:\n{to_summarize}")
        # Replace old messages with summary
        for doc_id in results["ids"][:-5]:
            self.collection.delete(ids=[doc_id])
        self.collection.add(
            documents=[summary["choices"][0]["message"]["content"]],
            metadatas=[{"session_id": session_id, "source_type": "episodic_chat", "summarized": True}],
            ids=[hashlib.sha256((session_id + "summary").encode("utf-8")).hexdigest()]
        )
