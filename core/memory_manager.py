"""
Memory Manager for MIMIR-AI.

RAG (Retrieval-Augmented Generation) system with Hybrid Search.
Combines Dense Retrieval (embeddings) and Sparse Retrieval (BM25) for optimal results.

Components:
- RecursiveCharacterTextSplitter: Splits documents into chunks
- NomicEmbeddingFunction: Generates embeddings via llama-server or sentence-transformers
- BM25Retriever: Sparse retrieval using BM25 algorithm
- RAGMemoryManager: Main manager that orchestrates storage, retrieval, and context injection

Features:
- Hybrid Search: Combines semantic (dense) and lexical (sparse) retrieval
- Reciprocal Rank Fusion (RRF): Merges results from both methods
- Persistent storage: ChromaDB for vectors, BM25 index in memory
- Automatic chunking: Smart text splitting with overlap
"""

import logging
import uuid
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import chromadb
from chromadb.config import Settings

from config import (
    STORAGE_DIR, MODELS_DIR, RAG_EMBEDDING_MODEL, USE_LOCAL_EMBEDDINGS
)

logger = logging.getLogger(__name__)


class RecursiveCharacterTextSplitter:
    """
    Recursive character text splitter for document chunking.
    Splits text recursively by characters, respecting chunk size and overlap.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize the text splitter.
        
        Args:
            chunk_size: Maximum size of chunks
            chunk_overlap: Overlap between chunks
            separators: List of separators to try (in order of preference)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            " ",     # Words
            ""       # Characters
        ]
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks recursively.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        current_text = text
        
        while len(current_text) > self.chunk_size:
            # Try each separator in order
            split_success = False
            for separator in self.separators:
                if separator == "":
                    # Character-level split (last resort)
                    chunk = current_text[:self.chunk_size]
                    chunks.append(chunk)
                    current_text = current_text[self.chunk_size - self.chunk_overlap:]
                    split_success = True
                    break
                
                # Find the last occurrence of separator within chunk_size
                split_idx = current_text.rfind(separator, 0, self.chunk_size)
                
                if split_idx != -1:
                    # Split at the separator
                    chunk = current_text[:split_idx + len(separator)].strip()
                    if chunk:
                        chunks.append(chunk)
                    current_text = current_text[split_idx + len(separator) - self.chunk_overlap:].strip()
                    split_success = True
                    break
            
            if not split_success:
                # Fallback: force split at chunk_size
                chunk = current_text[:self.chunk_size]
                chunks.append(chunk)
                current_text = current_text[self.chunk_size - self.chunk_overlap:]
        
        # Add remaining text
        if current_text.strip():
            chunks.append(current_text.strip())
        
        return chunks


class NomicEmbeddingFunction:
    """
    Custom embedding function using Nomic-Embed-Text-v1.5.
    Uses llama-server API if available, otherwise sentence-transformers.
    """
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        embedding_api_url: Optional[str] = None
    ):
        """
        Initialize the embedding function.
        
        Args:
            model_name: Name of the sentence-transformers model (default: from config)
            embedding_api_url: URL for llama-server embedding API (e.g., http://localhost:8081/v1)
        """
        self.model_name = model_name or RAG_EMBEDDING_MODEL
        self.embedding_api_url = embedding_api_url
        self._st_model = None
        self._use_llama_server = embedding_api_url is not None and USE_LOCAL_EMBEDDINGS
        
        if self._use_llama_server:
            try:
                from openai import OpenAI
                self.embedding_client = OpenAI(
                    base_url=embedding_api_url,
                    api_key="not-needed"
                )
                logger.info(f"Using llama-server API for embeddings at {embedding_api_url}")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client for embeddings: {e}, falling back to sentence-transformers")
                self._use_llama_server = False
        else:
            logger.info(f"Using sentence-transformers model: {self.model_name}")
    
    def name(self) -> str:
        """
        Return the name of the embedding function.
        Required by ChromaDB for embedding function identification.
        
        Returns:
            Name identifier for the embedding function
        """
        if self._use_llama_server:
            return "llama-server-embeddings"
        else:
            return f"sentence-transformers-{self.model_name.replace('/', '-').replace('\\', '-')}"
    
    def embed_query(self, input: str) -> List[float]:
        """
        Generate embedding for a single query text.
        Required by ChromaDB for query embeddings.
        
        Args:
            input: Query text string (ChromaDB passes this as 'input')
            
        Returns:
            Embedding vector as a list of floats
        """
        # Ensure input is a string
        if not isinstance(input, str):
            input = str(input)
        
        # Use llama-server API directly for single query
        if self._use_llama_server:
            try:
                response = self.embedding_client.embeddings.create(
                    model="llama",
                    input=input  # Single string
                )
                embedding = response.data[0].embedding
                
                # Debug: log the type of embedding
                logger.debug(f"embed_query: embedding type={type(embedding)}, is_list={isinstance(embedding, list)}")
                
                # Ensure we return a list of floats - handle all possible formats
                if isinstance(embedding, list):
                    # Already a list, convert all elements to float
                    result = [float(x) for x in embedding]
                elif hasattr(embedding, '__iter__') and not isinstance(embedding, str):
                    # Iterable but not a list (e.g., numpy array, tuple)
                    result = [float(x) for x in embedding]
                else:
                    # Single value or unexpected type
                    logger.error(f"embed_query: unexpected embedding type {type(embedding)}, value={embedding}")
                    raise ValueError(f"embed_query: embedding must be a list or iterable, got {type(embedding)}")
                
                # Validate that result is a non-empty list of floats
                if not isinstance(result, list) or len(result) == 0:
                    raise ValueError(f"embed_query returned invalid result: {type(result)}, length={len(result) if hasattr(result, '__len__') else 'N/A'}")
                
                # Double-check all elements are floats
                result = [float(x) for x in result]
                
                logger.debug(f"embed_query: returning list of length {len(result)}, first few values: {result[:3] if len(result) >= 3 else result}")
                return result
            except Exception as e:
                logger.warning(f"llama-server embedding API failed in embed_query: {e}, falling back to sentence-transformers")
                self._use_llama_server = False
        
        # Fallback to sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            
            # Load model if not already loaded
            if not hasattr(self, '_st_model') or self._st_model is None:
                logger.info(f"Loading Nomic embedding model: {self.model_name}")
                self._st_model = SentenceTransformer(self.model_name)
            
            embedding = self._st_model.encode(input, convert_to_numpy=True)
            # Ensure we return a list of floats
            result = [float(x) for x in embedding.tolist()]
            logger.debug(f"embed_query (sentence-transformers): returning list of length {len(result)}")
            return result
        except Exception as e:
            logger.error(f"Error generating embedding in embed_query: {e}")
            raise
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            input: List of text strings
            
        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        # Use llama-server API if available
        if self._use_llama_server:
            try:
                embeddings = []
                # llama-server with --embeddings flag expects input as strings
                # Process each input individually to avoid format issues
                for text in input:
                    response = self.embedding_client.embeddings.create(
                        model="llama",  # Model name doesn't matter for llama-server
                        input=text  # Single string, not a list
                    )
                    embedding = response.data[0].embedding
                    
                    # Ensure each embedding is a list of floats
                    if isinstance(embedding, list):
                        embedding_list = [float(x) for x in embedding]
                    elif hasattr(embedding, 'tolist'):
                        embedding_list = [float(x) for x in embedding.tolist()]
                    else:
                        embedding_list = [float(x) for x in list(embedding)]
                    
                    embeddings.append(embedding_list)
                return embeddings
            except Exception as e:
                logger.warning(f"llama-server embedding API failed: {e}, falling back to sentence-transformers")
                self._use_llama_server = False
        
        # Fallback to sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            
            # Load model if not already loaded
            if not hasattr(self, '_st_model') or self._st_model is None:
                logger.info(f"Loading Nomic embedding model: {self.model_name}")
                self._st_model = SentenceTransformer(self.model_name)
            
            embeddings = self._st_model.encode(input, convert_to_numpy=True)
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise


class BM25Retriever:
    """
    BM25 (Best Matching 25) retriever for sparse retrieval.
    Used in Hybrid Search alongside dense embeddings.
    """
    
    def __init__(self):
        """Initialize BM25 retriever."""
        try:
            from rank_bm25 import BM25Okapi
            self.BM25Okapi = BM25Okapi
            self.bm25: Optional[BM25Okapi] = None
            self.chunk_texts: List[str] = []
            self.chunk_ids: List[str] = []
            self._needs_rebuild = True
            logger.info("BM25 Retriever initialized")
        except ImportError:
            logger.warning("rank-bm25 not installed. BM25 retrieval will be disabled.")
            logger.warning("Install with: pip install rank-bm25")
            self.BM25Okapi = None
            self.bm25 = None
            self.chunk_texts = []
            self.chunk_ids = []
            self._needs_rebuild = True
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens (words)
        """
        # Simple tokenization: split by whitespace and lowercase
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def add_chunks(self, chunk_ids: List[str], chunk_texts: List[str]) -> None:
        """
        Add chunks to BM25 index.
        
        Args:
            chunk_ids: List of chunk IDs
            chunk_texts: List of chunk texts
        """
        if not self.BM25Okapi:
            return
        
        self.chunk_ids.extend(chunk_ids)
        self.chunk_texts.extend(chunk_texts)
        self._needs_rebuild = True
    
    def remove_chunks(self, chunk_ids: List[str]) -> None:
        """
        Remove chunks from BM25 index.
        
        Args:
            chunk_ids: List of chunk IDs to remove
        """
        if not self.BM25Okapi:
            return
        
        # Remove chunks by ID
        indices_to_remove = [
            i for i, cid in enumerate(self.chunk_ids) 
            if cid in chunk_ids
        ]
        
        for i in sorted(indices_to_remove, reverse=True):
            del self.chunk_ids[i]
            del self.chunk_texts[i]
        
        self._needs_rebuild = True
    
    def rebuild_index(self) -> None:
        """Rebuild BM25 index from current chunks."""
        if not self.BM25Okapi or not self.chunk_texts:
            return
        
        try:
            # Tokenize all chunks
            tokenized_corpus = [self._tokenize(text) for text in self.chunk_texts]
            
            # Build BM25 index
            self.bm25 = self.BM25Okapi(tokenized_corpus)
            self._needs_rebuild = False
            logger.debug(f"BM25 index rebuilt with {len(self.chunk_texts)} chunks")
        except Exception as e:
            logger.error(f"Error rebuilding BM25 index: {e}")
            self.bm25 = None
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search using BM25.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (chunk_id, score) tuples, sorted by score descending
        """
        if not self.BM25Okapi or not self.bm25:
            return []
        
        if self._needs_rebuild:
            self.rebuild_index()
        
        if not self.bm25:
            return []
        
        try:
            # Tokenize query
            query_tokens = self._tokenize(query)
            
            if not query_tokens:
                return []
            
            # Get BM25 scores
            scores = self.bm25.get_scores(query_tokens)
            
            # Get top-k results
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:top_k]
            
            # Return (chunk_id, score) tuples
            results = [
                (self.chunk_ids[i], float(scores[i]))
                for i in top_indices
                if scores[i] > 0
            ]
            
            return results
        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []
    
    def clear(self) -> None:
        """Clear all chunks from BM25 index."""
        self.chunk_texts = []
        self.chunk_ids = []
        self.bm25 = None
        self._needs_rebuild = True


class RAGMemoryManager:
    """
    RAG Memory Manager for MIMIR-AI.
    
    Handles document storage, retrieval, and context injection using Hybrid Search.
    
    This manager combines:
    - Dense Retrieval: Semantic search using embeddings (ChromaDB)
    - Sparse Retrieval: Lexical search using BM25
    - Reciprocal Rank Fusion: Combines both results for optimal retrieval
    
    Documents are automatically chunked and indexed in both systems.
    When searching, results from both methods are combined using RRF.
    """
    
    def __init__(
        self,
        enabled: bool = True,
        storage_dir: Optional[Path] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 4,
        collection_name: str = "mimir_memory",
        embedding_api_url: Optional[str] = None
    ):
        """
        Initialize the RAG Memory Manager.
        
        Args:
            enabled: Whether RAG is enabled
            storage_dir: Directory for ChromaDB storage
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            top_k: Number of documents to retrieve
            collection_name: ChromaDB collection name
            embedding_api_url: URL for llama-server embedding API (optional)
        """
        self.enabled = enabled
        self.storage_dir = storage_dir or (STORAGE_DIR / "vector_db")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.collection_name = collection_name
        self.embedding_api_url = embedding_api_url
        
        # Initialize components
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.vector_store: Optional[chromadb.Collection] = None
        self.embedding_function: Optional[NomicEmbeddingFunction] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.use_hybrid_search: bool = True  # Enable hybrid search by default
        
        if self.enabled:
            self._initialize()
        else:
            logger.info("RAG Memory Manager is disabled")
    
    def _initialize(self) -> None:
        """Initialize ChromaDB and embedding function."""
        try:
            # Create storage directory
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize embedding function
            self.embedding_function = NomicEmbeddingFunction(
                embedding_api_url=self.embedding_api_url
            )
            
            # Initialize ChromaDB (persistent mode)
            logger.info(f"Initializing ChromaDB at: {self.storage_dir}")
            client = chromadb.PersistentClient(
                path=str(self.storage_dir),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.vector_store = client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            
            # Initialize BM25 retriever for hybrid search
            self.bm25_retriever = BM25Retriever()
            
            # Load existing chunks into BM25 index
            self._rebuild_bm25_index()
            
            logger.info(f"RAG Memory Manager initialized (collection: {self.collection_name})")
            if self.use_hybrid_search and self.bm25_retriever.BM25Okapi:
                logger.info("Hybrid Search enabled: Dense (embeddings) + Sparse (BM25)")
            elif self.use_hybrid_search:
                logger.warning("Hybrid Search requested but BM25 not available. Using Dense only.")
                self.use_hybrid_search = False
            
        except Exception as e:
            logger.error(f"Error initializing RAG Memory Manager: {e}", exc_info=True)
            self.enabled = False
            raise
    
    def _rebuild_bm25_index(self) -> None:
        """Rebuild BM25 index from existing ChromaDB chunks."""
        if not self.bm25_retriever or not self.vector_store:
            return
        
        try:
            # Get all chunks from ChromaDB
            all_data = self.vector_store.get(include=['documents', 'metadatas'])
            
            if not all_data['ids']:
                logger.debug("No existing chunks to load into BM25 index")
                return
            
            # Add all chunks to BM25
            chunk_ids = all_data['ids']
            chunk_texts = all_data['documents']
            
            self.bm25_retriever.add_chunks(chunk_ids, chunk_texts)
            self.bm25_retriever.rebuild_index()
            
            logger.info(f"Loaded {len(chunk_ids)} chunks into BM25 index")
        except Exception as e:
            logger.warning(f"Error rebuilding BM25 index: {e}")
    
    def add_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> str:
        """
        Add a document to the memory.
        
        Args:
            text: Document text
            metadata: Optional metadata dictionary
            doc_id: Optional document ID (generated if not provided)
            
        Returns:
            Document ID
        """
        if not self.enabled or not self.vector_store:
            raise RuntimeError("RAG Memory Manager is not enabled")
        
        # Generate ID if not provided
        if not doc_id:
            doc_id = str(uuid.uuid4())
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Prepare metadata
        doc_metadata = metadata or {}
        doc_metadata["doc_id"] = doc_id
        doc_metadata["num_chunks"] = len(chunks)
        
        # Add chunks to vector store
        chunk_ids = []
        chunk_texts = []
        chunk_metadatas = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_ids.append(chunk_id)
            chunk_texts.append(chunk)
            chunk_metadatas.append({
                **doc_metadata,
                "chunk_index": i,
                "chunk_text": chunk[:100] + "..." if len(chunk) > 100 else chunk  # Preview
            })
        
        # Add to ChromaDB
        self.vector_store.add(
            ids=chunk_ids,
            documents=chunk_texts,
            metadatas=chunk_metadatas
        )
        
        # Add to BM25 index for hybrid search
        if self.bm25_retriever:
            self.bm25_retriever.add_chunks(chunk_ids, chunk_texts)
            self.bm25_retriever.rebuild_index()
        
        logger.info(f"Added document {doc_id} with {len(chunks)} chunks")
        return doc_id
    
    def add_document_from_file(self, file_path: Path, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a document from a file.
        
        Args:
            file_path: Path to the file
            metadata: Optional metadata
            
        Returns:
            Document ID
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read file based on extension
        file_ext = file_path.suffix.lower()
        
        # Supported text file extensions
        text_extensions = {'.txt', '.md', '.markdown', '.py', '.js', '.ts', 
                          '.json', '.yaml', '.yml', '.xml', '.html', '.css',
                          '.csv', '.log', '.sh', '.bat', '.ps1', '.sql',
                          '.cpp', '.c', '.h', '.hpp', '.java', '.go', '.rs',
                          '.php', '.rb', '.swift', '.kt', '.scala', '.r',
                          '.m', '.mm', '.pl', '.lua', '.vim', '.conf', '.ini',
                          '.cfg', '.toml', '.properties', '.env', '.gitignore'}
        
        if file_ext in text_extensions:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
        else:
            raise ValueError(
                f"Tipo de archivo no soportado: {file_ext}\n"
                f"Tipos soportados: {', '.join(sorted(text_extensions))}"
            )
        
        # Prepare metadata
        file_metadata = metadata or {}
        file_metadata.setdefault("source", str(file_path))
        file_metadata.setdefault("filename", file_path.name)
        file_metadata.setdefault("filetype", file_ext[1:])
        
        return self.add_document(text, file_metadata)
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Combine dense and sparse results using Reciprocal Rank Fusion (RRF).
        
        Args:
            dense_results: Results from dense retrieval (embeddings)
            sparse_results: Results from sparse retrieval (BM25)
            k: RRF constant (typically 60)
            
        Returns:
            Combined and ranked results
        """
        # Create score maps
        dense_scores = {}
        sparse_scores = {}
        
        # Map dense results by chunk_id (from metadata or direct)
        for rank, result in enumerate(dense_results, 1):
            chunk_id = (
                result.get('chunk_id') or 
                result.get('id') or
                result.get('metadata', {}).get('chunk_id') or 
                result.get('metadata', {}).get('id')
            )
            if not chunk_id:
                # Use content hash as fallback
                content = result.get('content', '')
                chunk_id = hashlib.md5(content.encode()).hexdigest()[:16]
            dense_scores[chunk_id] = 1.0 / (k + rank)
        
        # Map sparse results by chunk_id
        for rank, result in enumerate(sparse_results, 1):
            chunk_id = (
                result.get('chunk_id') or 
                result.get('id') or
                result.get('metadata', {}).get('chunk_id') or
                result.get('metadata', {}).get('id')
            )
            if not chunk_id:
                # Use content hash as fallback
                content = result.get('content', '')
                chunk_id = hashlib.md5(content.encode()).hexdigest()[:16]
            sparse_scores[chunk_id] = 1.0 / (k + rank)
        
        # Combine scores
        combined_scores = {}
        all_chunk_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
        
        for chunk_id in all_chunk_ids:
            combined_scores[chunk_id] = (
                dense_scores.get(chunk_id, 0) + 
                sparse_scores.get(chunk_id, 0)
            )
        
        # Create result map from both sources
        result_map = {}
        for result in dense_results:
            chunk_id = result.get('metadata', {}).get('chunk_id') or result.get('metadata', {}).get('id')
            if not chunk_id:
                continue
            result_map[chunk_id] = result
        
        for result in sparse_results:
            chunk_id = result.get('chunk_id') or result.get('id')
            if not chunk_id:
                continue
            # Prefer dense result if both exist (has more metadata)
            if chunk_id not in result_map:
                result_map[chunk_id] = result
        
        # Sort by combined score
        sorted_chunk_ids = sorted(
            combined_scores.keys(),
            key=lambda x: combined_scores[x],
            reverse=True
        )
        
        # Build final results
        final_results = []
        for chunk_id in sorted_chunk_ids:
            if chunk_id in result_map:
                result = result_map[chunk_id].copy()
                result['hybrid_score'] = combined_scores[chunk_id]
                result['dense_score'] = dense_scores.get(chunk_id, 0)
                result['sparse_score'] = sparse_scores.get(chunk_id, 0)
                final_results.append(result)
        
        return final_results
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
        use_hybrid: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using Hybrid Search (Dense + Sparse).
        
        Args:
            query: Search query
            top_k: Number of results (defaults to self.top_k)
            where: Optional metadata filter
            use_hybrid: Override hybrid search setting (defaults to self.use_hybrid_search)
            
        Returns:
            List of results with content, metadata, and scores
        """
        if not self.enabled or not self.vector_store:
            return []
        
        k = top_k or self.top_k
        use_hybrid = use_hybrid if use_hybrid is not None else self.use_hybrid_search
        
        try:
            # Dense retrieval (embeddings)
            dense_results = []
            if self.embedding_function:
                query_embedding = self.embedding_function.embed_query(query)
                
                # Ensure query_embedding is a list of floats
                if not isinstance(query_embedding, list):
                    logger.error(f"embed_query returned non-list type: {type(query_embedding)}")
                    query_embedding = list(query_embedding) if hasattr(query_embedding, '__iter__') else [float(query_embedding)]
                
                # Ensure all elements are floats
                query_embedding = [float(x) for x in query_embedding]
                
                # Validate the embedding
                if len(query_embedding) > 0:
                    # Retrieve more results for hybrid fusion (2x for better coverage)
                    dense_k = k * 2 if use_hybrid else k
                    
                    results = self.vector_store.query(
                        query_embeddings=[query_embedding],
                        n_results=dense_k,
                        where=where,
                        include=['documents', 'metadatas', 'distances']
                    )
                    
                    # Format dense results
                    if results['documents'] and results['documents'][0]:
                        for doc, metadata, distance, chunk_id in zip(
                            results['documents'][0],
                            results['metadatas'][0],
                            results['distances'][0],
                            results['ids'][0]
                        ):
                            dense_results.append({
                                "content": doc,
                                "metadata": {**(metadata or {}), "chunk_id": chunk_id, "id": chunk_id},
                                "distance": float(distance),
                                "score": 1.0 - float(distance),
                                "chunk_id": chunk_id,
                                "id": chunk_id
                            })
            
            # Sparse retrieval (BM25) - only if hybrid search is enabled
            sparse_results = []
            if use_hybrid and self.bm25_retriever and self.bm25_retriever.BM25Okapi:
                try:
                    # Get BM25 results
                    bm25_results = self.bm25_retriever.search(query, top_k=k * 2)
                    
                    # Fetch full documents from ChromaDB
                    if bm25_results:
                        chunk_ids = [chunk_id for chunk_id, _ in bm25_results]
                        bm25_scores = {chunk_id: score for chunk_id, score in bm25_results}
                        
                        # Get documents from ChromaDB
                        bm25_data = self.vector_store.get(
                            ids=chunk_ids,
                            include=['documents', 'metadatas']
                        )
                        
                        # Format sparse results
                        for i, chunk_id in enumerate(bm25_data.get('ids', [])):
                            if i < len(bm25_data.get('documents', [])):
                                sparse_results.append({
                                    "content": bm25_data['documents'][i],
                                    "metadata": bm25_data['metadatas'][i] if bm25_data.get('metadatas') else {},
                                    "score": bm25_scores.get(chunk_id, 0),
                                    "chunk_id": chunk_id,
                                    "id": chunk_id
                                })
                except Exception as e:
                    logger.warning(f"Error in BM25 search: {e}, using dense only")
            
            # Combine results using RRF if hybrid search
            if use_hybrid and dense_results and sparse_results:
                combined_results = self._reciprocal_rank_fusion(dense_results, sparse_results)
                # Return top-k
                return combined_results[:k]
            elif use_hybrid and sparse_results:
                # Only sparse results available
                return sparse_results[:k]
            else:
                # Dense only or hybrid disabled
                return dense_results[:k]
            
        except Exception as e:
            logger.error(f"Error searching memory: {e}", exc_info=True)
            return []
    
    def get_context(self, query: str, top_k: Optional[int] = None) -> str:
        """
        Get formatted context for a query.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            Formatted context string
        """
        results = self.search(query, top_k=top_k)
        
        if not results:
            return ""
        
        # Format context
        context_parts = []
        for result in results:
            context_parts.append(result["content"])
        
        return "\n\n".join(context_parts)
    
    def inject_context(self, query: str, user_message: str) -> str:
        """
        Inject context into user message using the template.
        
        Args:
            query: Query for context retrieval
            user_message: Original user message
            
        Returns:
            Message with injected context
        """
        context = self.get_context(query)
        
        if not context:
            return user_message
        
        # Context injection template
        formatted_message = f"""Use the following context to answer:

{context}

Question: {user_message}"""
        
        return formatted_message
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and all its chunks.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if deleted, False otherwise
        """
        if not self.enabled or not self.vector_store:
            return False
        
        try:
            # Get all chunks for this document
            results = self.vector_store.get(
                where={"doc_id": doc_id},
                include=['metadatas']
            )
            
            if results['ids']:
                # Delete all chunks from ChromaDB
                self.vector_store.delete(ids=results['ids'])
                
                # Remove from BM25 index
                if self.bm25_retriever:
                    self.bm25_retriever.remove_chunks(results['ids'])
                    self.bm25_retriever.rebuild_index()
                
                logger.info(f"Deleted document {doc_id} ({len(results['ids'])} chunks)")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all unique documents in the memory.
        
        Returns:
            List of document metadata
        """
        if not self.enabled or not self.vector_store:
            return []
        
        try:
            # Get all documents
            results = self.vector_store.get(include=['metadatas'])
            
            # Group by doc_id
            docs = {}
            for metadata in results['metadatas']:
                doc_id = metadata.get('doc_id')
                if doc_id and doc_id not in docs:
                    docs[doc_id] = {
                        "doc_id": doc_id,
                        "source": metadata.get("source", "unknown"),
                        "filename": metadata.get("filename", "unknown"),
                        "num_chunks": metadata.get("num_chunks", 0)
                    }
            
            return list(docs.values())
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []
    
    def clear(self) -> bool:
        """
        Clear all documents from memory.
        
        Returns:
            True if cleared, False otherwise
        """
        if not self.enabled or not self.vector_store:
            return False
        
        try:
            # Delete collection and recreate
            client = chromadb.PersistentClient(
                path=str(self.storage_dir),
                settings=Settings(anonymized_telemetry=False)
            )
            client.delete_collection(name=self.collection_name)
            
            # Recreate collection
            self.vector_store = client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            
            # Clear BM25 index
            if self.bm25_retriever:
                self.bm25_retriever.clear()
            
            logger.info("Memory cleared")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing memory: {e}")
            return False

