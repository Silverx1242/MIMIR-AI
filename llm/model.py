"""
LLM Model Interface for MIMIR-AI.

Provides interface for text generation and embeddings using OpenAI-compatible API.
Supports embeddings via llama-server API or sentence-transformers as fallback.

This module provides a unified interface for interacting with language models
through the OpenAI API client, which connects to local llama-server instances.
"""

import logging
from typing import Dict, List, Optional, Union
from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAICompatibleModel:
    """
    Model interface using OpenAI-compatible API (llama-server).
    For embeddings, uses llama-server API if available, otherwise sentence-transformers.
    """
    
    def __init__(
        self,
        api_base_url: str,
        model_name: str = "llama",
        embedding_api_url: Optional[str] = None,
        embedding_model_name: Optional[str] = None
    ):
        """
        Initialize the model.
        
        Args:
            api_base_url: Base URL for OpenAI API (e.g., http://localhost:8080/v1)
            model_name: Model name (not critical for llama-server)
            embedding_api_url: Optional URL for embedding API (if different from api_base_url)
            embedding_model_name: Sentence transformer model name for embeddings (fallback)
        """
        self.api_base_url = api_base_url
        self.model_name = model_name
        self.embedding_api_url = embedding_api_url or api_base_url
        self.embedding_model_name = embedding_model_name or "nomic-ai/nomic-embed-text-v1.5"
        
        # Initialize OpenAI client for text generation
        self.client = OpenAI(
            base_url=api_base_url,
            api_key="not-needed"
        )
        
        # Initialize OpenAI client for embeddings (may be same or different)
        self.embedding_client = OpenAI(
            base_url=self.embedding_api_url,
            api_key="not-needed"
        )
        
        # Initialize embedding model (lazy loading, for fallback)
        self._embedding_model = None
        self._use_llama_server_embeddings = True  # Try llama-server first
        
        logger.info(f"OpenAICompatibleModel initialized with API at {api_base_url}")
    
    @property
    def embedding_model(self):
        """Lazy load embedding model (fallback only)."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"Loaded embedding model (fallback): {self.embedding_model_name}")
            except ImportError:
                logger.error("sentence-transformers not available for embeddings")
                raise RuntimeError("sentence-transformers is required for embeddings fallback")
        return self._embedding_model
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        max_tokens: int = 2048,
        repeat_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None
    ) -> Dict[str, Union[str, int]]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text
            system_prompt: System message
            temperature: Temperature for generation
            top_p: Value for nucleus sampling
            top_k: Number of tokens to consider (ignored, OpenAI API doesn't support it)
            max_tokens: Maximum number of tokens to generate
            repeat_penalty: Repetition penalty (ignored, OpenAI API doesn't support it)
            stop: List of sequences to stop generation
            
        Returns:
            Dictionary with generated text and metadata
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop if stop else None
            )
            
            generated_text = response.choices[0].message.content
            
            return {
                "text": generated_text,
                "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else 0,
                "finish_reason": response.choices[0].finish_reason
            }
            
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            return {
                "text": f"Error generating response: {str(e)}",
                "tokens_used": 0,
                "finish_reason": "error"
            }
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embeddings from text using llama-server API or sentence-transformers fallback.
        
        Args:
            text: Text to get embeddings from
            
        Returns:
            List of float values representing the embedding
        """
        # Try llama-server API first
        if self._use_llama_server_embeddings:
            try:
                response = self.embedding_client.embeddings.create(
                    model=self.model_name,
                    input=text
                )
                embedding = response.data[0].embedding
                return embedding
            except Exception as e:
                logger.warning(f"llama-server embedding API failed: {e}, falling back to sentence-transformers")
                self._use_llama_server_embeddings = False  # Disable for future calls
        
        # Fallback to sentence-transformers
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise
