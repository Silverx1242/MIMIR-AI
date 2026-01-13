# config.py
"""
Centralized configuration for MIMIR-AI.

This module loads configuration from environment variables (.env file).
It handles model paths, generation parameters, memory settings, and RAG configuration.

Configuration is loaded in the following order:
1. Environment variables from .env file
2. Default values (if not specified)

Key configurations:
- Model paths: Auto-detects GGUF models in MODELS_DIR
- Generation parameters: Temperature, top_p, top_k, max_tokens
- Memory settings: History length, context window, expiry
- RAG settings: Chunk size, overlap, top_k, collection name
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(os.getenv('MODELS_DIR', './models'))

# Model names (automatically combined with MODELS_DIR)
_model_name = os.getenv('MODEL_NAME', '').strip()
if _model_name:
    _model_path = Path(_model_name)
    _model_path_str = str(_model_path).replace('\\', '/')
    _model_path = Path(_model_path_str)
    
    if _model_path.is_absolute():
        MODEL_PATH = _model_path
    elif str(_model_path).startswith('./') or str(_model_path).startswith('.\\'):
        MODEL_PATH = BASE_DIR / _model_path
    else:
        if _model_path.exists():
            MODEL_PATH = _model_path
        elif (MODELS_DIR / _model_path).exists():
            MODEL_PATH = MODELS_DIR / _model_path
        else:
            MODEL_PATH = MODELS_DIR / _model_path
else:
    # Auto-detect model
    MODEL_PATH = None
    if MODELS_DIR.exists():
        gguf_files = list(MODELS_DIR.glob("*.gguf"))
        embedding_keywords = ['embed', 'embedding', 'nomic']
        main_models = [f for f in gguf_files if not any(kw in f.name.lower() for kw in embedding_keywords)]
        if main_models:
            MODEL_PATH = sorted(main_models, key=lambda x: x.stat().st_mtime, reverse=True)[0]

# Generation configuration
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.7))
TOP_P = float(os.getenv('TOP_P', 0.9))
TOP_K = int(os.getenv('TOP_K', 40))
MAX_TOKENS = int(os.getenv('MAX_TOKENS', 2048))

# Memory configuration
MAX_HISTORY_LENGTH = int(os.getenv('MAX_HISTORY_LENGTH', 10))
CONTEXT_WINDOW_MESSAGES = int(os.getenv('CONTEXT_WINDOW_MESSAGES', 5))
MEMORY_EXPIRY = int(os.getenv('MEMORY_EXPIRY', 3600))

# v2 Configuration: Engine Manager
BIN_DIR = Path(os.getenv('BIN_DIR', './bin'))
ENGINE_PORT = int(os.getenv('ENGINE_PORT', 8080))
EMBEDDING_ENGINE_PORT = int(os.getenv('EMBEDDING_ENGINE_PORT', 8081))
STORAGE_DIR = Path(os.getenv('STORAGE_DIR', './storage'))

# Embedding Model Configuration
_embedding_model_name = os.getenv('EMBEDDING_MODEL_NAME', '').strip()
if _embedding_model_name:
    _embedding_model_path = Path(_embedding_model_name)
    _embedding_model_path_str = str(_embedding_model_path).replace('\\', '/')
    _embedding_model_path = Path(_embedding_model_path_str)
    
    if _embedding_model_path.is_absolute():
        EMBEDDING_MODEL_PATH = _embedding_model_path
    elif str(_embedding_model_path).startswith('./') or str(_embedding_model_path).startswith('.\\'):
        EMBEDDING_MODEL_PATH = BASE_DIR / _embedding_model_path
    else:
        if _embedding_model_path.exists():
            EMBEDDING_MODEL_PATH = _embedding_model_path
        elif (MODELS_DIR / _embedding_model_path).exists():
            EMBEDDING_MODEL_PATH = MODELS_DIR / _embedding_model_path
        else:
            EMBEDDING_MODEL_PATH = MODELS_DIR / _embedding_model_path
else:
    # Auto-detect embedding model
    EMBEDDING_MODEL_PATH = None
    if MODELS_DIR.exists():
        gguf_files = list(MODELS_DIR.glob("*.gguf"))
        embedding_keywords = ['embed', 'embedding', 'nomic']
        embedding_models = [f for f in gguf_files if any(kw in f.name.lower() for kw in embedding_keywords)]
        if embedding_models:
            EMBEDDING_MODEL_PATH = sorted(embedding_models, key=lambda x: x.stat().st_mtime, reverse=True)[0]

# RAG Memory Manager Configuration
RAG_ENABLED = os.getenv('RAG_ENABLED', 'true').lower() in ('true', '1', 'yes', 'on')
# Legacy: kept for backwards compatibility, but will use local embedding model if available
RAG_EMBEDDING_MODEL = os.getenv('RAG_EMBEDDING_MODEL', 'nomic-ai/nomic-embed-text-v1.5')
RAG_CHUNK_SIZE = int(os.getenv('RAG_CHUNK_SIZE', 512))
RAG_CHUNK_OVERLAP = int(os.getenv('RAG_CHUNK_OVERLAP', 50))
RAG_TOP_K = int(os.getenv('RAG_TOP_K', 4))
RAG_COLLECTION_NAME = os.getenv('RAG_COLLECTION_NAME', 'mimir_memory')
# Use local embeddings via llama-server if embedding model path is configured
USE_LOCAL_EMBEDDINGS = os.getenv('USE_LOCAL_EMBEDDINGS', 'true').lower() in ('true', '1', 'yes', 'on')

# System prompt
SYSTEM_PROMPT = """You are a helpful AI assistant named "TARS" that answers questions based on information from available documents.
If you don't know the answer, say so clearly instead of inventing information.
You can use tools to interact with the system when needed."""
