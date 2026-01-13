# MIMIR-AI (ex Atlas-AI): Zero-Compile System Agent

**MIMIR-AI** (formerly known as **Atlas-AI**) is a local system agent that uses GGUF models through precompiled binaries of `llama-server`. Zero-Compile architecture: no need to compile CUDA code.

## 🔄 Name Change: Atlas-AI → MIMIR-AI

This project was previously known as **Atlas-AI**. The name has been changed to **MIMIR-AI** to better reflect the project's evolution and vision.

**Why MIMIR?**
- **Mímir** is a figure in Norse mythology known as the god of wisdom and knowledge, who guards the Well of Wisdom
- This name better represents the system's role as a knowledge keeper and intelligent assistant
- The name change reflects the project's growth from a basic agent to a more sophisticated RAG-powered system

**What changed:**
- Project name: `Atlas-AI` → `MIMIR-AI`
- Log files: `atlas.log` → `mimir.log`
- Collection names: `atlas_memory` → `mimir_memory`
- All references updated throughout the codebase

**Backward compatibility:**
- Existing data and configurations remain compatible
- Collection names can be migrated if needed
- No breaking changes to the API or functionality

## 🌟 Features

- **Zero-Compile**: Uses precompiled llama-server binaries (no CUDA compilation required)
- **Automatic hardware detection**: NVIDIA GPU (CUDA), Apple Silicon (Metal), CPU (AVX2)
- **Tool system**: FileManager, Interpreter, SystemControl with security confirmation
- **RAG System**: Hybrid Search (Dense + Sparse) with ChromaDB and BM25
- **Function Calling**: Supports system tools through function calls
- **Conversational memory**: Maintains context from previous conversations
- **Local embeddings**: Uses llama-server for embeddings or sentence-transformers fallback

## 🛠️ System Requirements

- Python 3.10 or higher
- llama-server binaries (download from GitHub releases)
- Minimum 8GB RAM (recommended)
- GPU optional but recommended for better performance

## 📁 Project Structure

```
MIMIR-AI/
├── bin/                    # Precompiled llama-server binaries
├── core/                   # Main system modules
│   ├── engine_manager.py   # llama-server management and hardware detection
│   ├── agent_controller.py # Agent controller (OpenAI API)
│   ├── tool_registry.py    # System tools registry
│   └── memory_manager.py   # RAG Memory Manager with Hybrid Search
├── models/                 # GGUF models (.gguf files)
├── storage/               # Local storage
│   └── vector_db/         # ChromaDB vector database
├── config.py              # Centralized configuration
├── main.py                # Entry point
├── .env.example           # Environment variables example
└── requirements.txt       # Python dependencies
```

## ⚡ Installation

### 1. Clone repository and create virtual environment

```bash
git clone "https://github.com/Silverx1242/MIMIR-AI.git"
cd MIMIR-AI
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

**Main dependencies:**
- `openai` - API client for llama-server
- `psutil` - Hardware detection
- `chromadb` - Vector database for RAG
- `rank-bm25` - BM25 for hybrid search
- `sentence-transformers` - Embeddings (fallback)
- `requests` - HTTP communication

### 3. Download llama-server binaries

Binaries must be downloaded manually from:
**https://github.com/ggerganov/llama.cpp/releases**

Place the appropriate binary in the `/bin` directory:

- **Windows + NVIDIA GPU**: `llama-server-cuda.exe`
- **Windows + CPU**: `llama-server-avx2.exe`
- **Linux + NVIDIA GPU**: `llama-server-cuda`
- **Linux + CPU**: `llama-server-avx2`
- **macOS (Apple Silicon)**: `llama-server-metal`

### 4. Download GGUF Models

**IMPORTANT**: You must download two types of GGUF models for local operation:

1. **Generative Model (LLM)**: For text generation and conversation
   - Recommended: LLaMA-3.2, LLaMA-3, Mistral, or other compatible models
   - Download from: [HuggingFace](https://huggingface.co/models?library=gguf) or [TheBloke](https://huggingface.co/TheBloke)
   - Example: `llama-3.2-3b-instruct-q8_0.gguf`

2. **Embedding Model**: For RAG (Retrieval-Augmented Generation) functionality
   - Recommended: `nomic-embed-text-v1.5.Q5_K_M.gguf` or similar embedding models
   - Download from: [HuggingFace](https://huggingface.co/models?library=gguf&search=embed)
   - Example: `nomic-embed-text-v1.5.Q5_K_M.gguf`

**Place both models in the `/models` directory** (create it if it doesn't exist):

```bash
mkdir models
# Download and place your models here
# models/
#   ├── llama-3.2-3b-instruct-q8_0.gguf      (generative model)
#   └── nomic-embed-text-v1.5.Q5_K_M.gguf     (embedding model)
```

**Note**: Both models are required for full functionality. The generative model handles conversations, while the embedding model enables RAG features for document search and retrieval.

### 5. Configuration

Copy `.env.example` to `.env` and adjust values:

```bash
cp .env.example .env
```

Edit `.env` with your configuration. **You must specify the paths to both models**:

```env
# ============================================
# Model Configuration
# ============================================
# Generative Model (LLM) - Required for text generation
MODEL_NAME=llama-3.2-3b-instruct-q8_0.gguf
MODELS_DIR=./models

# Embedding Model - Required for RAG functionality
# Specify the name of your embedding model file
EMBEDDING_MODEL_NAME=nomic-embed-text-v1.5.Q5_K_M.gguf

# ============================================
# Engine Configuration
# ============================================
ENGINE_PORT=8080
EMBEDDING_ENGINE_PORT=8081
BIN_DIR=./bin
STORAGE_DIR=./storage

# ============================================
# Generation Parameters
# ============================================
TEMPERATURE=0.7
MAX_TOKENS=2048

# ============================================
# RAG Configuration
# ============================================
RAG_ENABLED=true
USE_LOCAL_EMBEDDINGS=true
RAG_CHUNK_SIZE=512
RAG_TOP_K=4
```

**Configuration Notes**:
- `MODEL_NAME`: Name of your generative GGUF model file (must be in `MODELS_DIR`)
- `MODELS_DIR`: Directory where your GGUF models are stored (default: `./models`)
- `EMBEDDING_MODEL_NAME`: Name of your embedding GGUF model file (must be in `MODELS_DIR`)
- Both models must be downloaded and placed in the `MODELS_DIR` directory before starting the system

## 💻 Usage

### Start the system

```bash
python main.py
```

Or with custom arguments:

```bash
python main.py --model ./models/llama-3.2-3b-instruct-q8_0.gguf --port 8080
```

### Initialization flow

1. **Hardware detection**: System automatically detects GPU/CPU
2. **llama-server startup**: Launches appropriate binary with optimized parameters
3. **RAG initialization**: Sets up ChromaDB and BM25 for hybrid search
4. **Agent initialization**: Connects AgentController with local server
5. **Ready to use**: System is ready to receive commands

### Interactive usage

Once started, you can interact with MIMIR-AI:

```
>>> Hello, what can you do?

>>> Can you load a file for me?
/load documents/notes.txt

>>> What is the user's name?
```

The system will automatically use available tools when needed (with confirmation for dangerous operations).

### RAG Commands

MIMIR-AI includes a RAG (Retrieval-Augmented Generation) system with hybrid search:

```
/load <file>     - Load a file (.txt, .md, .py, etc.) into RAG
/add <text>      - Add text directly to RAG
/list            - List all loaded documents
/clear           - Clear RAG memory
/help            - Show help
```

## 🔧 Main Components

### EngineManager

- **Automatic hardware detection**: Identifies NVIDIA GPU, Apple Silicon, or CPU
- **Lifecycle management**: Starts, monitors, and stops llama-server
- **Automatic optimization**: Calculates optimal number of GPU layers based on available VRAM

### AgentController

- **OpenAI client**: Communicates with llama-server via OpenAI-compatible API
- **Conversation management**: Maintains history with automatic expiration
- **Function Calling**: Executes system tools when model requests them
- **RAG integration**: Automatically injects context from loaded documents

### RAGMemoryManager

- **Hybrid Search**: Combines Dense (embeddings) and Sparse (BM25) retrieval
- **ChromaDB**: Vector database for storing document embeddings
- **BM25**: Sparse retrieval for exact term matching
- **Reciprocal Rank Fusion (RRF)**: Combines results from both methods
- **Automatic chunking**: Splits documents into optimal chunks with overlap

### ToolRegistry

System tools available to the agent:

- **FileManager**: `file_read`, `file_write`, `file_delete`, `file_list`, `file_exists`
- **Interpreter**: `python_execute` (execute Python code)
- **SystemControl**: `system_command` (execute system commands)

**Security**: All dangerous tools require explicit confirmation (y/n).

## 📦 Compatible Models

MIMIR-AI requires **two GGUF models** for full functionality:

### 1. Generative Model (LLM) - Required
- **Purpose**: Text generation, conversations, and reasoning
- **Recommended**: LLaMA-3.2, LLaMA-3, Mistral, or other compatible models
- **Download**: [HuggingFace](https://huggingface.co/models?library=gguf) or [TheBloke](https://huggingface.co/TheBloke)
- **Configuration**: Set `MODEL_NAME` in `.env` with the filename
- **Location**: Must be placed in the directory specified by `MODELS_DIR` (default: `./models`)

### 2. Embedding Model - Required for RAG
- **Purpose**: Document embeddings for RAG (Retrieval-Augmented Generation)
- **Recommended**: `nomic-embed-text-v1.5.Q5_K_M.gguf` or similar embedding models
- **Download**: [HuggingFace](https://huggingface.co/models?library=gguf&search=embed)
- **Configuration**: Set `EMBEDDING_MODEL_NAME` in `.env` with the filename
- **Location**: Must be placed in the same directory as the generative model (`MODELS_DIR`)

**Important**: 
- Both models must be downloaded manually and placed in the `MODELS_DIR` directory
- Both models must be configured in `.env` for local operation
- The embedding model is required for RAG functionality; if not available, the system will fall back to `sentence-transformers` (requires internet)

**Example `.env` configuration**:

```env
MODEL_NAME=llama-3.2-3b-instruct-q8_0.gguf
MODELS_DIR=./models
EMBEDDING_MODEL_NAME=nomic-embed-text-v1.5.Q5_K_M.gguf
USE_LOCAL_EMBEDDINGS=true
```

## 🔍 Technical Features

- **Zero-Compile Architecture**: No need to compile CUDA code
- **REST API**: Communication with llama-server via HTTP
- **Intelligent detection**: Automatically selects correct binary
- **Process management**: Robust handling of llama-server lifecycle
- **Detailed logs**: Complete logging in `mimir.log`
- **Hybrid Search**: Best of both worlds - semantic (dense) and lexical (sparse) retrieval

## ⚠️ Limitations and Considerations

1. **Function Calling**: Depends on support in llama-server (may require recent version)
2. **Embeddings**: 
   - By default uses llama-server with GGUF models if available
   - Falls back to `sentence-transformers` with models from HuggingFace
3. **Binary download**: Not automated (must be done manually)
4. **Performance**: Depends on available hardware and model size

## 🛠️ Troubleshooting

### Error: Binary not found

```bash
# Verify binary is in /bin directory
ls bin/

# Download correct binary from:
# https://github.com/ggerganov/llama.cpp/releases
```

### Error: Model not found

```bash
# Verify both models are in MODELS_DIR (default: ./models)
ls models/

# Verify model names in .env match the actual filenames
# Check MODEL_NAME and EMBEDDING_MODEL_NAME

# Or use --model argument for generative model
python main.py --model /full/path/to/model.gguf
```

**Common issues**:
- Model file not downloaded or in wrong location
- Model filename in `.env` doesn't match actual filename (case-sensitive)
- Missing embedding model when `USE_LOCAL_EMBEDDINGS=true`

### Error: Port in use

```bash
# Change port in .env or use --port
python main.py --port 8081
```

### Logs

Logs are saved in `mimir.log`. Check this file for more details on errors.

## 📜 Logging

The system maintains detailed logs in `mimir.log`:
- Hardware detection
- llama-server start/stop
- Agent operations
- RAG operations
- Errors and warnings

## 🤝 Contributing

Contributions are welcome. Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

AGPL v3 License
