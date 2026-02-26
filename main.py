"""
MIMIR-AI - Entry Point
Zero-Compile Architecture using llama-server binaries.
"""

import os
import sys
import argparse
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# Configure UTF-8 for Windows
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)
    except Exception:
        pass

from config import (
    SYSTEM_PROMPT, MODELS_DIR, MODEL_PATH, EMBEDDING_MODEL_PATH,
    ENGINE_PORT, EMBEDDING_ENGINE_PORT, BIN_DIR, RAG_ENABLED, STORAGE_DIR,
    USE_LOCAL_EMBEDDINGS
)
from orchestrator import ContextOrchestrator
from llm_client import LLMClient
from knowledge_manager import KnowledgeBaseManager
from memory_manager import MemoryManager
from reranker import FlashRankReranker

# Setup logger
logger = logging.getLogger("mimir")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler("mimir.log", maxBytes=1_000_000, backupCount=2)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class MimirSystem:
    def __init__(self, model_path, engine_port, bin_dir):
        self.model_path = model_path
        self.engine_port = engine_port
        self.bin_dir = bin_dir
        self.memory_manager = None  # Should be initialized in initialize()
        # Add other components as needed

    def initialize(self):
        # Initialize memory manager and other components here
        try:
            db_path = os.path.join(STORAGE_DIR, "rag_memory.db")
            self.memory_manager = MemoryManager(db_path=db_path)
            # Initialize other components as needed
            return True
        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            return False

    def print_welcome(self):
        print("Welcome to MIMIR-AI!")

    def process_input(self, user_input):
        # Dummy command handling for demonstration
        if user_input.startswith("/add "):
            return self._handle_add_command(user_input[5:])
        elif user_input == "/list":
            return self._handle_list_command()
        elif user_input == "/clear":
            return self._handle_clear_command()
        elif user_input == "/help":
            return self._handle_help_command()
        elif user_input in ("/exit", "/quit"):
            return {"exit": True, "message": "Exiting..."}
        else:
            return {"exit": False, "message": "Comando no reconocido. Usa /help para ayuda."}

    def shutdown(self):
        # Clean up resources if needed
        pass

    def _handle_add_command(self, text: str) -> dict:
        """Handle /add command to add text directly to RAG."""
        if not self.memory_manager or not getattr(self.memory_manager, "enabled", False):
            return {
                "success": False,
                "message": "[ERROR] RAG Memory Manager no está habilitado"
            }

        if not text:
            return {
                "success": False,
                "message": "[ERROR] Uso: /add <texto>\nEjemplo: /add Este es un documento importante sobre..."
            }

        try:
            doc_id = self.memory_manager.add_document(text)

            return {
                "success": True,
                "message": f"[SUCCESS] Texto agregado exitosamente\n   ID: {doc_id}\n   Longitud: {len(text)} caracteres",
                "exit": False
            }
        except Exception as e:
            logger.error(f"Error adding text: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"[ERROR] Error al agregar texto: {str(e)}"
            }

    def _handle_list_command(self) -> dict:
        """Handle /list command to list documents in RAG."""
        if not self.memory_manager or not getattr(self.memory_manager, "enabled", False):
            return {
                "success": False,
                "message": "[ERROR] RAG Memory Manager no está habilitado"
            }

        try:
            documents = self.memory_manager.list_documents()

            if not documents:
                return {
                    "success": True,
                    "message": "[INFO] No hay documentos cargados en la memoria RAG",
                    "exit": False
                }

            message = f"[INFO] Documentos en la memoria RAG ({len(documents)}):\n\n"
            for i, doc in enumerate(documents, 1):
                message += f"{i}. {doc.get('filename', 'Sin nombre')}\n"
                message += f"   ID: {doc.get('doc_id', 'N/A')}\n"
                message += f"   Fuente: {doc.get('source', 'N/A')}\n"
                message += f"   Chunks: {doc.get('num_chunks', 0)}\n\n"

            return {
                "success": True,
                "message": message.strip(),
                "exit": False
            }
        except Exception as e:
            logger.error(f"Error listing documents: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"[ERROR] Error al listar documentos: {str(e)}"
            }

    def _handle_clear_command(self) -> dict:
        """Handle /clear command to clear RAG memory."""
        if not self.memory_manager or not getattr(self.memory_manager, "enabled", False):
            return {
                "success": False,
                "message": "[ERROR] RAG Memory Manager no está habilitado"
            }

        try:
            if self.memory_manager.clear():
                return {
                    "success": True,
                    "message": "[SUCCESS] Memoria RAG limpiada exitosamente",
                    "exit": False
                }
            else:
                return {
                    "success": False,
                    "message": "[ERROR] Error al limpiar la memoria RAG"
                }
        except Exception as e:
            logger.error(f"Error clearing memory: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"[ERROR] Error al limpiar memoria: {str(e)}"
            }

    def _handle_help_command(self) -> dict:
        """Handle /help command to show available commands."""
        help_text = """
[HELP] Comandos disponibles para RAG:

  /load <archivo>     - Cargar un archivo (.txt, .md) al RAG
  /add <texto>        - Agregar texto directamente al RAG
  /list               - Listar todos los documentos cargados
  /clear              - Limpiar toda la memoria RAG
  /help               - Mostrar esta ayuda

Ejemplos:
  /load documentos/notas.txt
  /add Este es un documento importante sobre Python
  /list
  /clear

Nota: El RAG se usa automáticamente cuando haces preguntas.
Los documentos cargados se buscarán automáticamente para proporcionar contexto.
        """

        return {
            "success": True,
            "message": help_text.strip(),
            "exit": False
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MIMIR-AI - Zero-Compile System Agent")
    parser.add_argument(
        "--model", "-m",
        help="Path to GGUF model",
        type=str,
        default=None
    )
    parser.add_argument(
        "--port", "-p",
        help="Port for llama-server",
        type=int,
        default=ENGINE_PORT
    )
    parser.add_argument(
        "--bin-dir",
        help="Directory containing llama-server binaries",
        type=str,
        default=None
    )
    args = parser.parse_args()
    
    # Determine model path
    model_path = Path(args.model) if args.model else MODEL_PATH
    if model_path:
        model_path = Path(model_path)
    
    # Determine bin directory
    bin_dir = Path(args.bin_dir) if args.bin_dir else BIN_DIR
    
    # Initialize system
    system = MimirSystem(
        model_path=model_path,
        engine_port=args.port,
        bin_dir=bin_dir
    )
    
    # Initialize components
    if not system.initialize():
        print("\n[ERROR] Failed to initialize system. Exiting...")
        sys.exit(1)
    
    # Print welcome
    system.print_welcome()
    
    # Interactive loop
    try:
        while True:
            try:
                user_input = input(">>> ").strip()
                
                if not user_input:
                    continue
                
                result = system.process_input(user_input)
                
                # Check for exit
                if result.get("exit", False):
                    break
                
                # Print response
                print("\n" + "="*60)
                print(result.get("message", ""))
                print("="*60 + "\n")
                
            except KeyboardInterrupt:
                print("\n\nInterruption detected. Shutting down...")
                break
            except EOFError:
                print("\n\nInput not available. Shutting down...")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)
                print(f"\nError: {e}\n")
    
    finally:
        # Shutdown gracefully
        system.shutdown()
        print("\n[SUCCESS] System shutdown complete. Goodbye!")


if __name__ == "__main__":
    main()
