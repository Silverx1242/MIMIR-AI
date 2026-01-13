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
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from config import (
    SYSTEM_PROMPT, MODELS_DIR, MODEL_PATH, EMBEDDING_MODEL_PATH,
    ENGINE_PORT, EMBEDDING_ENGINE_PORT, BIN_DIR, RAG_ENABLED, STORAGE_DIR,
    USE_LOCAL_EMBEDDINGS
)
from core.engine_manager import EngineManager
from core.agent_controller import AgentController
from core.tool_registry import ToolRegistry
from core.memory_manager import RAGMemoryManager

# Configure logging
console_handler = logging.StreamHandler()
file_handler = RotatingFileHandler(
    "mimir.log",
    maxBytes=10 * 1024 * 1024,  # 10 MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[file_handler, console_handler]
)

logger = logging.getLogger(__name__)


class MimirSystem:
    """
    Main system class for MIMIR-AI.
    Orchestrates engine manager, agent controller, and tool registry.
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        engine_port: int = ENGINE_PORT,
        bin_dir: Path = BIN_DIR,
        embedding_model_path: Optional[Path] = None,
        embedding_engine_port: int = EMBEDDING_ENGINE_PORT
    ):
        """
        Initialize the MIMIR-AI system.
        
        Args:
            model_path: Path to GGUF model file for generation
            engine_port: Port for llama-server (generation)
            bin_dir: Directory containing llama-server binaries
            embedding_model_path: Path to GGUF model file for embeddings
            embedding_engine_port: Port for llama-server (embeddings)
        """
        self.model_path = model_path or MODEL_PATH
        self.embedding_model_path = embedding_model_path or (EMBEDDING_MODEL_PATH if USE_LOCAL_EMBEDDINGS else None)
        self.engine_port = engine_port
        self.embedding_engine_port = embedding_engine_port
        self.bin_dir = bin_dir
        
        # Initialize components
        self.engine_manager: Optional[EngineManager] = None
        self.embedding_engine_manager: Optional[EngineManager] = None
        self.agent_controller: Optional[AgentController] = None
        self.tool_registry = ToolRegistry(require_confirmation=True)
        
        # Initialize RAG Memory Manager if enabled
        self.memory_manager: Optional[RAGMemoryManager] = None
        # Will be initialized after embedding server starts (if using local embeddings)
        
        logger.info("MIMIR-AI System initialized")
    
    def initialize(self) -> bool:
        """
        Initialize the system: detect hardware, start engine, init agent.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Step 1: Initialize Engine Manager and detect hardware
            logger.info("Step 1: Detecting hardware...")
            self.engine_manager = EngineManager(
                bin_dir=self.bin_dir,
                base_port=self.engine_port
            )
            hardware_info = self.engine_manager.detect_hardware()
            
            print("\n" + "="*60)
            print("MIMIR-AI - Hardware Detection")
            print("="*60)
            print(f"Platform: {hardware_info['platform']} ({hardware_info['architecture']})")
            print(f"CPU Cores: {hardware_info['cpu_count']}")
            
            if hardware_info.get('has_nvidia_gpu'):
                print(f"GPU: {hardware_info.get('gpu_name', 'NVIDIA GPU')}")
                if hardware_info.get('vram_gb'):
                    print(f"VRAM: {hardware_info['vram_gb']:.2f} GB")
                if hardware_info.get('cuda_version'):
                    print(f"CUDA: {hardware_info['cuda_version']}")
            elif hardware_info.get('has_apple_silicon'):
                print("GPU: Apple Silicon (Metal)")
            else:
                print("GPU: None (CPU mode)")
            print("="*60 + "\n")
            
            # Step 2: Start llama-server for generation
            if not self.model_path:
                logger.error("No model path configured")
                print("[ERROR] Error: No model path configured")
                print(f"   Please set MODEL_NAME in .env or use --model argument")
                return False
            
            if not Path(self.model_path).exists():
                logger.error(f"Model not found: {self.model_path}")
                print(f"[ERROR] Error: Model not found: {self.model_path}")
                return False
            
            logger.info("Step 2: Starting llama-server for generation...")
            print(f"Starting generation engine with model: {self.model_path}")
            
            if not self.engine_manager.start_server(Path(self.model_path)):
                logger.error("Failed to start llama-server for generation")
                print("[ERROR] Error: Failed to start llama-server for generation")
                return False
            
            # Step 2.5: Start llama-server for embeddings (if using local embeddings)
            embedding_api_url = None
            if USE_LOCAL_EMBEDDINGS and self.embedding_model_path:
                if not Path(self.embedding_model_path).exists():
                    logger.warning(f"Embedding model not found: {self.embedding_model_path}, using sentence-transformers fallback")
                else:
                    logger.info("Step 2.5: Starting llama-server for embeddings...")
                    print(f"Starting embedding engine with model: {self.embedding_model_path}")
                    
                    self.embedding_engine_manager = EngineManager(
                        bin_dir=self.bin_dir,
                        base_port=self.embedding_engine_port
                    )
                    self.embedding_engine_manager.hardware_info = hardware_info  # Reuse hardware detection
                    
                    if self.embedding_engine_manager.start_server(Path(self.embedding_model_path), enable_embeddings=True):
                        embedding_api_url = self.embedding_engine_manager.get_server_url()
                        logger.info("[SUCCESS] Embedding engine started successfully")
                    else:
                        logger.warning("Failed to start embedding engine, using sentence-transformers fallback")
                        self.embedding_engine_manager = None
            
            # Step 3: Initialize RAG Memory Manager if enabled
            if RAG_ENABLED:
                try:
                    from config import RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP, RAG_TOP_K, RAG_COLLECTION_NAME
                    self.memory_manager = RAGMemoryManager(
                        enabled=True,
                        storage_dir=STORAGE_DIR / "vector_db",
                        chunk_size=RAG_CHUNK_SIZE,
                        chunk_overlap=RAG_CHUNK_OVERLAP,
                        top_k=RAG_TOP_K,
                        collection_name=RAG_COLLECTION_NAME,
                        embedding_api_url=embedding_api_url
                    )
                    logger.info("RAG Memory Manager initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize RAG Memory Manager: {e}")
                    self.memory_manager = None
            
            # Step 4: Initialize Agent Controller
            logger.info("Step 4: Initializing agent controller...")
            api_url = self.engine_manager.get_server_url()
            self.agent_controller = AgentController(
                api_base_url=api_url,
                system_prompt=SYSTEM_PROMPT,
                tool_registry=self.tool_registry,
                memory_manager=self.memory_manager
            )
            
            logger.info("[SUCCESS] System initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Error during initialization: {e}", exc_info=True)
            print(f"[ERROR] Error during initialization: {e}")
            return False
    
    def shutdown(self) -> None:
        """Shutdown the system gracefully."""
        logger.info("Shutting down system...")
        if self.engine_manager:
            self.engine_manager.stop_server()
        if self.embedding_engine_manager:
            self.embedding_engine_manager.stop_server()
        logger.info("System shutdown complete")
    
    def print_welcome(self) -> None:
        """Print welcome message."""
        print("\n" + "="*60)
        print("MIMIR-AI - System Agent")
        print("="*60)
        if self.engine_manager and self.engine_manager.is_running():
            hardware_info = self.engine_manager.hardware_info
            if hardware_info.get('has_nvidia_gpu'):
                device_info = f"GPU ({hardware_info.get('gpu_name', 'NVIDIA')})"
            elif hardware_info.get('has_apple_silicon'):
                device_info = "Apple Silicon (Metal)"
            else:
                device_info = "CPU"
            print(f"Generation Engine: Running on {device_info}")
        else:
            print("Generation Engine: Not running")
        
        if self.embedding_engine_manager and self.embedding_engine_manager.is_running():
            print(f"Embedding Engine: Running")
        else:
            print(f"Embedding Engine: Using sentence-transformers (fallback)")
        
        print(f"Model: {self.model_path}")
        if self.embedding_model_path:
            print(f"Embedding Model: {self.embedding_model_path}")
        print(f"API: {self.engine_manager.get_server_url() if self.engine_manager else 'N/A'}")
        if self.memory_manager and self.memory_manager.enabled:
            print(f"RAG: Habilitado (usa /help para ver comandos)")
        print(f"\nType 'exit' or 'quit' to exit")
        print("Type '/help' for RAG commands")
        print("="*60 + "\n")
    
    def process_input(self, user_input: str) -> dict:
        """
        Process user input.
        
        Args:
            user_input: User input text
            
        Returns:
            Processing result dictionary
        """
        if not self.agent_controller:
            return {
                "success": False,
                "message": "Agent controller not initialized"
            }
        
        # Check for exit commands
        if user_input.lower().strip() in ['exit', 'quit']:
            return {
                "success": True,
                "message": "Exiting...",
                "exit": True
            }
        
        # Handle RAG commands
        user_input_lower = user_input.lower().strip()
        
        # /load command - Load a file into RAG
        if user_input_lower.startswith('/load '):
            return self._handle_load_command(user_input[6:].strip())
        
        # /add command - Add text directly to RAG
        if user_input_lower.startswith('/add '):
            return self._handle_add_command(user_input[5:].strip())
        
        # /list command - List documents in RAG
        if user_input_lower == '/list':
            return self._handle_list_command()
        
        # /help command - Show help
        if user_input_lower in ['/help', '/?']:
            return self._handle_help_command()
        
        # /clear command - Clear RAG memory
        if user_input_lower == '/clear':
            return self._handle_clear_command()
        
        # Process with agent controller
        result = self.agent_controller.process_message(user_input, use_tools=True)
        
        return {
            "success": result.get("success", False),
            "message": result.get("message", ""),
            "tool_calls": result.get("tool_calls", False),
            "exit": False
        }
    
    def _handle_load_command(self, file_path_str: str) -> dict:
        """Handle /load command to load a file into RAG."""
        if not self.memory_manager or not self.memory_manager.enabled:
            return {
                "success": False,
                "message": "[ERROR] RAG Memory Manager no está habilitado"
            }
        
        if not file_path_str:
            return {
                "success": False,
                "message": "[ERROR] Uso: /load <ruta_al_archivo>\nEjemplo: /load documentos/mi_archivo.txt"
            }
        
        try:
            # Expandir ~ al directorio home
            if file_path_str.startswith('~'):
                home_dir = Path.home()
                file_path_str = str(home_dir / file_path_str[2:].lstrip('/\\'))
            
            file_path = Path(file_path_str)
            
            # Si la ruta no existe, intentar varias ubicaciones
            if not file_path.exists():
                possible_paths = []
                
                # 1. Ruta tal como está (por si es absoluta)
                if file_path.is_absolute():
                    possible_paths.append(file_path)
                
                # 2. Desde el directorio actual de trabajo
                possible_paths.append(Path.cwd() / file_path_str)
                
                # 3. Desde el directorio del proyecto
                project_root = Path(__file__).parent
                possible_paths.append(project_root / file_path_str)
                
                # 4. Manejar rutas comunes de Windows (desktop, documents, etc.)
                if sys.platform == 'win32':
                    home_dir = Path.home()
                    # desktop/user.txt -> C:\Users\...\Desktop\user.txt
                    if file_path_str.lower().startswith('desktop'):
                        desktop_path = home_dir / "Desktop" / file_path_str[7:].lstrip('/\\')
                        possible_paths.append(desktop_path)
                    # documents/user.txt -> C:\Users\...\Documents\user.txt
                    elif file_path_str.lower().startswith('documents') or file_path_str.lower().startswith('documentos'):
                        docs_path = home_dir / "Documents" / file_path_str.split('/', 1)[-1].split('\\', 1)[-1]
                        possible_paths.append(docs_path)
                    # downloads/user.txt -> C:\Users\...\Downloads\user.txt
                    elif file_path_str.lower().startswith('downloads') or file_path_str.lower().startswith('descargas'):
                        downloads_path = home_dir / "Downloads" / file_path_str.split('/', 1)[-1].split('\\', 1)[-1]
                        possible_paths.append(downloads_path)
                
                # Buscar el primer archivo que exista
                found = False
                for possible_path in possible_paths:
                    if possible_path.exists() and possible_path.is_file():
                        file_path = possible_path
                        found = True
                        break
                
                if not found:
                    # Construir mensaje de error con sugerencias
                    error_msg = f"[ERROR] Archivo no encontrado: {file_path_str}\n\n"
                    error_msg += "Ubicaciones intentadas:\n"
                    for i, path in enumerate(possible_paths[:5], 1):  # Mostrar solo las primeras 5
                        error_msg += f"   {i}. {path}\n"
                    error_msg += "\n[TIP] Sugerencias:\n"
                    error_msg += "   - Usa rutas absolutas: C:\\Users\\...\\archivo.txt\n"
                    error_msg += "   - O rutas relativas desde el proyecto: documentos/archivo.txt\n"
                    error_msg += "   - Para Desktop: desktop/archivo.txt o ~/Desktop/archivo.txt\n"
                    return {
                        "success": False,
                        "message": error_msg
                    }
            
            # Verificar que es un archivo, no un directorio
            if file_path.is_dir():
                return {
                    "success": False,
                    "message": f"[ERROR] La ruta es un directorio, no un archivo: {file_path}"
                }
            
            doc_id = self.memory_manager.add_document_from_file(file_path)
            
            return {
                "success": True,
                "message": f"[SUCCESS] Archivo cargado exitosamente\n   Archivo: {file_path.name}\n   Ruta: {file_path}\n   ID: {doc_id}",
                "exit": False
            }
        except FileNotFoundError:
            return {
                "success": False,
                "message": f"[ERROR] Archivo no encontrado: {file_path_str}"
            }
        except ValueError as e:
            return {
                "success": False,
                "message": f"[ERROR] Error: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Error loading file: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"[ERROR] Error al cargar archivo: {str(e)}"
            }
    
    def _handle_add_command(self, text: str) -> dict:
        """Handle /add command to add text directly to RAG."""
        if not self.memory_manager or not self.memory_manager.enabled:
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
        if not self.memory_manager or not self.memory_manager.enabled:
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
        if not self.memory_manager or not self.memory_manager.enabled:
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
