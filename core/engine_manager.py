"""
Engine Manager for MIMIR-AI.

Manages llama-server binary processes (Zero-Compile architecture).
Handles hardware detection, binary selection, and server lifecycle management.

Features:
- Automatic hardware detection (NVIDIA GPU, Apple Silicon, CPU)
- Binary selection based on detected hardware
- Process lifecycle management (start, monitor, stop)
- Health checks and automatic recovery
- Port management and conflict resolution
"""

import os
import sys
import platform
import subprocess
import logging
import time
import requests
import json
import zipfile
import tarfile
import io
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import psutil

logger = logging.getLogger(__name__)

# Try to import GPUtil for GPU detection (optional)
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    logger.warning("GPUtil not available. GPU detection will be limited.")

# Try to import nvidia-ml-py for NVIDIA GPU detection (replaces deprecated pynvml)
try:
    import pynvml  # nvidia-ml-py provides pynvml module
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class EngineManager:
    """
    Manages llama-server binary processes.
    Handles hardware detection, binary selection, and server lifecycle.
    """
    
    def __init__(self, bin_dir: Path = Path("bin"), base_port: int = 8080):
        """
        Initialize the Engine Manager.
        
        Args:
            bin_dir: Directory containing llama-server binaries
            base_port: Base port for the server
        """
        self.bin_dir = Path(bin_dir)
        self.base_port = base_port
        self.server_process: Optional[subprocess.Popen] = None
        self.server_port = base_port
        self.hardware_info: Dict[str, Any] = {}
        self.binary_path: Optional[Path] = None
        
        # Ensure bin directory exists
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        
    def detect_hardware(self) -> Dict[str, Any]:
        """
        Detect available hardware (GPU, CPU architecture).
        
        Returns:
            Dictionary with hardware information
        """
        info = {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "cpu_count": psutil.cpu_count(),
            "has_nvidia_gpu": False,
            "has_apple_silicon": False,
            "gpu_name": None,
            "vram_gb": None,
            "cuda_version": None
        }
        
        # Detect NVIDIA GPU
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_name_raw = pynvml.nvmlDeviceGetName(handle)
                if isinstance(gpu_name_raw, bytes):
                    info["gpu_name"] = gpu_name_raw.decode('utf-8')
                else:
                    info["gpu_name"] = str(gpu_name_raw)
                
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                info["vram_gb"] = mem_info.total / (1024**3)
                info["has_nvidia_gpu"] = True
                
                # Try to detect CUDA version (approximate)
                try:
                    cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
                    info["cuda_version"] = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"
                except:
                    pass
                    
                logger.info(f"NVIDIA GPU detected: {info['gpu_name']} ({info['vram_gb']:.2f} GB VRAM)")
            except Exception as e:
                logger.debug(f"NVML initialization failed: {e}")
        
        # Try GPUtil as fallback
        if not info["has_nvidia_gpu"] and GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    info["has_nvidia_gpu"] = True
                    info["gpu_name"] = gpu.name
                    info["vram_gb"] = gpu.memoryTotal
                    logger.info(f"NVIDIA GPU detected (via GPUtil): {info['gpu_name']} ({info['vram_gb']:.2f} GB VRAM)")
            except Exception as e:
                logger.debug(f"GPUtil detection failed: {e}")
        
        # Detect Apple Silicon
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            info["has_apple_silicon"] = True
            logger.info("Apple Silicon (Metal) detected")
        
        self.hardware_info = info
        return info
    
    def get_required_binary_name(self) -> Tuple[str, str]:
        """
        Determine which binary is required based on hardware.
        
        Returns:
            Tuple of (binary_name, description)
        """
        if self.hardware_info.get("has_nvidia_gpu"):
            # Windows: llama-server-cuda.exe
            # Linux: llama-server-cuda
            if platform.system() == "Windows":
                return "llama-server-cuda.exe", "NVIDIA GPU (CUDA) - Windows"
            else:
                return "llama-server-cuda", "NVIDIA GPU (CUDA) - Linux"
        elif self.hardware_info.get("has_apple_silicon"):
            return "llama-server-metal", "Apple Silicon (Metal)"
        else:
            # Generic CPU (AVX2)
            if platform.system() == "Windows":
                return "llama-server-avx2.exe", "CPU (AVX2) - Windows"
            else:
                return "llama-server-avx2", "CPU (AVX2) - Linux"
    
    def check_binary_exists(self, binary_name: str) -> bool:
        """Check if the required binary exists."""
        binary_path = self.bin_dir / binary_name
        return binary_path.exists() and binary_path.is_file()
    
    def check_binary_dependencies(self, binary_name: str) -> bool:
        """
        Check if the binary has required DLLs (Windows only).
        Returns True if dependencies are present or not needed, False if critical DLLs are missing.
        """
        if platform.system() != "Windows":
            return True  # Not needed on non-Windows systems
        
        binary_path = self.bin_dir / binary_name
        if not binary_path.exists():
            return False
        
        # Check for critical DLLs that llama-server needs
        critical_dlls = []
        if "cuda" in binary_name.lower():
            critical_dlls = ["ggml-cuda.dll", "ggml.dll", "llama.dll"]
        elif "avx2" in binary_name.lower():
            critical_dlls = ["ggml.dll", "llama.dll"]
        
        # Check if any critical DLLs are missing
        for dll in critical_dlls:
            dll_path = self.bin_dir / dll
            if not dll_path.exists():
                logger.warning(f"Missing critical DLL: {dll}")
                return False
        
        return True
    
    def get_binary_path(self) -> Optional[Path]:
        """Get the path to the required binary, or None if not found."""
        binary_name, _ = self.get_required_binary_name()
        binary_path = self.bin_dir / binary_name
        
        if binary_path.exists():
            return binary_path
        return None
    
    def offer_binary_download(self, binary_name: str) -> bool:
        """
        Offer to download the binary from GitHub releases.
        
        Args:
            binary_name: Name of the binary to download
            
        Returns:
            True if user wants to download, False otherwise
        """
        print(f"\n[WARNING] Binary '{binary_name}' not found in {self.bin_dir}")
        print(f"   The system needs this binary to run llama-server.")
        print(f"   You can download it from: https://github.com/ggerganov/llama.cpp/releases")
        print(f"\n   Would you like to download it now? (y/n): ", end="", flush=True)
        
        try:
            response = input().strip().lower()
            return response in ['y', 'yes', 's', 'sí']
        except (EOFError, KeyboardInterrupt):
            return False
    
    def _get_binary_asset_patterns(self, binary_name: str) -> Tuple[list, Optional[str]]:
        """
        Get possible asset patterns to search for in GitHub releases.
        
        Args:
            binary_name: Name of the binary (e.g., 'llama-server-cuda.exe')
            
        Returns:
            Tuple of (list of patterns, archive_type) or (None, None) if not found
        """
        system = platform.system().lower()
        is_windows = system == "windows"
        is_linux = system == "linux"
        is_macos = system == "darwin"
        
        patterns = []
        
        # Map binary names to asset patterns (llama.cpp uses different naming)
        if "cuda" in binary_name.lower():
            if is_windows:
                patterns = ["win-cuda", "windows-x64-cuda"]
                return patterns, "zip"
            elif is_linux:
                patterns = ["linux-x64-cuda", "linux-cuda"]
                return patterns, "tar.xz"
        elif "metal" in binary_name.lower():
            if is_macos:
                patterns = ["macos-arm64", "mac-arm64"]
                return patterns, "tar.gz"
        elif "avx2" in binary_name.lower():
            if is_windows:
                patterns = ["win-avx2", "windows-x64-avx2"]
                return patterns, "zip"
            elif is_linux:
                patterns = ["linux-x64-avx2", "linux-avx2"]
                return patterns, "tar.xz"
        
        return [], None
    
    def _download_binary(self, binary_name: str) -> bool:
        """
        Download binary from GitHub releases.
        
        Args:
            binary_name: Name of the binary to download
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            logger.info(f"Downloading binary '{binary_name}' from GitHub releases...")
            
            # Get asset patterns
            asset_patterns, archive_type = self._get_binary_asset_patterns(binary_name)
            if not asset_patterns or not archive_type:
                logger.error(f"Could not determine asset pattern for {binary_name}")
                return False
            
            # Get latest release from GitHub API
            api_url = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
            logger.info(f"Fetching latest release info from GitHub...")
            
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            release_data = response.json()
            
            tag_name = release_data.get("tag_name", "")
            assets = release_data.get("assets", [])
            
            logger.info(f"Latest release: {tag_name}")
            
            # Find matching asset (prioritize CUDA version match, exclude cudart)
            matching_asset = None
            cuda_version = self.hardware_info.get("cuda_version")
            cuda_major_int = None
            if cuda_version:
                try:
                    cuda_major = float(cuda_version)
                    cuda_major_int = int(cuda_major)
                except:
                    cuda_major_int = None
            
            # First pass: try to find exact CUDA version match (prefer llama-bin over cudart)
            if cuda_major_int and "cuda" in binary_name.lower():
                for asset in assets:
                    asset_name = asset.get("name", "").lower()
                    # Skip cudart packages (they don't contain server binaries)
                    if "cudart" in asset_name:
                        continue
                    # Check if it matches pattern, archive type, and CUDA version
                    for pattern in asset_patterns:
                        if pattern.lower() in asset_name and archive_type in asset_name:
                            # Prefer matching CUDA version (e.g., cuda-13.1)
                            if cuda_version and cuda_version.replace(".", "-") in asset_name:
                                matching_asset = asset
                                break
                            elif cuda_major_int and f"cuda-{cuda_major_int}" in asset_name:
                                matching_asset = asset
                                break
                    if matching_asset:
                        break
            
            # Second pass: fallback to any matching pattern (still exclude cudart)
            if not matching_asset:
                for asset in assets:
                    asset_name = asset.get("name", "").lower()
                    # Skip cudart packages
                    if "cudart" in asset_name:
                        continue
                    # Check if any pattern matches and archive type matches
                    for pattern in asset_patterns:
                        if pattern.lower() in asset_name and archive_type in asset_name:
                            matching_asset = asset
                            break
                    if matching_asset:
                        break
            
            if not matching_asset:
                logger.error(f"Could not find asset matching patterns {asset_patterns}")
                logger.error(f"Available assets: {[a.get('name') for a in assets[:10]]}")
                return False
            
            asset_url = matching_asset.get("browser_download_url")
            asset_size = matching_asset.get("size", 0)
            asset_name = matching_asset.get("name", "")
            
            logger.info(f"Found asset: {asset_name} ({asset_size / (1024*1024):.1f} MB)")
            logger.info(f"Downloading from: {asset_url}")
            
            # Download asset
            print(f"\n📥 Downloading {asset_name}...")
            download_response = requests.get(asset_url, stream=True, timeout=60)
            download_response.raise_for_status()
            
            # Read content into memory
            content = download_response.content
            logger.info(f"Downloaded {len(content) / (1024*1024):.1f} MB")
            
            # Extract binary from archive
            print(f"[INFO] Extracting binary...")
            binary_extracted = False
            
            if archive_type == "zip":
                with zipfile.ZipFile(io.BytesIO(content)) as zip_file:
                    # Log all files in zip for debugging
                    all_files = zip_file.namelist()
                    logger.info(f"Files in archive (first 30): {all_files[:30]}")
                    
                    # Find the server binary in the zip
                    server_member = None
                    for member in all_files:
                        member_lower = member.lower().replace("\\", "/")
                        # Look for server executable (can be llama-server.exe, server.exe, etc.)
                        if member_lower.endswith(".exe"):
                            # Check if it's a server binary (not test or example)
                            if "server" in member_lower and not any(x in member_lower for x in ["test", "example", "benchmark"]):
                                server_member = member
                                break
                    
                    if not server_member:
                        logger.error("Could not find server executable in archive")
                        return False
                    
                    logger.info(f"Found server binary in archive: {server_member}")
                    
                    # Extract ALL files to bin directory (server needs DLLs)
                    logger.info("Extracting all files from archive (server needs DLLs)...")
                    zip_file.extractall(self.bin_dir)
                    
                    # Find and rename the server binary
                    extracted_server_path = self.bin_dir / Path(server_member)
                    final_server_path = self.bin_dir / binary_name
                    
                    if extracted_server_path.exists():
                        if extracted_server_path != final_server_path:
                            # Remove target if exists
                            if final_server_path.exists():
                                final_server_path.unlink()
                            extracted_server_path.rename(final_server_path)
                            binary_extracted = True
                        else:
                            binary_extracted = True
                        
                        # Clean up empty subdirectories (but keep DLLs)
                        try:
                            # Only remove directories, not files
                            for root, dirs, files in os.walk(self.bin_dir):
                                for d in dirs:
                                    dir_path = Path(root) / d
                                    try:
                                        if not any(dir_path.iterdir()):
                                            dir_path.rmdir()
                                    except:
                                        pass
                        except:
                            pass
                    else:
                        logger.error(f"Server binary not found after extraction: {extracted_server_path}")
                        return False
            
            elif archive_type in ["tar.xz", "tar.gz"]:
                mode = "r:xz" if archive_type == "tar.xz" else "r:gz"
                with tarfile.open(fileobj=io.BytesIO(content), mode=mode) as tar_file:
                    # Find the binary in the tar
                    for member in tar_file.getmembers():
                        # Look for llama-server binary (could be in subdirectory)
                        if "llama-server" in member.name and member.isfile():
                            # Extract to bin directory
                            tar_file.extract(member, self.bin_dir)
                            extracted_path = self.bin_dir / member.name
                            # Move to bin root with correct name
                            final_path = self.bin_dir / binary_name
                            if extracted_path != final_path:
                                if extracted_path.exists():
                                    extracted_path.rename(final_path)
                                # Remove empty subdirectories
                                try:
                                    for parent in extracted_path.parents:
                                        if parent != self.bin_dir and parent.exists():
                                            try:
                                                parent.rmdir()
                                            except:
                                                pass
                                except:
                                    pass
                            binary_extracted = True
                            break
            
            if not binary_extracted:
                logger.error(f"Could not find server binary in archive")
                logger.error(f"Looking for: {binary_name}")
                if archive_type == "zip":
                    logger.error(f"Please check the archive contents manually")
                return False
            
            # Make executable on Unix systems
            binary_path = self.bin_dir / binary_name
            if platform.system() != "Windows":
                os.chmod(binary_path, 0o755)
            
            logger.info(f"[SUCCESS] Binary downloaded and extracted to: {binary_path}")
            print(f"[SUCCESS] Binary downloaded successfully!")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading binary: {e}")
            print(f"[ERROR] Error downloading binary: {e}")
            return False
        except Exception as e:
            logger.error(f"Error extracting binary: {e}", exc_info=True)
            print(f"[ERROR] Error extracting binary: {e}")
            return False
    
    def calculate_ngl(self, model_path: Path) -> int:
        """
        Calculate optimal number of GPU layers based on available VRAM.
        
        Args:
            model_path: Path to the GGUF model
            
        Returns:
            Number of GPU layers (-1 for all, or specific number)
        """
        vram_gb = self.hardware_info.get("vram_gb")
        if not self.hardware_info.get("has_nvidia_gpu") or not vram_gb:
            return 0
        
        # Estimate model size (rough approximation)
        try:
            model_size_gb = model_path.stat().st_size / (1024**3)
        except:
            model_size_gb = 4.0  # Default estimate
        
        # Reserve some VRAM for system
        available_vram = vram_gb * 0.85  # Use 85% of VRAM
        
        # If model fits in VRAM, use all layers
        if model_size_gb * 1.2 < available_vram:  # 1.2x safety margin
            return -1  # All layers
        
        # Otherwise, estimate based on available VRAM
        # Rough estimate: 1 layer ≈ 100-200 MB (varies by model)
        estimated_layers = int((available_vram / model_size_gb) * 32)  # Rough estimate
        
        logger.info(f"Calculated -ngl={estimated_layers} based on {vram_gb:.2f} GB VRAM")
        return min(estimated_layers, 32)  # Cap at reasonable number
    
    def start_server(
        self,
        model_path: Path,
        port: Optional[int] = None,
        ngl: Optional[int] = None,
        enable_embeddings: bool = False
    ) -> bool:
        """
        Start the llama-server process.
        
        Args:
            model_path: Path to the GGUF model file
            port: Port for the server (default: self.base_port)
            ngl: Number of GPU layers (-1 for all)
            enable_embeddings: Enable embeddings support (adds --embeddings flag)
            
        Returns:
            True if server started successfully, False otherwise
        """
        if self.server_process is not None:
            logger.warning("Server is already running")
            return False
        
        # Detect hardware if not done
        if not self.hardware_info:
            self.detect_hardware()
        
        # Get binary path
        binary_name, description = self.get_required_binary_name()
        self.binary_path = self.get_binary_path()
        
        # Check if binary exists and has dependencies
        if not self.binary_path or not self.check_binary_dependencies(binary_name):
            # If binary exists but dependencies are missing, offer to re-download
            if self.binary_path and not self.check_binary_dependencies(binary_name):
                logger.warning(f"Binary '{binary_name}' exists but is missing required DLLs")
                logger.warning("This usually happens if the binary was downloaded with an older version")
                print(f"\n[WARNING] Binary '{binary_name}' is missing required DLLs")
                print(f"   Would you like to re-download it? (y/n): ", end="", flush=True)
                try:
                    response = input().strip().lower()
                    if response not in ['y', 'yes', 's', 'sí']:
                        logger.error("Re-download declined. Please delete the binary and try again.")
                        return False
                    # Remove the incomplete binary
                    self.binary_path.unlink()
                    self.binary_path = None
                except (EOFError, KeyboardInterrupt):
                    return False
            
            if not self.binary_path:
                if not self.offer_binary_download(binary_name):
                    logger.error(f"Binary '{binary_name}' not found and download declined")
                    return False
            
            # Attempt to download the binary
            if not self._download_binary(binary_name):
                logger.error("Binary download failed. Please download manually.")
                logger.info(f"Download URL: https://github.com/ggerganov/llama.cpp/releases")
                logger.info(f"Place the binary at: {self.bin_dir / binary_name}")
                return False
            
            # Verify the binary was downloaded
            self.binary_path = self.get_binary_path()
            if not self.binary_path:
                logger.error("Binary download completed but file not found")
                return False
            
            # Verify dependencies are present after download
            if not self.check_binary_dependencies(binary_name):
                logger.error("Binary downloaded but critical DLLs are missing")
                return False
        
        # Validate model path
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return False
        
        # Set port
        self.server_port = port or self.base_port
        
        # Calculate ngl if not provided
        if ngl is None:
            ngl = self.calculate_ngl(model_path)
        
        # Build command
        cmd = [
            str(self.binary_path),
            "-m", str(model_path),
            "--port", str(self.server_port),
            "-ngl", str(ngl),
            "--host", "127.0.0.1"
        ]
        
        # Add embeddings flag if enabled
        if enable_embeddings:
            cmd.append("--embeddings")
        
        logger.info(f"Starting llama-server: {description}")
        logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            # Start server process
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Wait a bit for server to start
            time.sleep(2)
            
            # Check if process is still running
            if self.server_process.poll() is not None:
                # Process died, read error
                stdout, stderr = self.server_process.communicate()
                logger.error(f"Server process exited with code {self.server_process.returncode}")
                logger.error(f"STDOUT: {stdout}")
                logger.error(f"STDERR: {stderr}")
                self.server_process = None
                return False
            
            # Wait for server to be ready
            if self.wait_for_server(timeout=30):
                logger.info(f"[SUCCESS] llama-server started successfully on port {self.server_port}")
                return True
            else:
                logger.error("Server started but did not become ready in time")
                self.stop_server()
                return False
                
        except Exception as e:
            logger.error(f"Error starting server: {e}", exc_info=True)
            if self.server_process:
                self.server_process.terminate()
                self.server_process = None
            return False
    
    def wait_for_server(self, timeout: int = 30) -> bool:
        """
        Wait for the server to become ready.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if server is ready, False otherwise
        """
        url = f"http://localhost:{self.server_port}/health"
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            
            # Check if process is still running
            if self.server_process and self.server_process.poll() is not None:
                logger.error("Server process died")
                return False
            
            time.sleep(0.5)
        
        return False
    
    def stop_server(self) -> None:
        """Stop the llama-server process."""
        if self.server_process is None:
            return
        
        logger.info("Stopping llama-server...")
        try:
            self.server_process.terminate()
            # Wait up to 5 seconds for graceful shutdown
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate
                logger.warning("Server did not terminate gracefully, forcing kill...")
                self.server_process.kill()
                self.server_process.wait()
            
            logger.info("[SUCCESS] llama-server stopped")
        except Exception as e:
            logger.error(f"Error stopping server: {e}")
        finally:
            self.server_process = None
    
    def is_running(self) -> bool:
        """Check if the server is running."""
        if self.server_process is None:
            return False
        
        # Check if process is still alive
        if self.server_process.poll() is not None:
            self.server_process = None
            return False
        
        # Check if server responds
        try:
            response = requests.get(f"http://localhost:{self.server_port}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def get_server_url(self) -> str:
        """Get the base URL for the server API."""
        return f"http://localhost:{self.server_port}/v1"
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop server."""
        self.stop_server()

