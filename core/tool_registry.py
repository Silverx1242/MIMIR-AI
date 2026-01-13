"""
Tool Registry for MIMIR-AI.

Manages system tools available to the agent.
Provides secure tool execution with confirmation for dangerous operations.

Available tools:
- FileManager: File operations (read, write, delete, list, exists)
- Interpreter: Python code execution
- SystemControl: System command execution
Implements system tools for the agent: FileManager, Interpreter, SystemControl.
"""

import os
import sys
import subprocess
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import shutil

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry for system tools available to the agent.
    All tools require user confirmation for safety.
    """
    
    def __init__(self, require_confirmation: bool = True):
        """
        Initialize the Tool Registry.
        
        Args:
            require_confirmation: Whether to require user confirmation for tools
        """
        self.require_confirmation = require_confirmation
        self.tools: Dict[str, Dict[str, Any]] = {}
        self._register_tools()
    
    def _register_tools(self) -> None:
        """Register all available tools."""
        # FileManager tools
        self.register_tool(
            "file_read",
            "Read content from a file",
            self._file_read,
            {"file_path": "Path to the file to read"}
        )
        self.register_tool(
            "file_write",
            "Write content to a file",
            self._file_write,
            {"file_path": "Path to the file to write", "content": "Content to write"}
        )
        self.register_tool(
            "file_delete",
            "Delete a file",
            self._file_delete,
            {"file_path": "Path to the file to delete"}
        )
        self.register_tool(
            "file_list",
            "List files in a directory",
            self._file_list,
            {"directory": "Directory path to list"}
        )
        self.register_tool(
            "file_exists",
            "Check if a file exists",
            self._file_exists,
            {"file_path": "Path to check"}
        )
        
        # Interpreter tools
        self.register_tool(
            "python_execute",
            "Execute Python code",
            self._python_execute,
            {"code": "Python code to execute"}
        )
        
        # SystemControl tools
        self.register_tool(
            "system_command",
            "Execute a system command (requires confirmation)",
            self._system_command,
            {"command": "Command to execute"}
        )
    
    def register_tool(
        self,
        name: str,
        description: str,
        handler: Callable,
        parameters: Dict[str, str]
    ) -> None:
        """
        Register a tool.
        
        Args:
            name: Tool name
            description: Tool description
            handler: Function to handle the tool call
            parameters: Dictionary of parameter names and descriptions
        """
        self.tools[name] = {
            "name": name,
            "description": description,
            "handler": handler,
            "parameters": parameters
        }
        logger.debug(f"Registered tool: {name}")
    
    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """
        Get OpenAI function calling schema for all tools.
        
        Returns:
            List of tool schemas in OpenAI format
        """
        schemas = []
        for tool_name, tool_info in self.tools.items():
            # Skip system_command from automatic tool calling (too dangerous)
            if tool_name == "system_command":
                continue
                
            properties = {}
            required = []
            for param_name, param_desc in tool_info["parameters"].items():
                properties[param_name] = {
                    "type": "string",
                    "description": param_desc
                }
                required.append(param_name)
            
            schema = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_info["description"],
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            }
            schemas.append(schema)
        
        return schemas
    
    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a tool.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool parameters
            
        Returns:
            Dictionary with result or error
        """
        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found"
            }
        
        tool_info = self.tools[tool_name]
        handler = tool_info["handler"]
        
        # Special handling for dangerous tools
        if tool_name in ["system_command", "python_execute", "file_delete", "file_write"]:
            if self.require_confirmation:
                if not self._request_confirmation(tool_name, kwargs):
                    return {
                        "success": False,
                        "error": "User declined to execute the tool"
                    }
        
        try:
            result = handler(**kwargs)
            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def _request_confirmation(self, tool_name: str, parameters: Dict[str, Any]) -> bool:
        """
        Request user confirmation for tool execution.
        
        Args:
            tool_name: Name of the tool
            parameters: Tool parameters
            
        Returns:
            True if user confirms, False otherwise
        """
        print(f"\n[WARNING] Tool execution request: {tool_name}")
        print(f"   Parameters: {json.dumps(parameters, indent=2)}")
        print(f"   Execute this tool? (y/n): ", end="", flush=True)
        
        try:
            response = input().strip().lower()
            return response in ['y', 'yes', 's', 'sí']
        except (EOFError, KeyboardInterrupt):
            return False
    
    # FileManager tools implementation
    def _file_read(self, file_path: str) -> Dict[str, Any]:
        """Read content from a file."""
        try:
            path = Path(file_path)
            if not path.exists():
                return {"error": f"File not found: {file_path}"}
            
            if not path.is_file():
                return {"error": f"Path is not a file: {file_path}"}
            
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            return {
                "content": content,
                "size": len(content),
                "path": str(path)
            }
        except Exception as e:
            raise Exception(f"Error reading file: {e}")
    
    def _file_write(self, file_path: str, content: str) -> Dict[str, Any]:
        """Write content to a file."""
        try:
            path = Path(file_path)
            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                "success": True,
                "path": str(path),
                "size": len(content)
            }
        except Exception as e:
            raise Exception(f"Error writing file: {e}")
    
    def _file_delete(self, file_path: str) -> Dict[str, Any]:
        """Delete a file."""
        try:
            path = Path(file_path)
            if not path.exists():
                return {"error": f"File not found: {file_path}"}
            
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                return {"error": "Use file_list to list directory contents. Directory deletion not supported via file_delete."}
            else:
                return {"error": f"Path is not a file: {file_path}"}
            
            return {
                "success": True,
                "path": str(path)
            }
        except Exception as e:
            raise Exception(f"Error deleting file: {e}")
    
    def _file_list(self, directory: str) -> Dict[str, Any]:
        """List files in a directory."""
        try:
            path = Path(directory)
            if not path.exists():
                return {"error": f"Directory not found: {directory}"}
            
            if not path.is_dir():
                return {"error": f"Path is not a directory: {directory}"}
            
            files = []
            dirs = []
            
            for item in sorted(path.iterdir()):
                item_info = {
                    "name": item.name,
                    "path": str(item)
                }
                if item.is_file():
                    item_info["type"] = "file"
                    item_info["size"] = item.stat().st_size
                    files.append(item_info)
                elif item.is_dir():
                    item_info["type"] = "directory"
                    dirs.append(item_info)
            
            return {
                "directory": str(path),
                "files": files,
                "directories": dirs,
                "total_files": len(files),
                "total_directories": len(dirs)
            }
        except Exception as e:
            raise Exception(f"Error listing directory: {e}")
    
    def _file_exists(self, file_path: str) -> Dict[str, Any]:
        """Check if a file exists."""
        try:
            path = Path(file_path)
            exists = path.exists()
            
            result = {
                "exists": exists,
                "path": str(path)
            }
            
            if exists:
                stat = path.stat()
                result["is_file"] = path.is_file()
                result["is_directory"] = path.is_dir()
                result["size"] = stat.st_size if path.is_file() else None
            
            return result
        except Exception as e:
            raise Exception(f"Error checking file: {e}")
    
    # Interpreter tools implementation
    def _python_execute(self, code: str) -> Dict[str, Any]:
        """Execute Python code."""
        try:
            # Create a safe execution environment
            safe_builtins = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'str': str,
                    'int': int,
                    'float': float,
                    'list': list,
                    'dict': dict,
                    'tuple': tuple,
                    'set': set,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'min': min,
                    'max': max,
                    'sum': sum,
                    'abs': abs,
                    'round': round,
                    'sorted': sorted,
                    'reversed': reversed,
                }
            }
            
            # Capture stdout
            import io
            from contextlib import redirect_stdout
            
            output_buffer = io.StringIO()
            error_buffer = io.StringIO()
            
            try:
                with redirect_stdout(output_buffer):
                    result = eval(code, safe_builtins)
            except SyntaxError:
                # Try exec if eval fails
                with redirect_stdout(output_buffer):
                    exec(code, safe_builtins)
                result = None
            
            output = output_buffer.getvalue()
            
            return {
                "success": True,
                "result": str(result) if result is not None else None,
                "output": output
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "type": type(e).__name__
            }
    
    # SystemControl tools implementation
    def _system_command(self, command: str) -> Dict[str, Any]:
        """Execute a system command."""
        try:
            # Use shell=True for cross-platform compatibility
            # But be careful with user input
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,  # 60 second timeout
                cwd=os.getcwd()
            )
            
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Command timed out after 60 seconds"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

