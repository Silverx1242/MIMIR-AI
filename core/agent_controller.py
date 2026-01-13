"""
Agent Controller for MIMIR-AI.

Orchestrates the agent using OpenAI API client pointing to local llama-server.
Manages conversations, tool calls, and RAG context injection.

The controller:
- Maintains conversation history with automatic expiry
- Handles function calling for system tools
- Integrates with RAG Memory Manager for context injection
- Manages tool execution with security confirmations
"""

import logging
import time
from typing import Dict, List, Optional, Any
from openai import OpenAI
import json

from core.tool_registry import ToolRegistry
from core.memory_manager import RAGMemoryManager
from config import (
    SYSTEM_PROMPT, TEMPERATURE, TOP_P, TOP_K, MAX_TOKENS,
    MAX_HISTORY_LENGTH, CONTEXT_WINDOW_MESSAGES, MEMORY_EXPIRY,
    RAG_ENABLED
)

logger = logging.getLogger(__name__)


class AgentController:
    """
    Controller for the agent system.
    Manages conversations, tool calls, and interaction with the LLM via OpenAI API.
    """
    
    def __init__(
        self,
        api_base_url: str,
        system_prompt: Optional[str] = None,
        tool_registry: Optional[ToolRegistry] = None,
        memory_manager: Optional[RAGMemoryManager] = None
    ):
        """
        Initialize the Agent Controller.
        
        Args:
            api_base_url: Base URL for the OpenAI-compatible API (e.g., http://localhost:8080/v1)
            system_prompt: System prompt for the agent
            tool_registry: Tool registry instance
            memory_manager: RAG Memory Manager instance (optional)
        """
        self.api_base_url = api_base_url
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.tool_registry = tool_registry or ToolRegistry()
        self.memory_manager = memory_manager
        
        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=api_base_url,
            api_key="not-needed"  # llama-server doesn't require API key
        )
        
        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []
        self.last_interaction_time = time.time()
        
        logger.info(f"Agent Controller initialized with API at {api_base_url}")
        if self.memory_manager and self.memory_manager.enabled:
            logger.info("RAG Memory Manager is enabled")
    
    def _update_conversation_history(self, role: str, content: str) -> None:
        """Update conversation history with expiry logic."""
        current_time = time.time()
        
        # Reset history if too much time has passed
        if current_time - self.last_interaction_time > MEMORY_EXPIRY:
            self.conversation_history = []
            logger.info("Conversation memory reset due to inactivity")
        
        # Update last interaction time
        self.last_interaction_time = current_time
        
        # Add new message
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        
        # Keep only the last MAX_HISTORY_LENGTH messages
        if len(self.conversation_history) > MAX_HISTORY_LENGTH:
            self.conversation_history = self.conversation_history[-MAX_HISTORY_LENGTH:]
    
    def _get_conversation_messages(self) -> List[Dict[str, str]]:
        """Get formatted conversation messages for the API."""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add recent conversation history
        recent_messages = self.conversation_history[-CONTEXT_WINDOW_MESSAGES:]
        messages.extend(recent_messages)
        
        return messages
    
    def process_message(
        self,
        user_message: str,
        use_tools: bool = True
    ) -> Dict[str, Any]:
        """
        Process a user message and generate a response.
        
        Args:
            user_message: User's message
            use_tools: Whether to enable tool calling
            
        Returns:
            Dictionary with response and metadata
        """
        # Inject RAG context if memory manager is enabled
        processed_message = user_message
        if self.memory_manager and self.memory_manager.enabled:
            try:
                processed_message = self.memory_manager.inject_context(user_message, user_message)
                logger.debug("RAG context injected into user message")
            except Exception as e:
                logger.warning(f"Error injecting RAG context: {e}, using original message")
                processed_message = user_message
        
        # Update history with processed message
        self._update_conversation_history("user", processed_message)
        
        # Build messages
        messages = self._get_conversation_messages()
        
        # Prepare tool calling if enabled
        tools = None
        tool_choice = None
        if use_tools:
            tools = self.tool_registry.get_tools_schema()
            if tools:
                tool_choice = "auto"
        
        try:
            # Call the API
            logger.debug(f"Sending message to LLM: {user_message[:100]}...")
            
            response = self.client.chat.completions.create(
                model="llama",  # Model name doesn't matter for llama-server
                messages=messages,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
                tools=tools,
                tool_choice=tool_choice
            )
            
            # Get the response
            choice = response.choices[0]
            message = choice.message
            
            # Handle tool calls
            if message.tool_calls:
                return self._handle_tool_calls(message, messages, user_message)
            
            # Regular text response
            response_text = message.content or ""
            self._update_conversation_history("assistant", response_text)
            
            return {
                "success": True,
                "message": response_text,
                "tool_calls": False,
                "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else None
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "error": str(e)
            }
    
    def _handle_tool_calls(
        self,
        message: Any,
        messages: List[Dict[str, str]],
        original_user_message: str
    ) -> Dict[str, Any]:
        """
        Handle tool calls from the LLM response.
        
        Args:
            message: Message object with tool_calls
            messages: Current conversation messages
            original_user_message: Original user message
            
        Returns:
            Dictionary with response and tool results
        """
        # Add assistant message with tool calls to conversation
        tool_call_message = {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]
        }
        messages.append(tool_call_message)
        
        # Execute each tool call
        tool_results = []
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing tool arguments: {e}")
                tool_results.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps({"error": f"Invalid JSON in arguments: {e}"})
                })
                continue
            
            logger.info(f"Executing tool: {function_name} with args: {function_args}")
            
            # Execute tool
            tool_result = self.tool_registry.execute_tool(function_name, **function_args)
            
            # Format result for API
            tool_results.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": json.dumps(tool_result)
            })
        
        # Add tool results to messages
        messages.extend(tool_results)
        
        # Get final response from LLM with tool results
        try:
            response = self.client.chat.completions.create(
                model="llama",
                messages=messages,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS
            )
            
            final_message = response.choices[0].message
            final_text = final_message.content or ""
            
            self._update_conversation_history("assistant", final_text)
            
            return {
                "success": True,
                "message": final_text,
                "tool_calls": True,
                "tools_used": [tc.function.name for tc in message.tool_calls],
                "tool_results": tool_results,
                "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else None
            }
            
        except Exception as e:
            logger.error(f"Error getting final response after tool calls: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Error processing tool results: {str(e)}",
                "tool_calls": True,
                "tool_results": tool_results
            }
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        self.last_interaction_time = time.time()
        logger.info("Conversation history cleared")
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Get conversation history."""
        if limit:
            return self.conversation_history[-limit:]
        return self.conversation_history.copy()

