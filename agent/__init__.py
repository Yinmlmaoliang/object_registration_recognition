"""
Object Registration & Recognition Conversation Agent.

This module provides a terminal-based conversation agent that uses
DeepSeek API with function calling to interact with the object
registration and recognition system.

Usage:
    export DEEPSEEK_API_KEY="your_key"
    python -m agent.main
"""

from .config import AgentConfig
from .agent_service import AgentService
from .conversation_agent import ConversationAgent

__all__ = [
    'AgentConfig',
    'AgentService',
    'ConversationAgent'
]
