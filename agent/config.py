"""
Configuration management for the conversation agent.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AgentConfig:
    """Configuration for the conversation agent."""

    # DeepSeek API configuration
    api_key: str
    api_base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"

    # System paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    default_templates_path: str = "templates/objects"
    default_output_path: str = "recognition_results"

    # Model parameters
    device: str = "cuda"
    verbose: bool = False

    # Recognition parameters
    default_confidence_threshold: float = 0.6
    default_text_threshold: float = 0.3

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """
        Load configuration from environment variables.

        Environment variables:
            DEEPSEEK_API_KEY: Required. Your DeepSeek API key.
            DEEPSEEK_API_BASE: Optional. API base URL (default: https://api.deepseek.com)
            DEEPSEEK_MODEL: Optional. Model name (default: deepseek-chat)
            AGENT_DEVICE: Optional. Device to use (default: cuda)
            AGENT_VERBOSE: Optional. Verbose mode (default: false)

        Returns:
            AgentConfig instance

        Raises:
            ValueError: If DEEPSEEK_API_KEY is not set
        """
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY environment variable not set.\n"
                "Please set it with: export DEEPSEEK_API_KEY='your_api_key'"
            )

        return cls(
            api_key=api_key,
            api_base_url=os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
            device=os.environ.get("AGENT_DEVICE", "cuda"),
            verbose=os.environ.get("AGENT_VERBOSE", "").lower() in ("true", "1", "yes")
        )

    def get_templates_path(self, custom_path: Optional[str] = None) -> Path:
        """Get resolved templates path."""
        path = Path(custom_path) if custom_path else Path(self.default_templates_path)
        if not path.is_absolute():
            path = self.project_root / path
        return path

    def get_output_path(self, custom_path: Optional[str] = None) -> Path:
        """Get resolved output path."""
        path = Path(custom_path) if custom_path else Path(self.default_output_path)
        if not path.is_absolute():
            path = self.project_root / path
        return path
