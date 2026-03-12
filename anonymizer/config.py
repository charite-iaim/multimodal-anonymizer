"""
Configuration for the anonymization pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Supported LLM providers
LLMProvider = Literal["openrouter", "local"]


@dataclass
class AnonymizerConfig:
    """Configuration for the anonymization system."""

    # LLM Provider Selection
    llm_provider: LLMProvider = "openrouter"

    # LLM Configuration
    temperature: float = 0.0

    # Input/Output directories
    input_dir: str = "data/input"
    output_dir: str = "data/output"

    # Debug configuration
    save_debug_files: bool = False  # If True, save JSON metadata and intermediate files

    # Image processing configuration
    max_image_dimension: int = 1024

    # OpenRouter Configuration (OpenAI-compatible)
    openrouter_api_key: Optional[str] = None
    openrouter_model: str = "anthropic/claude-sonnet-4"
    openrouter_vision_model: str = "anthropic/claude-sonnet-4"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Local LLM Configuration (OpenAI-compatible: Ollama, LM Studio, vLLM, LocalAI, etc.)
    local_base_url: Optional[str] = None  # e.g., http://localhost:11434/v1
    local_model: str = "llama3.2"  # Model name as known by the local server
    local_vision_model: Optional[str] = None  # Optional separate vision model
    local_api_key: Optional[str] = None  # Most local servers don't need this
    local_thinking: bool = False  # Enable/disable thinking mode for reasoning models (e.g., kimi-k2.5)

    def __post_init__(self):
        """Load configuration from environment if not provided."""
        # Load LLM provider from environment if set
        env_provider = os.getenv("LLM_PROVIDER")
        if env_provider and env_provider.lower() in ("openrouter", "local"):
            self.llm_provider = env_provider.lower()

        # Load OpenRouter configuration
        if self.openrouter_api_key is None:
            self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

        env_openrouter_model = os.getenv("OPENROUTER_MODEL")
        if env_openrouter_model:
            self.openrouter_model = env_openrouter_model

        env_openrouter_vision_model = os.getenv("OPENROUTER_VISION_MODEL")
        if env_openrouter_vision_model:
            self.openrouter_vision_model = env_openrouter_vision_model

        env_openrouter_base_url = os.getenv("OPENROUTER_BASE_URL")
        if env_openrouter_base_url:
            self.openrouter_base_url = env_openrouter_base_url

        # Load Local LLM configuration
        env_local_base_url = os.getenv("LOCAL_BASE_URL")
        if env_local_base_url:
            self.local_base_url = env_local_base_url

        env_local_model = os.getenv("LOCAL_MODEL")
        if env_local_model:
            self.local_model = env_local_model

        env_local_vision_model = os.getenv("LOCAL_VISION_MODEL")
        if env_local_vision_model:
            self.local_vision_model = env_local_vision_model

        env_local_api_key = os.getenv("LOCAL_API_KEY")
        if env_local_api_key:
            self.local_api_key = env_local_api_key

        env_local_thinking = os.getenv("LOCAL_THINKING")
        if env_local_thinking is not None:
            self.local_thinking = env_local_thinking.lower() in ("true", "1", "yes")

        # Validate required fields based on provider
        if self.llm_provider == "openrouter":
            if not self.openrouter_api_key:
                raise ValueError("OPENROUTER_API_KEY must be set in environment or config (get it from https://openrouter.ai/keys)")

        elif self.llm_provider == "local":
            if not self.local_base_url:
                raise ValueError("local_base_url must be set for local LLM provider (e.g., http://localhost:11434/v1 for Ollama)")
