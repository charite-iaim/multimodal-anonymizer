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
LLMProvider = Literal["azure", "fireworks", "poe", "openrouter"]


@dataclass
class AnonymizerConfig:
    """Configuration for the anonymization system."""

    # LLM Provider Selection
    llm_provider: LLMProvider = "poe"  # "azure", "fireworks", "poe", "openrouter"

    # LLM Configuration
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0

    # Input/Output directories
    input_dir: str = "data/input"
    output_dir: str = "data/output"

    # Debug configuration
    save_debug_files: bool = False  # If True, save JSON metadata and intermediate files

    # Image processing configuration
    # Maximum dimension for vision model input (1024 for Mistral, 2000 for GPT-4 Vision)
    max_image_dimension: int = 1024

    # Azure OpenAI Configuration
    azure_endpoint: Optional[str] = None
    azure_api_key: Optional[str] = None
    azure_api_version: str = "2024-08-01-preview"
    azure_deployment_name: Optional[str] = None

    # Fireworks AI Configuration
    fireworks_api_key: Optional[str] = None
    fireworks_model: str = "accounts/fireworks/models/glm-4p7"
    fireworks_vision_model: str = "accounts/fireworks/models/glm-4p7"

    # Poe API Configuration (OpenAI-compatible)
    poe_api_key: Optional[str] = None
    poe_model: str = "Claude-Sonnet-4"
    poe_vision_model: str = "Claude-Sonnet-4"
    poe_base_url: str = "https://api.poe.com/v1"

    # OpenRouter Configuration (OpenAI-compatible)
    openrouter_api_key: Optional[str] = None
    openrouter_model: str = "anthropic/claude-sonnet-4"
    openrouter_vision_model: str = "anthropic/claude-sonnet-4"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    def __post_init__(self):
        """Load configuration from environment if not provided."""
        # Load LLM provider from environment if set
        env_provider = os.getenv("LLM_PROVIDER")
        if env_provider and env_provider.lower() in ("azure", "fireworks", "poe", "openrouter"):
            self.llm_provider = env_provider.lower()

        # Load Azure configuration
        if self.azure_endpoint is None:
            self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        if self.azure_api_key is None:
            self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")

        if self.azure_deployment_name is None:
            self.azure_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

        # Load API version from environment if set
        env_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        if env_api_version:
            self.azure_api_version = env_api_version

        # Load Fireworks configuration
        if self.fireworks_api_key is None:
            self.fireworks_api_key = os.getenv("FIREWORKS_API_KEY")

        env_fireworks_model = os.getenv("FIREWORKS_MODEL")
        if env_fireworks_model:
            self.fireworks_model = env_fireworks_model

        env_fireworks_vision_model = os.getenv("FIREWORKS_VISION_MODEL")
        if env_fireworks_vision_model:
            self.fireworks_vision_model = env_fireworks_vision_model

        # Load Poe configuration
        if self.poe_api_key is None:
            self.poe_api_key = os.getenv("POE_API_KEY")

        env_poe_model = os.getenv("POE_MODEL")
        if env_poe_model:
            self.poe_model = env_poe_model

        env_poe_vision_model = os.getenv("POE_VISION_MODEL")
        if env_poe_vision_model:
            self.poe_vision_model = env_poe_vision_model

        env_poe_base_url = os.getenv("POE_BASE_URL")
        if env_poe_base_url:
            self.poe_base_url = env_poe_base_url

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

        # Validate required fields based on provider
        if self.llm_provider == "azure":
            if not self.azure_endpoint:
                raise ValueError("AZURE_OPENAI_ENDPOINT must be set in environment or config")

            if not self.azure_api_key:
                raise ValueError("AZURE_OPENAI_API_KEY must be set in environment or config")

            if not self.azure_deployment_name:
                raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME must be set in environment or config")

        elif self.llm_provider == "fireworks":
            if not self.fireworks_api_key:
                raise ValueError("FIREWORKS_API_KEY must be set in environment or config")

        elif self.llm_provider == "poe":
            if not self.poe_api_key:
                raise ValueError("POE_API_KEY must be set in environment or config (get it from https://poe.com/api_key)")

        elif self.llm_provider == "openrouter":
            if not self.openrouter_api_key:
                raise ValueError("OPENROUTER_API_KEY must be set in environment or config (get it from https://openrouter.ai/keys)")
