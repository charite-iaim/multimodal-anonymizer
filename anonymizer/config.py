"""
Configuration for the anonymization pipeline.
"""

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


@dataclass
class AnonymizerConfig:
    """Configuration for the anonymization system."""

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

    def __post_init__(self):
        """Load Azure configuration from environment if not provided."""
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

        # Validate required fields
        if not self.azure_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT must be set in environment or config")

        if not self.azure_api_key:
            raise ValueError("AZURE_OPENAI_API_KEY must be set in environment or config")

        if not self.azure_deployment_name:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME must be set in environment or config")
