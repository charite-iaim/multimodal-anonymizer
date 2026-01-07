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
    model_name: str = "accounts/fireworks/models/llama-v3p2-11b-vision-instruct"
    temperature: float = 0.0

    # Input/Output directories
    input_dir: str = "data/input"
    output_dir: str = "data/output"

    # Image processing configuration
    # Maximum dimension for vision model input (1024 for Mistral, 2000 for GPT-4 Vision)
    max_image_dimension: int = 1024

    # Fireworks API Configuration
    fireworks_api_key: Optional[str] = None
    fireworks_base_url: str = "https://api.fireworks.ai/inference/v1"

    def __post_init__(self):
        """Load Fireworks configuration from environment if not provided."""
        if self.fireworks_api_key is None:
            self.fireworks_api_key = os.getenv("FIREWORKS_API_KEY")

        # Load model name from environment if set
        env_model_name = os.getenv("FIREWORKS_MODEL_NAME")
        if env_model_name:
            self.model_name = env_model_name

        # Validate required fields
        if not self.fireworks_api_key:
            raise ValueError("FIREWORKS_API_KEY must be set in environment or config")
