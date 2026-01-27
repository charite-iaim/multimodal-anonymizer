"""
Base class for file processors.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from .config import AnonymizerConfig


class FileProcessor(ABC):
    """Base class for processing and anonymizing files."""

    def __init__(self, config: AnonymizerConfig):
        """
        Initialize the processor.

        Args:
            config: Configuration object with LLM settings
        """
        self.config = config
        self.warnings: list[str] = []

    @abstractmethod
    def can_process(self, file_path: Path) -> bool:
        """
        Check if this processor can handle the given file.

        Args:
            file_path: Path to the file

        Returns:
            True if this processor can handle the file
        """
        pass

    @abstractmethod
    def anonymize(self, input_path: Path, output_path: Path) -> None:
        """
        Anonymize the file and save to output path.

        Args:
            input_path: Path to input file
            output_path: Path to save anonymized file
        """
        pass

    @abstractmethod
    def extract_content(self, file_path: Path) -> Any:
        """
        Extract content from file for LLM processing.

        Args:
            file_path: Path to the file

        Returns:
            Extracted content in a format suitable for LLM
        """
        pass
