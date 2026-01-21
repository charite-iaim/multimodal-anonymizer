"""
File type detector using multimodal LLM to determine data type and processing path.
"""

from enum import Enum
from pathlib import Path
from typing import Optional
import base64

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from .config import AnonymizerConfig
from .llm_factory import create_chat_llm


class DataType(str, Enum):
    """Data types for file processing."""
    TEXT = "text"
    IMAGE = "image"
    UNKNOWN = "unknown"


class FileTypeResult(BaseModel):
    """Result of file type detection."""
    data_type: DataType = Field(description="The detected data type: 'text', 'image', or 'unknown'")
    reasoning: str = Field(description="Brief explanation of why this data type was chosen")
    suggested_processor: str = Field(description="Suggested processor: 'csv', 'ocr', or 'vision'")


class FileTypeDetector:
    """Detector that uses multimodal LLM to determine file data type and processing path."""

    def __init__(self, config: AnonymizerConfig):
        """
        Initialize file type detector.

        Args:
            config: Anonymizer configuration
        """
        self.config = config

        # Initialize LLM for file type detection
        self.llm = create_chat_llm(
            config=config,
            temperature=0.0,  # Use deterministic classification
            structured_output=FileTypeResult,
            use_vision_model=True,  # File type detection may need vision capability
        )

    def detect_file_type(self, file_path: Path) -> FileTypeResult:
        """
        Detect the data type of a file using multimodal LLM.

        Args:
            file_path: Path to the file to analyze

        Returns:
            FileTypeResult with detected data type and suggested processor
        """
        # Read file extension
        file_extension = file_path.suffix.lower()

        # Determine if we can read the file as an image
        is_image_extension = file_extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']

        # Prepare the prompt
        prompt = self._create_detection_prompt(file_path, file_extension)

        # Create message with file content
        if is_image_extension:
            # For image files, encode and send as image
            message_content = self._create_image_message(file_path, prompt)
        else:
            # For text files, read content and send as text
            message_content = self._create_text_message(file_path, prompt)

        message = HumanMessage(content=message_content)

        try:
            result: FileTypeResult = self.llm.invoke([message])
            print(f"File type detection result:")
            print(f"  Data type: {result.data_type}")
            print(f"  Reasoning: {result.reasoning}")
            print(f"  Suggested processor: {result.suggested_processor}")
            return result

        except Exception as e:
            print(f"Error during file type detection: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to unknown
            return FileTypeResult(
                data_type=DataType.UNKNOWN,
                reasoning=f"Detection failed: {str(e)}",
                suggested_processor="none"
            )

    def _create_detection_prompt(self, file_path: Path, file_extension: str) -> str:
        """
        Create prompt for file type detection.

        Args:
            file_path: Path to the file
            file_extension: File extension

        Returns:
            Prompt string
        """
        return f"""Analyze this file and determine its data type for processing in a medical data anonymization pipeline.

File name: {file_path.name}
File extension: {file_extension}

Your task is to classify this file into one of these data types:
1. TEXT - Any text-based document including:
   - Plain text documents (.txt files with narrative content like discharge summaries, clinical notes)
   - Structured tabular data (CSV, TSV files with rows and columns)
2. IMAGE - Image data (scanned documents, photos, medical images, screenshots)
3. UNKNOWN - Cannot determine or unsupported format

Processing paths available:
- For TEXT data: Use 'text' processor for plain text documents, 'csv' processor for tabular data
- For IMAGE data: Use 'ocr' processor (extracts text from images)

Based on the file content, determine:
1. data_type: Which category (text, image, or unknown)
2. reasoning: Why you chose this category (be specific about what you observe in the file)
3. suggested_processor: Which processor to use ('text' for plain text documents, 'csv' for tabular data, 'ocr' for images, 'none' for unknown)

Guidelines:
- If the file contains plain text narrative (discharge summaries, clinical notes, reports) → TEXT → 'text'
- If the file contains tabular data with rows and columns → TEXT → 'csv'
- If the file is a scanned document or medical image → IMAGE → 'ocr'
- If the file extension is .txt with narrative content → TEXT → 'text'
- If the file extension is .csv but contains non-tabular content → base decision on actual content
- If the file extension is .png/.jpg but contains tables/text → IMAGE (use 'ocr' to extract text)
"""

    def _create_image_message(self, file_path: Path, prompt: str) -> list:
        """
        Create message content for image files.

        Args:
            file_path: Path to image file
            prompt: Detection prompt

        Returns:
            Message content list with text and image
        """
        # Read and encode image
        with open(file_path, "rb") as f:
            image_data = f.read()

        image_base64 = base64.b64encode(image_data).decode("utf-8")

        # Determine MIME type from extension
        extension = file_path.suffix.lower()
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff'
        }
        mime_type = mime_types.get(extension, 'image/png')

        return [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_base64}"
                }
            }
        ]

    def _create_text_message(self, file_path: Path, prompt: str) -> list:
        """
        Create message content for text files.

        Args:
            file_path: Path to text file
            prompt: Detection prompt

        Returns:
            Message content list with text and file preview
        """
        try:
            # Read first 1000 characters of the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content_preview = f.read(1000)

            file_content = f"""
File content preview (first 1000 characters):
{content_preview}
"""
            return [
                {"type": "text", "text": prompt},
                {"type": "text", "text": file_content}
            ]

        except UnicodeDecodeError:
            # If file can't be read as text, it might be binary
            return [
                {"type": "text", "text": prompt},
                {"type": "text", "text": f"[Binary file - cannot read as text]"}
            ]
