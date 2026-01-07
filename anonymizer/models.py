"""
Pydantic models for structured LLM output.
"""

from pydantic import BaseModel, Field
from typing import List


class BoundingBox(BaseModel):
    """Bounding box coordinates for a PII element in pixel coordinates."""

    x: int = Field(..., description="X coordinate of top-left corner in pixels")
    y: int = Field(..., description="Y coordinate of top-left corner in pixels")
    width: int = Field(..., description="Width of the bounding box in pixels")
    height: int = Field(..., description="Height of the bounding box in pixels")


class PIIElement(BaseModel):
    """A single PII element detected in the image."""

    type: str = Field(description="Type of PII (e.g., 'name', 'date_of_birth', 'address')")
    text: str = Field(description="The actual text content of the PII")
    bbox: BoundingBox = Field(description="Bounding box coordinates")


class PIIDetectionResult(BaseModel):
    """Complete result of PII detection in an image."""

    pii_elements: List[PIIElement] = Field(
        default_factory=list,
        description="List of detected PII elements with their bounding boxes"
    )
