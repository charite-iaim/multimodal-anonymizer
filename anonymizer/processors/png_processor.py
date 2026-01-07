"""
PNG image processor for anonymization using vision-capable LLM.
"""

import base64
import io
import json
from pathlib import Path
from PIL import Image, ImageDraw
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from ..base_processor import FileProcessor
from ..config import AnonymizerConfig
from ..models import PIIDetectionResult, PIIElement, BoundingBox


class PNGProcessor(FileProcessor):
    """Processor for PNG images using vision LLM."""

    def __init__(self, config: AnonymizerConfig):
        """Initialize PNG processor with Fireworks vision-capable LLM."""
        super().__init__(config)
        self.llm = ChatOpenAI(
            model=config.model_name,
            api_key=config.fireworks_api_key,
            base_url=config.fireworks_base_url,
            temperature=config.temperature,
        ).with_structured_output(PIIDetectionResult)

    def can_process(self, file_path: Path) -> bool:
        """Check if file is a PNG image."""
        return file_path.suffix.lower() == ".png"

    def extract_content(self, file_path: Path) -> str:
        """
        Encode PNG image to base64 for LLM vision input.

        Args:
            file_path: Path to PNG file

        Returns:
            Base64 encoded image string
        """
        with open(file_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def anonymize(self, input_path: Path, output_path: Path) -> None:
        """
        Anonymize PNG image by detecting and redacting PII.

        Args:
            input_path: Path to input PNG
            output_path: Path to save anonymized PNG
        """
        print(f"Processing: {input_path.name}")

        # Load and resize image to max 512x512
        original_image = Image.open(input_path)
        original_width, original_height = original_image.size
        print(f"Original image size: {original_width}x{original_height}")

        # Resize to max 512x512 while maintaining aspect ratio
        max_dimension = 512
        if original_width > max_dimension or original_height > max_dimension:
            scale_factor = min(max_dimension / original_width, max_dimension / original_height)
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            resized_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"Resized to: {new_width}x{new_height}")
        else:
            resized_image = original_image
            new_width, new_height = original_width, original_height

        # Encode resized image for LLM
        buffer = io.BytesIO()
        resized_image.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Create prompt for PII detection
        prompt = f"""Analyze this medical image (size: {new_width}x{new_height} pixels) and identify all personal identifiable information (PII) that needs to be redacted.

Provide bounding box coordinates in PIXELS:
- x, y: top-left corner position in pixels
- width, height: dimensions in pixels
- Make sure the bounding boxes fully cover the text

PII to identify:
- Patient names (including labels like "Name:")
- Physician/doctor names
- Dates of birth
- Patient ID numbers
- Medical record numbers
- Addresses
- Phone numbers
- Email addresses

For each PII element, provide:
1. type: The category (e.g., "name", "date_of_birth", "id_number")
2. text: The actual text content you see
3. bbox: Bounding box in pixels (x, y, width, height)
"""

        # Call LLM with vision and structured output
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ]
        )

        print("Analyzing image with LLM...")
        try:
            # Get structured output from LLM
            pii_result: PIIDetectionResult = self.llm.invoke([message])
            print(f"Found {len(pii_result.pii_elements)} PII elements")

            # Apply redactions to resized image
            if pii_result.pii_elements:
                resized_image = self._apply_redactions(resized_image.copy(), pii_result.pii_elements)

            # Save anonymized resized image
            output_path.parent.mkdir(parents=True, exist_ok=True)
            resized_image.save(output_path)
            print(f"Saved anonymized resized image to: {output_path}")

            # Scale bounding boxes back to original dimensions and apply to original image
            if pii_result.pii_elements:
                scale_x = original_width / new_width
                scale_y = original_height / new_height

                # Create scaled bounding boxes for original image
                scaled_elements = []
                for element in pii_result.pii_elements:
                    scaled_bbox = BoundingBox(
                        x=int(element.bbox.x * scale_x),
                        y=int(element.bbox.y * scale_y),
                        width=int(element.bbox.width * scale_x),
                        height=int(element.bbox.height * scale_y)
                    )
                    scaled_element = PIIElement(
                        type=element.type,
                        text=element.text,
                        bbox=scaled_bbox
                    )
                    scaled_elements.append(scaled_element)

                # Apply redactions to original image
                original_image = self._apply_redactions(original_image, scaled_elements)

            # Save anonymized original image
            original_output_path = output_path.with_name(f"{output_path.stem}_original{output_path.suffix}")
            original_image.save(original_output_path)
            print(f"Saved anonymized original image to: {original_output_path}")

            # Save JSON with bounding boxes (from resized image)
            json_output_path = output_path.with_suffix(".json")
            self._save_json_output(pii_result, input_path, output_path, json_output_path)
            print(f"Saved detection results to: {json_output_path}")

        except Exception as e:
            import traceback
            print(f"Error during processing: {e}")
            print(traceback.format_exc())
            # Save resized image if processing fails
            output_path.parent.mkdir(parents=True, exist_ok=True)
            resized_image.save(output_path)

    def _apply_redactions(self, image: Image.Image, pii_elements: list[PIIElement]) -> Image.Image:
        """
        Apply black rectangles to redact PII regions.

        Args:
            image: PIL Image object
            pii_elements: List of PIIElement objects with bounding boxes

        Returns:
            Image with redactions applied
        """
        draw = ImageDraw.Draw(image)

        for element in pii_elements:
            bbox = element.bbox
            if bbox.width > 0 and bbox.height > 0:
                # Draw black rectangle
                draw.rectangle(
                    [bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height],
                    fill="black",
                    outline="black",
                )
                print(f"  Redacted {element.type}: {element.text}")

        return image

    def _save_json_output(
        self,
        pii_result: PIIDetectionResult,
        input_path: Path,
        output_path: Path,
        json_output_path: Path
    ) -> None:
        """
        Save PII detection results as JSON.

        Args:
            pii_result: PIIDetectionResult object
            input_path: Path to original input file
            output_path: Path to anonymized output file
            json_output_path: Path to save JSON output
        """
        output_data = {
            "metadata": {
                "input_file": str(input_path.name),
                "output_file": str(output_path.name),
                "timestamp": datetime.now().isoformat(),
                "total_pii_elements": len(pii_result.pii_elements)
            },
            "pii_elements": [
                {
                    "type": element.type,
                    "text": element.text,
                    "bbox": {
                        "x": element.bbox.x,
                        "y": element.bbox.y,
                        "width": element.bbox.width,
                        "height": element.bbox.height
                    }
                }
                for element in pii_result.pii_elements
            ]
        }

        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
