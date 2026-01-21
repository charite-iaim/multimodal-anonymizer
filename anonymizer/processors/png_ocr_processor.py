"""
PNG image processor using OCR + LLM for anonymization.
This approach uses OCR to extract text with precise bounding boxes,
then uses LLM to classify which text contains PII.
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw
from datetime import datetime
from typing import List, Dict, Any

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from ..base_processor import FileProcessor
from ..config import AnonymizerConfig
from ..llm_factory import create_chat_llm
from ..models import PIIDetectionResult, PIIElement, BoundingBox
from ..prompt_config import PromptConfig, DEFAULT_PROMPT_CONFIG


class OCRText(BaseModel):
    """Text extracted from OCR with bounding box."""
    text: str
    bbox: BoundingBox


class PIITextItem(BaseModel):
    """A single PII text item."""
    text: str = Field(description="The exact matching text from OCR")
    type: str = Field(description="PII category (name, date_of_birth, id_number, address, phone, email)")


class PIIClassificationResult(BaseModel):
    """Result of PII classification for OCR text."""

    pii_texts: List[PIITextItem] = Field(
        description="List of texts that contain PII with their types"
    )


class PNGOCRProcessor(FileProcessor):
    """Processor for PNG images using OCR + LLM classification."""

    def __init__(self, config: AnonymizerConfig, prompt_config: PromptConfig = None):
        """Initialize PNG OCR processor."""
        super().__init__(config)
        self.prompt_config = prompt_config or DEFAULT_PROMPT_CONFIG

        if not EASYOCR_AVAILABLE:
            raise ImportError(
                "EasyOCR is required for PNGOCRProcessor. "
                "Install it with: pip install easyocr"
            )

        # Initialize EasyOCR reader (English)
        # Check if GPU is available
        use_gpu = False
        if TORCH_AVAILABLE:
            use_gpu = torch.cuda.is_available()
            if use_gpu:
                print("GPU detected - using GPU for OCR processing")
            else:
                print("No GPU detected - using CPU for OCR processing")
        else:
            print("PyTorch not available - using CPU for OCR processing")

        print("Initializing EasyOCR reader...")
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)

        # Initialize LLM for PII classification
        self.llm = create_chat_llm(
            config=config,
            structured_output=PIIClassificationResult,
        )

    def can_process(self, file_path: Path) -> bool:
        """Check if file is a supported image format (PNG, JPG, JPEG)."""
        return file_path.suffix.lower() in [".png", ".jpg", ".jpeg"]

    def extract_content(self, file_path: Path) -> str:
        """Not used in OCR processor."""
        return ""

    def anonymize(self, input_path: Path, output_path: Path) -> None:
        """
        Anonymize PNG image using OCR + LLM classification.

        Steps:
        1. Run OCR to extract all text with precise bounding boxes
        2. Use LLM to classify which texts contain PII
        3. Redact identified PII texts using OCR bounding boxes
        4. Save anonymized image

        Args:
            input_path: Path to input PNG
            output_path: Path to save anonymized PNG
        """
        print(f"Processing: {input_path.name}")

        # Load image
        image = Image.open(input_path)
        width, height = image.size
        print(f"Image size: {width}x{height}")

        # Step 1: Extract text using OCR
        print("Running OCR to extract text...")
        ocr_results = self._extract_text_with_ocr(input_path)
        print(f"Found {len(ocr_results)} text regions")

        if not ocr_results:
            print("No text found in image, saving original")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
            return

        # Step 2: Classify which texts contain PII using LLM
        print("Classifying PII using LLM...")
        pii_elements = self._classify_pii(ocr_results)
        print(f"Identified {len(pii_elements)} PII elements")

        # Step 3: Apply redactions
        if pii_elements:
            image = self._apply_redactions(image, pii_elements)

        # Step 4: Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        print(f"Saved anonymized image to: {output_path}")

        # Save JSON with detection results (only if debug mode is enabled)
        if self.config.save_debug_files:
            json_output_path = output_path.with_suffix(".json")
            pii_result = PIIDetectionResult(pii_elements=pii_elements)
            self._save_json_output(pii_result, input_path, output_path, json_output_path)
            print(f"Saved detection results to: {json_output_path}")

    def _extract_text_with_ocr(self, image_path: Path) -> List[OCRText]:
        """
        Extract text from image using EasyOCR.

        Args:
            image_path: Path to image file

        Returns:
            List of OCRText objects with text and bounding boxes
        """
        # Run OCR
        results = self.reader.readtext(str(image_path))

        ocr_texts = []
        for detection in results:
            bbox_points, text, confidence = detection

            # Convert bbox points to x, y, width, height
            # bbox_points is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            x_coords = [point[0] for point in bbox_points]
            y_coords = [point[1] for point in bbox_points]

            x = int(min(x_coords))
            y = int(min(y_coords))
            width = int(max(x_coords) - x)
            height = int(max(y_coords) - y)

            bbox = BoundingBox(x=x, y=y, width=width, height=height)
            ocr_texts.append(OCRText(text=text, bbox=bbox))

            print(f"  OCR: '{text}' at ({x}, {y}, {width}, {height}) [confidence: {confidence:.2f}]")

        return ocr_texts

    def _classify_pii(self, ocr_texts: List[OCRText]) -> List[PIIElement]:
        """
        Use LLM to classify which OCR texts contain PII.

        Args:
            ocr_texts: List of texts extracted by OCR

        Returns:
            List of PIIElement objects for texts classified as PII
        """
        if not ocr_texts:
            return []

        # Prepare text list for LLM
        text_list = "\n".join([f"{i+1}. {ocr.text}" for i, ocr in enumerate(ocr_texts)])

        # Get prompt from config (reusing PDF prompt since it has same structure)
        prompt = self.prompt_config.get_pdf_anonymization_prompt(text_list)

        message = HumanMessage(content=prompt)

        try:
            classification: PIIClassificationResult = self.llm.invoke([message])

            # Match classified PII back to OCR results
            pii_elements = []
            for pii_text in classification.pii_texts:
                text_to_find = pii_text.text
                pii_type = pii_text.type

                # Find matching OCR text
                for ocr in ocr_texts:
                    if ocr.text.strip() == text_to_find.strip():
                        pii_element = PIIElement(
                            type=pii_type,
                            text=ocr.text,
                            bbox=ocr.bbox
                        )
                        pii_elements.append(pii_element)
                        print(f"  Classified as PII: {pii_type} - '{ocr.text}'")
                        break
                else:
                    # Try fuzzy matching if exact match fails
                    for ocr in ocr_texts:
                        if text_to_find.lower() in ocr.text.lower() or ocr.text.lower() in text_to_find.lower():
                            pii_element = PIIElement(
                                type=pii_type,
                                text=ocr.text,
                                bbox=ocr.bbox
                            )
                            pii_elements.append(pii_element)
                            print(f"  Classified as PII (fuzzy): {pii_type} - '{ocr.text}'")
                            break

            return pii_elements

        except Exception as e:
            print(f"Error during PII classification: {e}")
            import traceback
            traceback.print_exc()
            return []

    def detect_pii_bboxes(self, image: Image.Image) -> List[PIIElement]:
        """
        Detect PII in an image and return bounding boxes without applying redactions.

        Args:
            image: PIL Image object

        Returns:
            List of PIIElement objects with bounding boxes
        """
        # Save image temporarily for OCR
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            image.save(tmp_path)

        try:
            # Extract text using OCR
            ocr_results = self._extract_text_with_ocr(tmp_path)

            if not ocr_results:
                return []

            # Classify which texts contain PII
            pii_elements = self._classify_pii(ocr_results)
            return pii_elements

        finally:
            # Clean up temp file
            if tmp_path.exists():
                tmp_path.unlink()

    def _apply_redactions(self, image: Image.Image, pii_elements: List[PIIElement]) -> Image.Image:
        """
        Apply black rectangles to redact PII regions.

        Args:
            image: PIL Image object
            pii_elements: List of PIIElement objects with bounding boxes

        Returns:
            Image with redactions applied
        """
        draw = ImageDraw.Draw(image)

        # Add padding to ensure complete coverage (in pixels)
        padding = 5

        for element in pii_elements:
            bbox = element.bbox
            if bbox.width > 0 and bbox.height > 0:
                # Expand bbox with padding
                x1 = max(0, bbox.x - padding)
                y1 = max(0, bbox.y - padding)
                x2 = bbox.x + bbox.width + padding
                y2 = bbox.y + bbox.height + padding

                # Draw black rectangle
                draw.rectangle(
                    [x1, y1, x2, y2],
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
                "processing_method": "ocr",
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
