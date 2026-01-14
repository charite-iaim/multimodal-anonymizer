"""
PNG image processor using Vision LLM + OCR for anonymization.
This approach uses a vision-capable LLM to identify PII text in the image,
then matches the identified text against OCR results to get precise bounding boxes.

Advantages over pure OCR approach:
- LLM can understand context and identify PII more accurately
- LLM can see the entire image and understand relationships between text elements

Advantages over pure Vision approach:
- OCR provides precise bounding boxes (LLM bbox estimates are often inaccurate)
- More reliable redaction coverage
"""

import base64
import io
import json
from pathlib import Path
from PIL import Image, ImageDraw
from datetime import datetime
from typing import List, Optional
from difflib import SequenceMatcher

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

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from ..base_processor import FileProcessor
from ..config import AnonymizerConfig
from ..models import PIIDetectionResult, PIIElement, BoundingBox
from .image_verification_agent import ImageVerificationAgent, VerificationResult


class PIITextItem(BaseModel):
    """A single PII text item identified by the vision LLM."""
    text: str = Field(description="The exact text that contains PII as seen in the image")
    type: str = Field(description="PII category (name, date_of_birth, id_number, address, phone, email, location, dates)")


class VisionPIIResult(BaseModel):
    """Result of PII detection by vision LLM."""
    pii_texts: List[PIITextItem] = Field(
        default_factory=list,
        description="List of texts that contain PII with their types"
    )


class OCRText(BaseModel):
    """Text extracted from OCR with bounding box."""
    text: str
    bbox: BoundingBox
    confidence: float = 0.0


class PNGVisionOCRProcessor(FileProcessor):
    """
    Processor for PNG/JPEG/DICOM images using Vision LLM + OCR.

    This processor combines the best of both approaches:
    1. Vision LLM identifies which text in the image contains PII
    2. OCR provides precise bounding boxes for all text
    3. Smart matching connects LLM-identified PII to OCR bounding boxes
    4. Optional: Verification agent checks for remaining PII after redaction
    """

    def __init__(
        self,
        config: AnonymizerConfig,
        similarity_threshold: float = 0.6,
        enable_verification: bool = True,
        check_over_redaction: bool = False,
        max_verification_rounds: int = 2
    ):
        """
        Initialize the Vision+OCR processor.

        Args:
            config: Anonymizer configuration
            similarity_threshold: Minimum similarity ratio for fuzzy text matching (0.0-1.0)
            enable_verification: If True, run verification agent after initial redaction
            check_over_redaction: If True, also check for over-redaction (needs original image)
            max_verification_rounds: Maximum rounds of verify-and-redact
        """
        super().__init__(config)
        self.similarity_threshold = similarity_threshold
        self.enable_verification = enable_verification
        self.check_over_redaction = check_over_redaction
        self.max_verification_rounds = max_verification_rounds

        if not EASYOCR_AVAILABLE:
            raise ImportError(
                "EasyOCR is required for PNGVisionOCRProcessor. "
                "Install it with: pip install easyocr"
            )

        # Initialize EasyOCR reader
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

        # Initialize Vision LLM for PII identification
        self.vision_llm = AzureChatOpenAI(
            azure_deployment=config.azure_deployment_name,
            azure_endpoint=config.azure_endpoint,
            api_key=config.azure_api_key,
            api_version=config.azure_api_version,
            temperature=config.temperature,
        ).with_structured_output(VisionPIIResult)

        # Initialize Verification Agent (lazy - only if enabled)
        self._verification_agent = None
        if self.enable_verification:
            self._verification_agent = ImageVerificationAgent(
                config=config,
                check_over_redaction=check_over_redaction,
                similarity_threshold=similarity_threshold
            )

    def can_process(self, file_path: Path) -> bool:
        """Check if file is a supported image format (PNG, JPG, JPEG)."""
        return file_path.suffix.lower() in [".png", ".jpg", ".jpeg"]

    def extract_content(self, file_path: Path) -> str:
        """Not used directly - image is processed via vision LLM."""
        return ""

    def anonymize(self, input_path: Path, output_path: Path) -> None:
        """
        Anonymize image using Vision LLM + OCR matching.

        Steps:
        1. Run OCR to extract all text with precise bounding boxes
        2. Send image to Vision LLM to identify which text contains PII
        3. Match LLM-identified PII against OCR results using fuzzy matching
        4. Redact matched regions using OCR bounding boxes
        5. (Optional) Verification agent checks for remaining PII
        6. Save anonymized image

        Args:
            input_path: Path to input image
            output_path: Path to save anonymized image
        """
        print(f"Processing: {input_path.name}")

        # Load image
        image = Image.open(input_path)
        original_image = image.copy()  # Keep for verification
        width, height = image.size
        print(f"Image size: {width}x{height}")

        # Step 1: Extract text using OCR
        print("Running OCR to extract text with bounding boxes...")
        ocr_results = self._extract_text_with_ocr(input_path)
        print(f"Found {len(ocr_results)} text regions via OCR")

        if not ocr_results:
            print("No text found in image, saving original")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
            return

        # Step 2: Send image to Vision LLM to identify PII
        print("Sending image to Vision LLM for PII identification...")
        pii_texts = self._identify_pii_with_vision(image, ocr_results)
        print(f"Vision LLM identified {len(pii_texts)} PII elements")

        # Step 3: Match LLM-identified PII to OCR bounding boxes
        print("Matching PII texts to OCR bounding boxes...")
        pii_elements = self._match_pii_to_ocr(pii_texts, ocr_results)
        print(f"Successfully matched {len(pii_elements)} PII elements to bounding boxes")

        # Step 4: Apply redactions
        if pii_elements:
            image = self._apply_redactions(image.copy(), pii_elements)

        # Step 5: Verification (if enabled)
        verification_result = None
        additional_elements = []
        if self.enable_verification and self._verification_agent is not None:
            print("\n=== Verification Phase ===")
            image, verification_result, additional_elements = self._run_verification(
                image,
                original_image if self.check_over_redaction else None
            )
            pii_elements.extend(additional_elements)

        # Step 6: Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        print(f"Saved anonymized image to: {output_path}")

        # Save JSON with detection results (only if debug mode is enabled)
        if self.config.save_debug_files:
            json_output_path = output_path.with_suffix(".json")
            pii_result = PIIDetectionResult(pii_elements=pii_elements)
            self._save_json_output(
                pii_result,
                input_path,
                output_path,
                json_output_path,
                ocr_results,
                pii_texts,
                verification_result,
                additional_elements
            )
            print(f"Saved detection results to: {json_output_path}")

    def _extract_text_with_ocr(self, image_path: Path) -> List[OCRText]:
        """
        Extract text from image using EasyOCR.

        Args:
            image_path: Path to image file

        Returns:
            List of OCRText objects with text and bounding boxes
        """
        results = self.reader.readtext(str(image_path))

        ocr_texts = []
        for detection in results:
            bbox_points, text, confidence = detection

            # Convert bbox points to x, y, width, height
            x_coords = [point[0] for point in bbox_points]
            y_coords = [point[1] for point in bbox_points]

            x = int(min(x_coords))
            y = int(min(y_coords))
            width = int(max(x_coords) - x)
            height = int(max(y_coords) - y)

            bbox = BoundingBox(x=x, y=y, width=width, height=height)
            ocr_texts.append(OCRText(text=text, bbox=bbox, confidence=confidence))

            print(f"  OCR: '{text}' at ({x}, {y}, {width}, {height}) [conf: {confidence:.2f}]")

        return ocr_texts

    def _extract_text_with_ocr_from_image(self, image: Image.Image) -> List[OCRText]:
        """
        Extract text from PIL Image using EasyOCR.

        Args:
            image: PIL Image object

        Returns:
            List of OCRText objects with text and bounding boxes
        """
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            image.save(tmp_path)

        try:
            return self._extract_text_with_ocr(tmp_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def _identify_pii_with_vision(
        self,
        image: Image.Image,
        ocr_results: List[OCRText]
    ) -> List[PIITextItem]:
        """
        Use Vision LLM to identify which text in the image contains PII.

        Args:
            image: PIL Image object
            ocr_results: List of OCR-extracted texts (provided as context)

        Returns:
            List of PIITextItem objects identifying PII
        """
        # Prepare image for LLM (resize if needed for efficiency)
        max_dimension = 1024
        width, height = image.size

        if width > max_dimension or height > max_dimension:
            scale = min(max_dimension / width, max_dimension / height)
            new_size = (int(width * scale), int(height * scale))
            resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
        else:
            resized_image = image

        # Encode image to base64
        buffer = io.BytesIO()
        # Convert to RGB if necessary (for JPEG compatibility)
        if resized_image.mode in ('L', 'LA', 'P'):
            resized_image = resized_image.convert('RGB')
        resized_image.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Create prompt with OCR context
        ocr_text_list = "\n".join([f"- \"{ocr.text}\"" for ocr in ocr_results])

        prompt = f"""Analyze this medical image and identify ALL text that contains Personal Identifiable Information (PII) that should be redacted for patient privacy.

The following texts were detected in the image by OCR:
{ocr_text_list}

For EACH piece of PII you identify:
1. text: Copy the EXACT text as it appears (must match one of the OCR texts above as closely as possible)
2. type: Classify the PII type

PII categories to identify:
- name: Patient names, physician/doctor names, any person names
- date_of_birth: Dates of birth, DOB
- id_number: Patient IDs, medical record numbers (MRN), accession numbers, study IDs
- address: Physical addresses, street addresses
- location: Hospital names, clinic names, facility names, cities, institutions
- phone: Phone numbers, fax numbers
- email: Email addresses
- dates: Specific dates (admission date, study date, exam date, etc.) that could identify a patient

IMPORTANT:
- Focus on text that could identify a specific patient or person
- Include ALL identifying information, not just obvious names
- Medical record numbers, accession numbers, and study IDs are PII
- Hospital/facility names are PII (location)
- Dates associated with studies or admissions are PII
- Generic medical terms and measurements are NOT PII
"""

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ]
        )

        try:
            result: VisionPIIResult = self.vision_llm.invoke([message])

            for pii in result.pii_texts:
                print(f"  Vision LLM identified: [{pii.type}] \"{pii.text}\"")

            return result.pii_texts

        except Exception as e:
            print(f"Error during Vision LLM PII identification: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity ratio between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity ratio (0.0 to 1.0)
        """
        # Normalize texts
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()

        # Use SequenceMatcher for fuzzy matching
        return SequenceMatcher(None, t1, t2).ratio()

    def _match_pii_to_ocr(
        self,
        pii_texts: List[PIITextItem],
        ocr_results: List[OCRText]
    ) -> List[PIIElement]:
        """
        Match LLM-identified PII texts to OCR results with bounding boxes.

        Uses multiple matching strategies:
        1. Exact match
        2. Substring match (PII text contained in OCR text or vice versa)
        3. Fuzzy match using similarity ratio

        Args:
            pii_texts: List of PII texts identified by Vision LLM
            ocr_results: List of OCR results with bounding boxes

        Returns:
            List of PIIElement objects with matched bounding boxes
        """
        pii_elements = []
        used_ocr_indices = set()

        for pii in pii_texts:
            pii_text = pii.text.strip()
            pii_lower = pii_text.lower()
            best_match: Optional[OCRText] = None
            best_score = 0.0
            best_idx = -1

            for idx, ocr in enumerate(ocr_results):
                if idx in used_ocr_indices:
                    continue

                ocr_text = ocr.text.strip()
                ocr_lower = ocr_text.lower()

                # Strategy 1: Exact match
                if pii_lower == ocr_lower:
                    best_match = ocr
                    best_score = 1.0
                    best_idx = idx
                    break

                # Strategy 2: Substring match
                if pii_lower in ocr_lower or ocr_lower in pii_lower:
                    score = 0.9  # High score for substring match
                    if score > best_score:
                        best_match = ocr
                        best_score = score
                        best_idx = idx
                    continue

                # Strategy 3: Fuzzy match
                similarity = self._calculate_similarity(pii_text, ocr_text)
                if similarity > best_score and similarity >= self.similarity_threshold:
                    best_match = ocr
                    best_score = similarity
                    best_idx = idx

            if best_match:
                used_ocr_indices.add(best_idx)
                pii_element = PIIElement(
                    type=pii.type,
                    text=best_match.text,
                    bbox=best_match.bbox
                )
                pii_elements.append(pii_element)
                match_type = "exact" if best_score == 1.0 else ("substring" if best_score == 0.9 else f"fuzzy({best_score:.2f})")
                print(f"  Matched [{pii.type}]: \"{pii.text}\" -> \"{best_match.text}\" ({match_type})")
            else:
                print(f"  No match found for: \"{pii.text}\" (type: {pii.type})")

        return pii_elements

    def detect_pii_bboxes(self, image: Image.Image) -> List[PIIElement]:
        """
        Detect PII in an image and return bounding boxes without applying redactions.
        Useful for multi-frame processing (e.g., DICOM videos).

        Args:
            image: PIL Image object

        Returns:
            List of PIIElement objects with bounding boxes
        """
        # Extract text using OCR
        ocr_results = self._extract_text_with_ocr_from_image(image)

        if not ocr_results:
            return []

        # Identify PII using Vision LLM
        pii_texts = self._identify_pii_with_vision(image, ocr_results)

        if not pii_texts:
            return []

        # Match PII to OCR bounding boxes
        pii_elements = self._match_pii_to_ocr(pii_texts, ocr_results)

        return pii_elements

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

        # Add padding to ensure complete coverage
        padding = 5

        for element in pii_elements:
            bbox = element.bbox
            if bbox.width > 0 and bbox.height > 0:
                x1 = max(0, bbox.x - padding)
                y1 = max(0, bbox.y - padding)
                x2 = bbox.x + bbox.width + padding
                y2 = bbox.y + bbox.height + padding

                draw.rectangle(
                    [x1, y1, x2, y2],
                    fill="black",
                    outline="black",
                )
                print(f"  Redacted [{element.type}]: \"{element.text}\"")

        return image

    def _run_verification(
        self,
        redacted_image: Image.Image,
        original_image: Optional[Image.Image] = None
    ) -> tuple[Image.Image, Optional[VerificationResult], List[PIIElement]]:
        """
        Run verification agent to check for remaining PII and apply additional redactions.

        Args:
            redacted_image: Image after initial redaction
            original_image: Original image (optional, for over-redaction check)

        Returns:
            Tuple of (final image, verification result, additional elements redacted)
        """
        current_image = redacted_image
        all_additional_elements = []
        final_result = None

        for round_num in range(self.max_verification_rounds):
            print(f"\n  --- Verification Round {round_num + 1}/{self.max_verification_rounds} ---")

            # Verify current state
            result = self._verification_agent.verify_redaction(
                current_image,
                original_image
            )
            final_result = result

            # If clean, we're done
            if result.is_clean:
                print(f"  Verification passed after {round_num + 1} round(s)")
                break

            # Apply additional redactions if remaining PII found
            if result.remaining_pii:
                current_image, additional = self._verification_agent.apply_additional_redactions(
                    current_image,
                    result.remaining_pii,
                    self._extract_text_with_ocr_from_image
                )
                all_additional_elements.extend(additional)

                # If we couldn't redact anything new, stop
                if not additional:
                    print("  Warning: Could not locate remaining PII for redaction")
                    break
            else:
                break

        return current_image, final_result, all_additional_elements

    def _save_json_output(
        self,
        pii_result: PIIDetectionResult,
        input_path: Path,
        output_path: Path,
        json_output_path: Path,
        ocr_results: List[OCRText],
        vision_pii_texts: List[PIITextItem],
        verification_result: Optional[VerificationResult] = None,
        additional_elements: Optional[List[PIIElement]] = None
    ) -> None:
        """
        Save PII detection results as JSON with additional debug info.

        Args:
            pii_result: PIIDetectionResult object
            input_path: Path to original input file
            output_path: Path to anonymized output file
            json_output_path: Path to save JSON output
            ocr_results: OCR results for debugging
            vision_pii_texts: Vision LLM identified texts for debugging
            verification_result: Optional verification result
            additional_elements: Optional list of additionally redacted elements
        """
        output_data = {
            "metadata": {
                "input_file": str(input_path.name),
                "output_file": str(output_path.name),
                "timestamp": datetime.now().isoformat(),
                "processing_method": "vision_ocr",
                "verification_enabled": self.enable_verification,
                "total_ocr_texts": len(ocr_results),
                "total_vision_pii": len(vision_pii_texts),
                "total_matched_pii": len(pii_result.pii_elements)
            },
            "ocr_texts": [
                {
                    "text": ocr.text,
                    "confidence": ocr.confidence,
                    "bbox": {
                        "x": ocr.bbox.x,
                        "y": ocr.bbox.y,
                        "width": ocr.bbox.width,
                        "height": ocr.bbox.height
                    }
                }
                for ocr in ocr_results
            ],
            "vision_identified_pii": [
                {
                    "text": pii.text,
                    "type": pii.type
                }
                for pii in vision_pii_texts
            ],
            "matched_pii_elements": [
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

        # Add verification results if available
        if verification_result is not None:
            output_data["verification"] = {
                "is_clean": verification_result.is_clean,
                "confidence": verification_result.confidence,
                "notes": verification_result.notes,
                "remaining_pii_found": [
                    {
                        "text": pii.text,
                        "type": pii.type,
                        "reason": pii.reason
                    }
                    for pii in verification_result.remaining_pii
                ],
                "over_redactions": [
                    {
                        "description": over.description,
                        "reason": over.reason,
                        "can_recover": over.can_recover
                    }
                    for over in verification_result.over_redactions
                ]
            }

        # Add additional elements redacted during verification
        if additional_elements:
            output_data["verification_additional_redactions"] = [
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
                for element in additional_elements
            ]

        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
