"""
Image Verification Agent for post-redaction PII detection.

This agent verifies that redacted images no longer contain visible PII.
If PII is still detected, it identifies it and applies additional redactions.

The agent uses a multi-step approach:
1. Vision LLM scans the redacted image for any remaining PII
2. If PII found: OCR + matching to get precise bounding boxes for additional redaction
3. Face detection tool can be called to detect and redact any visible faces

Optional: Over-redaction detection (checks if non-PII was unnecessarily redacted)
"""

import base64
import io
from typing import List, Optional, Tuple
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, ToolMessage

from ..config import AnonymizerConfig
from ..llm_factory import create_chat_llm
from ..models import PIIElement, BoundingBox
from ..prompt_config import PromptConfig, DEFAULT_PROMPT_CONFIG
from ..tools.face_detection_tool import (
    detect_faces,
    detect_faces_from_pil,
    redact_faces_in_pil_image,
    get_face_bounding_boxes,
)


class RemainingPII(BaseModel):
    """A PII element that was not properly redacted."""
    text: str = Field(description="The exact text that contains PII still visible in the image")
    type: str = Field(description="PII category (name, date_of_birth, id_number, address, phone, email, location, dates)")
    reason: str = Field(description="Why this is considered PII that should be redacted")


class OverRedaction(BaseModel):
    """Information that was unnecessarily redacted."""
    description: str = Field(description="Description of what was over-redacted")
    reason: str = Field(description="Why this should NOT have been redacted")
    can_recover: bool = Field(description="Whether this can potentially be recovered from context")


class VerificationResult(BaseModel):
    """Result of the verification check."""
    is_clean: bool = Field(description="True if no remaining PII was found")
    remaining_pii: List[RemainingPII] = Field(
        default_factory=list,
        description="List of PII elements that are still visible and need additional redaction"
    )
    over_redactions: List[OverRedaction] = Field(
        default_factory=list,
        description="List of items that were unnecessarily redacted (optional analysis)"
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence score of the verification (0.0-1.0)"
    )
    notes: str = Field(
        default="",
        description="Additional notes about the verification"
    )


class ImageVerificationAgent:
    """
    Agent that verifies redacted images for remaining PII.

    This agent performs a second-pass analysis on redacted images to ensure
    no PII was missed during the initial redaction process. It can also
    detect and redact faces using the face detection tool.
    """

    def __init__(
        self,
        config: AnonymizerConfig,
        check_over_redaction: bool = False,
        similarity_threshold: float = 0.6,
        prompt_config: PromptConfig = None,
        verification_prompt_getter: callable = None,
        enable_face_detection: bool = True
    ):
        """
        Initialize the verification agent.

        Args:
            config: Anonymizer configuration
            check_over_redaction: If True, also check for over-redaction (requires original image)
            similarity_threshold: Minimum similarity for fuzzy text matching
            prompt_config: Optional custom prompt configuration
            verification_prompt_getter: Optional callable that returns the verification prompt.
                                       If provided, overrides prompt_config.get_image_verification_prompt()
            enable_face_detection: If True, enables the face detection tool for the verification LLM
        """
        self.config = config
        self.check_over_redaction = check_over_redaction
        self.similarity_threshold = similarity_threshold
        self.prompt_config = prompt_config or DEFAULT_PROMPT_CONFIG
        self._verification_prompt_getter = verification_prompt_getter
        self.enable_face_detection = enable_face_detection

        # Initialize Vision LLM for verification (structured output mode)
        self.vision_llm = create_chat_llm(
            config=config,
            temperature=0,  # Use 0 temperature for consistent verification
            structured_output=VerificationResult,
            use_vision_model=True,
        )

        # Initialize Vision LLM with face detection tool (agentic mode)
        if enable_face_detection:
            self.vision_llm_with_tools = create_chat_llm(
                config=config,
                temperature=0,
                use_vision_model=True,
                tools=[detect_faces],
            )
        else:
            self.vision_llm_with_tools = None

    def verify_redaction(
        self,
        redacted_image: Image.Image,
        original_image: Optional[Image.Image] = None
    ) -> VerificationResult:
        """
        Verify that a redacted image has no remaining PII.

        Args:
            redacted_image: The image after redaction
            original_image: Optional original image for over-redaction check

        Returns:
            VerificationResult with findings
        """
        print("  Verification Agent: Analyzing redacted image for remaining PII...")

        # Prepare redacted image for LLM
        redacted_b64 = self._image_to_base64(redacted_image)

        # Build prompt based on whether we're checking over-redaction
        if self.check_over_redaction and original_image is not None:
            original_b64 = self._image_to_base64(original_image)
            result = self._verify_with_original(redacted_b64, original_b64)
        else:
            result = self._verify_redacted_only(redacted_b64)

        # Log results
        if result.is_clean:
            print("  Verification Agent: Image is clean - no remaining PII detected")
        else:
            print(f"  Verification Agent: Found {len(result.remaining_pii)} remaining PII elements!")
            for pii in result.remaining_pii:
                print(f"    - [{pii.type}] \"{pii.text}\": {pii.reason}")

        if result.over_redactions:
            print(f"  Verification Agent: Detected {len(result.over_redactions)} potential over-redactions")
            for over in result.over_redactions:
                print(f"    - {over.description}: {over.reason}")

        return result

    def _image_to_base64(self, image: Image.Image, max_dimension: int = 1024) -> str:
        """Convert PIL Image to base64 string, resizing if needed."""
        width, height = image.size

        if width > max_dimension or height > max_dimension:
            scale = min(max_dimension / width, max_dimension / height)
            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Convert to RGB if necessary
        if image.mode in ('L', 'LA', 'P'):
            image = image.convert('RGB')

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _verify_redacted_only(self, redacted_b64: str) -> VerificationResult:
        """Verify redacted image without comparing to original."""
        # Use custom prompt getter if provided, otherwise use prompt_config
        if self._verification_prompt_getter is not None:
            prompt = self._verification_prompt_getter()
        else:
            prompt = self.prompt_config.get_image_verification_prompt()

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{redacted_b64}"},
                },
            ]
        )

        try:
            result: VerificationResult = self.vision_llm.invoke([message])
            return result
        except Exception as e:
            print(f"  Verification Agent ERROR: {e}")
            # Return a cautious result on error
            return VerificationResult(
                is_clean=False,
                confidence=0.0,
                notes=f"Verification failed with error: {str(e)}"
            )

    def _verify_with_original(self, redacted_b64: str, original_b64: str) -> VerificationResult:
        """Verify redacted image by comparing to original (includes over-redaction check)."""
        prompt = """Analyze these two medical images:
1. FIRST IMAGE: The ORIGINAL image (before redaction)
2. SECOND IMAGE: The REDACTED image (after redaction - black rectangles cover sensitive info)

Perform TWO checks:

## CHECK 1: Remaining PII (CRITICAL)
Look at the REDACTED image and identify ANY PII that is STILL VISIBLE:
- Patient names, physician names
- Dates of birth
- Patient IDs, medical record numbers, accession numbers
- Addresses
- Hospital/facility names
- Phone numbers, emails
- Dates that could identify a patient

Report these as remaining_pii. Set is_clean=FALSE if any PII remains.

## CHECK 2: Over-Redaction (INFORMATIONAL)
Compare the original to the redacted image and identify if anything was UNNECESSARILY redacted:
- Medical terms that are not PII
- Measurements and values
- Generic labels
- Information that is important for medical interpretation

Report these as over_redactions. Note: Some over-redaction is acceptable to ensure privacy.

IMPORTANT:
- Privacy is the priority - under-redaction is worse than over-redaction
- Only flag over-redaction if it significantly impacts medical utility
- Set can_recover=true if the info might be recoverable from context"""

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{original_b64}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{redacted_b64}"},
                },
            ]
        )

        try:
            result: VerificationResult = self.vision_llm.invoke([message])
            return result
        except Exception as e:
            print(f"  Verification Agent ERROR: {e}")
            return VerificationResult(
                is_clean=False,
                confidence=0.0,
                notes=f"Verification failed with error: {str(e)}"
            )

    def apply_additional_redactions(
        self,
        image: Image.Image,
        remaining_pii: List[RemainingPII],
        ocr_func: callable
    ) -> Tuple[Image.Image, List[PIIElement]]:
        """
        Apply additional redactions for remaining PII.

        This method uses OCR to find precise bounding boxes for the remaining PII
        and applies redactions.

        Args:
            image: The current (partially redacted) image
            remaining_pii: List of PII that needs to be redacted
            ocr_func: Function to extract OCR text with bounding boxes from image

        Returns:
            Tuple of (redacted image, list of additional PIIElements redacted)
        """
        if not remaining_pii:
            return image, []

        print(f"  Verification Agent: Applying additional redactions for {len(remaining_pii)} items...")

        # Get OCR results for the current image
        ocr_results = ocr_func(image)

        # Match remaining PII to OCR bounding boxes
        additional_elements = []
        for pii in remaining_pii:
            pii_text = pii.text.strip().lower()

            for ocr in ocr_results:
                ocr_text = ocr.text.strip().lower()

                # Check for match (exact, substring, or fuzzy)
                if (pii_text == ocr_text or
                    pii_text in ocr_text or
                    ocr_text in pii_text or
                    self._fuzzy_match(pii_text, ocr_text)):

                    element = PIIElement(
                        type=pii.type,
                        text=ocr.text,
                        bbox=ocr.bbox
                    )
                    additional_elements.append(element)
                    print(f"    Matched: \"{pii.text}\" -> \"{ocr.text}\"")
                    break
            else:
                print(f"    Warning: Could not find bbox for \"{pii.text}\"")

        # Apply redactions
        if additional_elements:
            image = self._apply_redactions(image.copy(), additional_elements)

        return image, additional_elements

    def _fuzzy_match(self, text1: str, text2: str) -> bool:
        """Check if two texts match with fuzzy matching."""
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, text1, text2).ratio()
        return ratio >= self.similarity_threshold

    def _apply_redactions(self, image: Image.Image, pii_elements: List[PIIElement]) -> Image.Image:
        """Apply black rectangle redactions to image."""
        draw = ImageDraw.Draw(image)
        padding = 5

        for element in pii_elements:
            bbox = element.bbox
            if bbox.width > 0 and bbox.height > 0:
                x1 = max(0, bbox.x - padding)
                y1 = max(0, bbox.y - padding)
                x2 = bbox.x + bbox.width + padding
                y2 = bbox.y + bbox.height + padding

                draw.rectangle([x1, y1, x2, y2], fill="black", outline="black")
                print(f"    Additional redaction: [{element.type}] \"{element.text}\"")

        return image

    def detect_and_redact_faces(
        self,
        image: Image.Image,
        padding: int = 10
    ) -> Tuple[Image.Image, List[BoundingBox]]:
        """
        Detect faces in the image and redact them.

        This method directly uses the face detection library to find and
        redact any faces in the image.

        Args:
            image: PIL Image to process
            padding: Extra padding around each detected face

        Returns:
            Tuple of (image with faces redacted, list of face bounding boxes)
        """
        if not self.enable_face_detection:
            return image, []

        print("  Verification Agent: Running face detection...")

        # Detect faces using the face detection library
        face_bboxes = get_face_bounding_boxes(image)

        if not face_bboxes:
            print("  Verification Agent: No faces detected")
            return image, []

        print(f"  Verification Agent: Detected {len(face_bboxes)} face(s)")

        # Apply redactions
        redacted_image = image.copy()
        draw = ImageDraw.Draw(redacted_image)

        for i, bbox in enumerate(face_bboxes):
            x1 = max(0, bbox.x - padding)
            y1 = max(0, bbox.y - padding)
            x2 = min(image.width, bbox.x + bbox.width + padding)
            y2 = min(image.height, bbox.y + bbox.height + padding)

            draw.rectangle([x1, y1, x2, y2], fill="black", outline="black")
            print(f"    Redacted face {i + 1}: ({bbox.x}, {bbox.y}, {bbox.width}x{bbox.height})")

        return redacted_image, face_bboxes

    def verify_with_face_detection(
        self,
        redacted_image: Image.Image,
        original_image: Optional[Image.Image] = None
    ) -> Tuple[Image.Image, VerificationResult, List[BoundingBox]]:
        """
        Verify a redacted image and also check for faces using the agentic tool-calling approach.

        This method uses the Vision LLM with the face detection tool to:
        1. Analyze the image for remaining PII
        2. Call the detect_faces tool if the LLM determines faces are present

        Args:
            redacted_image: The image after redaction
            original_image: Optional original image for over-redaction check

        Returns:
            Tuple of (image with faces redacted, verification result, face bounding boxes)
        """
        if not self.enable_face_detection or self.vision_llm_with_tools is None:
            # Fall back to standard verification without face detection
            result = self.verify_redaction(redacted_image, original_image)
            return redacted_image, result, []

        print("  Verification Agent: Running agentic verification with face detection tool...")

        # Prepare image for LLM
        redacted_b64 = self._image_to_base64(redacted_image)

        # Build prompt that includes face detection tool usage
        prompt = self._get_face_detection_verification_prompt()

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{redacted_b64}"},
                },
            ]
        )

        messages = [message]
        face_bboxes = []
        current_image = redacted_image

        # Agentic loop - allow LLM to call face detection tool
        max_iterations = 5
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            try:
                response = self.vision_llm_with_tools.invoke(messages)
                messages.append(response)

                # Check if LLM wants to call tools
                if not response.tool_calls:
                    break

                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    if tool_name == "detect_faces":
                        print("  Verification Agent: LLM called detect_faces tool")

                        # The LLM called the face detection tool
                        # We run face detection directly on the image
                        current_image, detected_faces = self.detect_and_redact_faces(current_image)
                        face_bboxes.extend(detected_faces)

                        # Provide tool result back to LLM
                        if detected_faces:
                            tool_result = f"Detected and redacted {len(detected_faces)} face(s) in the image."
                        else:
                            tool_result = "No faces detected in the image."

                        messages.append(ToolMessage(
                            content=tool_result,
                            tool_call_id=tool_call["id"]
                        ))

            except Exception as e:
                print(f"  Verification Agent ERROR during agentic loop: {e}")
                break

        # Now run standard verification to get the VerificationResult
        result = self.verify_redaction(current_image, original_image)

        return current_image, result, face_bboxes

    def _get_face_detection_verification_prompt(self) -> str:
        """Get the prompt for verification with face detection tool."""
        base_prompt = """Analyze this image for any remaining personally identifiable information (PII) that should be redacted.

## Your Task

1. **Check for visible faces**: If you see any human faces in this image that could identify a person, you MUST call the `detect_faces` tool to detect and redact them. This is critical for privacy protection.

2. **Check for remaining PII text**: Look for any visible text that contains:
   - Patient names, physician names, or any person's name
   - Dates of birth
   - Patient IDs, medical record numbers, accession numbers
   - Addresses
   - Hospital/facility names (specific named hospitals, not generic "Hospital")
   - Phone numbers, emails
   - Any other identifying information

## Tool Usage

- If you see ANY faces in the image, call the `detect_faces` tool immediately
- The tool will automatically detect and redact faces in the image

## Important Notes

- Black rectangles indicate already-redacted areas - these are good
- Focus on finding PII that is still VISIBLE (not redacted)
- Err on the side of caution - if you're unsure whether something is PII, it should be redacted
- Faces of any person (patients, doctors, visitors) should be redacted for privacy

After analysis, provide your findings about any remaining PII text that needs redaction."""

        return base_prompt


def create_verification_step(
    config: AnonymizerConfig,
    check_over_redaction: bool = False,
    max_verification_rounds: int = 2,
    enable_face_detection: bool = True
) -> callable:
    """
    Factory function to create a verification step that can be added to processors.

    Args:
        config: Anonymizer configuration
        check_over_redaction: Whether to check for over-redaction
        max_verification_rounds: Maximum rounds of verify-and-redact
        enable_face_detection: Whether to enable face detection during verification

    Returns:
        A function that performs verification and additional redaction
    """
    agent = ImageVerificationAgent(
        config=config,
        check_over_redaction=check_over_redaction,
        enable_face_detection=enable_face_detection
    )

    def verify_and_redact(
        redacted_image: Image.Image,
        original_image: Optional[Image.Image],
        ocr_func: callable
    ) -> Tuple[Image.Image, VerificationResult, List[PIIElement]]:
        """
        Verify and apply additional redactions if needed.

        Args:
            redacted_image: Image after initial redaction
            original_image: Original image (optional, for over-redaction check)
            ocr_func: Function to extract OCR text from image

        Returns:
            Tuple of (final image, verification result, additional elements redacted)
        """
        current_image = redacted_image
        all_additional_elements = []
        final_result = None

        # First, run face detection if enabled
        if enable_face_detection:
            print("\n  === Face Detection Phase ===")
            current_image, face_bboxes = agent.detect_and_redact_faces(current_image)
            if face_bboxes:
                # Add face detections as PIIElements
                for i, bbox in enumerate(face_bboxes):
                    face_element = PIIElement(
                        type="face",
                        text=f"face_{i+1}",
                        bbox=bbox
                    )
                    all_additional_elements.append(face_element)

        for round_num in range(max_verification_rounds):
            print(f"\n  === Verification Round {round_num + 1}/{max_verification_rounds} ===")

            # Verify current state
            result = agent.verify_redaction(
                current_image,
                original_image if check_over_redaction else None
            )
            final_result = result

            # If clean, we're done
            if result.is_clean:
                print(f"  Verification passed after {round_num + 1} round(s)")
                break

            # Apply additional redactions
            current_image, additional = agent.apply_additional_redactions(
                current_image,
                result.remaining_pii,
                ocr_func
            )
            all_additional_elements.extend(additional)

            # If we couldn't redact anything new, stop
            if not additional:
                print("  Warning: Could not locate remaining PII for redaction")
                break

        return current_image, final_result, all_additional_elements

    return verify_and_redact
