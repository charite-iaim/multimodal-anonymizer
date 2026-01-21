"""
Image Verification Agent for post-redaction PII detection.

This agent verifies that redacted images no longer contain visible PII.
If PII is still detected, it identifies it and applies additional redactions.

The agent uses a two-step approach:
1. Vision LLM scans the redacted image for any remaining PII
2. If PII found: OCR + matching to get precise bounding boxes for additional redaction

Optional: Over-redaction detection (checks if non-PII was unnecessarily redacted)
"""

import base64
import io
from typing import List, Optional, Tuple
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage

from ..config import AnonymizerConfig
from ..llm_factory import create_chat_llm
from ..models import PIIElement, BoundingBox
from ..prompt_config import PromptConfig, DEFAULT_PROMPT_CONFIG


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
    no PII was missed during the initial redaction process.
    """

    def __init__(
        self,
        config: AnonymizerConfig,
        check_over_redaction: bool = False,
        similarity_threshold: float = 0.6,
        prompt_config: PromptConfig = None,
        verification_prompt_getter: callable = None
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
        """
        self.config = config
        self.check_over_redaction = check_over_redaction
        self.similarity_threshold = similarity_threshold
        self.prompt_config = prompt_config or DEFAULT_PROMPT_CONFIG
        self._verification_prompt_getter = verification_prompt_getter

        # Initialize Vision LLM for verification
        self.vision_llm = create_chat_llm(
            config=config,
            temperature=0,  # Use 0 temperature for consistent verification
            structured_output=VerificationResult,
            use_vision_model=True,
        )

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


def create_verification_step(
    config: AnonymizerConfig,
    check_over_redaction: bool = False,
    max_verification_rounds: int = 2
) -> callable:
    """
    Factory function to create a verification step that can be added to processors.

    Args:
        config: Anonymizer configuration
        check_over_redaction: Whether to check for over-redaction
        max_verification_rounds: Maximum rounds of verify-and-redact

    Returns:
        A function that performs verification and additional redaction
    """
    agent = ImageVerificationAgent(
        config=config,
        check_over_redaction=check_over_redaction
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
