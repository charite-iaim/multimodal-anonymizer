"""
PDF processor for anonymization using Vision LLM + OCR.
Converts PDF pages to images, processes with PNGVisionOCRProcessor, then saves back as PDF.

This approach combines:
- Vision LLM to identify PII (understands context and image content)
- OCR for precise bounding boxes (accurate redaction coverage)
"""

import json
from pathlib import Path
from PIL import Image
from datetime import datetime
from typing import List

Image.MAX_IMAGE_PIXELS = 300000000  # 300 million pixels

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

from ..base_processor import FileProcessor
from ..config import AnonymizerConfig
from .image_processor import PNGVisionOCRProcessor
from ..models import PIIDetectionResult
from ..prompt_config import PromptConfig, DEFAULT_PROMPT_CONFIG


class PDFVisionOCRProcessor(FileProcessor):
    """Processor for PDF files using Vision LLM + OCR approach."""

    def __init__(
        self,
        config: AnonymizerConfig,
        save_intermediate: bool = None,
        similarity_threshold: float = 0.6,
        dpi: int = 300,
        enable_verification: bool = True,
        check_over_redaction: bool = False,
        max_verification_rounds: int = 2,
        prompt_config: PromptConfig = None,
        enable_face_detection: bool = True
    ):
        """
        Initialize PDF Vision+OCR processor.

        Args:
            config: Anonymizer configuration
            save_intermediate: If True, save intermediate PNG files for development.
                             If None, uses config.save_debug_files
            similarity_threshold: Minimum similarity for fuzzy text matching (0.0-1.0)
            dpi: DPI for PDF to image conversion (higher = better quality but slower)
            enable_verification: If True, run verification agent after initial redaction
            check_over_redaction: If True, also check for over-redaction
            max_verification_rounds: Maximum rounds of verify-and-redact
            prompt_config: Optional custom prompt configuration
            enable_face_detection: If True, enables face detection to redact visible faces
        """
        super().__init__(config)

        if not PDF2IMAGE_AVAILABLE:
            raise ImportError(
                "pdf2image is required for PDFVisionOCRProcessor. "
                "Install it with: pip install pdf2image\n"
                "Also requires poppler: brew install poppler (macOS) or apt-get install poppler-utils (Linux)"
            )

        self.save_intermediate = save_intermediate if save_intermediate is not None else config.save_debug_files
        self.dpi = dpi
        self.enable_verification = enable_verification
        self.check_over_redaction = check_over_redaction
        self.max_verification_rounds = max_verification_rounds
        self.prompt_config = prompt_config or DEFAULT_PROMPT_CONFIG
        self.enable_face_detection = enable_face_detection

        # Create PDF-specific prompt getters that use pdf_anonymization_prompt and pdf_verification_prompt
        def pdf_anonymization_prompt_getter(ocr_text_list: str) -> str:
            return self.prompt_config.get_pdf_anonymization_prompt(ocr_text_list)

        def pdf_verification_prompt_getter() -> str:
            return self.prompt_config.get_pdf_verification_prompt()

        self.png_processor = PNGVisionOCRProcessor(
            config,
            similarity_threshold=similarity_threshold,
            enable_verification=enable_verification,
            check_over_redaction=check_over_redaction,
            max_verification_rounds=max_verification_rounds,
            prompt_config=self.prompt_config,
            anonymization_prompt_getter=pdf_anonymization_prompt_getter,
            verification_prompt_getter=pdf_verification_prompt_getter,
            enable_face_detection=enable_face_detection
        )

    def can_process(self, file_path: Path) -> bool:
        """Check if file is a PDF."""
        return file_path.suffix.lower() == ".pdf"

    def extract_content(self, file_path: Path) -> str:
        """Extract content from PDF file (not used in this processor)."""
        return ""

    def _pdf_to_images(self, pdf_path: Path) -> List[Image.Image]:
        """
        Convert PDF file to list of PIL Images.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of PIL Images (one per page)
        """
        print(f"Converting PDF to images at {self.dpi} DPI...")
        images = convert_from_path(pdf_path, dpi=self.dpi)
        print(f"Converted {len(images)} pages")
        return images

    def _images_to_pdf(self, images: List[Image.Image], output_path: Path) -> None:
        """
        Save list of PIL Images as a multi-page PDF.

        Args:
            images: List of PIL Images
            output_path: Path to save PDF file
        """
        if not images:
            print("No images to save!")
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert images to RGB if necessary (PDF doesn't support all modes)
        rgb_images = []
        for img in images:
            if img.mode != 'RGB':
                rgb_images.append(img.convert('RGB'))
            else:
                rgb_images.append(img)

        # Save as multi-page PDF
        print(f"Saving {len(rgb_images)} pages to PDF...")
        rgb_images[0].save(
            output_path,
            save_all=True,
            append_images=rgb_images[1:] if len(rgb_images) > 1 else [],
            resolution=self.dpi,
            quality=95
        )

    def anonymize(self, input_path: Path, output_path: Path) -> None:
        """
        Anonymize PDF using Vision LLM + OCR approach.

        Args:
            input_path: Path to input PDF file
            output_path: Path to save anonymized PDF file
        """
        print(f"Processing PDF (Vision+OCR): {input_path.name}")

        # Step 1: Convert PDF to images
        images = self._pdf_to_images(input_path)

        all_pii_elements = []
        all_verification_results = []
        all_additional_elements = []
        redacted_images = []

        # Step 2: Process each page
        for page_num, image in enumerate(images, start=1):
            print(f"\n--- Page {page_num}/{len(images)} ---")
            original_image = image.copy()  # Keep for verification
            width, height = image.size
            print(f"Page size: {width}x{height}")

            # Save intermediate image if requested
            if self.save_intermediate:
                intermediate_dir = output_path.parent / "intermediate"
                intermediate_dir.mkdir(parents=True, exist_ok=True)
                original_path = intermediate_dir / f"{input_path.stem}_page{page_num:04d}_original.png"
                image.save(original_path)
                print(f"Saved intermediate: {original_path}")

            # Detect PII using Vision+OCR
            print("Detecting PII using Vision+OCR approach...")
            pii_elements = self.png_processor.detect_pii_bboxes(image)
            print(f"Found {len(pii_elements)} PII elements on page {page_num}")

            # Store elements with page info for JSON output
            for element in pii_elements:
                all_pii_elements.append({
                    "page": page_num,
                    "element": element
                })

            # Apply redactions
            if pii_elements:
                redacted_image = self.png_processor._apply_redactions(image.copy(), pii_elements)
            else:
                redacted_image = image.copy()

            # Verification phase for this page
            if self.enable_verification and self.png_processor._verification_agent is not None:
                print(f"\n=== Verification Phase (Page {page_num}) ===")
                redacted_image, verification_result, additional_elements = self.png_processor._run_verification(
                    redacted_image,
                    original_image if self.check_over_redaction else None
                )

                if verification_result is not None:
                    all_verification_results.append({
                        "page": page_num,
                        "result": verification_result
                    })

                for element in additional_elements:
                    all_additional_elements.append({
                        "page": page_num,
                        "element": element
                    })
                    all_pii_elements.append({
                        "page": page_num,
                        "element": element
                    })

            # Save intermediate redacted image if requested
            if self.save_intermediate:
                redacted_path = intermediate_dir / f"{input_path.stem}_page{page_num:04d}_redacted.png"
                redacted_image.save(redacted_path)
                print(f"Saved redacted intermediate: {redacted_path}")

            redacted_images.append(redacted_image)

        # Step 3: Save results as PDF
        self._images_to_pdf(redacted_images, output_path)
        print(f"\nSaved anonymized PDF to: {output_path}")

        # Save JSON with detection results
        if self.config.save_debug_files:
            json_dest = output_path.with_suffix(".json")
            self._save_json_output(
                all_pii_elements,
                input_path,
                output_path,
                json_dest,
                len(images),
                all_verification_results,
                all_additional_elements
            )
            print(f"Saved detection results to: {json_dest}")

    def _save_json_output(
        self,
        all_pii_elements: List[dict],
        input_path: Path,
        output_path: Path,
        json_output_path: Path,
        total_pages: int,
        verification_results: List[dict] = None,
        additional_elements: List[dict] = None
    ) -> None:
        """
        Save PII detection results as JSON.

        Args:
            all_pii_elements: List of dicts with page number and PIIElement
            input_path: Path to original input file
            output_path: Path to anonymized output file
            json_output_path: Path to save JSON output
            total_pages: Total number of pages in PDF
            verification_results: Optional list of verification results per page
            additional_elements: Optional list of additionally redacted elements per page
        """
        output_data = {
            "metadata": {
                "input_file": str(input_path.name),
                "output_file": str(output_path.name),
                "timestamp": datetime.now().isoformat(),
                "processing_method": "pdf_vision_ocr",
                "verification_enabled": self.enable_verification,
                "total_pages": total_pages,
                "total_pii_elements": len(all_pii_elements)
            },
            "pii_elements": [
                {
                    "page": item["page"],
                    "type": item["element"].type,
                    "text": item["element"].text,
                    "bbox": {
                        "x": item["element"].bbox.x,
                        "y": item["element"].bbox.y,
                        "width": item["element"].bbox.width,
                        "height": item["element"].bbox.height
                    }
                }
                for item in all_pii_elements
            ]
        }

        # Add verification results if available
        if verification_results:
            output_data["verification"] = [
                {
                    "page": item["page"],
                    "is_clean": item["result"].is_clean,
                    "confidence": item["result"].confidence,
                    "notes": item["result"].notes,
                    "remaining_pii_found": [
                        {"text": pii.text, "type": pii.type, "reason": pii.reason}
                        for pii in item["result"].remaining_pii
                    ],
                    "over_redactions": [
                        {"description": o.description, "reason": o.reason, "can_recover": o.can_recover}
                        for o in item["result"].over_redactions
                    ]
                }
                for item in verification_results
            ]

        # Add additional elements redacted during verification
        if additional_elements:
            output_data["verification_additional_redactions"] = [
                {
                    "page": item["page"],
                    "type": item["element"].type,
                    "text": item["element"].text,
                    "bbox": {
                        "x": item["element"].bbox.x,
                        "y": item["element"].bbox.y,
                        "width": item["element"].bbox.width,
                        "height": item["element"].bbox.height
                    }
                }
                for item in additional_elements
            ]

        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
