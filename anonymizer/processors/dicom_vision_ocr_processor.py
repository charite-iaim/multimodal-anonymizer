"""
DICOM image processor for anonymization using Vision LLM + OCR.
Converts DICOM to PNG, processes with PNGVisionOCRProcessor, then converts back to DICOM.

This approach combines:
- Vision LLM to identify PII (understands context and image content)
- OCR for precise bounding boxes (accurate redaction coverage)
"""

import json
import random
import numpy as np
from pathlib import Path
from PIL import Image
from datetime import datetime, timedelta
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2

from langchain_core.messages import HumanMessage, ToolMessage

# Increase PIL's max image pixels limit to handle large medical images
Image.MAX_IMAGE_PIXELS = 300000000  # 300 million pixels

from ..base_processor import FileProcessor
from ..config import AnonymizerConfig
from ..prompt_config import PromptConfig, DEFAULT_PROMPT_CONFIG
from ..llm_factory import create_chat_llm
from ..tools.time_shift_tool import shift_datetime_value, redact_text
from ..retry_utils import retry_with_backoff, RetryConfig, create_retry_callback
from .png_vision_ocr_processor import PNGVisionOCRProcessor


# ── DICOM metadata tag categories for anonymization ──

# Category 1: Tags to DELETE/BLANK (direct patient/provider identifiers)
TAGS_TO_BLANK = [
    "PatientName", "PatientID", "PatientBirthDate", "PatientBirthTime",
    "PatientAddress", "PatientTelephoneNumbers",
    "ReferringPhysicianName", "ReferringPhysicianAddress", "ReferringPhysicianTelephoneNumbers",
    "PerformingPhysicianName", "NameOfPhysiciansReadingStudy", "PhysiciansOfRecord",
    "InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName",
    "StationName", "AccessionNumber",
    "OtherPatientIDs", "OtherPatientNames", "OtherPatientIDsSequence",
    "PatientBirthName", "PatientMotherBirthName", "MedicalRecordLocator",
    "ResponsiblePerson", "ResponsibleOrganization",
    "OperatorsName",
]

# Category 2: Date/time tags to SHIFT
DATE_TAGS_TO_SHIFT = [
    "StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate",
    "InstanceCreationDate", "PerformedProcedureStepStartDate",
    "PerformedProcedureStepEndDate",
]

TIME_TAGS_TO_SHIFT = [
    "StudyTime", "SeriesTime", "AcquisitionTime", "ContentTime",
    "InstanceCreationTime", "PerformedProcedureStepStartTime",
    "PerformedProcedureStepEndTime",
]

# Category 3: Free-text tags to anonymize via LLM
FREE_TEXT_TAGS = [
    "StudyDescription", "SeriesDescription", "ImageComments",
    "AdditionalPatientHistory", "PatientComments",
    "RequestedProcedureDescription", "PerformedProcedureStepDescription",
    "ReasonForTheRequestedProcedure", "AdmittingDiagnosesDescription",
    "ClinicalTrialProtocolName", "ClinicalTrialSiteName",
]

# Category 4: UIDs to regenerate
UIDS_TO_REGENERATE = [
    "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID",
]


class DICOMVisionOCRProcessor(FileProcessor):
    """Processor for DICOM images using Vision LLM + OCR approach."""

    def __init__(
        self,
        config: AnonymizerConfig,
        save_intermediate: bool = None,
        similarity_threshold: float = 0.6,
        enable_verification: bool = True,
        check_over_redaction: bool = False,
        max_verification_rounds: int = 2,
        prompt_config: PromptConfig = None
    ):
        """
        Initialize DICOM Vision+OCR processor.

        Args:
            config: Anonymizer configuration
            save_intermediate: If True, save intermediate PNG files for development.
                             If None, uses config.save_debug_files
            similarity_threshold: Minimum similarity for fuzzy text matching (0.0-1.0)
            enable_verification: If True, run verification agent after initial redaction
            check_over_redaction: If True, also check for over-redaction
            max_verification_rounds: Maximum rounds of verify-and-redact
            prompt_config: Optional custom prompt configuration
        """
        super().__init__(config)
        self.save_intermediate = save_intermediate if save_intermediate is not None else config.save_debug_files
        self.enable_verification = enable_verification
        self.check_over_redaction = check_over_redaction
        self.max_verification_rounds = max_verification_rounds
        self.prompt_config = prompt_config or DEFAULT_PROMPT_CONFIG
        self.png_processor = PNGVisionOCRProcessor(
            config,
            similarity_threshold=similarity_threshold,
            enable_verification=enable_verification,
            check_over_redaction=check_over_redaction,
            max_verification_rounds=max_verification_rounds,
            prompt_config=self.prompt_config
        )

        # Metadata anonymization settings
        self.time_offset_days = random.randint(-365, 365)
        self.retry_config = RetryConfig(
            max_retries=3,
            initial_delay=2.0,
            max_delay=60.0,
            exponential_base=2.0,
            jitter=True,
        )
        self.llm_metadata = create_chat_llm(
            config=config,
            timeout=600,
            max_tokens=16000,
            tools=[redact_text],
        )

    def can_process(self, file_path: Path) -> bool:
        """Check if file is a DICOM image."""
        # First check extension
        if file_path.suffix.lower() in ['.dcm', '.dicom']:
            return True

        # For files without DICOM extension, check for DICOM magic bytes
        # DICOM files have "DICM" at byte offset 128
        try:
            with open(file_path, 'rb') as f:
                f.seek(128)
                magic = f.read(4)
                if magic == b'DICM':
                    return True
        except (IOError, OSError):
            pass

        # Don't try pydicom.dcmread with force=True for unknown files
        # as it will accept almost any file and fail later on pixel decoding
        return False

    def extract_content(self, file_path: Path) -> str:
        """Extract content from DICOM file."""
        ds = pydicom.dcmread(file_path, force=True)
        return str(ds)

    # ── DICOM metadata anonymization ──

    @staticmethod
    def _shift_dicom_date(dicom_date: str, offset_days: int) -> str:
        """
        Shift a DICOM date (YYYYMMDD) by offset_days.

        Args:
            dicom_date: Date string in DICOM format YYYYMMDD
            offset_days: Number of days to shift

        Returns:
            Shifted date in YYYYMMDD format, or original if parsing fails.
        """
        dicom_date = dicom_date.strip()
        if not dicom_date:
            return dicom_date
        try:
            dt = datetime.strptime(dicom_date, "%Y%m%d")
            shifted = dt + timedelta(days=offset_days)
            return shifted.strftime("%Y%m%d")
        except ValueError:
            # Try via shift_datetime_value with dashes (YYYY-MM-DD)
            if len(dicom_date) == 8 and dicom_date.isdigit():
                iso = f"{dicom_date[:4]}-{dicom_date[4:6]}-{dicom_date[6:8]}"
                result = shift_datetime_value(iso, offset_days)
                if "[SHIFT_FAILED]" not in result:
                    return result.replace("-", "")
            return dicom_date

    @staticmethod
    def _shift_dicom_time(dicom_time: str, offset_days: int) -> str:
        """
        Shift a DICOM time (HHMMSS.FFFFFF) by offset_days.

        Time-of-day shifting is generally not needed for de-identification,
        so this returns the original value. Override if time shifting is required.
        """
        return dicom_time

    def _anonymize_free_text_tags(self, ds: pydicom.Dataset) -> dict:
        """
        Use LLM to anonymize free-text DICOM tags that may contain embedded PHI.

        Args:
            ds: pydicom Dataset (modified in-place)

        Returns:
            Dict mapping tag names to {"original": ..., "anonymized": ...}
        """
        # Collect non-empty free-text tag values
        tag_entries = {}
        for tag_name in FREE_TEXT_TAGS:
            if hasattr(ds, tag_name):
                value = str(getattr(ds, tag_name)).strip()
                if value:
                    tag_entries[tag_name] = value

        if not tag_entries:
            return {}

        # Format tag data for the LLM prompt
        tag_data_lines = []
        for tag_name, value in tag_entries.items():
            tag_data_lines.append(f"{tag_name}: {value}")
        tag_data = "\n".join(tag_data_lines)

        prompt = self.prompt_config.get_dicom_metadata_anonymization_prompt(tag_data=tag_data)
        messages = [HumanMessage(content=prompt)]

        changes = {}
        max_iterations = 30
        iteration = 0

        def invoke_with_retry(msgs):
            return retry_with_backoff(
                lambda: self.llm_metadata.invoke(msgs),
                config=self.retry_config,
                on_retry=create_retry_callback(prefix="    [Metadata LLM] "),
            )

        while iteration < max_iterations:
            iteration += 1
            try:
                response = invoke_with_retry(messages)
                messages.append(response)

                if not response.tool_calls:
                    break

                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    if tool_name == "redact_text":
                        text_to_redact = tool_args.get("text_to_redact", "")
                        result = redact_text.invoke(tool_args)

                        if "[REDACT_FAILED" not in result and text_to_redact:
                            # Apply redaction to matching tag values
                            for tag_name, value in list(tag_entries.items()):
                                if text_to_redact in value:
                                    new_value = value.replace(text_to_redact, result)
                                    if tag_name not in changes:
                                        changes[tag_name] = {
                                            "original": value,
                                            "anonymized": new_value,
                                        }
                                    else:
                                        changes[tag_name]["anonymized"] = new_value
                                    tag_entries[tag_name] = new_value
                                    setattr(ds, tag_name, new_value)
                                    print(f"    Redacted in {tag_name}: '{text_to_redact}'")

                        messages.append(ToolMessage(
                            content=f"Redacted: '{text_to_redact}' -> '{result}'",
                            tool_call_id=tool_call["id"],
                        ))

            except Exception as e:
                print(f"    Error during metadata anonymization: {e}")
                break

        return changes

    def _anonymize_metadata(self, ds: pydicom.Dataset) -> dict:
        """
        Anonymize all DICOM metadata on the dataset (modified in-place).

        Phases:
        1. Blank known PHI tags (names, IDs, addresses, etc.)
        2. Shift date/time tags
        3. Regenerate UIDs
        4. LLM-based anonymization of free-text tags

        Args:
            ds: pydicom Dataset (modified in-place)

        Returns:
            Dict of all changes made, for debug logging.
        """
        metadata_changes = {
            "blanked_tags": {},
            "shifted_dates": {},
            "regenerated_uids": {},
            "free_text_redactions": {},
        }

        # Phase 1: Blank known PHI tags
        print("  Phase 1: Blanking known PHI tags...")
        for tag_name in TAGS_TO_BLANK:
            if hasattr(ds, tag_name):
                original = str(getattr(ds, tag_name))
                if original.strip():
                    setattr(ds, tag_name, "")
                    metadata_changes["blanked_tags"][tag_name] = original
                    print(f"    Blanked: {tag_name}")

        # Phase 2: Shift date/time tags
        print(f"  Phase 2: Shifting dates by {self.time_offset_days} days...")
        for tag_name in DATE_TAGS_TO_SHIFT:
            if hasattr(ds, tag_name):
                original = str(getattr(ds, tag_name)).strip()
                if original:
                    shifted = self._shift_dicom_date(original, self.time_offset_days)
                    if shifted != original:
                        setattr(ds, tag_name, shifted)
                        metadata_changes["shifted_dates"][tag_name] = {
                            "original": original,
                            "shifted": shifted,
                        }
                        print(f"    Shifted: {tag_name} {original} -> {shifted}")

        for tag_name in TIME_TAGS_TO_SHIFT:
            if hasattr(ds, tag_name):
                original = str(getattr(ds, tag_name)).strip()
                if original:
                    shifted = self._shift_dicom_time(original, self.time_offset_days)
                    if shifted != original:
                        setattr(ds, tag_name, shifted)
                        metadata_changes["shifted_dates"][tag_name] = {
                            "original": original,
                            "shifted": shifted,
                        }

        # Phase 3: Regenerate UIDs
        print("  Phase 3: Regenerating UIDs...")
        for tag_name in UIDS_TO_REGENERATE:
            if hasattr(ds, tag_name):
                original = str(getattr(ds, tag_name))
                new_uid = pydicom.uid.generate_uid()
                setattr(ds, tag_name, new_uid)
                metadata_changes["regenerated_uids"][tag_name] = {
                    "original": original,
                    "new": str(new_uid),
                }
                print(f"    Regenerated: {tag_name}")

        # Keep file_meta.MediaStorageSOPInstanceUID in sync with SOPInstanceUID
        if hasattr(ds, "SOPInstanceUID") and hasattr(ds, "file_meta"):
            if hasattr(ds.file_meta, "MediaStorageSOPInstanceUID"):
                ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID

        # Phase 4: LLM anonymization of free-text tags
        print("  Phase 4: Anonymizing free-text tags via LLM...")
        free_text_changes = self._anonymize_free_text_tags(ds)
        metadata_changes["free_text_redactions"] = free_text_changes
        if not free_text_changes:
            print("    No free-text PHI found.")

        return metadata_changes

    # ── DICOM pixel data conversion ──

    def _dicom_to_images(self, dicom_path: Path) -> tuple[list[Image.Image], pydicom.Dataset, bool]:
        """
        Convert DICOM file to PIL Image(s).

        Args:
            dicom_path: Path to DICOM file

        Returns:
            Tuple of (list of PIL Images, DICOM dataset, is_multiframe)
        """
        ds = pydicom.dcmread(dicom_path, force=True)
        
        # Handle missing transfer syntax by setting a default
        if not hasattr(ds, 'file_meta') or ds.file_meta is None:
            ds.file_meta = pydicom.dataset.FileMetaDataset()
        
        if not hasattr(ds.file_meta, 'TransferSyntaxUID') or ds.file_meta.TransferSyntaxUID is None:
            # Try to infer transfer syntax from the data
            # Common transfer syntaxes to try in order
            transfer_syntaxes = [
                pydicom.uid.ImplicitVRLittleEndian,  # Most common for files without header
                pydicom.uid.ExplicitVRLittleEndian,
                pydicom.uid.ExplicitVRBigEndian,
            ]
            
            pixel_array = None
            for ts in transfer_syntaxes:
                try:
                    ds.file_meta.TransferSyntaxUID = ts
                    pixel_array = ds.pixel_array
                    print(f"Successfully decoded with Transfer Syntax: {ts.name}")
                    break
                except Exception as e:
                    continue
            
            if pixel_array is None:
                raise ValueError(
                    "Unable to decode pixel data with any common transfer syntax. "
                    "The DICOM file may be corrupted or use an unsupported format."
                )
        else:
            pixel_array = ds.pixel_array

        is_multiframe = len(pixel_array.shape) == 4

        if is_multiframe:
            print(f"Multi-frame DICOM detected: {pixel_array.shape[0]} frames")
            frames = []
            for i in range(pixel_array.shape[0]):
                frame = self._process_frame(pixel_array[i], ds)
                frames.append(frame)
            return frames, ds, True
        else:
            frame = self._process_frame(pixel_array, ds)
            return [frame], ds, False

    def _process_frame(self, pixel_array: np.ndarray, ds: pydicom.Dataset) -> Image.Image:
        """
        Process a single frame/image from pixel array.

        Args:
            pixel_array: Pixel data array (2D or 3D)
            ds: DICOM dataset for metadata

        Returns:
            PIL Image
        """
        try:
            pixel_array = apply_voi_lut(pixel_array, ds)
        except:
            pass

        pixel_array = pixel_array.astype(float)
        pixel_min = pixel_array.min()
        pixel_max = pixel_array.max()

        if pixel_max > pixel_min:
            pixel_array = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255.0)

        pixel_array = pixel_array.astype(np.uint8)

        if hasattr(ds, 'PhotometricInterpretation'):
            if ds.PhotometricInterpretation == 'MONOCHROME1':
                pixel_array = 255 - pixel_array

        if len(pixel_array.shape) == 2:
            image = Image.fromarray(pixel_array, mode='L')
        elif len(pixel_array.shape) == 3:
            image = Image.fromarray(pixel_array, mode='RGB')
        else:
            raise ValueError(f"Unexpected pixel array shape: {pixel_array.shape}")

        return image

    def _images_to_dicom(self, images: list[Image.Image], original_ds: pydicom.Dataset,
                         output_path: Path, is_multiframe: bool) -> dict:
        """
        Convert PIL Image(s) back to DICOM format, anonymizing metadata.

        Args:
            images: List of PIL Images with redactions applied
            original_ds: Original DICOM dataset
            output_path: Path to save DICOM file
            is_multiframe: Whether the original was multi-frame

        Returns:
            Dict of metadata changes made during anonymization.
        """
        new_ds = original_ds.copy()

        if is_multiframe:
            frame_arrays = []
            for image in images:
                if image.mode == 'RGB':
                    frame_array = np.array(image)
                elif image.mode == 'L':
                    frame_array = np.array(image)
                else:
                    frame_array = np.array(image.convert('L'))

                if hasattr(original_ds, 'PhotometricInterpretation'):
                    if original_ds.PhotometricInterpretation == 'MONOCHROME1':
                        frame_array = 255 - frame_array

                frame_arrays.append(frame_array)

            pixel_array = np.stack(frame_arrays, axis=0)
            new_ds.PixelData = pixel_array.tobytes()
            new_ds.NumberOfFrames = len(images)
            new_ds.Rows = pixel_array.shape[1]
            new_ds.Columns = pixel_array.shape[2]

            if len(pixel_array.shape) == 4:
                new_ds.SamplesPerPixel = pixel_array.shape[3]
                new_ds.PhotometricInterpretation = 'RGB'
            else:
                new_ds.SamplesPerPixel = 1
                new_ds.PhotometricInterpretation = 'MONOCHROME2'
        else:
            image = images[0]

            if image.mode == 'RGB':
                pixel_array = np.array(image)
            elif image.mode == 'L':
                pixel_array = np.array(image)
            else:
                pixel_array = np.array(image.convert('L'))

            if hasattr(original_ds, 'PhotometricInterpretation'):
                if original_ds.PhotometricInterpretation == 'MONOCHROME1':
                    pixel_array = 255 - pixel_array

            new_ds.PixelData = pixel_array.tobytes()
            new_ds.Rows = pixel_array.shape[0]
            new_ds.Columns = pixel_array.shape[1]
            new_ds.SamplesPerPixel = 1

            if len(pixel_array.shape) == 2:
                new_ds.PhotometricInterpretation = 'MONOCHROME2'
            elif len(pixel_array.shape) == 3:
                new_ds.SamplesPerPixel = pixel_array.shape[2]
                new_ds.PhotometricInterpretation = 'RGB'

        new_ds.BitsAllocated = 8
        new_ds.BitsStored = 8
        new_ds.HighBit = 7
        new_ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        # Anonymize DICOM metadata (blank PHI tags, shift dates, regenerate UIDs, LLM free-text)
        print("Anonymizing DICOM metadata...")
        metadata_changes = self._anonymize_metadata(new_ds)

        # Add processing note after metadata anonymization (which may have cleared ImageComments)
        if hasattr(new_ds, 'ImageComments') and new_ds.ImageComments:
            new_ds.ImageComments = f"{new_ds.ImageComments}; Anonymized by Vision+OCR LLM processor"
        else:
            new_ds.ImageComments = "Anonymized by Vision+OCR LLM processor"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        new_ds.save_as(output_path)

        return metadata_changes

    def _create_video_from_frames(self, frames: list[Image.Image], output_path: Path, fps: int = 10) -> None:
        """
        Create an MP4 video from a list of frames for debugging purposes.

        Args:
            frames: List of PIL Images
            output_path: Path to save the MP4 file
            fps: Frames per second for the video
        """
        if not frames:
            return

        first_frame = frames[0]
        width, height = first_frame.size

        rgb_frames = []
        for frame in frames:
            if frame.mode != 'RGB':
                frame = frame.convert('RGB')
            rgb_frames.append(np.array(frame))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        for frame_array in rgb_frames:
            frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

        video_writer.release()
        print(f"Created debug video: {output_path}")

    def anonymize(self, input_path: Path, output_path: Path) -> None:
        """
        Anonymize DICOM image using Vision LLM + OCR approach.

        For multi-frame DICOMs, PHI detection is performed only on the first frame
        and the detected bounding boxes are applied to all frames.

        Args:
            input_path: Path to input DICOM file
            output_path: Path to save anonymized DICOM file
        """
        print(f"Processing DICOM (Vision+OCR): {input_path.name}")

        print("Converting DICOM to PNG(s)...")
        images, dicom_dataset, is_multiframe = self._dicom_to_images(input_path)

        if is_multiframe:
            print(f"Multi-frame DICOM: Using first-frame-only detection strategy")
            redacted_images = self._anonymize_multiframe_optimized(
                images, input_path, output_path
            )
        else:
            redacted_images = self._anonymize_singleframe(images[0], input_path, output_path)

        try:
            if is_multiframe and self.save_intermediate:
                video_path = output_path.parent / "intermediate" / f"{input_path.stem}_redacted.mp4"
                print(f"Creating debug video from {len(redacted_images)} frames...")
                self._create_video_from_frames(redacted_images, video_path, fps=10)

            print("Converting redacted PNG(s) back to DICOM...")
            metadata_changes = self._images_to_dicom(redacted_images, dicom_dataset, output_path, is_multiframe)
            print(f"Saved anonymized DICOM to: {output_path}")

            # Append metadata anonymization details to the debug JSON if it exists
            if self.config.save_debug_files and metadata_changes:
                json_dest = output_path.with_suffix(".json")
                if json_dest.exists():
                    with open(json_dest, "r", encoding="utf-8") as f:
                        output_data = json.load(f)
                    output_data["metadata_anonymization"] = metadata_changes
                    with open(json_dest, "w", encoding="utf-8") as f:
                        json.dump(output_data, f, indent=2, ensure_ascii=False)

        finally:
            pass

    def _anonymize_singleframe(self, image: Image.Image, input_path: Path, output_path: Path) -> list[Image.Image]:
        """
        Anonymize a single frame using Vision LLM + OCR pipeline.

        Args:
            image: PIL Image
            input_path: Original input path
            output_path: Output path

        Returns:
            List containing single redacted image
        """
        original_image = image.copy()  # Keep for verification

        if self.save_intermediate:
            intermediate_dir = output_path.parent / "intermediate"
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            intermediate_png_path = intermediate_dir / f"{input_path.stem}_original.png"
            image.save(intermediate_png_path)
            print(f"Saved intermediate PNG to: {intermediate_png_path}")

        print("Detecting and redacting PHI using Vision+OCR approach...")
        pii_elements = self.png_processor.detect_pii_bboxes(image)
        print(f"Found {len(pii_elements)} PHI elements")

        redacted_image = self.png_processor._apply_redactions(image.copy(), pii_elements)

        # Verification phase
        verification_result = None
        additional_elements = []
        if self.enable_verification and self.png_processor._verification_agent is not None:
            print("\n=== Verification Phase ===")
            redacted_image, verification_result, additional_elements = self.png_processor._run_verification(
                redacted_image,
                original_image if self.check_over_redaction else None
            )
            pii_elements.extend(additional_elements)

        if self.save_intermediate:
            redacted_intermediate_path = intermediate_dir / f"{input_path.stem}_redacted.png"
            redacted_image.save(redacted_intermediate_path)
            print(f"Saved redacted intermediate PNG to: {redacted_intermediate_path}")

        if self.config.save_debug_files:
            from ..models import PIIDetectionResult
            pii_result = PIIDetectionResult(pii_elements=pii_elements)

            json_dest = output_path.with_suffix(".json")
            output_data = {
                "metadata": {
                    "input_file": str(input_path.name),
                    "output_file": str(output_path.name),
                    "timestamp": datetime.now().isoformat(),
                    "processing_method": "vision_ocr_singleframe",
                    "verification_enabled": self.enable_verification,
                    "total_pii_elements": len(pii_elements)
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
                    for element in pii_elements
                ]
            }

            # Add verification results if available
            if verification_result is not None:
                output_data["verification"] = {
                    "is_clean": verification_result.is_clean,
                    "confidence": verification_result.confidence,
                    "notes": verification_result.notes,
                    "remaining_pii_found": [
                        {"text": pii.text, "type": pii.type, "reason": pii.reason}
                        for pii in verification_result.remaining_pii
                    ],
                    "over_redactions": [
                        {"description": o.description, "reason": o.reason, "can_recover": o.can_recover}
                        for o in verification_result.over_redactions
                    ]
                }

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

            with open(json_dest, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"Saved detection results to: {json_dest}")

        return [redacted_image]

    def _anonymize_multiframe_optimized(
        self,
        images: list[Image.Image],
        input_path: Path,
        output_path: Path
    ) -> list[Image.Image]:
        """
        Anonymize multi-frame DICOM using first-frame-only detection.

        For multi-frame DICOMs, verification is performed on the first frame after
        initial redaction. Any additional bounding boxes found are applied to all frames.

        Args:
            images: List of all frame images
            input_path: Original input path
            output_path: Output path

        Returns:
            List of all redacted images
        """
        num_frames = len(images)
        original_first_frame = images[0].copy()  # Keep for verification
        print(f"Analyzing first frame only (out of {num_frames} total frames)...")

        print(f"Detecting PHI in first frame using Vision+OCR...")
        first_frame_bboxes = self.png_processor.detect_pii_bboxes(images[0])
        print(f"  Found {len(first_frame_bboxes)} PHI elements")

        if self.save_intermediate:
            intermediate_dir = output_path.parent / "intermediate"
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            first_frame_path = intermediate_dir / f"{input_path.stem}_frame0000_original.png"
            images[0].save(first_frame_path)
            print(f"Saved first frame to: {first_frame_path}")

        # Apply initial redactions to first frame for verification
        redacted_first_frame = self.png_processor._apply_redactions(images[0].copy(), first_frame_bboxes)

        # Verification phase on first frame
        verification_result = None
        additional_elements = []
        all_bboxes = list(first_frame_bboxes)  # Start with initial bboxes

        if self.enable_verification and self.png_processor._verification_agent is not None:
            print("\n=== Verification Phase (First Frame) ===")
            redacted_first_frame, verification_result, additional_elements = self.png_processor._run_verification(
                redacted_first_frame,
                original_first_frame if self.check_over_redaction else None
            )
            all_bboxes.extend(additional_elements)

        # Now apply ALL bounding boxes (initial + verification) to all frames
        print(f"Applying redactions to all {num_frames} frames...")
        redacted_images = []
        for i, image in enumerate(images):
            if i % 10 == 0 or i == num_frames - 1:
                print(f"  Redacting frame {i + 1}/{num_frames}...")

            redacted_image = self.png_processor._apply_redactions(image.copy(), all_bboxes)
            redacted_images.append(redacted_image)

        if self.save_intermediate:
            redacted_path = intermediate_dir / f"{input_path.stem}_frame0000_redacted.png"
            redacted_images[0].save(redacted_path)
            print(f"Saved redacted first frame to: {redacted_path}")

        if self.config.save_debug_files:
            from ..models import PIIDetectionResult
            pii_result = PIIDetectionResult(pii_elements=all_bboxes)

            json_dest = output_path.with_suffix(".json")
            output_data = {
                "metadata": {
                    "input_file": str(input_path.name),
                    "output_file": str(output_path.name),
                    "timestamp": datetime.now().isoformat(),
                    "processing_method": "vision_ocr_multiframe_first_frame_only",
                    "verification_enabled": self.enable_verification,
                    "total_frames": num_frames,
                    "analyzed_frames": 1,
                    "total_pii_elements": len(all_bboxes),
                    "initial_pii_elements": len(first_frame_bboxes),
                    "verification_additional_elements": len(additional_elements)
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
                    for element in all_bboxes
                ]
            }

            # Add verification results if available
            if verification_result is not None:
                output_data["verification"] = {
                    "is_clean": verification_result.is_clean,
                    "confidence": verification_result.confidence,
                    "notes": verification_result.notes,
                    "remaining_pii_found": [
                        {"text": pii.text, "type": pii.type, "reason": pii.reason}
                        for pii in verification_result.remaining_pii
                    ],
                    "over_redactions": [
                        {"description": o.description, "reason": o.reason, "can_recover": o.can_recover}
                        for o in verification_result.over_redactions
                    ]
                }

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

            with open(json_dest, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"Saved detection results to: {json_dest}")

        return redacted_images
