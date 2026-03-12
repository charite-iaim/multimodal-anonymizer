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
from ..tools.time_shift_tool import shift_datetime_value
from ..tools.redact_tool import redact_text
from ..retry_utils import retry_with_backoff, RetryConfig, create_retry_callback
from ..llm_response_utils import extract_content_from_response, get_reasoning_content_from_response
from .image_processor import PNGVisionOCRProcessor
from ..tools.face_detection_tool import detect_faces_in_image, redact_faces_in_pil_image
from .dicom_face_redaction_processor import get_face_redaction_masks_for_frames


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


def is_dicom_video(file_path: Path) -> tuple[bool, int]:
    """
    Check if a DICOM file contains video data (multiple frames).

    This function can be used to detect multi-frame DICOMs early, allowing
    users to choose frame-by-frame processing mode before anonymization.

    Args:
        file_path: Path to the DICOM file

    Returns:
        Tuple of (is_video, frame_count) where:
        - is_video: True if the DICOM has multiple frames (is a video)
        - frame_count: Number of frames in the DICOM (1 for single-frame)

    Raises:
        ValueError: If the file cannot be read as a DICOM or has no pixel data

    Example:
        >>> is_video, num_frames = is_dicom_video(Path("scan.dcm"))
        >>> if is_video:
        ...     print(f"DICOM video detected with {num_frames} frames")
        ...     # Offer user choice for frame-by-frame processing
    """
    ds = pydicom.dcmread(file_path, force=True)

    # Handle missing transfer syntax
    if not hasattr(ds, 'file_meta') or ds.file_meta is None:
        ds.file_meta = pydicom.dataset.FileMetaDataset()

    if not hasattr(ds.file_meta, 'TransferSyntaxUID') or ds.file_meta.TransferSyntaxUID is None:
        transfer_syntaxes = [
            pydicom.uid.ImplicitVRLittleEndian,
            pydicom.uid.ExplicitVRLittleEndian,
            pydicom.uid.ExplicitVRBigEndian,
        ]

        pixel_array = None
        for ts in transfer_syntaxes:
            try:
                ds.file_meta.TransferSyntaxUID = ts
                pixel_array = ds.pixel_array
                break
            except Exception:
                continue

        if pixel_array is None:
            raise ValueError(
                "Unable to decode pixel data. The DICOM file may be corrupted "
                "or use an unsupported format."
            )
    else:
        pixel_array = ds.pixel_array

    # 4D array indicates multi-frame DICOM (video)
    # Shape: (frames, height, width) for grayscale or (frames, height, width, channels) for color
    is_video = len(pixel_array.shape) == 4 or (
        len(pixel_array.shape) == 3 and
        hasattr(ds, 'NumberOfFrames') and
        int(ds.NumberOfFrames) > 1
    )

    if is_video:
        frame_count = pixel_array.shape[0]
    else:
        frame_count = 1

    return is_video, frame_count


def get_dicom_info(file_path: Path) -> dict:
    """
    Get detailed information about a DICOM file.

    Args:
        file_path: Path to the DICOM file

    Returns:
        Dictionary containing:
        - is_video: True if multi-frame DICOM
        - frame_count: Number of frames
        - dimensions: (width, height) of frames
        - is_color: True if RGB/color image
        - modality: DICOM modality (CT, MR, US, etc.) if available
        - bits_stored: Bits per pixel

    Example:
        >>> info = get_dicom_info(Path("scan.dcm"))
        >>> if info['is_video']:
        ...     print(f"Video with {info['frame_count']} frames")
    """
    ds = pydicom.dcmread(file_path, force=True)

    # Handle missing transfer syntax
    if not hasattr(ds, 'file_meta') or ds.file_meta is None:
        ds.file_meta = pydicom.dataset.FileMetaDataset()

    if not hasattr(ds.file_meta, 'TransferSyntaxUID') or ds.file_meta.TransferSyntaxUID is None:
        transfer_syntaxes = [
            pydicom.uid.ImplicitVRLittleEndian,
            pydicom.uid.ExplicitVRLittleEndian,
            pydicom.uid.ExplicitVRBigEndian,
        ]

        pixel_array = None
        for ts in transfer_syntaxes:
            try:
                ds.file_meta.TransferSyntaxUID = ts
                pixel_array = ds.pixel_array
                break
            except Exception:
                continue

        if pixel_array is None:
            raise ValueError(
                "Unable to decode pixel data. The DICOM file may be corrupted "
                "or use an unsupported format."
            )
    else:
        pixel_array = ds.pixel_array

    # Determine if multi-frame
    is_video = len(pixel_array.shape) == 4 or (
        len(pixel_array.shape) == 3 and
        hasattr(ds, 'NumberOfFrames') and
        int(ds.NumberOfFrames) > 1
    )

    if is_video:
        frame_count = pixel_array.shape[0]
        height = pixel_array.shape[1]
        width = pixel_array.shape[2]
        is_color = len(pixel_array.shape) == 4 and pixel_array.shape[3] >= 3
    else:
        frame_count = 1
        height = pixel_array.shape[0]
        width = pixel_array.shape[1]
        is_color = len(pixel_array.shape) == 3 and pixel_array.shape[2] >= 3

    return {
        'is_video': is_video,
        'frame_count': frame_count,
        'dimensions': (width, height),
        'is_color': is_color,
        'modality': getattr(ds, 'Modality', None),
        'bits_stored': getattr(ds, 'BitsStored', None),
    }


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
        prompt_config: PromptConfig = None,
        process_all_frames: bool = False
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
            process_all_frames: If True, run PHI detection on every frame of multi-frame
                              DICOMs (resource-intensive). If False (default), detect on
                              first frame only and apply to all frames.
        """
        super().__init__(config)
        self.save_intermediate = save_intermediate if save_intermediate is not None else config.save_debug_files
        self.enable_verification = enable_verification
        self.check_over_redaction = check_over_redaction
        self.max_verification_rounds = max_verification_rounds
        self.prompt_config = prompt_config or DEFAULT_PROMPT_CONFIG
        self.process_all_frames = process_all_frames
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

        # Vision LLM for head scan classification
        self.llm_vision = create_chat_llm(
            config=config,
            timeout=120,
            max_tokens=2048,
            use_vision_model=True,
        )

    def _is_head_scan(self, image: Image.Image) -> bool:
        """
        Ask the vision LLM whether a DICOM image is a CT or MRI scan of the head/face.

        Args:
            image: A representative frame from the DICOM (first frame).

        Returns:
            True if the LLM determines this is a CT/MRI head scan showing the face.
        """
        import base64
        import io

        # Convert image to base64 JPEG for the vision model
        rgb_image = image.convert("RGB")
        buffer = io.BytesIO()
        rgb_image.save(buffer, format="JPEG", quality=80)
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        prompt = (
            "You are a medical imaging expert. Look at this DICOM image and determine "
            "whether it is a CT or MRI scan of the head/face region.\n\n"
            "Answer ONLY with 'YES' or 'NO'.\n"
            "- Answer 'YES' if this is a CT or MRI scan of the head or face, or skull\n"
            "- Answer 'NO' for all other types of medical images (chest CT, abdominal "
            "scans, X-rays, ultrasounds, etc.).\n\n"
            "Your answer (YES or NO):"
        )

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ]
        )

        try:
            response = retry_with_backoff(
                lambda: self.llm_vision.invoke([message]),
                config=self.retry_config,
                on_retry=create_retry_callback(prefix="    [Head scan detection] "),
            )
            import re
            raw_answer = response.content.strip()
            cleaned = extract_content_from_response(response).upper()
            yes_no_matches = re.findall(r'\b(YES|NO)\b', cleaned)
            if yes_no_matches:
                final_answer = yes_no_matches[-1]
            else:
                # No clear YES/NO found — default to NO (safe: triggers face detection)
                final_answer = "NO"
                print(f"  Head scan detection: WARNING - could not parse YES/NO from response")
            is_head = final_answer == "YES"
            print(f"  Head scan detection: LLM answered '{raw_answer[:200]}{'...' if len(raw_answer) > 200 else ''}' -> final='{final_answer}' -> {'head scan' if is_head else 'not a head scan'}")
            return is_head
        except Exception as e:
            print(f"  Head scan detection failed: {e}. Falling back to standard processing.")
            return False

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

    def _dicom_to_images(self, dicom_path: Path) -> tuple[list[Image.Image], pydicom.Dataset, bool, np.ndarray]:
        """
        Convert DICOM file to display-quality PIL Image(s) and preserve the raw pixel array.

        The PIL images are used for LLM/OCR detection only. The raw pixel array
        is preserved so redactions can be applied directly without lossy round-tripping.

        Args:
            dicom_path: Path to DICOM file

        Returns:
            Tuple of (list of PIL Images, DICOM dataset, is_multiframe, raw_pixel_array)
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

        # Keep a copy of the raw pixel array before any display transformations
        raw_pixel_array = pixel_array.copy()

        is_multiframe = len(pixel_array.shape) == 4

        if is_multiframe:
            print(f"Multi-frame DICOM detected: {pixel_array.shape[0]} frames")
            frames = []
            for i in range(pixel_array.shape[0]):
                frame = self._make_display_image(pixel_array[i], ds)
                frames.append(frame)
            return frames, ds, True, raw_pixel_array
        else:
            frame = self._make_display_image(pixel_array, ds)
            return [frame], ds, False, raw_pixel_array

    def _make_display_image(self, pixel_array: np.ndarray, ds: pydicom.Dataset) -> Image.Image:
        """
        Create a display-quality 8-bit PIL Image from raw DICOM pixel data.

        This is used ONLY for sending to the LLM/OCR for PII detection and for
        saving intermediate debug PNGs. The original pixel data is never modified.

        Args:
            pixel_array: Pixel data array (2D grayscale or 3D color)
            ds: DICOM dataset for metadata

        Returns:
            PIL Image suitable for display/LLM consumption
        """
        is_color = len(pixel_array.shape) == 3 and pixel_array.shape[2] >= 3

        if not is_color:
            # Simple min-max normalization for display
            display = pixel_array.copy().astype(float)
            pixel_min = display.min()
            pixel_max = display.max()

            if pixel_max > pixel_min:
                display = ((display - pixel_min) / (pixel_max - pixel_min) * 255.0)

            display = display.astype(np.uint8)

            if hasattr(ds, 'PhotometricInterpretation'):
                if ds.PhotometricInterpretation == 'MONOCHROME1':
                    display = 255 - display

            return Image.fromarray(display, mode='L')
        else:
            photometric = getattr(ds, 'PhotometricInterpretation', 'RGB')

            if photometric.startswith('YBR'):
                display = pixel_array.astype(np.uint8)
                image = Image.fromarray(display, mode='YCbCr')
                return image.convert('RGB')
            else:
                if pixel_array.dtype != np.uint8:
                    display = pixel_array.astype(float)
                    pixel_min = display.min()
                    pixel_max = display.max()

                    if pixel_max > pixel_min:
                        display = ((display - pixel_min) / (pixel_max - pixel_min) * 255.0)

                    display = display.astype(np.uint8)
                else:
                    display = pixel_array

                return Image.fromarray(display, mode='RGB')

    @staticmethod
    def _apply_redaction_to_pixel_array(
        pixel_array: np.ndarray,
        pii_elements: list,
        ds: pydicom.Dataset,
    ) -> np.ndarray:
        """
        Apply redaction bounding boxes directly on the raw DICOM pixel array.

        For grayscale: fills redacted regions with the minimum pixel value
        (appears black in MONOCHROME2, white in MONOCHROME1 — both are "blank").
        For color: fills with zeros (black).

        Args:
            pixel_array: Raw pixel array (2D for grayscale, 3D for color).
                         Modified in-place.
            pii_elements: List of PIIElement objects with bounding boxes.
            ds: DICOM dataset for metadata.

        Returns:
            The modified pixel array.
        """
        is_color = len(pixel_array.shape) == 3 and pixel_array.shape[2] >= 3
        padding = 5

        # Determine the "black" fill value for this image
        if is_color:
            fill_value = 0
        else:
            photometric = getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2')
            if photometric == 'MONOCHROME1':
                # MONOCHROME1: higher values = darker, so use max to get black
                fill_value = int(pixel_array.max())
            else:
                # MONOCHROME2: lower values = darker, so use min to get black
                fill_value = int(pixel_array.min())

        rows, cols = pixel_array.shape[0], pixel_array.shape[1]

        for element in pii_elements:
            bbox = element.bbox
            if bbox.width > 0 and bbox.height > 0:
                x1 = max(0, bbox.x - padding)
                y1 = max(0, bbox.y - padding)
                x2 = min(cols, bbox.x + bbox.width + padding)
                y2 = min(rows, bbox.y + bbox.height + padding)

                pixel_array[y1:y2, x1:x2] = fill_value
                print(f"  Redacted [{element.type}]: \"{element.text}\"")

        return pixel_array

    def _save_dicom_with_pixel_array(
        self,
        original_ds: pydicom.Dataset,
        pixel_array: np.ndarray,
        output_path: Path,
    ) -> dict:
        """
        Save a DICOM file using the original dataset and a (possibly redacted) pixel array.

        Preserves the original bit depth, transfer syntax, and all pixel-related DICOM tags.
        Only metadata is anonymized; pixel encoding parameters are left intact.

        Args:
            original_ds: Original DICOM dataset (used as template).
            pixel_array: Pixel array to write (same dtype/shape as original).
            output_path: Path to save the output DICOM file.

        Returns:
            Dict of metadata changes made during anonymization.
        """
        new_ds = original_ds.copy()
        new_ds.PixelData = pixel_array.tobytes()

        # If the original had a compressed transfer syntax, change to uncompressed
        # since we've modified the raw pixel data
        if hasattr(new_ds, 'file_meta') and hasattr(new_ds.file_meta, 'TransferSyntaxUID'):
            ts = new_ds.file_meta.TransferSyntaxUID
            # Check if it's a compressed transfer syntax
            if ts not in [pydicom.uid.ImplicitVRLittleEndian,
                          pydicom.uid.ExplicitVRLittleEndian,
                          pydicom.uid.ExplicitVRBigEndian]:
                # Change to uncompressed
                new_ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
                print(f"  Changed compressed transfer syntax {ts.name} to ExplicitVRLittleEndian")

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

        The pipeline uses display-quality images for PII detection (LLM + OCR),
        but applies redaction bounding boxes directly on the original raw pixel
        data to preserve the full fidelity of the DICOM.

        If the DICOM is detected as a CT/MRI head scan (via LLM), a trained
        ResNet U-Net model is used to redact facial features. Otherwise the
        standard Vision+OCR pipeline is used for text-based PII redaction.

        For multi-frame DICOMs, PHI detection is performed only on the first frame
        and the detected bounding boxes are applied to all frames.

        Args:
            input_path: Path to input DICOM file
            output_path: Path to save anonymized DICOM file
        """
        print(f"Processing DICOM (Vision+OCR): {input_path.name}")

        print("Converting DICOM to display images...")
        display_images, dicom_dataset, is_multiframe, raw_pixel_array = self._dicom_to_images(input_path)

        # ── Head scan detection: ask LLM if this is a CT/MRI head scan ──
        print("Checking if DICOM is a CT/MRI head scan...")
        is_head_scan = self._is_head_scan(display_images[0])

        # Initialize pixel array for redaction
        redacted_pixel_array = raw_pixel_array.copy()

        # Apply appropriate face/defacing based on scan type:
        # - Head scan (CT/MRI): Use U-Net defacing model (trained for medical imaging)
        # - Not head scan: Use RetinaNet face detection (for photos/videos with faces)

        if is_head_scan:
            # HEAD SCAN: Apply U-Net defacing model for CT/MRI head scans
            print("Step 1: Using U-Net defacing model for CT/MRI head scan...")

            # Get defacing masks from the U-Net model
            face_masks = get_face_redaction_masks_for_frames(display_images)

            # Save intermediate images if enabled
            if self.save_intermediate:
                intermediate_dir = output_path.parent / "intermediate"
                intermediate_dir.mkdir(parents=True, exist_ok=True)
                for i, img in enumerate(display_images):
                    img.save(intermediate_dir / f"{input_path.stem}_frame{i:04d}_original.png")
                # Save the masks for debugging
                for i, mask in enumerate(face_masks):
                    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
                    mask_img.save(intermediate_dir / f"{input_path.stem}_frame{i:04d}_deface_mask.png")

            # Apply defacing masks to the raw pixel array (preserves original bit depth)
            print("Applying defacing masks to original DICOM pixel data...")

            if is_multiframe:
                # Multi-frame: apply mask to each frame
                for frame_idx, mask in enumerate(face_masks):
                    if frame_idx < redacted_pixel_array.shape[0]:
                        redacted_pixel_array[frame_idx][mask] = 0
            else:
                # Single frame: apply mask directly
                redacted_pixel_array[face_masks[0]] = 0

            # Save DICOM with the defaced pixel array
            print("Saving defaced DICOM...")
            metadata_changes = self._save_dicom_with_pixel_array(
                dicom_dataset, redacted_pixel_array, output_path
            )
            print(f"Saved anonymized DICOM to: {output_path}")

            # Save debug JSON for head scan processing
            if self.config.save_debug_files:
                json_dest = output_path.with_suffix(".json")
                output_data = {
                    "metadata": {
                        "input_file": str(input_path.name),
                        "output_file": str(output_path.name),
                        "timestamp": datetime.now().isoformat(),
                        "processing_method": "unet_defacing",
                        "is_head_scan": True,
                        "total_frames": len(display_images),
                        "bit_depth_preserved": True,
                        "face_redaction_applied": True,
                        "text_redaction_applied": False,
                    }
                }
                if metadata_changes:
                    output_data["metadata_anonymization"] = metadata_changes
                with open(json_dest, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                print(f"Saved detection results to: {json_dest}")

            return  # Exit early for head scan processing

        else:
            # NOT A HEAD SCAN: Apply RetinaNet face detection for photos/videos
            print("Step 1: Using RetinaNet face detection (not a head scan)...")

            # Detect faces in each frame using RetinaNet
            all_face_detections = []
            total_faces = 0
            for i, img in enumerate(display_images):
                faces = detect_faces_in_image(img, score_threshold=0.5, min_face_size=30)
                all_face_detections.append(faces)
                total_faces += len(faces)
                if i % 10 == 0 or i == len(display_images) - 1:
                    print(f"  Frame {i + 1}/{len(display_images)}: {len(faces)} face(s) detected")

            print(f"  Total faces detected: {total_faces}")

            # Save intermediate images if enabled
            if self.save_intermediate:
                intermediate_dir = output_path.parent / "intermediate"
                intermediate_dir.mkdir(parents=True, exist_ok=True)
                for i, img in enumerate(display_images):
                    img.save(intermediate_dir / f"{input_path.stem}_frame{i:04d}_original.png")
                # Save face-redacted display images for debugging
                for i, (img, faces) in enumerate(zip(display_images, all_face_detections)):
                    if faces:
                        redacted_img = redact_faces_in_pil_image(img, faces, padding=10)
                        redacted_img.save(intermediate_dir / f"{input_path.stem}_frame{i:04d}_face_redacted.png")

            # Apply face redaction bounding boxes to the raw pixel array (preserves original bit depth)
            print("Applying face redaction to original DICOM pixel data...")
            padding = 10

            if is_multiframe:
                # Multi-frame: apply face bboxes to each frame
                for frame_idx, faces in enumerate(all_face_detections):
                    if frame_idx < redacted_pixel_array.shape[0]:
                        for face in faces:
                            x1 = max(0, face.x - padding)
                            y1 = max(0, face.y - padding)
                            x2 = min(redacted_pixel_array.shape[2], face.x + face.width + padding)
                            y2 = min(redacted_pixel_array.shape[1], face.y + face.height + padding)
                            redacted_pixel_array[frame_idx, y1:y2, x1:x2] = 0
            else:
                # Single frame: apply face bboxes directly
                for face in all_face_detections[0]:
                    x1 = max(0, face.x - padding)
                    y1 = max(0, face.y - padding)
                    x2 = min(redacted_pixel_array.shape[1], face.x + face.width + padding)
                    y2 = min(redacted_pixel_array.shape[0], face.y + face.height + padding)
                    redacted_pixel_array[y1:y2, x1:x2] = 0

            # Step 2: Also run text detection on face-redacted images
            # Create display images from the face-redacted pixel array for text detection
            print("Step 2: Running Vision+OCR text detection on face-redacted images...")
            face_redacted_display_images = []
            if is_multiframe:
                for i in range(redacted_pixel_array.shape[0]):
                    face_redacted_display_images.append(
                        self._make_display_image(redacted_pixel_array[i], dicom_dataset)
                    )
            else:
                face_redacted_display_images.append(
                    self._make_display_image(redacted_pixel_array, dicom_dataset)
                )

            # Detect text PII on the face-redacted images
            if is_multiframe:
                print(f"Multi-frame DICOM: Using first-frame-only detection strategy for text")
                pii_elements = self._detect_pii_multiframe(
                    face_redacted_display_images, input_path, output_path
                )
            else:
                pii_elements = self._detect_pii_singleframe(
                    face_redacted_display_images[0], input_path, output_path
                )

            # Apply text redaction bounding boxes on top of face redaction
            print("Applying text redactions to DICOM pixel data...")
            if is_multiframe:
                for i in range(redacted_pixel_array.shape[0]):
                    self._apply_redaction_to_pixel_array(
                        redacted_pixel_array[i], pii_elements, dicom_dataset
                    )
            else:
                self._apply_redaction_to_pixel_array(
                    redacted_pixel_array, pii_elements, dicom_dataset
                )

            # Save intermediate fully redacted images for debugging
            if self.save_intermediate:
                intermediate_dir = output_path.parent / "intermediate"
                intermediate_dir.mkdir(parents=True, exist_ok=True)
                if is_multiframe:
                    for i in range(redacted_pixel_array.shape[0]):
                        redacted_display = self._make_display_image(redacted_pixel_array[i], dicom_dataset)
                        redacted_display.save(intermediate_dir / f"{input_path.stem}_frame{i:04d}_fully_redacted.png")
                else:
                    redacted_display = self._make_display_image(redacted_pixel_array, dicom_dataset)
                    redacted_display.save(intermediate_dir / f"{input_path.stem}_fully_redacted.png")

            # Update debug JSON with combined processing info
            if self.config.save_debug_files:
                json_dest = output_path.with_suffix(".json")
                if json_dest.exists():
                    with open(json_dest, "r", encoding="utf-8") as f:
                        output_data = json.load(f)
                else:
                    output_data = {}

                output_data["metadata"] = output_data.get("metadata", {})
                output_data["metadata"].update({
                    "processing_method": "face_redaction_and_vision_ocr",
                    "is_head_scan": is_head_scan,
                    "total_frames": len(display_images),
                    "bit_depth_preserved": True,
                    "face_redaction_applied": True,
                    "text_redaction_applied": True,
                    "total_text_pii_elements": len(pii_elements),
                })
                with open(json_dest, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)

            # Save DICOM with the fully redacted pixel array
            print("Saving fully anonymized DICOM (face anonymization + text redaction)...")
            metadata_changes = self._save_dicom_with_pixel_array(
                dicom_dataset, redacted_pixel_array, output_path
            )
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

    def _save_dicom_from_pil(
        self,
        images: list[Image.Image],
        original_ds: pydicom.Dataset,
        output_path: Path,
        is_multiframe: bool,
    ) -> dict:
        """
        Save DICOM from PIL images (used only for face redaction model output).

        The face redaction model produces 8-bit RGB PIL images, so this path
        necessarily converts to 8-bit. Used only when the head scan model is invoked.

        Args:
            images: List of PIL Images with redactions applied.
            original_ds: Original DICOM dataset.
            output_path: Path to save DICOM file.
            is_multiframe: Whether the original was multi-frame.

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
        new_ds.PixelRepresentation = 0  # unsigned
        new_ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        # The display transform (VOI LUT, windowing, rescale) has already been
        # baked into the 8-bit pixel values by _make_display_image.  Remove the
        # interpretation tags so DICOM viewers don't re-apply them on the
        # already-windowed data.
        for tag in ('RescaleIntercept', 'RescaleSlope',
                    'WindowCenter', 'WindowWidth',
                    'VOILUTFunction', 'VOILUTSequence'):
            if hasattr(new_ds, tag):
                delattr(new_ds, tag)

        # Anonymize DICOM metadata
        print("Anonymizing DICOM metadata...")
        metadata_changes = self._anonymize_metadata(new_ds)

        if hasattr(new_ds, 'ImageComments') and new_ds.ImageComments:
            new_ds.ImageComments = f"{new_ds.ImageComments}; Anonymized by Vision+OCR LLM processor"
        else:
            new_ds.ImageComments = "Anonymized by Vision+OCR LLM processor"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        new_ds.save_as(output_path)
        print(f"Saved anonymized DICOM to: {output_path}")

        return metadata_changes

    def _detect_pii_singleframe(self, image: Image.Image, input_path: Path, output_path: Path) -> list:
        """
        Detect PII bounding boxes in a single frame using Vision LLM + OCR.

        Does NOT apply redactions — only returns the list of detected PIIElements.
        Redaction is applied later directly on the raw pixel array.

        Args:
            image: Display-quality PIL Image for LLM/OCR
            input_path: Original input path
            output_path: Output path

        Returns:
            List of PIIElement objects with bounding boxes
        """
        original_image = image.copy()  # Keep for verification

        if self.save_intermediate:
            intermediate_dir = output_path.parent / "intermediate"
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            intermediate_png_path = intermediate_dir / f"{input_path.stem}_original.png"
            image.save(intermediate_png_path)
            print(f"Saved intermediate PNG to: {intermediate_png_path}")

        print("Detecting PHI using Vision+OCR approach...")
        pii_elements = self.png_processor.detect_pii_bboxes(image)
        print(f"Found {len(pii_elements)} PHI elements")

        # Verification phase (uses display images only for visual checks)
        verification_result = None
        additional_elements = []
        if self.enable_verification and self.png_processor._verification_agent is not None:
            print("\n=== Verification Phase ===")
            # Apply redactions on a display copy for verification
            redacted_display = self.png_processor._apply_redactions(image.copy(), pii_elements)
            redacted_display, verification_result, additional_elements = self.png_processor._run_verification(
                redacted_display,
                original_image if self.check_over_redaction else None
            )
            pii_elements.extend(additional_elements)

        if self.config.save_debug_files:
            from ..models import PIIDetectionResult
            PIIDetectionResult(pii_elements=pii_elements)

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

        return pii_elements

    def _detect_pii_multiframe(
        self,
        images: list[Image.Image],
        input_path: Path,
        output_path: Path
    ) -> list:
        """
        Detect PII bounding boxes using first-frame-only detection for multi-frame DICOMs.

        Does NOT apply redactions — only returns the list of detected PIIElements.
        Verification is performed on a display copy of the first frame.

        Args:
            images: List of all display-quality frame images
            input_path: Original input path
            output_path: Output path

        Returns:
            List of PIIElement objects with bounding boxes
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

        # Verification phase on a display copy of the first frame
        verification_result = None
        additional_elements = []
        all_bboxes = list(first_frame_bboxes)

        if self.enable_verification and self.png_processor._verification_agent is not None:
            print("\n=== Verification Phase (First Frame) ===")
            redacted_first_display = self.png_processor._apply_redactions(images[0].copy(), first_frame_bboxes)
            redacted_first_display, verification_result, additional_elements = self.png_processor._run_verification(
                redacted_first_display,
                original_first_frame if self.check_over_redaction else None
            )
            all_bboxes.extend(additional_elements)

        if self.config.save_debug_files:
            from ..models import PIIDetectionResult
            PIIDetectionResult(pii_elements=all_bboxes)

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

        return all_bboxes
