"""
File format specific processors (agentic/vision-based).
"""

from .png_vision_ocr_processor import PNGVisionOCRProcessor
from .dicom_vision_ocr_processor import DICOMVisionOCRProcessor
from .pdf_vision_ocr_processor import PDFVisionOCRProcessor
from .video_vision_ocr_processor import VideoVisionOCRProcessor
from .agentic_text_processor import AgenticTextProcessor
from .agentic_csv_processor import AgenticCSVProcessor
from .agentic_excel_processor import AgenticExcelProcessor
from .agentic_audio_processor import AgenticAudioProcessor
from .image_verification_agent import ImageVerificationAgent, VerificationResult, create_verification_step

# Optional CT/MRI processor (requires additional setup)
from .mede_processor import (
    MedeProcessor,
    is_mede_available,
    get_mede_status,
    process_ct_mri_file,
    CT_MRI_EXTENSIONS,
    EXTENDED_3D_IMAGE_SUFFIX,
    is_extended_3d_image_folder,
    find_3d_image_folders_in_path,
)

__all__ = [
    "PNGVisionOCRProcessor",
    "DICOMVisionOCRProcessor",
    "PDFVisionOCRProcessor",
    "VideoVisionOCRProcessor",
    "AgenticTextProcessor",
    "AgenticCSVProcessor",
    "AgenticExcelProcessor",
    "AgenticAudioProcessor",
    "ImageVerificationAgent",
    "VerificationResult",
    "create_verification_step",
    # Optional CT/MRI support
    "MedeProcessor",
    "is_mede_available",
    "get_mede_status",
    "process_ct_mri_file",
    "CT_MRI_EXTENSIONS",
    "EXTENDED_3D_IMAGE_SUFFIX",
    "is_extended_3d_image_folder",
    "find_3d_image_folders_in_path",
]
