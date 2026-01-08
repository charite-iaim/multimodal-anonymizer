"""
DICOM image processor for anonymization using OCR-based LLM.
Converts DICOM to PNG, processes with PNGOCRProcessor, then converts back to DICOM.
"""

import io
import json
import numpy as np
from pathlib import Path
from PIL import Image
from datetime import datetime
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

from ..base_processor import FileProcessor
from ..config import AnonymizerConfig
from .png_ocr_processor import PNGOCRProcessor


class DICOMProcessor(FileProcessor):
    """Processor for DICOM images using OCR-based LLM via PNG conversion."""

    def __init__(self, config: AnonymizerConfig, save_intermediate: bool = True):
        """
        Initialize DICOM processor.

        Args:
            config: Anonymizer configuration
            save_intermediate: If True, save intermediate PNG files for development
        """
        super().__init__(config)
        self.save_intermediate = save_intermediate
        self.png_processor = PNGOCRProcessor(config)

    def can_process(self, file_path: Path) -> bool:
        """Check if file is a DICOM image."""
        # Check file extension
        if file_path.suffix.lower() in ['.dcm', '.dicom']:
            return True

        # Try to read as DICOM (some DICOM files have no extension)
        try:
            pydicom.dcmread(file_path, stop_before_pixels=True)
            return True
        except:
            return False

    def extract_content(self, file_path: Path) -> str:
        """
        Extract content from DICOM file.

        Args:
            file_path: Path to DICOM file

        Returns:
            String representation of DICOM metadata
        """
        ds = pydicom.dcmread(file_path)
        return str(ds)

    def _dicom_to_image(self, dicom_path: Path) -> tuple[Image.Image, pydicom.Dataset]:
        """
        Convert DICOM file to PIL Image.

        Args:
            dicom_path: Path to DICOM file

        Returns:
            Tuple of (PIL Image, DICOM dataset)
        """
        # Read DICOM file
        ds = pydicom.dcmread(dicom_path)

        # Get pixel array
        pixel_array = ds.pixel_array

        # Apply VOI LUT (Value of Interest Lookup Table) for proper windowing
        try:
            pixel_array = apply_voi_lut(pixel_array, ds)
        except:
            # If VOI LUT fails, continue with raw pixel array
            pass

        # Normalize to 0-255 range
        pixel_array = pixel_array.astype(float)
        pixel_min = pixel_array.min()
        pixel_max = pixel_array.max()

        if pixel_max > pixel_min:
            pixel_array = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255.0)

        pixel_array = pixel_array.astype(np.uint8)

        # Convert to PIL Image
        # Handle different photometric interpretations
        if hasattr(ds, 'PhotometricInterpretation'):
            if ds.PhotometricInterpretation == 'MONOCHROME1':
                # Invert for MONOCHROME1 (lower values = brighter)
                pixel_array = 255 - pixel_array

        # Create PIL Image
        if len(pixel_array.shape) == 2:
            # Grayscale image
            image = Image.fromarray(pixel_array, mode='L')
        elif len(pixel_array.shape) == 3:
            # Color image
            image = Image.fromarray(pixel_array, mode='RGB')
        else:
            raise ValueError(f"Unexpected pixel array shape: {pixel_array.shape}")

        return image, ds

    def _image_to_dicom(self, image: Image.Image, original_ds: pydicom.Dataset, output_path: Path) -> None:
        """
        Convert PIL Image back to DICOM format, preserving original DICOM metadata.

        Args:
            image: PIL Image with redactions applied
            original_ds: Original DICOM dataset
            output_path: Path to save DICOM file
        """
        # Convert PIL Image to numpy array
        if image.mode == 'RGB':
            pixel_array = np.array(image)
        elif image.mode == 'L':
            pixel_array = np.array(image)
        else:
            # Convert to grayscale if in different mode
            pixel_array = np.array(image.convert('L'))

        # Handle MONOCHROME1 inversion (if original was MONOCHROME1)
        if hasattr(original_ds, 'PhotometricInterpretation'):
            if original_ds.PhotometricInterpretation == 'MONOCHROME1':
                pixel_array = 255 - pixel_array

        # Create a copy of the original dataset
        new_ds = original_ds.copy()

        # Update pixel data
        new_ds.PixelData = pixel_array.tobytes()

        # Update relevant metadata
        new_ds.Rows = pixel_array.shape[0]
        new_ds.Columns = pixel_array.shape[1]

        # Set bits allocated/stored based on our 8-bit data
        new_ds.BitsAllocated = 8
        new_ds.BitsStored = 8
        new_ds.HighBit = 7
        new_ds.SamplesPerPixel = 1

        # Update photometric interpretation
        if len(pixel_array.shape) == 2:
            new_ds.PhotometricInterpretation = 'MONOCHROME2'

        # Add processing note to DICOM metadata
        if hasattr(new_ds, 'ImageComments'):
            new_ds.ImageComments = f"{new_ds.ImageComments}; Anonymized by LLM-based processor"
        else:
            new_ds.ImageComments = "Anonymized by LLM-based processor"

        # Save DICOM file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        new_ds.save_as(output_path)

    def anonymize(self, input_path: Path, output_path: Path) -> None:
        """
        Anonymize DICOM image by converting to PNG, detecting and redacting PII,
        then converting back to DICOM.

        Args:
            input_path: Path to input DICOM file
            output_path: Path to save anonymized DICOM file
        """
        print(f"Processing DICOM: {input_path.name}")

        # Convert DICOM to PNG
        print("Converting DICOM to PNG...")
        image, dicom_dataset = self._dicom_to_image(input_path)

        # Save intermediate PNG if requested
        intermediate_png_path = None
        if self.save_intermediate:
            intermediate_dir = output_path.parent / "intermediate"
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            intermediate_png_path = intermediate_dir / f"{input_path.stem}_original.png"
            image.save(intermediate_png_path)
            print(f"Saved intermediate PNG to: {intermediate_png_path}")

        # Create temporary PNG file for processing
        temp_png_path = output_path.parent / f"temp_{input_path.stem}.png"
        temp_output_png_path = output_path.parent / f"temp_{input_path.stem}_redacted.png"

        try:
            # Save image as PNG
            image.save(temp_png_path)

            # Process with PNGOCRProcessor to detect and redact PII
            print("Detecting and redacting PHI using PNGOCRProcessor (OCR-based)...")
            self.png_processor.anonymize(temp_png_path, temp_output_png_path)

            # Load the redacted PNG
            # The PNGOCRProcessor saves both resized and original versions
            # We want to use the original size version
            original_redacted_path = temp_output_png_path.with_name(
                f"{temp_output_png_path.stem}_original{temp_output_png_path.suffix}"
            )

            if original_redacted_path.exists():
                redacted_image = Image.open(original_redacted_path)
            else:
                # Fallback to resized version if original not found
                redacted_image = Image.open(temp_output_png_path)

            # Save intermediate redacted PNG if requested
            if self.save_intermediate and intermediate_png_path:
                redacted_intermediate_path = intermediate_png_path.parent / f"{input_path.stem}_redacted.png"
                redacted_image.save(redacted_intermediate_path)
                print(f"Saved redacted intermediate PNG to: {redacted_intermediate_path}")

            # Convert redacted PNG back to DICOM
            print("Converting redacted PNG back to DICOM...")
            self._image_to_dicom(redacted_image, dicom_dataset, output_path)
            print(f"Saved anonymized DICOM to: {output_path}")

            # Copy the JSON output from PNGProcessor if it exists
            json_source = temp_output_png_path.with_suffix(".json")
            if json_source.exists():
                json_dest = output_path.with_suffix(".json")
                import shutil
                shutil.copy2(json_source, json_dest)
                print(f"Saved detection results to: {json_dest}")

        finally:
            # Clean up temporary files
            if temp_png_path.exists():
                temp_png_path.unlink()
            if temp_output_png_path.exists():
                temp_output_png_path.unlink()
            # Clean up the original version created by PNGProcessor
            original_temp = temp_output_png_path.with_name(
                f"{temp_output_png_path.stem}_original{temp_output_png_path.suffix}"
            )
            if original_temp.exists():
                original_temp.unlink()
            # Clean up JSON temp file
            json_temp = temp_output_png_path.with_suffix(".json")
            if json_temp.exists():
                json_temp.unlink()
