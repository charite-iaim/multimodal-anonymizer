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
import cv2

from ..base_processor import FileProcessor
from ..config import AnonymizerConfig
from .png_ocr_processor import PNGOCRProcessor


class DICOMProcessor(FileProcessor):
    """Processor for DICOM images using OCR-based LLM via PNG conversion."""

    def __init__(self, config: AnonymizerConfig, save_intermediate: bool = None):
        """
        Initialize DICOM processor.

        Args:
            config: Anonymizer configuration
            save_intermediate: If True, save intermediate PNG files for development.
                             If None, uses config.save_debug_files
        """
        super().__init__(config)
        self.save_intermediate = save_intermediate if save_intermediate is not None else config.save_debug_files
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

    def _dicom_to_images(self, dicom_path: Path) -> tuple[list[Image.Image], pydicom.Dataset, bool]:
        """
        Convert DICOM file to PIL Image(s).

        Args:
            dicom_path: Path to DICOM file

        Returns:
            Tuple of (list of PIL Images, DICOM dataset, is_multiframe)
        """
        # Read DICOM file
        ds = pydicom.dcmread(dicom_path)

        # Get pixel array
        pixel_array = ds.pixel_array

        is_multiframe = len(pixel_array.shape) == 4

        # Handle multi-frame DICOM (4D array)
        if is_multiframe:
            print(f"Multi-frame DICOM detected: {pixel_array.shape[0]} frames")
            frames = []
            for i in range(pixel_array.shape[0]):
                frame = self._process_frame(pixel_array[i], ds)
                frames.append(frame)
            return frames, ds, True
        else:
            # Single frame
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

        return image

    def _images_to_dicom(self, images: list[Image.Image], original_ds: pydicom.Dataset,
                         output_path: Path, is_multiframe: bool) -> None:
        """
        Convert PIL Image(s) back to DICOM format, preserving original DICOM metadata.

        Args:
            images: List of PIL Images with redactions applied
            original_ds: Original DICOM dataset
            output_path: Path to save DICOM file
            is_multiframe: Whether the original was multi-frame
        """
        # Create a copy of the original dataset
        new_ds = original_ds.copy()

        if is_multiframe:
            # Stack all frames into a 4D array
            frame_arrays = []
            for image in images:
                # Convert PIL Image to numpy array
                if image.mode == 'RGB':
                    frame_array = np.array(image)
                elif image.mode == 'L':
                    frame_array = np.array(image)
                else:
                    # Convert to grayscale if in different mode
                    frame_array = np.array(image.convert('L'))

                # Handle MONOCHROME1 inversion (if original was MONOCHROME1)
                if hasattr(original_ds, 'PhotometricInterpretation'):
                    if original_ds.PhotometricInterpretation == 'MONOCHROME1':
                        frame_array = 255 - frame_array

                frame_arrays.append(frame_array)

            # Stack frames: (num_frames, height, width, channels)
            pixel_array = np.stack(frame_arrays, axis=0)

            # Update pixel data
            new_ds.PixelData = pixel_array.tobytes()

            # Update metadata for multi-frame
            new_ds.NumberOfFrames = len(images)
            new_ds.Rows = pixel_array.shape[1]
            new_ds.Columns = pixel_array.shape[2]

            # Set samples per pixel based on array shape
            if len(pixel_array.shape) == 4:
                new_ds.SamplesPerPixel = pixel_array.shape[3]
                new_ds.PhotometricInterpretation = 'RGB'
            else:
                new_ds.SamplesPerPixel = 1
                new_ds.PhotometricInterpretation = 'MONOCHROME2'
        else:
            # Single frame
            image = images[0]

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

            # Update pixel data
            new_ds.PixelData = pixel_array.tobytes()

            # Update relevant metadata
            new_ds.Rows = pixel_array.shape[0]
            new_ds.Columns = pixel_array.shape[1]

            # Set bits allocated/stored based on our 8-bit data
            new_ds.SamplesPerPixel = 1

            # Update photometric interpretation
            if len(pixel_array.shape) == 2:
                new_ds.PhotometricInterpretation = 'MONOCHROME2'
            elif len(pixel_array.shape) == 3:
                new_ds.SamplesPerPixel = pixel_array.shape[2]
                new_ds.PhotometricInterpretation = 'RGB'

        # Set bits allocated/stored based on our 8-bit data
        new_ds.BitsAllocated = 8
        new_ds.BitsStored = 8
        new_ds.HighBit = 7

        # Change to uncompressed transfer syntax since we're using raw pixel data
        # This is required when the original DICOM uses compressed transfer syntax
        new_ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        # Add processing note to DICOM metadata
        if hasattr(new_ds, 'ImageComments'):
            new_ds.ImageComments = f"{new_ds.ImageComments}; Anonymized by LLM-based processor"
        else:
            new_ds.ImageComments = "Anonymized by LLM-based processor"

        # Save DICOM file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        new_ds.save_as(output_path)

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

        # Get dimensions from first frame
        first_frame = frames[0]
        width, height = first_frame.size

        # Ensure frames are in RGB format for video
        rgb_frames = []
        for frame in frames:
            if frame.mode != 'RGB':
                frame = frame.convert('RGB')
            rgb_frames.append(np.array(frame))

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Write frames
        for frame_array in rgb_frames:
            # OpenCV uses BGR, PIL uses RGB, so convert
            frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

        video_writer.release()
        print(f"Created debug video: {output_path}")

    def anonymize(self, input_path: Path, output_path: Path) -> None:
        """
        Anonymize DICOM image by converting to PNG, detecting and redacting PII,
        then converting back to DICOM.

        For multi-frame DICOMs, PHI detection is performed only on the first frame
        and the detected bounding boxes are applied to all frames.
        This significantly reduces processing time.

        Args:
            input_path: Path to input DICOM file
            output_path: Path to save anonymized DICOM file
        """
        print(f"Processing DICOM: {input_path.name}")

        # Convert DICOM to PNG(s)
        print("Converting DICOM to PNG(s)...")
        images, dicom_dataset, is_multiframe = self._dicom_to_images(input_path)

        # For multi-frame DICOMs, use optimized first-frame-only approach
        if is_multiframe:
            print(f"Multi-frame DICOM: Using first-frame-only detection strategy")
            redacted_images = self._anonymize_multiframe_optimized(
                images, input_path, output_path
            )
        else:
            # Single frame - use standard processing
            redacted_images = self._anonymize_singleframe(images[0], input_path, output_path)

        try:
            # Create debug video if multi-frame
            if is_multiframe and self.save_intermediate:
                video_path = output_path.parent / "intermediate" / f"{input_path.stem}_redacted.mp4"
                print(f"Creating debug video from {len(redacted_images)} frames...")
                self._create_video_from_frames(redacted_images, video_path, fps=10)

            # Convert redacted PNG(s) back to DICOM
            print("Converting redacted PNG(s) back to DICOM...")
            self._images_to_dicom(redacted_images, dicom_dataset, output_path, is_multiframe)
            print(f"Saved anonymized DICOM to: {output_path}")

        finally:
            pass

    def _anonymize_singleframe(self, image: Image.Image, input_path: Path, output_path: Path) -> list[Image.Image]:
        """
        Anonymize a single frame using standard OCR + LLM pipeline.

        Args:
            image: PIL Image
            input_path: Original input path
            output_path: Output path

        Returns:
            List containing single redacted image
        """
        temp_files = []

        try:
            # Save intermediate PNG if requested
            if self.save_intermediate:
                intermediate_dir = output_path.parent / "intermediate"
                intermediate_dir.mkdir(parents=True, exist_ok=True)
                intermediate_png_path = intermediate_dir / f"{input_path.stem}_original.png"
                image.save(intermediate_png_path)
                print(f"Saved intermediate PNG to: {intermediate_png_path}")

            # Create temporary PNG file for processing
            temp_png_path = output_path.parent / f"temp_{input_path.stem}.png"
            temp_output_png_path = output_path.parent / f"temp_{input_path.stem}_redacted.png"
            temp_files.extend([temp_png_path, temp_output_png_path])

            # Save image as PNG
            image.save(temp_png_path)

            # Process with PNGOCRProcessor to detect and redact PII
            print("Detecting and redacting PHI using PNGOCRProcessor (OCR-based)...")
            self.png_processor.anonymize(temp_png_path, temp_output_png_path)

            # Load the redacted PNG
            original_redacted_path = temp_output_png_path.with_name(
                f"{temp_output_png_path.stem}_original{temp_output_png_path.suffix}"
            )
            temp_files.append(original_redacted_path)

            if original_redacted_path.exists():
                redacted_image = Image.open(original_redacted_path).copy()
            else:
                redacted_image = Image.open(temp_output_png_path).copy()

            # Save intermediate redacted PNG if requested
            if self.save_intermediate:
                redacted_intermediate_path = intermediate_png_path.parent / f"{input_path.stem}_redacted.png"
                redacted_image.save(redacted_intermediate_path)
                print(f"Saved redacted intermediate PNG to: {redacted_intermediate_path}")

            # Copy the JSON output only if debug mode is enabled
            json_source = temp_output_png_path.with_suffix(".json")
            if json_source.exists():
                if self.config.save_debug_files:
                    json_dest = output_path.with_suffix(".json")
                    import shutil
                    shutil.copy2(json_source, json_dest)
                    print(f"Saved detection results to: {json_dest}")
            temp_files.append(json_source)

            return [redacted_image]

        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                if temp_file.exists():
                    temp_file.unlink()

    def _anonymize_multiframe_optimized(
        self,
        images: list[Image.Image],
        input_path: Path,
        output_path: Path
    ) -> list[Image.Image]:
        """
        Anonymize multi-frame DICOM using first-frame-only detection.

        Analyzes only the first frame and applies detected bounding boxes to all frames.

        Args:
            images: List of all frame images
            input_path: Original input path
            output_path: Output path

        Returns:
            List of all redacted images
        """
        num_frames = len(images)
        print(f"Analyzing first frame only (out of {num_frames} total frames)...")

        # Detect PHI in first frame only
        print(f"Detecting PHI in first frame...")
        first_frame_bboxes = self.png_processor.detect_pii_bboxes(images[0])
        print(f"  Found {len(first_frame_bboxes)} PHI elements")

        # Save intermediate first frame if requested
        if self.save_intermediate:
            intermediate_dir = output_path.parent / "intermediate"
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            first_frame_path = intermediate_dir / f"{input_path.stem}_frame0000_original.png"
            images[0].save(first_frame_path)
            print(f"Saved first frame to: {first_frame_path}")

        # Apply redactions to all frames using first frame's bounding boxes
        print(f"Applying redactions to all {num_frames} frames...")
        redacted_images = []
        for i, image in enumerate(images):
            # Show progress for every 10 frames
            if i % 10 == 0 or i == num_frames - 1:
                print(f"  Redacting frame {i + 1}/{num_frames}...")

            # Apply redactions using first frame's bboxes
            redacted_image = self.png_processor._apply_redactions(image.copy(), first_frame_bboxes)
            redacted_images.append(redacted_image)

        # Save intermediate redacted first frame if requested
        if self.save_intermediate:
            redacted_path = intermediate_dir / f"{input_path.stem}_frame0000_redacted.png"
            redacted_images[0].save(redacted_path)
            print(f"Saved redacted first frame to: {redacted_path}")

        # Save JSON with detection results only if debug mode is enabled
        if self.config.save_debug_files:
            from ..models import PIIDetectionResult
            pii_result = PIIDetectionResult(pii_elements=first_frame_bboxes)

            json_dest = output_path.with_suffix(".json")
            output_data = {
                "metadata": {
                    "input_file": str(input_path.name),
                    "output_file": str(output_path.name),
                    "timestamp": datetime.now().isoformat(),
                    "processing_method": "ocr_multiframe_first_frame_only",
                    "total_frames": num_frames,
                    "analyzed_frames": 1,
                    "total_pii_elements": len(first_frame_bboxes)
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
                    for element in first_frame_bboxes
                ]
            }

            import json
            with open(json_dest, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"Saved detection results to: {json_dest}")

        return redacted_images
