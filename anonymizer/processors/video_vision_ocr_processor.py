"""
Video processor for anonymization using Vision LLM + OCR.
Supports MP4 files and DICOM videos (multi-frame DICOMs).

This processor can operate in two modes:
1. First-frame-only (default): PHI detection on first frame, applied to all frames
2. Frame-by-frame: PHI detection on every frame (resource-intensive)
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
from datetime import datetime
import cv2

# Increase PIL's max image pixels limit to handle large video frames
Image.MAX_IMAGE_PIXELS = 300000000  # 300 million pixels

from ..base_processor import FileProcessor
from ..config import AnonymizerConfig
from ..prompt_config import PromptConfig, DEFAULT_PROMPT_CONFIG
from .png_vision_ocr_processor import PNGVisionOCRProcessor


class VideoVisionOCRProcessor(FileProcessor):
    """Processor for video files (MP4) using Vision LLM + OCR approach."""

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
        Initialize Video Vision+OCR processor.

        Args:
            config: Anonymizer configuration
            save_intermediate: If True, save intermediate files for development.
                             If None, uses config.save_debug_files
            similarity_threshold: Minimum similarity for fuzzy text matching (0.0-1.0)
            enable_verification: If True, run verification agent after initial redaction
            check_over_redaction: If True, also check for over-redaction
            max_verification_rounds: Maximum rounds of verify-and-redact
            prompt_config: Optional custom prompt configuration
            process_all_frames: If True, run PHI detection on every frame (resource-intensive).
                              If False (default), detect on first frame only.
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

    def can_process(self, file_path: Path) -> bool:
        """Check if file is a video file (MP4)."""
        return file_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']

    def extract_content(self, file_path: Path) -> str:
        """Extract content description from video file."""
        cap = cv2.VideoCapture(str(file_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        return f"Video file: {frame_count} frames, {fps} FPS, {width}x{height}"

    def _extract_frames(self, video_path: Path) -> tuple[list[Image.Image], float, tuple[int, int]]:
        """
        Extract all frames from video file.

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (list of PIL Images, fps, (width, height))
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video info: {frame_count} frames, {fps} FPS, {width}x{height}")

        frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  Extracted {frame_idx}/{frame_count} frames...")

        cap.release()
        print(f"Extracted {len(frames)} frames total")

        return frames, fps, (width, height)

    def _create_video_from_frames(self, frames: list[Image.Image], output_path: Path, fps: float) -> None:
        """
        Create an MP4 video from a list of frames.

        Args:
            frames: List of PIL Images
            output_path: Path to save the MP4 file
            fps: Frames per second for the video
        """
        if not frames:
            return

        first_frame = frames[0]
        width, height = first_frame.size

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use H.264 codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        for idx, frame in enumerate(frames):
            if frame.mode != 'RGB':
                frame = frame.convert('RGB')
            frame_array = np.array(frame)
            frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

            if idx % 100 == 0:
                print(f"  Writing frame {idx + 1}/{len(frames)}...")

        video_writer.release()
        print(f"Created video: {output_path}")

    def anonymize(self, input_path: Path, output_path: Path) -> None:
        """
        Anonymize video using Vision LLM + OCR approach.

        The processing mode depends on self.process_all_frames:
        - False (default): PHI detection on first frame only, bboxes applied to all frames
        - True: PHI detection on every frame individually (resource-intensive)

        Args:
            input_path: Path to input video file
            output_path: Path to save anonymized video file
        """
        print(f"Processing Video (Vision+OCR): {input_path.name}")
        mode_str = "frame-by-frame" if self.process_all_frames else "first-frame-only"
        print(f"Processing mode: {mode_str}")

        print("Extracting frames from video...")
        frames, fps, (width, height) = self._extract_frames(input_path)
        num_frames = len(frames)

        if num_frames == 0:
            raise ValueError("No frames extracted from video")

        if self.process_all_frames:
            print(f"Frame-by-frame processing: Analyzing all {num_frames} frames")
            redacted_frames = self._anonymize_all_frames(frames, input_path, output_path)
        else:
            print(f"First-frame-only processing: Analyzing first frame, applying to all {num_frames} frames")
            redacted_frames = self._anonymize_first_frame_only(frames, input_path, output_path)

        # Save intermediate frames if enabled
        if self.save_intermediate:
            intermediate_dir = output_path.parent / "intermediate"
            intermediate_dir.mkdir(parents=True, exist_ok=True)

            # Save first and last frame for comparison
            frames[0].save(intermediate_dir / f"{input_path.stem}_frame0000_original.png")
            redacted_frames[0].save(intermediate_dir / f"{input_path.stem}_frame0000_redacted.png")
            if len(frames) > 1:
                frames[-1].save(intermediate_dir / f"{input_path.stem}_frame{len(frames)-1:04d}_original.png")
                redacted_frames[-1].save(intermediate_dir / f"{input_path.stem}_frame{len(frames)-1:04d}_redacted.png")

        # Create output video
        print("Creating output video...")
        self._create_video_from_frames(redacted_frames, output_path, fps)
        print(f"Saved anonymized video to: {output_path}")

    def _anonymize_first_frame_only(
        self,
        frames: list[Image.Image],
        input_path: Path,
        output_path: Path
    ) -> list[Image.Image]:
        """
        Anonymize video using first-frame-only detection.

        PHI detection is performed only on the first frame and the detected
        bounding boxes are applied to all frames.

        Args:
            frames: List of all frame images
            input_path: Original input path
            output_path: Output path

        Returns:
            List of all redacted images
        """
        num_frames = len(frames)
        original_first_frame = frames[0].copy()

        print(f"Detecting PHI in first frame using Vision+OCR...")
        first_frame_bboxes = self.png_processor.detect_pii_bboxes(frames[0])
        print(f"  Found {len(first_frame_bboxes)} PHI elements")

        # Apply initial redactions to first frame for verification
        redacted_first_frame = self.png_processor._apply_redactions(frames[0].copy(), first_frame_bboxes)

        # Verification phase on first frame
        verification_result = None
        additional_elements = []
        all_bboxes = list(first_frame_bboxes)

        if self.enable_verification and self.png_processor._verification_agent is not None:
            print("\n=== Verification Phase (First Frame) ===")
            redacted_first_frame, verification_result, additional_elements = self.png_processor._run_verification(
                redacted_first_frame,
                original_first_frame if self.check_over_redaction else None
            )
            all_bboxes.extend(additional_elements)

        # Apply ALL bounding boxes to all frames
        print(f"Applying redactions to all {num_frames} frames...")
        redacted_frames = []
        for i, frame in enumerate(frames):
            if i % 50 == 0 or i == num_frames - 1:
                print(f"  Redacting frame {i + 1}/{num_frames}...")

            redacted_frame = self.png_processor._apply_redactions(frame.copy(), all_bboxes)
            redacted_frames.append(redacted_frame)

        # Save debug JSON
        if self.config.save_debug_files:
            self._save_debug_json(
                output_path,
                input_path,
                all_bboxes,
                first_frame_bboxes,
                additional_elements,
                verification_result,
                num_frames,
                processing_method="vision_ocr_video_first_frame_only"
            )

        return redacted_frames

    def _anonymize_all_frames(
        self,
        frames: list[Image.Image],
        input_path: Path,
        output_path: Path
    ) -> list[Image.Image]:
        """
        Anonymize video using frame-by-frame detection.

        PHI detection is performed on every frame individually. This is
        resource-intensive but can catch PHI that only appears in certain frames.

        Args:
            frames: List of all frame images
            input_path: Original input path
            output_path: Output path

        Returns:
            List of all redacted images
        """
        num_frames = len(frames)
        redacted_frames = []
        all_frame_bboxes = []

        for i, frame in enumerate(frames):
            print(f"\n=== Processing Frame {i + 1}/{num_frames} ===")
            original_frame = frame.copy()

            # Detect PHI in this frame
            frame_bboxes = self.png_processor.detect_pii_bboxes(frame)
            print(f"  Found {len(frame_bboxes)} PHI elements")

            # Apply redactions
            redacted_frame = self.png_processor._apply_redactions(frame.copy(), frame_bboxes)

            # Verification phase for this frame
            verification_result = None
            additional_elements = []
            frame_all_bboxes = list(frame_bboxes)

            if self.enable_verification and self.png_processor._verification_agent is not None:
                print(f"  Verification Phase (Frame {i + 1})...")
                redacted_frame, verification_result, additional_elements = self.png_processor._run_verification(
                    redacted_frame,
                    original_frame if self.check_over_redaction else None
                )
                frame_all_bboxes.extend(additional_elements)

            redacted_frames.append(redacted_frame)
            all_frame_bboxes.append({
                "frame_index": i,
                "bboxes": frame_all_bboxes,
                "initial_count": len(frame_bboxes),
                "verification_additional": len(additional_elements)
            })

        # Save debug JSON with all frame data
        if self.config.save_debug_files:
            self._save_debug_json_all_frames(
                output_path,
                input_path,
                all_frame_bboxes,
                num_frames
            )

        return redacted_frames

    def _save_debug_json(
        self,
        output_path: Path,
        input_path: Path,
        all_bboxes: list,
        initial_bboxes: list,
        additional_elements: list,
        verification_result,
        num_frames: int,
        processing_method: str
    ) -> None:
        """Save debug JSON file for first-frame-only processing."""
        json_dest = output_path.with_suffix(".json")
        output_data = {
            "metadata": {
                "input_file": str(input_path.name),
                "output_file": str(output_path.name),
                "timestamp": datetime.now().isoformat(),
                "processing_method": processing_method,
                "verification_enabled": self.enable_verification,
                "total_frames": num_frames,
                "analyzed_frames": 1,
                "total_pii_elements": len(all_bboxes),
                "initial_pii_elements": len(initial_bboxes),
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

        with open(json_dest, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Saved detection results to: {json_dest}")

    def _save_debug_json_all_frames(
        self,
        output_path: Path,
        input_path: Path,
        all_frame_bboxes: list,
        num_frames: int
    ) -> None:
        """Save debug JSON file for frame-by-frame processing."""
        json_dest = output_path.with_suffix(".json")

        total_pii = sum(len(f["bboxes"]) for f in all_frame_bboxes)
        total_initial = sum(f["initial_count"] for f in all_frame_bboxes)
        total_verification = sum(f["verification_additional"] for f in all_frame_bboxes)

        output_data = {
            "metadata": {
                "input_file": str(input_path.name),
                "output_file": str(output_path.name),
                "timestamp": datetime.now().isoformat(),
                "processing_method": "vision_ocr_video_all_frames",
                "verification_enabled": self.enable_verification,
                "total_frames": num_frames,
                "analyzed_frames": num_frames,
                "total_pii_elements": total_pii,
                "total_initial_elements": total_initial,
                "total_verification_additional": total_verification
            },
            "frames": [
                {
                    "frame_index": frame_data["frame_index"],
                    "pii_count": len(frame_data["bboxes"]),
                    "initial_count": frame_data["initial_count"],
                    "verification_additional": frame_data["verification_additional"],
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
                        for element in frame_data["bboxes"]
                    ]
                }
                for frame_data in all_frame_bboxes
            ]
        }

        with open(json_dest, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Saved detection results to: {json_dest}")
