"""
TrOCR-based handwriting recognition module.

This module provides a wrapper around a fine-tuned TrOCR model for handwritten
text recognition. TrOCR (Transformer-based Optical Character Recognition) is
particularly effective at recognizing handwritten text in medical images.

The module serves as the fallback OCR engine in the ImageVerificationAgent
for detecting handwritten text that EasyOCR may miss.
"""

import os
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
from PIL import Image

# Default paths for the TrOCR model (relative to this file's location)
# anonymizer/tools/trocr_ocr.py -> anonymizer/models/trocr/
_THIS_DIR = Path(__file__).parent  # anonymizer/tools/
_MODELS_DIR = _THIS_DIR.parent / "models" / "trocr"  # anonymizer/models/trocr/

DEFAULT_MODEL_WEIGHTS_PATH = _MODELS_DIR / "model.safetensors"
DEFAULT_MODEL_CONFIG_PATH = _MODELS_DIR

# Lazy imports to avoid loading heavy libraries at module import time
_trocr_model = None
_trocr_processor = None
_torch = None
_TROCR_AVAILABLE = None


def _check_trocr_available() -> bool:
    """Check if TrOCR dependencies are available."""
    global _TROCR_AVAILABLE
    if _TROCR_AVAILABLE is not None:
        return _TROCR_AVAILABLE

    try:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        _TROCR_AVAILABLE = True
    except ImportError:
        _TROCR_AVAILABLE = False

    return _TROCR_AVAILABLE


def is_trocr_available() -> bool:
    """Check if TrOCR is available for use."""
    return _check_trocr_available()


class TrOCRHandwritingRecognizer:
    """
    TrOCR-based handwriting recognition engine.

    This class wraps a fine-tuned TrOCR model for recognizing handwritten text
    in images. It's designed to work with cropped text regions from a text
    detection model or OCR bounding boxes.

    TrOCR uses a Vision Transformer (ViT) encoder and a text Transformer decoder
    to directly convert image patches into text tokens.
    """

    def __init__(
        self,
        model_weights_path: Optional[str] = None,
        model_config_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the TrOCR handwriting recognizer.

        Args:
            model_weights_path: Path to model.safetensors file. If None, uses default.
            model_config_path: Path to directory containing config.json, tokenizer files, etc.
                              If None, uses default checkpoint directory.
            device: Device to run inference on ('cuda', 'mps', 'cpu'). Auto-detected if None.
        """
        if not _check_trocr_available():
            raise ImportError(
                "TrOCR dependencies not available. Install with: "
                "pip install transformers torch safetensors"
            )

        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from safetensors.torch import load_file

        self._torch = torch

        # Resolve paths
        self.model_weights_path = Path(model_weights_path) if model_weights_path else DEFAULT_MODEL_WEIGHTS_PATH
        self.model_config_path = Path(model_config_path) if model_config_path else DEFAULT_MODEL_CONFIG_PATH

        # Validate paths
        if not self.model_weights_path.exists():
            raise FileNotFoundError(f"Model weights not found at: {self.model_weights_path}")
        if not self.model_config_path.exists():
            raise FileNotFoundError(f"Model config directory not found at: {self.model_config_path}")

        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"  TrOCR: Loading model on {self.device}...")

        # Load processor (tokenizer + image processor) from config directory
        self.processor = TrOCRProcessor.from_pretrained(str(self.model_config_path))

        # Load model architecture from config directory
        self.model = VisionEncoderDecoderModel.from_pretrained(str(self.model_config_path))

        # Load fine-tuned weights from safetensors file
        state_dict = load_file(str(self.model_weights_path))

        # Handle tied embeddings: if output_projection.weight is missing but embeddings are tied,
        # the model will use the embedding weights for output projection automatically
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

        # Only raise error if there are unexpected keys or missing keys that aren't tied weights
        if unexpected_keys:
            raise ValueError(f"Unexpected keys in state_dict: {unexpected_keys}")
        if missing_keys and "output_projection.weight" not in str(missing_keys):
            raise ValueError(f"Missing keys in state_dict: {missing_keys}")

        # Move to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        # Get model config for max length
        self.max_length = self.model.config.decoder.max_position_embeddings

        print(f"  TrOCR: Model loaded successfully (vocab_size={self.model.config.vocab_size})")

    def recognize_text(
        self,
        image: Image.Image,
    ) -> Tuple[str, float]:
        """
        Recognize text in a single image crop.

        This method expects an image containing a single line or word of text.
        For full-page OCR, use recognize_text_regions() with bounding boxes.

        Args:
            image: PIL Image containing handwritten text

        Returns:
            Tuple of (recognized_text, confidence_score)
        """
        import torch

        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        # Generate text with default parameters for maximum speed
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)

        # Decode output tokens to text
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Return fixed confidence since we're not computing scores for speed
        return generated_text.strip(), 0.8

    def recognize_text_regions(
        self,
        image: Image.Image,
        bboxes: List[Tuple[int, int, int, int]],
        padding: int = 5,
    ) -> List[Tuple[str, Tuple[int, int, int, int], float]]:
        """
        Recognize text in multiple regions of an image.

        Args:
            image: Full PIL Image
            bboxes: List of bounding boxes as (x, y, width, height) tuples
            padding: Extra padding around each bbox when cropping

        Returns:
            List of (text, bbox, confidence) tuples
        """
        results = []

        for bbox in bboxes:
            x, y, w, h = bbox

            # Add padding and clip to image bounds
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.width, x + w + padding)
            y2 = min(image.height, y + h + padding)

            # Crop region
            crop = image.crop((x1, y1, x2, y2))

            # Skip if crop is too small
            if crop.width < 10 or crop.height < 10:
                continue

            # Recognize text
            text, confidence = self.recognize_text(crop)

            if text:  # Only include non-empty results
                results.append((text, bbox, confidence))

        return results

    def scan_full_image(
        self,
        image: Image.Image,
        text_detector: Optional[callable] = None,
        min_confidence: float = 0.3
    ) -> List[Tuple[str, Tuple[int, int, int, int], float]]:
        """
        Scan a full image for handwritten text.

        If a text_detector function is provided, it's used to find text regions.
        Otherwise, a simple sliding window approach is used.

        Args:
            image: Full PIL Image to scan
            text_detector: Optional function that takes an image and returns
                          list of (x, y, w, h) bounding boxes
            min_confidence: Minimum confidence threshold for results

        Returns:
            List of (text, bbox, confidence) tuples
        """
        if text_detector is not None:
            # Use provided text detector
            bboxes = text_detector(image)
            results = self.recognize_text_regions(image, bboxes)
        else:
            # Simple approach: try the full image as a single text region
            # This works well for images that are already cropped to text
            text, confidence = self.recognize_text(image)
            if text and confidence >= min_confidence:
                results = [(text, (0, 0, image.width, image.height), confidence)]
            else:
                results = []

        # Filter by confidence
        results = [(t, b, c) for t, b, c in results if c >= min_confidence]

        return results


# Global instance for lazy initialization
_recognizer_instance: Optional[TrOCRHandwritingRecognizer] = None


def get_trocr_recognizer(
    model_weights_path: Optional[str] = None,
    model_config_path: Optional[str] = None,
    device: Optional[str] = None
) -> TrOCRHandwritingRecognizer:
    """
    Get or create the global TrOCR recognizer instance.

    This function provides lazy initialization and caching of the TrOCR model
    to avoid loading the model multiple times.

    Args:
        model_weights_path: Path to model weights (only used on first call)
        model_config_path: Path to model config (only used on first call)
        device: Device to use (only used on first call)

    Returns:
        TrOCRHandwritingRecognizer instance
    """
    global _recognizer_instance

    if _recognizer_instance is None:
        _recognizer_instance = TrOCRHandwritingRecognizer(
            model_weights_path=model_weights_path,
            model_config_path=model_config_path,
            device=device
        )

    return _recognizer_instance


def recognize_handwriting(
    image: Image.Image,
    bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
    model_weights_path: Optional[str] = None,
    model_config_path: Optional[str] = None
) -> List[Tuple[str, Tuple[int, int, int, int], float]]:
    """
    Convenience function to recognize handwriting in an image.

    Args:
        image: PIL Image containing handwritten text
        bboxes: Optional list of (x, y, w, h) bounding boxes. If None,
               treats the entire image as a single text region.
        model_weights_path: Optional custom model weights path
        model_config_path: Optional custom model config path

    Returns:
        List of (text, bbox, confidence) tuples
    """
    recognizer = get_trocr_recognizer(model_weights_path, model_config_path)

    if bboxes:
        return recognizer.recognize_text_regions(image, bboxes)
    else:
        return recognizer.scan_full_image(image)
