"""
Face detection tool for agentic image anonymization.

This tool detects faces in images and returns bounding boxes for redaction.
Uses a trained RetinaNet model with ResNet50 backbone for face detection.
"""

import base64
import io
import logging
from functools import partial
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw
from langchain_core.tools import tool
from pydantic import BaseModel, Field

import torch
import torch.nn as nn
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from ..models import BoundingBox

logger = logging.getLogger(__name__)

# Default model path (relative to project root)
DEFAULT_FACE_MODEL_PATH = Path(__file__).parent.parent / "models" / "face-redaction.pth"

# Model image size for detection
MODEL_IMAGE_SIZE = 640


# ── RetinaNet Face Detection Model ──

def _create_face_detector_model(
    num_classes: int = 2,  # 1 class (face) + background
    pretrained: bool = False,
    trainable_backbone_layers: int = 3,
    score_thresh: float = 0.5,
    nms_thresh: float = 0.5,
    detections_per_img: int = 100
) -> nn.Module:
    """
    Create a RetinaNet model with ResNet50 backbone for face detection.

    Args:
        num_classes: Number of classes (including background)
        pretrained: Whether to use pretrained COCO weights (False for inference with custom weights)
        trainable_backbone_layers: Number of backbone layers to train (0-5)
        score_thresh: Score threshold for detections
        nms_thresh: NMS threshold
        detections_per_img: Maximum detections per image

    Returns:
        RetinaNet model configured for face detection
    """
    if pretrained:
        weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        model = retinanet_resnet50_fpn_v2(
            weights=weights,
            trainable_backbone_layers=trainable_backbone_layers,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            detections_per_img=detections_per_img
        )
    else:
        model = retinanet_resnet50_fpn_v2(
            weights=None,
            trainable_backbone_layers=trainable_backbone_layers,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            detections_per_img=detections_per_img
        )

    # Replace the classification head for our number of classes
    num_anchors = model.head.classification_head.num_anchors
    in_channels = model.backbone.out_channels

    # Create new classification head for face detection (2 classes: background + face)
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=partial(nn.GroupNorm, 32)
    )

    return model


class FaceDetector(nn.Module):
    """
    Wrapper class for face detection model with convenient methods.
    """

    def __init__(
        self,
        pretrained: bool = False,
        trainable_backbone_layers: int = 3,
        score_thresh: float = 0.5,
        nms_thresh: float = 0.5
    ):
        super().__init__()
        self.model = _create_face_detector_model(
            num_classes=2,
            pretrained=pretrained,
            trainable_backbone_layers=trainable_backbone_layers,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh
        )

    def forward(self, images, targets=None):
        """
        Forward pass.

        In training mode with targets: returns loss dict
        In eval mode without targets: returns detections
        """
        return self.model(images, targets)

    @torch.no_grad()
    def detect(self, images: torch.Tensor, score_threshold: float = 0.5):
        """
        Detect faces in images.

        Args:
            images: Tensor of shape (B, C, H, W)
            score_threshold: Minimum confidence score

        Returns:
            List of dictionaries with 'boxes', 'labels', 'scores'
        """
        self.eval()
        predictions = self.model(images)

        # Filter by score threshold
        filtered_preds = []
        for pred in predictions:
            mask = pred['scores'] >= score_threshold
            filtered_preds.append({
                'boxes': pred['boxes'][mask],
                'labels': pred['labels'][mask],
                'scores': pred['scores'][mask]
            })

        return filtered_preds


# ── Model Loading (singleton) ──

_loaded_face_model = None
_loaded_face_device = None


def _get_device() -> torch.device:
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_face_detection_model(
    model_path: Path = None,
    score_threshold: float = 0.5,
    nms_threshold: float = 0.5
) -> tuple:
    """
    Load the trained face detection model (cached singleton).

    Args:
        model_path: Path to the .pth checkpoint file.
                    Defaults to anonymizer/models/face-redaction.pth
        score_threshold: Score threshold for detections
        nms_threshold: NMS threshold for detections

    Returns:
        Tuple of (model, device)
    """
    global _loaded_face_model, _loaded_face_device

    if _loaded_face_model is not None:
        return _loaded_face_model, _loaded_face_device

    if model_path is None:
        model_path = DEFAULT_FACE_MODEL_PATH

    if not model_path.exists():
        raise FileNotFoundError(
            f"Face detection model not found at {model_path}. "
            f"Please place the trained model file (face-redaction.pth) "
            f"in the anonymizer/models/ directory."
        )

    device = _get_device()
    logger.info(f"Loading face detection model from {model_path} (device: {device})")

    model = FaceDetector(
        pretrained=False,
        score_thresh=score_threshold,
        nms_thresh=nms_threshold
    )

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    _loaded_face_model = model
    _loaded_face_device = device

    logger.info("Face detection model loaded successfully")
    if 'metrics' in checkpoint:
        logger.info(f"Model metrics: F1={checkpoint['metrics'].get('f1', 'N/A')}")

    return model, device


class FaceDetectionInput(BaseModel):
    """Input schema for the face detection tool."""
    image_base64: str = Field(
        description="Base64-encoded image to detect faces in"
    )
    score_threshold: float = Field(
        default=0.5,
        description="Confidence threshold for face detections (0.0-1.0)"
    )
    min_face_size: int = Field(
        default=30,
        description="Minimum face size in pixels (width and height)"
    )


class DetectedFace(BaseModel):
    """A detected face with bounding box."""
    x: int = Field(description="X coordinate of top-left corner")
    y: int = Field(description="Y coordinate of top-left corner")
    width: int = Field(description="Width of the bounding box")
    height: int = Field(description="Height of the bounding box")
    confidence: float = Field(default=1.0, description="Detection confidence (0.0-1.0)")


class FaceDetectionResult(BaseModel):
    """Result of face detection."""
    faces_detected: int = Field(description="Number of faces detected")
    faces: List[DetectedFace] = Field(default_factory=list, description="List of detected faces")
    error: Optional[str] = Field(default=None, description="Error message if detection failed")


def _base64_to_pil_image(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data)).convert('RGB')


def _pil_to_tensor(pil_image: Image.Image, target_size: int = MODEL_IMAGE_SIZE) -> tuple:
    """
    Convert PIL Image to tensor for model input.

    Args:
        pil_image: PIL Image object
        target_size: Target size for model input

    Returns:
        Tuple of (tensor, original_size, scale_factors)
    """
    original_size = pil_image.size  # (width, height)

    # Resize image
    image_resized = pil_image.resize((target_size, target_size), Image.BILINEAR)

    # Calculate scale factors
    scale_x = original_size[0] / target_size
    scale_y = original_size[1] / target_size

    # Convert to tensor [0, 1]
    image_tensor = torch.from_numpy(np.array(image_resized)).permute(2, 0, 1).float() / 255.0

    return image_tensor, original_size, (scale_x, scale_y)


def detect_faces_in_image(
    image: Image.Image,
    score_threshold: float = 0.5,
    min_face_size: int = 30,
    model_path: Path = None
) -> List[DetectedFace]:
    """
    Detect faces in a PIL Image using the trained RetinaNet model.

    Args:
        image: PIL Image object (RGB)
        score_threshold: Confidence threshold for detections
        min_face_size: Minimum face size in pixels
        model_path: Optional path to model checkpoint

    Returns:
        List of DetectedFace objects
    """
    # Ensure RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Load model
    model, device = load_face_detection_model(model_path, score_threshold=score_threshold)

    # Preprocess image
    image_tensor, original_size, (scale_x, scale_y) = _pil_to_tensor(image)

    # Add batch dimension and move to device
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # Get predictions
    predictions = model.detect(image_tensor, score_threshold)

    # Process results
    faces = []
    if len(predictions) > 0:
        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()

        for box, score in zip(boxes, scores):
            # Scale boxes back to original image size
            x_min = int(box[0] * scale_x)
            y_min = int(box[1] * scale_y)
            x_max = int(box[2] * scale_x)
            y_max = int(box[3] * scale_y)

            # Clamp to image boundaries
            x_min = max(0, min(x_min, original_size[0] - 1))
            y_min = max(0, min(y_min, original_size[1] - 1))
            x_max = max(0, min(x_max, original_size[0]))
            y_max = max(0, min(y_max, original_size[1]))

            # Calculate width and height
            width = x_max - x_min
            height = y_max - y_min

            # Filter by minimum face size
            if width >= min_face_size and height >= min_face_size:
                faces.append(DetectedFace(
                    x=x_min,
                    y=y_min,
                    width=width,
                    height=height,
                    confidence=float(score)
                ))

    return faces


def detect_faces_from_pil(
    image: Image.Image,
    score_threshold: float = 0.5,
    min_face_size: int = 30
) -> List[DetectedFace]:
    """
    Detect faces in a PIL Image.

    Args:
        image: PIL Image object
        score_threshold: Confidence threshold for detections
        min_face_size: Minimum face size in pixels

    Returns:
        List of DetectedFace objects
    """
    return detect_faces_in_image(image, score_threshold, min_face_size)


def detect_faces_from_base64(
    image_base64: str,
    score_threshold: float = 0.5,
    min_face_size: int = 30
) -> List[DetectedFace]:
    """
    Detect faces in a base64-encoded image.

    Args:
        image_base64: Base64-encoded image string
        score_threshold: Confidence threshold for detections
        min_face_size: Minimum face size in pixels

    Returns:
        List of DetectedFace objects
    """
    pil_image = _base64_to_pil_image(image_base64)
    return detect_faces_in_image(pil_image, score_threshold, min_face_size)


def redact_faces_in_pil_image(
    image: Image.Image,
    faces: List[DetectedFace],
    padding: int = 10,
    fill_color: str = "black"
) -> Image.Image:
    """
    Redact (black out) detected faces in a PIL Image.

    Args:
        image: PIL Image object
        faces: List of DetectedFace objects to redact
        padding: Extra padding around each face
        fill_color: Color to use for redaction

    Returns:
        Image with faces redacted
    """
    image = image.copy()
    draw = ImageDraw.Draw(image)

    for face in faces:
        x1 = max(0, face.x - padding)
        y1 = max(0, face.y - padding)
        x2 = min(image.width, face.x + face.width + padding)
        y2 = min(image.height, face.y + face.height + padding)

        draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=fill_color)

    return image


@tool("detect_faces", args_schema=FaceDetectionInput)
def detect_faces(
    image_base64: str,
    score_threshold: float = 0.5,
    min_face_size: int = 30
) -> str:
    """
    Detect faces in an image for potential redaction.

    Use this tool when you suspect an image contains human faces that should be
    anonymized. The tool will return bounding box coordinates for each detected face.

    Uses a trained RetinaNet model with ResNet50 backbone for accurate face detection.

    Args:
        image_base64: Base64-encoded image to analyze
        score_threshold: Confidence threshold for detections (0.0-1.0, default: 0.5)
        min_face_size: Minimum face size in pixels (default: 30)

    Returns:
        JSON string with detection results including face bounding boxes
    """
    try:
        faces = detect_faces_from_base64(
            image_base64,
            score_threshold=score_threshold,
            min_face_size=min_face_size
        )

        if not faces:
            return "No faces detected in the image."

        result_parts = [f"Detected {len(faces)} face(s):"]
        for i, face in enumerate(faces, 1):
            result_parts.append(
                f"  Face {i}: x={face.x}, y={face.y}, width={face.width}, height={face.height}, confidence={face.confidence:.2f}"
            )

        return "\n".join(result_parts)

    except Exception as e:
        return f"[FACE_DETECTION_FAILED: {str(e)}]"


def get_face_bounding_boxes(
    image: Image.Image,
    score_threshold: float = 0.5,
    min_face_size: int = 30
) -> List[BoundingBox]:
    """
    Detect faces in a PIL image and return as BoundingBox objects.

    This is a convenience function for integration with the image processors.

    Args:
        image: PIL Image object
        score_threshold: Confidence threshold for detections (0.0-1.0)
        min_face_size: Minimum face size in pixels

    Returns:
        List of BoundingBox objects for detected faces
    """
    faces = detect_faces_from_pil(image, score_threshold, min_face_size)

    return [
        BoundingBox(x=face.x, y=face.y, width=face.width, height=face.height)
        for face in faces
    ]
