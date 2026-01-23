"""
Face detection tool for agentic image anonymization.

This tool detects faces in images and returns bounding boxes for redaction.
Uses OpenCV's Haar cascade classifier for face detection.
"""

import base64
import io
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..models import BoundingBox


class FaceDetectionInput(BaseModel):
    """Input schema for the face detection tool."""
    image_base64: str = Field(
        description="Base64-encoded image to detect faces in"
    )
    scale_factor: float = Field(
        default=1.1,
        description="Scale factor for multi-scale detection (1.1 = 10% increase per scale)"
    )
    min_neighbors: int = Field(
        default=5,
        description="Minimum number of neighbors for detection (higher = fewer false positives)"
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


def _base64_to_cv2_image(base64_str: str) -> np.ndarray:
    """Convert base64 string to OpenCV image."""
    image_data = base64.b64decode(base64_str)
    image_array = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


def _pil_to_cv2_image(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV image."""
    if pil_image.mode == 'RGBA':
        pil_image = pil_image.convert('RGB')
    elif pil_image.mode == 'L':
        pil_image = pil_image.convert('RGB')
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def _cv2_to_pil_image(cv2_image: np.ndarray) -> Image.Image:
    """Convert OpenCV image to PIL Image."""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


def detect_faces_in_image(
    image: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_face_size: int = 30
) -> List[DetectedFace]:
    """
    Detect faces in an OpenCV image using Haar cascade classifier.

    Args:
        image: OpenCV image (BGR format)
        scale_factor: Scale factor for multi-scale detection
        min_neighbors: Minimum neighbors for detection
        min_face_size: Minimum face size in pixels

    Returns:
        List of DetectedFace objects
    """
    # Load the Haar cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    # Also load profile face detector for side views
    profile_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_profileface.xml'
    )

    # Convert to grayscale for detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect frontal faces
    frontal_faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_face_size, min_face_size),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Detect profile faces (left-facing)
    profile_faces = profile_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_face_size, min_face_size),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Detect profile faces (right-facing by flipping)
    flipped = cv2.flip(gray, 1)
    profile_faces_right = profile_cascade.detectMultiScale(
        flipped,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_face_size, min_face_size),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Convert right-facing detections back to original coordinates
    img_width = image.shape[1]
    profile_faces_right_converted = []
    for (x, y, w, h) in profile_faces_right:
        # Flip x coordinate back
        new_x = img_width - x - w
        profile_faces_right_converted.append((new_x, y, w, h))

    # Combine all detections
    all_faces = []

    for (x, y, w, h) in frontal_faces:
        all_faces.append(DetectedFace(x=int(x), y=int(y), width=int(w), height=int(h)))

    for (x, y, w, h) in profile_faces:
        all_faces.append(DetectedFace(x=int(x), y=int(y), width=int(w), height=int(h)))

    for (x, y, w, h) in profile_faces_right_converted:
        all_faces.append(DetectedFace(x=int(x), y=int(y), width=int(w), height=int(h)))

    # Remove duplicate/overlapping detections
    all_faces = _remove_overlapping_faces(all_faces)

    return all_faces


def _remove_overlapping_faces(faces: List[DetectedFace], overlap_threshold: float = 0.3) -> List[DetectedFace]:
    """Remove overlapping face detections using non-maximum suppression."""
    if not faces:
        return []

    # Convert to numpy arrays for NMS
    boxes = np.array([[f.x, f.y, f.x + f.width, f.y + f.height] for f in faces])

    # Simple NMS implementation
    keep = []
    indices = list(range(len(faces)))

    while indices:
        # Take the first box
        i = indices[0]
        keep.append(i)
        indices = indices[1:]

        # Remove overlapping boxes
        remaining = []
        for j in indices:
            # Calculate IoU
            x1 = max(boxes[i][0], boxes[j][0])
            y1 = max(boxes[i][1], boxes[j][1])
            x2 = min(boxes[i][2], boxes[j][2])
            y2 = min(boxes[i][3], boxes[j][3])

            inter_area = max(0, x2 - x1) * max(0, y2 - y1)
            box_i_area = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
            box_j_area = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
            union_area = box_i_area + box_j_area - inter_area

            iou = inter_area / union_area if union_area > 0 else 0

            if iou < overlap_threshold:
                remaining.append(j)

        indices = remaining

    return [faces[i] for i in keep]


def detect_faces_from_pil(
    image: Image.Image,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_face_size: int = 30
) -> List[DetectedFace]:
    """
    Detect faces in a PIL Image.

    Args:
        image: PIL Image object
        scale_factor: Scale factor for multi-scale detection
        min_neighbors: Minimum neighbors for detection
        min_face_size: Minimum face size in pixels

    Returns:
        List of DetectedFace objects
    """
    cv2_image = _pil_to_cv2_image(image)
    return detect_faces_in_image(cv2_image, scale_factor, min_neighbors, min_face_size)


def detect_faces_from_base64(
    image_base64: str,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_face_size: int = 30
) -> List[DetectedFace]:
    """
    Detect faces in a base64-encoded image.

    Args:
        image_base64: Base64-encoded image string
        scale_factor: Scale factor for multi-scale detection
        min_neighbors: Minimum neighbors for detection
        min_face_size: Minimum face size in pixels

    Returns:
        List of DetectedFace objects
    """
    cv2_image = _base64_to_cv2_image(image_base64)
    return detect_faces_in_image(cv2_image, scale_factor, min_neighbors, min_face_size)


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
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_face_size: int = 30
) -> str:
    """
    Detect faces in an image for potential redaction.

    Use this tool when you suspect an image contains human faces that should be
    anonymized. The tool will return bounding box coordinates for each detected face.

    Args:
        image_base64: Base64-encoded image to analyze
        scale_factor: Scale factor for multi-scale detection (default: 1.1)
        min_neighbors: Minimum neighbors for detection (higher = fewer false positives, default: 5)
        min_face_size: Minimum face size in pixels (default: 30)

    Returns:
        JSON string with detection results including face bounding boxes
    """
    try:
        faces = detect_faces_from_base64(
            image_base64,
            scale_factor=scale_factor,
            min_neighbors=min_neighbors,
            min_face_size=min_face_size
        )

        if not faces:
            return "No faces detected in the image."

        result_parts = [f"Detected {len(faces)} face(s):"]
        for i, face in enumerate(faces, 1):
            result_parts.append(
                f"  Face {i}: x={face.x}, y={face.y}, width={face.width}, height={face.height}"
            )

        return "\n".join(result_parts)

    except Exception as e:
        return f"[FACE_DETECTION_FAILED: {str(e)}]"


def get_face_bounding_boxes(image: Image.Image) -> List[BoundingBox]:
    """
    Detect faces in a PIL image and return as BoundingBox objects.

    This is a convenience function for integration with the image processors.

    Args:
        image: PIL Image object

    Returns:
        List of BoundingBox objects for detected faces
    """
    faces = detect_faces_from_pil(image)

    return [
        BoundingBox(x=face.x, y=face.y, width=face.width, height=face.height)
        for face in faces
    ]
