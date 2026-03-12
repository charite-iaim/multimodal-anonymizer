"""
DICOM face redaction processor for CT/MRI head scans.

Uses a trained ResNet U-Net model to redact faces in DICOM images.
The model takes a DICOM image (converted to RGB) and outputs the redacted version
with faces blacked out.

This processor is invoked by DICOMVisionOCRProcessor when the LLM determines
that a DICOM image is a CT/MRI head scan.
"""

import json
import logging
import numpy as np
from pathlib import Path
from PIL import Image
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

import pydicom

logger = logging.getLogger(__name__)

# Default model path
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "redact-faces-ct-mri.pth"

# Image size expected by the model
MODEL_IMAGE_SIZE = 256

# Model Definition
class ResNetUNet(nn.Module):
    """U-Net with ResNet-18 encoder for face redaction in CT/MRI images."""

    def __init__(self):
        super().__init__()

        resnet = models.resnet18(weights=None)

        # Encoder layers from ResNet
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64 channels
        self.pool1 = resnet.maxpool
        self.encoder2 = resnet.layer1   # 64 channels
        self.encoder3 = resnet.layer2   # 128 channels
        self.encoder4 = resnet.layer3   # 256 channels
        self.encoder5 = resnet.layer4   # 512 channels

        # Decoder with skip connections
        self.upconv5 = self._upconv(512, 256)
        self.decoder5 = self._conv_block(512, 256)

        self.upconv4 = self._upconv(256, 128)
        self.decoder4 = self._conv_block(256, 128)

        self.upconv3 = self._upconv(128, 64)
        self.decoder3 = self._conv_block(128, 64)

        self.upconv2 = self._upconv(64, 64)
        self.decoder2 = self._conv_block(128, 64)

        self.upconv1 = self._upconv(64, 32)
        self.decoder1 = self._conv_block(32, 32)

        # Final output layer
        self.final = nn.Conv2d(32, 3, kernel_size=1)

    def _upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e1_pool = self.pool1(e1)
        e2 = self.encoder2(e1_pool)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        # Decoder with skip connections
        d5 = self.upconv5(e5)
        d5 = torch.cat([d5, e4], dim=1)
        d5 = self.decoder5(d5)

        d4 = self.upconv4(d5)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = self.decoder1(d1)

        out = self.final(d1)
        out = torch.sigmoid(out)

        return out


#  Model Loading (singleton)

_loaded_model = None
_loaded_device = None


def _get_device() -> torch.device:
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_face_redaction_model(model_path: Path = None) -> tuple[ResNetUNet, torch.device]:
    """
    Load the trained face redaction model (cached singleton).

    Args:
        model_path: Path to the .pth checkpoint file.
                    Defaults to anonymizer/models/redact-faces-ct-mri.pth

    Returns:
        Tuple of (model, device)
    """
    global _loaded_model, _loaded_device

    if _loaded_model is not None:
        return _loaded_model, _loaded_device

    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    if not model_path.exists():
        raise FileNotFoundError(
            f"Face redaction model not found at {model_path}. "
            f"Please place the trained model file (redact-faces-ct-mri.pth) "
            f"in the anonymizer/models/ directory."
        )

    device = _get_device()
    print(f"Loading face redaction model from {model_path} (device: {device})")

    model = ResNetUNet()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    _loaded_model = model
    _loaded_device = device

    print("Face redaction model loaded successfully")
    return model, device


# Preprocessing / Postprocessing

_preprocess_transform = transforms.Compose([
    transforms.Resize((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)),
    transforms.ToTensor(),
])


def redact_face_in_image(
    image: Image.Image,
    model: ResNetUNet,
    device: torch.device,
    debug_save_path: Path = None,
) -> Image.Image:
    """
    Run the face redaction model on a single PIL image.

    The model expects a 256x256 RGB input and produces a 256x256 RGB output.
    The output is then resized back to the original image dimensions.

    For grayscale inputs, the output is converted back to grayscale by averaging
    the RGB channels to avoid color tinting issues from the model.

    Args:
        image: Input PIL image (any mode, any size)
        model: Loaded ResNetUNet model
        device: torch device
        debug_save_path: Optional path to save intermediate debug images

    Returns:
        PIL image with face redacted, same size as input
    """
    original_size = image.size
    original_mode = image.mode

    # Convert to RGB for the model
    rgb_image = image.convert("RGB")

    # Save debug input
    if debug_save_path:
        debug_dir = debug_save_path.parent / "face_redaction_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        rgb_image.save(debug_dir / f"{debug_save_path.stem}_model_input_rgb.png")

    # Preprocess: resize to 256x256 and convert to tensor [0, 1]
    input_tensor = _preprocess_transform(rgb_image).unsqueeze(0).to(device)  # (1, 3, 256, 256)

    # Run inference
    with torch.no_grad():
        output_tensor = model(input_tensor)  # (1, 3, 256, 256) in [0, 1]

    # Convert output tensor to numpy array
    output_array = output_tensor[0].cpu().numpy()  # (3, 256, 256)

    # Save debug raw RGB
    if debug_save_path:
        raw_rgb_output = (output_array * 255).clip(0, 255).astype(np.uint8)
        raw_rgb_output = np.transpose(raw_rgb_output, (1, 2, 0))
        raw_rgb_image = Image.fromarray(raw_rgb_output, mode="RGB")
        raw_rgb_image.save(debug_dir / f"{debug_save_path.stem}_model_output_raw_rgb.png")

    if original_mode == "L":
        # Average across RGB channels to get clean grayscale
        grayscale_array = output_array.mean(axis=0)  # (256, 256)
        grayscale_array = (grayscale_array * 255).clip(0, 255).astype(np.uint8)
        redacted_image = Image.fromarray(grayscale_array, mode="L")
    else:
        # For color images, keep RGB output
        output_array = (output_array * 255).clip(0, 255).astype(np.uint8)
        output_array = np.transpose(output_array, (1, 2, 0))  # (256, 256, 3)
        redacted_image = Image.fromarray(output_array, mode="RGB")

    # Resize back to original dimensions
    if redacted_image.size != original_size:
        redacted_image = redacted_image.resize(original_size, Image.LANCZOS)

    # Save debug final output
    if debug_save_path:
        redacted_image.save(debug_dir / f"{debug_save_path.stem}_model_output_final.png")

    # Convert to original mode if needed (for modes other than L and RGB)
    if original_mode not in ("L", "RGB"):
        redacted_image = redacted_image.convert(original_mode)

    return redacted_image


def redact_faces_in_dicom_frames(
    images: list[Image.Image],
    model_path: Path = None,
    debug_output_path: Path = None,
) -> list[Image.Image]:
    """
    Redact faces in a list of DICOM frame images using the trained model.

    Args:
        images: List of PIL images (frames from a DICOM file)
        model_path: Optional path to the model checkpoint
        debug_output_path: Optional path for saving debug images

    Returns:
        List of redacted PIL images
    """
    model, device = load_face_redaction_model(model_path)

    redacted = []
    for i, image in enumerate(images):
        if i % 10 == 0 or i == len(images) - 1:
            print(f"  Redacting face in frame {i + 1}/{len(images)}...")
        
        # Only save debug for first frame to avoid too many files
        frame_debug_path = None
        if debug_output_path and i == 0:
            frame_debug_path = debug_output_path.with_stem(f"{debug_output_path.stem}_frame{i:04d}")
        
        redacted.append(redact_face_in_image(image, model, device, debug_save_path=frame_debug_path))

    return redacted


def get_face_redaction_mask(
    image: Image.Image,
    model: ResNetUNet,
    device: torch.device,
    threshold: float = 0.1,
) -> np.ndarray:
    """
    Get a binary mask indicating where face redaction should be applied.

    Compares the model input and output to identify regions that were modified
    (blacked out) by the face redaction model.

    Args:
        image: Input PIL image (any mode, any size)
        model: Loaded ResNetUNet model
        device: torch device
        threshold: Normalized intensity threshold below which output is considered "redacted"

    Returns:
        Boolean numpy array (H, W) where True = should be redacted (blacked out)
    """
    original_size = image.size  # (width, height)

    # Convert to RGB for the model
    rgb_image = image.convert("RGB")

    # Preprocess: resize to 256x256 and convert to tensor [0, 1]
    input_tensor = _preprocess_transform(rgb_image).unsqueeze(0).to(device)  # (1, 3, 256, 256)

    # Run inference
    with torch.no_grad():
        output_tensor = model(input_tensor)  # (1, 3, 256, 256) in [0, 1]

    # Convert output tensor to numpy array and average channels
    output_array = output_tensor[0].cpu().numpy()  # (3, 256, 256)
    output_gray = output_array.mean(axis=0)  # (256, 256) average of RGB

    # Create mask: True where output is dark (below threshold)
    mask_256 = output_gray < threshold

    # Resize mask to original dimensions
    mask_pil = Image.fromarray(mask_256.astype(np.uint8) * 255, mode="L")
    mask_pil = mask_pil.resize(original_size, Image.NEAREST)

    # Convert back to boolean array (H, W)
    mask = np.array(mask_pil) > 127

    return mask


def get_face_redaction_masks_for_frames(
    images: list[Image.Image],
    model_path: Path = None,
) -> list[np.ndarray]:
    """
    Get face redaction masks for a list of DICOM frame images.

    Args:
        images: List of PIL images (frames from a DICOM file)
        model_path: Optional path to the model checkpoint

    Returns:
        List of boolean numpy arrays (H, W) indicating redaction regions
    """
    model, device = load_face_redaction_model(model_path)

    masks = []
    for i, image in enumerate(images):
        if i % 10 == 0 or i == len(images) - 1:
            print(f"  Computing face redaction mask for frame {i + 1}/{len(images)}...")

        masks.append(get_face_redaction_mask(image, model, device))

    return masks
