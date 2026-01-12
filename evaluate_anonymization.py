#!/usr/bin/env python3
"""
Comprehensive Anonymization Evaluation Pipeline

This script evaluates the quality of anonymization across multiple file types:
- CSV files: Compares PHI-tagged labels with anonymized output
- DICOM images (CXR, Echo): IoU-based bounding box evaluation
- PDF files (ECG): Page-based redaction evaluation
- Filenames: Checks if PHI was removed from filenames

Labels directory structure:
    patient_records_labels/
        folder_annotations.csv
        patient_XXXX/
            annotations_csv/
                csv_filename_annotations.csv
                hosp_admissions_XXXX.csv  # with <PER> tags
            annotations_cxr/
                phi_annotations_cxr.csv  # bounding boxes
            annotations_ecg/
                phi_annotations_ecg.csv  # bounding boxes
            annotations_echo/
                phi_annotations_echo.csv  # bounding boxes

Results directory structure:
    results/
        folder_anonymization.csv
        patient_ID_ID_XXX/
            csv/
                filename_anonymization.csv
                hosp_admissions_ID.csv  # anonymized
            cxr/
                filename_anonymization.csv
                cxr_PERSON_ID_XXX.dcm  # anonymized
            ecg/
                filename_anonymization.csv
                ecg_PERSON_ID_ID.pdf  # anonymized

Usage:
    python evaluate_anonymization.py --labels <labels_dir> --results <results_dir> --output <output_dir>
"""

import os
import re
import csv
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime
import traceback

# Optional imports for image processing
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Image evaluation will be skipped.")

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    print("Warning: pydicom not available. DICOM evaluation will be limited.")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("Warning: PyMuPDF not available. PDF evaluation will be limited.")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available. Using basic CSV handling.")


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PHIAnnotation:
    """Represents a single PHI annotation with bounding box"""
    filename: str
    field: str
    text: str
    region: str
    x: int
    y: int
    width: int
    height: int


@dataclass
class FileMapping:
    """Mapping between original and anonymized filenames"""
    original: str
    anonymized: str
    phi_values: List[str] = field(default_factory=list)
    phi_categories: List[str] = field(default_factory=list)


@dataclass
class CSVEvaluationResult:
    """Results from evaluating a single CSV file"""
    filename: str
    total_phi: int = 0
    true_positives: int = 0
    false_negatives: int = 0
    phi_not_redacted: List[str] = field(default_factory=list)
    
    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 1.0


@dataclass
class ImageEvaluationResult:
    """Results from evaluating a single image file"""
    filename: str
    total_phi_regions: int = 0
    detected_redactions: int = 0
    mean_iou: float = 0.0
    mean_coverage: float = 0.0
    fully_covered: int = 0  # >= 95% coverage
    partially_covered: int = 0  # 50-95% coverage
    not_covered: int = 0  # < 50% coverage
    details: List[Dict] = field(default_factory=list)


@dataclass
class FilenameEvaluationResult:
    """Results from evaluating filename anonymization"""
    original: str
    anonymized: str
    expected_phi: List[str] = field(default_factory=list)
    phi_in_filename: List[str] = field(default_factory=list)
    is_anonymized: bool = True


@dataclass
class HEAEvaluationResult:
    """Results from evaluating a single HEA (text) file"""
    filename: str
    total_phi: int = 0
    true_positives: int = 0
    false_negatives: int = 0
    phi_not_redacted: List[str] = field(default_factory=list)
    
    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 1.0


@dataclass
class PatientEvaluationResult:
    """Aggregated results for a single patient"""
    original_folder: str
    anonymized_folder: str
    csv_results: List[CSVEvaluationResult] = field(default_factory=list)
    image_results: List[ImageEvaluationResult] = field(default_factory=list)
    pdf_results: List[ImageEvaluationResult] = field(default_factory=list)
    hea_results: List[HEAEvaluationResult] = field(default_factory=list)
    filename_results: List[FilenameEvaluationResult] = field(default_factory=list)


@dataclass
class OverallEvaluationResult:
    """Aggregated results across all patients"""
    timestamp: str = ""
    labels_dir: str = ""
    results_dir: str = ""
    patient_results: List[PatientEvaluationResult] = field(default_factory=list)

    # Aggregate CSV metrics
    csv_total_phi: int = 0
    csv_true_positives: int = 0
    csv_false_negatives: int = 0
    csv_recall: float = 0.0

    # Aggregate Image metrics (DICOM, PNG)
    image_total_regions: int = 0
    image_mean_coverage: float = 0.0
    image_fully_covered: int = 0
    image_not_covered: int = 0

    # Aggregate PDF metrics (separate from images)
    pdf_total_regions: int = 0
    pdf_mean_coverage: float = 0.0
    pdf_fully_covered: int = 0
    pdf_not_covered: int = 0

    # Aggregate HEA (text) metrics
    hea_total_phi: int = 0
    hea_true_positives: int = 0
    hea_false_negatives: int = 0
    hea_recall: float = 0.0

    # Aggregate Filename metrics
    filename_total: int = 0
    filename_anonymized: int = 0
    filename_phi_leaked: int = 0


# ============================================================================
# Utility Functions
# ============================================================================

def extract_phi_from_text(text: str) -> Set[str]:
    """Extract all PHI values from text marked with <PER> tags"""
    pattern = r'<PER>(.*?)</PER>'
    return set(re.findall(pattern, text))


def extract_phi_with_positions(text: str) -> List[Tuple[str, int, int]]:
    """
    Extract all PHI values from text with their positions (after tag removal).

    Returns list of (phi_value, start_pos, end_pos) tuples where positions
    refer to the text AFTER tags have been removed.

    Example:
        Input:  "Patient is <PER>67</PER> years old, BP: 120/67"
        Output: [("67", 11, 13)]

        After tag removal: "Patient is 67 years old, BP: 120/67"
        Position 11-13 corresponds to the PHI "67", not the BP "67"
    """
    phi_positions = []
    pattern = r'<PER>(.*?)</PER>'

    # Track how many characters we've seen in the original text
    current_pos = 0
    # Track position in text after tag removal
    adjusted_pos = 0

    for match in re.finditer(pattern, text):
        phi_value = match.group(1)
        match_start = match.start()
        match_end = match.end()

        # Add the length of text before this match (no tags)
        adjusted_pos += (match_start - current_pos)

        # Record the PHI position in tag-free text
        start_in_clean = adjusted_pos
        end_in_clean = adjusted_pos + len(phi_value)
        phi_positions.append((phi_value, start_in_clean, end_in_clean))

        # Update positions
        adjusted_pos += len(phi_value)
        current_pos = match_end

    return phi_positions


def remove_tags(text: str) -> str:
    """Remove XML-style tags from text"""
    return re.sub(r'<[^>]+>', '', text)


def load_folder_mapping(labels_dir: str, results_dir: str) -> Dict[str, str]:
    """
    Load mapping between original and anonymized folder names.
    
    Returns dict: original_folder -> anonymized_folder
    """
    mapping = {}
    
    # Load from results folder_anonymization.csv
    results_mapping_file = os.path.join(results_dir, "folder_anonymization.csv")
    if os.path.exists(results_mapping_file):
        with open(results_mapping_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                original = row.get('original_foldername', row.get('original_folder_name', ''))
                anonymized = row.get('anonymized_foldername', row.get('anonymized_folder_name', ''))
                if original and anonymized:
                    mapping[original] = anonymized
    
    return mapping


def load_filename_mapping(mapping_file: str) -> List[FileMapping]:
    """Load filename mappings from a CSV file"""
    mappings = []
    
    if not os.path.exists(mapping_file):
        return mappings
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            original = row.get('original_filename', '')
            anonymized = row.get('anonymized_filename', '')
            
            phi_values = []
            if 'phi_values' in row and row['phi_values']:
                phi_values = [v.strip() for v in row['phi_values'].split(';')]
            
            phi_categories = []
            if 'phi_categories' in row and row['phi_categories']:
                phi_categories = [c.strip() for c in row['phi_categories'].split(';')]
            
            if original and anonymized:
                mappings.append(FileMapping(
                    original=original,
                    anonymized=anonymized,
                    phi_values=phi_values,
                    phi_categories=phi_categories
                ))
    
    return mappings


def load_phi_annotations(csv_path: str) -> List[PHIAnnotation]:
    """Load PHI bounding box annotations from CSV"""
    annotations = []
    
    if not os.path.exists(csv_path):
        return annotations
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows without bounding box info
            if not row.get('x') or not row.get('y'):
                continue
            
            annotations.append(PHIAnnotation(
                filename=row.get('filename', ''),
                field=row.get('field', ''),
                text=row.get('text', ''),
                region=row.get('region', ''),
                x=int(float(row.get('x', 0))),
                y=int(float(row.get('y', 0))),
                width=int(float(row.get('width', 0))),
                height=int(float(row.get('height', 0)))
            ))
    
    return annotations


# ============================================================================
# CSV Evaluation
# ============================================================================

def evaluate_csv_field(source_value: str, redacted_value: str, min_phi_length: int = 3) -> Tuple[int, int, List[str]]:
    """
    Evaluate redaction for a single CSV field using position-based comparison.

    This function:
    1. Extracts PHI values and their positions from the labeled source text
    2. Removes tags from source to get clean text
    3. Compares each PHI region in the clean text with the corresponding region in redacted text
    4. Only marks as FN if the PHI value still appears at its ORIGINAL position

    Example:
        source_value:   "Patient is <PER>67</PER> years old, BP: 120/67"
        redacted_value: "Patient is [AGE] years old, BP: 120/67"

        - Extracts PHI "67" at position 11-13 (after tag removal)
        - Compares "67" with "[AGE]" at position 11-16 in redacted text
        - Different → TP (correctly redacted)
        - The "67" in "120/67" is ignored (different position)

    Returns: (true_positives, false_negatives, phi_not_redacted)
    """
    tp = 0
    fn = 0
    not_redacted = []

    # Extract PHI with their positions
    phi_positions = extract_phi_with_positions(source_value)

    if not phi_positions:
        # No PHI to evaluate
        return tp, fn, not_redacted

    # For each PHI occurrence at a specific position
    for phi_value, start_pos, end_pos in phi_positions:
        # Extract the text at this position in the redacted value
        # We need to check if the region has been changed

        # Handle cases where redacted text might be longer/shorter
        # We check a window around the position
        if start_pos >= len(redacted_value):
            # Position doesn't exist in redacted text (text was shortened)
            # This is acceptable - consider it redacted
            tp += 1
            continue

        # Check if the exact PHI value still appears at or near this position
        # We allow some flexibility for replacement tokens that might be different lengths
        window_start = max(0, start_pos - len(phi_value))
        window_end = min(len(redacted_value), end_pos + len(phi_value))
        window_text = redacted_value[window_start:window_end]

        # Check if PHI appears in this window
        if phi_value in window_text:
            # PHI still present at original position
            fn += 1
            not_redacted.append(f"{phi_value}@pos{start_pos}")
        else:
            # PHI successfully removed/replaced at this position
            tp += 1

    return tp, fn, not_redacted


def evaluate_csv_file(label_path: str, result_path: str) -> CSVEvaluationResult:
    """
    Evaluate a single CSV file by comparing labeled PHI with anonymized output.
    """
    filename = os.path.basename(label_path)
    result = CSVEvaluationResult(filename=filename)
    
    if not os.path.exists(label_path):
        print(f"  Warning: Label file not found: {label_path}")
        return result
    
    if not os.path.exists(result_path):
        print(f"  Warning: Result file not found: {result_path}")
        return result
    
    # Read both files
    with open(label_path, 'r', encoding='utf-8') as f:
        label_rows = list(csv.DictReader(f))
    
    with open(result_path, 'r', encoding='utf-8') as f:
        result_rows = list(csv.DictReader(f))
    
    if len(label_rows) != len(result_rows):
        print(f"  Warning: Row count mismatch in {filename}: {len(label_rows)} vs {len(result_rows)}")
    
    # Compare row by row
    for label_row, result_row in zip(label_rows, result_rows):
        for field_name in label_row.keys():
            if field_name not in result_row:
                continue
            
            label_value = label_row[field_name] or ''
            result_value = result_row[field_name] or ''
            
            if '<PER>' not in label_value:
                continue
            
            tp, fn, not_redacted = evaluate_csv_field(label_value, result_value)
            
            result.total_phi += tp + fn
            result.true_positives += tp
            result.false_negatives += fn
            result.phi_not_redacted.extend(not_redacted)
    
    return result


# ============================================================================
# HEA (Text) Evaluation
# ============================================================================

def evaluate_hea_file(label_path: str, result_path: str) -> HEAEvaluationResult:
    """
    Evaluate a single HEA text file by comparing labeled PHI with anonymized output.
    
    HEA files are ECG header files containing PHI marked with <PER> tags.
    The evaluation compares line by line to check if PHI was properly redacted.
    """
    filename = os.path.basename(label_path)
    result = HEAEvaluationResult(filename=filename)
    
    if not os.path.exists(label_path):
        print(f"  Warning: Label file not found: {label_path}")
        return result
    
    if not os.path.exists(result_path):
        print(f"  Warning: Result file not found: {result_path}")
        return result
    
    # Read both files
    with open(label_path, 'r', encoding='utf-8') as f:
        label_content = f.read()
    
    with open(result_path, 'r', encoding='utf-8') as f:
        result_content = f.read()
    
    # Split into lines for line-by-line comparison
    label_lines = label_content.split('\n')
    result_lines = result_content.split('\n')
    
    # Compare line by line
    for i, label_line in enumerate(label_lines):
        if '<PER>' not in label_line:
            continue
        
        # Get corresponding result line (if exists)
        result_line = result_lines[i] if i < len(result_lines) else ''
        
        # Use the same position-based evaluation as CSV fields
        tp, fn, not_redacted = evaluate_csv_field(label_line, result_line)
        
        result.total_phi += tp + fn
        result.true_positives += tp
        result.false_negatives += fn
        result.phi_not_redacted.extend(not_redacted)
    
    return result


# ============================================================================
# Image Evaluation (DICOM, PNG)
# ============================================================================

def calculate_iou(box1: Dict, box2: Dict) -> float:
    """Calculate Intersection over Union between two bounding boxes"""
    x1_min, y1_min = box1['x'], box1['y']
    x1_max, y1_max = x1_min + box1['width'], y1_min + box1['height']
    
    x2_min, y2_min = box2['x'], box2['y']
    x2_max, y2_max = x2_min + box2['width'], y2_min + box2['height']
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = box1['width'] * box1['height']
    area2 = box2['width'] * box2['height']
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_coverage(gt_box: Dict, red_box: Dict) -> float:
    """Calculate what fraction of ground truth box is covered by redaction"""
    if red_box is None:
        return 0.0
    
    gt_x_min, gt_y_min = gt_box['x'], gt_box['y']
    gt_x_max, gt_y_max = gt_x_min + gt_box['width'], gt_y_min + gt_box['height']
    
    red_x_min, red_y_min = red_box['x'], red_box['y']
    red_x_max, red_y_max = red_x_min + red_box['width'], red_y_min + red_box['height']
    
    inter_x_min = max(gt_x_min, red_x_min)
    inter_y_min = max(gt_y_min, red_y_min)
    inter_x_max = min(gt_x_max, red_x_max)
    inter_y_max = min(gt_y_max, red_y_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    gt_area = gt_box['width'] * gt_box['height']
    
    return intersection / gt_area if gt_area > 0 else 0.0


def load_dicom_image(dicom_path: str) -> Optional[np.ndarray]:
    """Load DICOM file and convert to image array"""
    if not PYDICOM_AVAILABLE:
        return None
    
    try:
        ds = pydicom.dcmread(dicom_path)
        
        # Handle multi-frame (video) DICOM - use first frame
        if hasattr(ds, 'NumberOfFrames') and ds.NumberOfFrames > 1:
            pixel_array = ds.pixel_array[0]
        else:
            pixel_array = ds.pixel_array
        
        # Apply VOI LUT if available
        try:
            pixel_array = apply_voi_lut(pixel_array, ds)
        except:
            pass
        
        # Normalize to 8-bit
        if pixel_array.dtype != np.uint8:
            pixel_array = ((pixel_array - pixel_array.min()) / 
                          (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
        
        # Convert to BGR for OpenCV
        if len(pixel_array.shape) == 2:
            pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2BGR)
        elif pixel_array.shape[2] == 4:
            pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_RGBA2BGR)
        
        return pixel_array
    except Exception as e:
        print(f"  Error loading DICOM {dicom_path}: {e}")
        return None


def detect_redaction_boxes(original_img: np.ndarray, redacted_img: np.ndarray,
                           threshold: int = 30) -> List[Dict]:
    """Detect redaction regions by comparing original and redacted images"""
    if not CV2_AVAILABLE:
        return []
    
    # Ensure same size
    if original_img.shape[:2] != redacted_img.shape[:2]:
        redacted_img = cv2.resize(redacted_img, (original_img.shape[1], original_img.shape[0]))
    
    # Convert to grayscale
    if len(original_img.shape) == 3:
        gray_original = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_original = original_img
    
    if len(redacted_img.shape) == 3:
        if redacted_img.shape[2] == 4:
            gray_redacted = cv2.cvtColor(redacted_img[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            gray_redacted = cv2.cvtColor(redacted_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_redacted = redacted_img
    
    # Compute absolute difference
    diff = cv2.absdiff(gray_original, gray_redacted)
    
    # Threshold
    _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    redaction_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:
            redaction_boxes.append({'x': x, 'y': y, 'width': w, 'height': h})
    
    return redaction_boxes


def match_boxes(gt_boxes: List[Dict], redaction_boxes: List[Dict]) -> List[Tuple[Dict, Optional[Dict], float, float]]:
    """Match ground truth boxes with detected redaction boxes (legacy box-based method)"""
    matches = []
    
    for gt_box in gt_boxes:
        best_iou = 0.0
        best_coverage = 0.0
        best_match = None
        
        for red_box in redaction_boxes:
            iou = calculate_iou(gt_box, red_box)
            coverage = calculate_coverage(gt_box, red_box)
            
            if coverage > best_coverage:
                best_coverage = coverage
                best_iou = iou
                best_match = red_box
        
        matches.append((gt_box, best_match, best_iou, best_coverage))
    
    return matches


def calculate_pixel_coverage(redacted_img: np.ndarray, gt_box: Dict, 
                              black_threshold: int = 10) -> float:
    """
    Calculate what fraction of a ground truth PHI region is black (redacted)
    in the redacted image using direct pixel analysis.
    
    This is more accurate than box-matching because:
    1. It doesn't depend on detecting redaction boxes via contours
    2. It directly measures if PHI regions are actually blacked out
    3. Works even when redactions span multiple fragmented areas
    
    Args:
        redacted_img: The redacted image as numpy array (BGR or grayscale)
        gt_box: Ground truth bounding box with 'x', 'y', 'width', 'height'
        black_threshold: Pixel values below this are considered black/redacted
    
    Returns:
        Float between 0.0 and 1.0 representing fraction of region that is black
    """
    x, y, w, h = gt_box['x'], gt_box['y'], gt_box['width'], gt_box['height']
    
    # Ensure coordinates are within image bounds
    img_h, img_w = redacted_img.shape[:2]
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    
    if w <= 0 or h <= 0:
        return 0.0
    
    # Extract the region from the redacted image
    region = redacted_img[y:y+h, x:x+w]
    
    # Convert to grayscale if needed
    if len(region.shape) == 3:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        gray = region
    
    # Count black pixels
    total_pixels = w * h
    black_pixels = np.sum(gray < black_threshold)
    
    return black_pixels / total_pixels if total_pixels > 0 else 0.0


def evaluate_image_file(original_path: str, redacted_path: str, 
                        annotations: List[PHIAnnotation],
                        original_filename: str) -> ImageEvaluationResult:
    """
    Evaluate a single image file (DICOM or PNG) using pixel-based coverage analysis.
    
    This method directly checks if PHI regions are blacked out in the redacted image,
    rather than trying to detect and match redaction boxes. This is more accurate
    because it doesn't depend on contour detection which can fragment redacted areas.
    """
    result = ImageEvaluationResult(filename=original_filename)
    
    # Filter annotations for this file
    file_annotations = [a for a in annotations if a.filename == original_filename]
    result.total_phi_regions = len(file_annotations)
    
    if result.total_phi_regions == 0:
        return result
    
    if not CV2_AVAILABLE:
        print(f"  Warning: OpenCV not available for image evaluation")
        return result
    
    # Load redacted image only (we just need to check if regions are black)
    if redacted_path.lower().endswith('.dcm'):
        redacted_img = load_dicom_image(redacted_path)
    else:
        redacted_img = cv2.imread(redacted_path, cv2.IMREAD_UNCHANGED)
    
    if redacted_img is None:
        print(f"  Warning: Could not load redacted image for {original_filename}")
        return result
    
    # Handle images with alpha channel
    if len(redacted_img.shape) == 3 and redacted_img.shape[2] == 4:
        redacted_img = cv2.cvtColor(redacted_img, cv2.COLOR_BGRA2BGR)
    
    # Convert annotations to box format and calculate pixel-based coverage
    coverages = []
    for ann in file_annotations:
        gt_box = {
            'x': ann.x, 'y': ann.y, 
            'width': ann.width, 'height': ann.height,
            'field': ann.field, 'text': ann.text
        }
        
        # Calculate pixel-based coverage (what fraction of region is black)
        coverage = calculate_pixel_coverage(redacted_img, gt_box)
        coverages.append(coverage)
        
        # Store details
        result.details.append({
            'field': ann.field,
            'text': ann.text,
            'iou': coverage,  # For compatibility, use coverage as IoU proxy
            'coverage': coverage,
            'status': 'covered' if coverage >= 0.95 else ('partial' if coverage >= 0.5 else 'missing')
        })
    
    # Calculate aggregate metrics
    result.mean_iou = sum(coverages) / len(coverages) if coverages else 0.0
    result.mean_coverage = sum(coverages) / len(coverages) if coverages else 0.0
    result.fully_covered = sum(1 for cov in coverages if cov >= 0.95)
    result.partially_covered = sum(1 for cov in coverages if 0.5 <= cov < 0.95)
    result.not_covered = sum(1 for cov in coverages if cov < 0.5)
    result.detected_redactions = result.fully_covered + result.partially_covered  # Approximate
    
    return result


# ============================================================================
# PDF Evaluation (Box-based with image differencing)
# ============================================================================

def evaluate_pdf_file(original_path: str, redacted_path: str,
                      annotations: List[PHIAnnotation],
                      original_filename: str) -> ImageEvaluationResult:
    """
    Evaluate a PDF file by converting pages to images and detecting redactions
    using image differencing and bounding box matching.
    """
    result = ImageEvaluationResult(filename=original_filename)
    
    # Filter annotations for this file
    file_annotations = [a for a in annotations if a.filename == original_filename]
    result.total_phi_regions = len(file_annotations)
    
    if result.total_phi_regions == 0:
        return result
    
    if not PYMUPDF_AVAILABLE or not CV2_AVAILABLE:
        print(f"  Warning: PyMuPDF or OpenCV not available for PDF evaluation")
        return result
    
    if not os.path.exists(original_path) or not os.path.exists(redacted_path):
        print(f"  Warning: PDF files not found for {original_filename}")
        return result
    
    try:
        # Open PDFs
        original_doc = fitz.open(original_path)
        redacted_doc = fitz.open(redacted_path)
        
        # Process first page (most PDFs are single-page)
        original_page = original_doc[0]
        redacted_page = redacted_doc[0]
        
        # Render to images
        zoom = 2.0  # Higher resolution for better detection
        mat = fitz.Matrix(zoom, zoom)
        
        original_pix = original_page.get_pixmap(matrix=mat)
        redacted_pix = redacted_page.get_pixmap(matrix=mat)
        
        # Convert to numpy arrays
        original_img = np.frombuffer(original_pix.samples, dtype=np.uint8).reshape(
            original_pix.height, original_pix.width, original_pix.n)
        redacted_img = np.frombuffer(redacted_pix.samples, dtype=np.uint8).reshape(
            redacted_pix.height, redacted_pix.width, redacted_pix.n)
        
        # Convert to BGR
        if original_pix.n == 4:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_RGBA2BGR)
        elif original_pix.n == 3:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        
        if redacted_pix.n == 4:
            redacted_img = cv2.cvtColor(redacted_img, cv2.COLOR_RGBA2BGR)
        elif redacted_pix.n == 3:
            redacted_img = cv2.cvtColor(redacted_img, cv2.COLOR_RGB2BGR)
        
        # Get page height for coordinate transformation
        # PDF coordinate system has y=0 at bottom, but image has y=0 at top
        page_height = original_page.rect.height
        
        # Detect redaction boxes by comparing original and redacted images
        redaction_boxes = detect_redaction_boxes(original_img, redacted_img)
        result.detected_redactions = len(redaction_boxes)
        
        # Convert annotations to ground truth boxes (with coordinate transformation)
        gt_boxes = []
        for ann in file_annotations:
            # Convert y from PDF coords (origin at bottom) to image coords (origin at top)
            image_y = page_height - ann.y - ann.height
            gt_boxes.append({
                'x': int(ann.x * zoom), 
                'y': int(image_y * zoom),
                'width': int(ann.width * zoom), 
                'height': int(ann.height * zoom),
                'field': ann.field, 
                'text': ann.text
            })
        
        # Match ground truth boxes with detected redaction boxes
        matches = match_boxes(gt_boxes, redaction_boxes)
        
        # Calculate metrics
        ious = [iou for _, _, iou, _ in matches]
        coverages = [cov for _, _, _, cov in matches]
        
        result.mean_iou = sum(ious) / len(ious) if ious else 0.0
        result.mean_coverage = sum(coverages) / len(coverages) if coverages else 0.0
        result.fully_covered = sum(1 for cov in coverages if cov >= 0.95)
        result.partially_covered = sum(1 for cov in coverages if 0.5 <= cov < 0.95)
        result.not_covered = sum(1 for cov in coverages if cov < 0.5)
        
        # Store details
        for gt_box, red_box, iou, coverage in matches:
            result.details.append({
                'field': gt_box.get('field', ''),
                'text': gt_box.get('text', ''),
                'iou': iou,
                'coverage': coverage,
                'status': 'covered' if coverage >= 0.95 else ('partial' if coverage >= 0.5 else 'missing')
            })
        
        original_doc.close()
        redacted_doc.close()
        
    except Exception as e:
        print(f"  Error evaluating PDF {original_filename}: {e}")
        traceback.print_exc()
    
    return result


# ============================================================================
# Filename Evaluation
# ============================================================================

def evaluate_filename_anonymization(label_mapping: FileMapping, 
                                    result_mapping: FileMapping) -> FilenameEvaluationResult:
    """Evaluate if filename was properly anonymized"""
    result = FilenameEvaluationResult(
        original=label_mapping.original,
        anonymized=result_mapping.anonymized
    )
    
    # Extract expected PHI from label filename (values in <PER> tags)
    label_annotated = label_mapping.anonymized if hasattr(label_mapping, 'annotated_filename') else ''
    
    # Use phi_values from result mapping if available
    if result_mapping.phi_values:
        result.expected_phi = result_mapping.phi_values
    
    # Check if any expected PHI appears in the anonymized filename
    for phi in result.expected_phi:
        if phi.lower() in result.anonymized.lower():
            result.phi_in_filename.append(phi)
            result.is_anonymized = False
    
    return result


# ============================================================================
# Patient-Level Evaluation
# ============================================================================

def evaluate_patient(labels_dir: str, results_dir: str,
                     original_folder: str, anonymized_folder: str) -> PatientEvaluationResult:
    """Evaluate all files for a single patient"""
    result = PatientEvaluationResult(
        original_folder=original_folder,
        anonymized_folder=anonymized_folder
    )
    
    label_base = os.path.join(labels_dir, original_folder)
    result_base = os.path.join(results_dir, anonymized_folder)
    
    print(f"\n{'='*60}")
    print(f"Evaluating patient: {original_folder} -> {anonymized_folder}")
    print(f"{'='*60}")
    
    # ---- CSV Evaluation ----
    csv_label_dir = os.path.join(label_base, "annotations_csv")
    csv_result_dir = os.path.join(result_base, "csv")
    
    if os.path.exists(csv_label_dir) and os.path.exists(csv_result_dir):
        print(f"\n📄 Evaluating CSV files...")
        
        # Load filename mapping to match files
        result_mapping_file = os.path.join(csv_result_dir, "filename_anonymization.csv")
        label_mapping_file = os.path.join(csv_label_dir, "csv_filename_annotations.csv")
        
        result_mappings = load_filename_mapping(result_mapping_file)
        
        # Create mapping: original -> anonymized
        file_map = {m.original: m.anonymized for m in result_mappings}
        
        for label_file in os.listdir(csv_label_dir):
            if not label_file.endswith('.csv') or label_file.startswith('csv_filename'):
                continue
            
            label_path = os.path.join(csv_label_dir, label_file)
            
            # Find corresponding result file
            result_file = file_map.get(label_file, label_file.replace('_', '_ID_').replace('.csv', '_ID.csv'))
            result_path = os.path.join(csv_result_dir, result_file)
            
            # Try to find the file with pattern matching if direct match fails
            if not os.path.exists(result_path):
                base_name = label_file.split('_')[0:2]  # e.g., ['hosp', 'admissions']
                for f in os.listdir(csv_result_dir):
                    if f.startswith('_'.join(base_name)):
                        result_path = os.path.join(csv_result_dir, f)
                        break
            
            csv_result = evaluate_csv_file(label_path, result_path)
            result.csv_results.append(csv_result)
            
            status = "✓" if csv_result.false_negatives == 0 else "✗"
            print(f"  {status} {label_file}: {csv_result.true_positives}/{csv_result.total_phi} PHI redacted "
                  f"(recall: {csv_result.recall:.2%})")
    
    # ---- CXR/Image Evaluation ----
    for modality in ['cxr', 'ecg', 'echo']:
        label_ann_dir = os.path.join(label_base, f"annotations_{modality}")
        result_modality_dir = os.path.join(result_base, modality)
        
        if not os.path.exists(label_ann_dir) or not os.path.exists(result_modality_dir):
            continue
        
        print(f"\n🖼️  Evaluating {modality.upper()} files...")
        
        # Load annotations
        ann_file = os.path.join(label_ann_dir, f"phi_annotations_{modality}.csv")
        annotations = load_phi_annotations(ann_file)
        
        if not annotations:
            print(f"  No annotations found in {ann_file}")
            continue
        
        # Load filename mapping
        result_mapping_file = os.path.join(result_modality_dir, "filename_anonymization.csv")
        result_mappings = load_filename_mapping(result_mapping_file)
        file_map = {m.original: m.anonymized for m in result_mappings}
        
        # Get unique files to evaluate
        unique_files = set(a.filename for a in annotations)
        
        for original_filename in unique_files:
            # Find original file
            original_path = os.path.join(label_ann_dir, original_filename)
            
            # Find anonymized file
            anonymized_filename = file_map.get(original_filename, '')
            redacted_path = os.path.join(result_modality_dir, anonymized_filename) if anonymized_filename else ''
            
            if not os.path.exists(original_path) or not os.path.exists(redacted_path):
                print(f"  Warning: Files not found for {original_filename}")
                continue
            
            # Evaluate based on file type
            if original_filename.lower().endswith('.pdf'):
                eval_result = evaluate_pdf_file(original_path, redacted_path, annotations, original_filename)
                result.pdf_results.append(eval_result)
            else:
                eval_result = evaluate_image_file(original_path, redacted_path, annotations, original_filename)
                result.image_results.append(eval_result)
            
            status = "✓" if eval_result.not_covered == 0 else "✗"
            print(f"  {status} {original_filename}: {eval_result.fully_covered}/{eval_result.total_phi_regions} "
                  f"fully covered (mean coverage: {eval_result.mean_coverage:.2%})")
        
        # ---- HEA File Evaluation (ECG modality only) ----
        if modality == 'ecg':
            # Find and evaluate .hea files
            hea_files_evaluated = False
            for label_file in os.listdir(label_ann_dir):
                if not label_file.endswith('.hea'):
                    continue
                
                if not hea_files_evaluated:
                    print(f"\n📝 Evaluating HEA files...")
                    hea_files_evaluated = True
                
                label_path = os.path.join(label_ann_dir, label_file)
                
                # Find corresponding anonymized .hea file
                anonymized_filename = file_map.get(label_file, '')
                result_path = os.path.join(result_modality_dir, anonymized_filename) if anonymized_filename else ''
                
                # If no mapping found, try to find any .hea file in results
                if not os.path.exists(result_path):
                    for f in os.listdir(result_modality_dir):
                        if f.endswith('.hea'):
                            result_path = os.path.join(result_modality_dir, f)
                            anonymized_filename = f
                            break
                
                if not os.path.exists(result_path):
                    print(f"  Warning: Result file not found for {label_file}")
                    continue
                
                hea_result = evaluate_hea_file(label_path, result_path)
                result.hea_results.append(hea_result)
                
                status = "✓" if hea_result.false_negatives == 0 else "✗"
                print(f"  {status} {label_file}: {hea_result.true_positives}/{hea_result.total_phi} PHI redacted "
                      f"(recall: {hea_result.recall:.2%})")
        
        # Evaluate filename anonymization
        for mapping in result_mappings:
            fn_result = FilenameEvaluationResult(
                original=mapping.original,
                anonymized=mapping.anonymized,
                expected_phi=mapping.phi_values
            )
            
            # Check if PHI leaked in filename
            for phi in mapping.phi_values:
                if phi.lower().replace('_', ' ') in mapping.anonymized.lower().replace('_', ' '):
                    fn_result.phi_in_filename.append(phi)
                    fn_result.is_anonymized = False
            
            result.filename_results.append(fn_result)
    
    return result


# ============================================================================
# Main Pipeline
# ============================================================================

def run_evaluation_pipeline(labels_dir: str, results_dir: str, output_dir: str) -> OverallEvaluationResult:
    """Run the complete evaluation pipeline"""
    overall = OverallEvaluationResult(
        timestamp=datetime.now().isoformat(),
        labels_dir=labels_dir,
        results_dir=results_dir
    )
    
    print("="*80)
    print("ANONYMIZATION EVALUATION PIPELINE")
    print("="*80)
    print(f"Labels directory:  {labels_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Output directory:  {output_dir}")
    
    # Load folder mapping
    folder_mapping = load_folder_mapping(labels_dir, results_dir)
    print(f"\nFound {len(folder_mapping)} patient folder mappings")
    
    if not folder_mapping:
        print("Warning: No folder mappings found. Trying to match folders by listing...")
        # Try to match folders manually
        label_folders = [f for f in os.listdir(labels_dir) 
                        if os.path.isdir(os.path.join(labels_dir, f)) and f.startswith('patient_')]
        result_folders = [f for f in os.listdir(results_dir) 
                         if os.path.isdir(os.path.join(results_dir, f)) and f.startswith('patient_')]
        
        print(f"  Found {len(label_folders)} label folders and {len(result_folders)} result folders")
        
        # Basic matching by order
        for label_folder, result_folder in zip(sorted(label_folders), sorted(result_folders)):
            folder_mapping[label_folder] = result_folder
    
    # Evaluate each patient
    for original_folder, anonymized_folder in folder_mapping.items():
        patient_result = evaluate_patient(labels_dir, results_dir, original_folder, anonymized_folder)
        overall.patient_results.append(patient_result)
    
    # Aggregate metrics
    for patient in overall.patient_results:
        # CSV metrics
        for csv_result in patient.csv_results:
            overall.csv_total_phi += csv_result.total_phi
            overall.csv_true_positives += csv_result.true_positives
            overall.csv_false_negatives += csv_result.false_negatives

        # Image metrics (DICOM, PNG only)
        for img_result in patient.image_results:
            overall.image_total_regions += img_result.total_phi_regions
            overall.image_fully_covered += img_result.fully_covered
            overall.image_not_covered += img_result.not_covered

        # PDF metrics (separate)
        for pdf_result in patient.pdf_results:
            overall.pdf_total_regions += pdf_result.total_phi_regions
            overall.pdf_fully_covered += pdf_result.fully_covered
            overall.pdf_not_covered += pdf_result.not_covered

        # HEA (text) metrics
        for hea_result in patient.hea_results:
            overall.hea_total_phi += hea_result.total_phi
            overall.hea_true_positives += hea_result.true_positives
            overall.hea_false_negatives += hea_result.false_negatives

        # Filename metrics
        for fn_result in patient.filename_results:
            overall.filename_total += 1
            if fn_result.is_anonymized:
                overall.filename_anonymized += 1
            else:
                overall.filename_phi_leaked += 1

    # Calculate rates
    if overall.csv_total_phi > 0:
        overall.csv_recall = overall.csv_true_positives / overall.csv_total_phi

    if overall.image_total_regions > 0:
        overall.image_mean_coverage = overall.image_fully_covered / overall.image_total_regions

    if overall.pdf_total_regions > 0:
        overall.pdf_mean_coverage = overall.pdf_fully_covered / overall.pdf_total_regions

    if overall.hea_total_phi > 0:
        overall.hea_recall = overall.hea_true_positives / overall.hea_total_phi
    
    return overall


def print_summary(overall: OverallEvaluationResult):
    """Print a summary of the evaluation results"""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)

    print(f"\n📊 CSV EVALUATION:")
    print(f"   Total PHI instances:      {overall.csv_total_phi}")
    print(f"   Correctly redacted (TP):  {overall.csv_true_positives}")
    print(f"   Not redacted (FN):        {overall.csv_false_negatives}")
    print(f"   Recall:                   {overall.csv_recall:.2%}")

    if overall.csv_false_negatives > 0:
        print(f"   ⚠️  WARNING: {overall.csv_false_negatives} PHI instances were NOT properly redacted!")
    else:
        print(f"   ✓ All CSV PHI successfully redacted!")

    print(f"\n🖼️  IMAGE EVALUATION (DICOM, PNG):")
    print(f"   Total PHI regions:        {overall.image_total_regions}")
    print(f"   Fully covered (≥95%):     {overall.image_fully_covered}")
    print(f"   Not adequately covered:   {overall.image_not_covered}")
    print(f"   Coverage rate:            {overall.image_mean_coverage:.2%}")

    if overall.image_not_covered > 0:
        print(f"   ⚠️  WARNING: {overall.image_not_covered} PHI regions are not adequately covered!")
    else:
        print(f"   ✓ All image PHI regions properly redacted!")

    print(f"\n📄 PDF EVALUATION (ECG):")
    print(f"   Total PHI regions:        {overall.pdf_total_regions}")
    print(f"   Fully covered (≥95%):     {overall.pdf_fully_covered}")
    print(f"   Not adequately covered:   {overall.pdf_not_covered}")
    print(f"   Coverage rate:            {overall.pdf_mean_coverage:.2%}")

    if overall.pdf_not_covered > 0:
        print(f"   ⚠️  WARNING: {overall.pdf_not_covered} PHI regions in PDFs are not adequately covered!")
    else:
        print(f"   ✓ All PDF PHI regions properly redacted!")

    print(f"\n📝 HEA EVALUATION (ECG Headers):")
    print(f"   Total PHI instances:      {overall.hea_total_phi}")
    print(f"   Correctly redacted (TP):  {overall.hea_true_positives}")
    print(f"   Not redacted (FN):        {overall.hea_false_negatives}")
    print(f"   Recall:                   {overall.hea_recall:.2%}")

    if overall.hea_false_negatives > 0:
        print(f"   ⚠️  WARNING: {overall.hea_false_negatives} PHI instances in HEA files were NOT properly redacted!")
    else:
        print(f"   ✓ All HEA file PHI successfully redacted!")

    print(f"\n📛 FILENAME EVALUATION:")
    print(f"   Total files:              {overall.filename_total}")
    print(f"   Properly anonymized:      {overall.filename_anonymized}")
    print(f"   PHI leaked in filename:   {overall.filename_phi_leaked}")

    if overall.filename_phi_leaked > 0:
        print(f"   ⚠️  WARNING: {overall.filename_phi_leaked} filenames contain PHI!")
    else:
        print(f"   ✓ All filenames properly anonymized!")

    # Overall assessment
    print(f"\n{'='*80}")
    total_issues = overall.csv_false_negatives + overall.image_not_covered + overall.pdf_not_covered + overall.hea_false_negatives + overall.filename_phi_leaked
    if total_issues == 0:
        print("🎉 OVERALL: All anonymization checks passed!")
    else:
        print(f"⚠️  OVERALL: {total_issues} issues found that need attention")
    print("="*80)


def save_results(overall: OverallEvaluationResult, output_dir: str):
    """Save evaluation results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary JSON
    summary_file = os.path.join(output_dir, f"evaluation_summary_{timestamp}.json")
    
    summary_dict = {
        'timestamp': overall.timestamp,
        'labels_dir': overall.labels_dir,
        'results_dir': overall.results_dir,
        'csv_metrics': {
            'total_phi': overall.csv_total_phi,
            'true_positives': overall.csv_true_positives,
            'false_negatives': overall.csv_false_negatives,
            'recall': overall.csv_recall
        },
        'image_metrics': {
            'total_regions': overall.image_total_regions,
            'fully_covered': overall.image_fully_covered,
            'not_covered': overall.image_not_covered,
            'coverage_rate': overall.image_mean_coverage
        },
        'pdf_metrics': {
            'total_regions': overall.pdf_total_regions,
            'fully_covered': overall.pdf_fully_covered,
            'not_covered': overall.pdf_not_covered,
            'coverage_rate': overall.pdf_mean_coverage
        },
        'hea_metrics': {
            'total_phi': overall.hea_total_phi,
            'true_positives': overall.hea_true_positives,
            'false_negatives': overall.hea_false_negatives,
            'recall': overall.hea_recall
        },
        'filename_metrics': {
            'total': overall.filename_total,
            'anonymized': overall.filename_anonymized,
            'phi_leaked': overall.filename_phi_leaked
        },
        'patients_evaluated': len(overall.patient_results)
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_dict, f, indent=2)
    
    print(f"\n📁 Results saved to: {summary_file}")
    
    # Save detailed CSV report
    csv_report_file = os.path.join(output_dir, f"evaluation_details_{timestamp}.csv")

    with open(csv_report_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['patient', 'type', 'filename', 'metric', 'value'])

        for patient in overall.patient_results:
            for csv_result in patient.csv_results:
                writer.writerow([patient.original_folder, 'csv', csv_result.filename, 'recall', f"{csv_result.recall:.4f}"])
                writer.writerow([patient.original_folder, 'csv', csv_result.filename, 'false_negatives', csv_result.false_negatives])

            for img_result in patient.image_results:
                writer.writerow([patient.original_folder, 'image', img_result.filename, 'mean_coverage', f"{img_result.mean_coverage:.4f}"])
                writer.writerow([patient.original_folder, 'image', img_result.filename, 'not_covered', img_result.not_covered])

            for pdf_result in patient.pdf_results:
                writer.writerow([patient.original_folder, 'pdf', pdf_result.filename, 'mean_coverage', f"{pdf_result.mean_coverage:.4f}"])
                writer.writerow([patient.original_folder, 'pdf', pdf_result.filename, 'not_covered', pdf_result.not_covered])

            for hea_result in patient.hea_results:
                writer.writerow([patient.original_folder, 'hea', hea_result.filename, 'recall', f"{hea_result.recall:.4f}"])
                writer.writerow([patient.original_folder, 'hea', hea_result.filename, 'false_negatives', hea_result.false_negatives])

            for fn_result in patient.filename_results:
                writer.writerow([patient.original_folder, 'filename', fn_result.original, 'is_anonymized', fn_result.is_anonymized])

    print(f"📁 Detailed report saved to: {csv_report_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate anonymization quality across CSV, DICOM, and PDF files'
    )
    parser.add_argument(
        '--labels', '-l',
        default='/Users/julian_anja/Documents/coding/mimiciv-anonymization-pipeline/data/primary/patient_records_labels',
        help='Path to labels directory with PHI annotations'
    )
    parser.add_argument(
        '--results', '-r',
        default='/Users/julian_anja/Documents/coding/mimiciv-anonymization-pipeline/data/results_copy',
        help='Path to results directory with anonymized files'
    )
    parser.add_argument(
        '--output', '-o',
        default='/Users/julian_anja/Documents/coding/mimiciv-anonymization-pipeline/data/evaluation_results',
        help='Path to output directory for evaluation results'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed output'
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    overall = run_evaluation_pipeline(args.labels, args.results, args.output)
    
    # Print summary
    print_summary(overall)
    
    # Save results
    save_results(overall, args.output)
    
    # Exit with error code if there are issues
    total_issues = overall.csv_false_negatives + overall.image_not_covered + overall.pdf_not_covered + overall.filename_phi_leaked
    if total_issues > 0:
        exit(1)


if __name__ == '__main__':
    main()
