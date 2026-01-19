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
    false_positives: int = 0  # Non-PHI tokens incorrectly redacted
    true_negatives: int = 0   # Non-PHI tokens correctly left unchanged
    phi_not_redacted: List[str] = field(default_factory=list)
    over_redacted: List[str] = field(default_factory=list)  # Non-PHI that was redacted
    
    @property
    def recall(self) -> float:
        """Sensitivity / True Positive Rate: TP / (TP + FN)"""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 1.0
    
    @property
    def sensitivity(self) -> float:
        """Same as recall: TP / (TP + FN)"""
        return self.recall
    
    @property
    def specificity(self) -> float:
        """True Negative Rate: TN / (TN + FP)"""
        denom = self.true_negatives + self.false_positives
        return self.true_negatives / denom if denom > 0 else 1.0
    
    @property
    def precision(self) -> float:
        """Positive Predictive Value: TP / (TP + FP)"""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 1.0
    
    @property
    def accuracy(self) -> float:
        """(TP + TN) / (TP + TN + FP + FN)"""
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        return (self.true_positives + self.true_negatives) / total if total > 0 else 1.0
    
    @property
    def false_negative_rate(self) -> float:
        """FNR = FN / (TP + FN) = 1 - Sensitivity"""
        return 1.0 - self.sensitivity
    
    @property
    def false_positive_rate(self) -> float:
        """FPR = FP / (TN + FP) = 1 - Specificity"""
        return 1.0 - self.specificity


@dataclass
class ImageEvaluationResult:
    """Results from evaluating a single image file"""
    filename: str
    total_phi_regions: int = 0
    detected_redactions: int = 0
    mean_iou: float = 0.0
    mean_coverage: float = 0.0
    iou_threshold_met: int = 0  # >= 0.5 IoU
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
    false_positives: int = 0  # Non-PHI tokens incorrectly redacted
    true_negatives: int = 0   # Non-PHI tokens correctly left unchanged
    phi_not_redacted: List[str] = field(default_factory=list)
    over_redacted: List[str] = field(default_factory=list)
    
    @property
    def recall(self) -> float:
        """Sensitivity / True Positive Rate: TP / (TP + FN)"""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 1.0
    
    @property
    def sensitivity(self) -> float:
        """Same as recall: TP / (TP + FN)"""
        return self.recall
    
    @property
    def specificity(self) -> float:
        """True Negative Rate: TN / (TN + FP)"""
        denom = self.true_negatives + self.false_positives
        return self.true_negatives / denom if denom > 0 else 1.0
    
    @property
    def precision(self) -> float:
        """Positive Predictive Value: TP / (TP + FP)"""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 1.0
    
    @property
    def accuracy(self) -> float:
        """(TP + TN) / (TP + TN + FP + FN)"""
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        return (self.true_positives + self.true_negatives) / total if total > 0 else 1.0
    
    @property
    def false_negative_rate(self) -> float:
        """FNR = FN / (TP + FN) = 1 - Sensitivity"""
        return 1.0 - self.sensitivity
    
    @property
    def false_positive_rate(self) -> float:
        """FPR = FP / (TN + FP) = 1 - Specificity"""
        return 1.0 - self.specificity


@dataclass
class DateShiftEvaluationResult:
    """Results from evaluating date shifting for a single file"""
    filename: str
    total_dates: int = 0
    correctly_shifted: int = 0  # Date was shifted (different from original)
    not_shifted: int = 0  # Date is still the same as original
    incorrectly_redacted: int = 0  # Date was redacted instead of shifted
    shift_failed: int = 0  # Date shifting failed (contains [SHIFT_FAILED] or similar)
    dates_not_shifted: List[str] = field(default_factory=list)
    dates_redacted: List[str] = field(default_factory=list)
    dates_failed: List[str] = field(default_factory=list)

    @property
    def shift_rate(self) -> float:
        """Fraction of dates that were correctly shifted"""
        return self.correctly_shifted / self.total_dates if self.total_dates > 0 else 1.0

    @property
    def redaction_error_rate(self) -> float:
        """Fraction of dates that were incorrectly redacted"""
        return self.incorrectly_redacted / self.total_dates if self.total_dates > 0 else 0.0


@dataclass
class PatientEvaluationResult:
    """Aggregated results for a single patient"""
    original_folder: str
    anonymized_folder: str
    csv_results: List[CSVEvaluationResult] = field(default_factory=list)
    csv_date_results: List[DateShiftEvaluationResult] = field(default_factory=list)
    image_results: List[ImageEvaluationResult] = field(default_factory=list)
    pdf_results: List[ImageEvaluationResult] = field(default_factory=list)
    hea_results: List[HEAEvaluationResult] = field(default_factory=list)
    hea_date_results: List[DateShiftEvaluationResult] = field(default_factory=list)
    filename_results: List[FilenameEvaluationResult] = field(default_factory=list)


@dataclass
class OverallEvaluationResult:
    """Aggregated results across all patients"""
    timestamp: str = ""
    labels_dir: str = ""
    results_dir: str = ""
    patient_results: List[PatientEvaluationResult] = field(default_factory=list)

    # Aggregate CSV metrics (micro-averaged / totals)
    csv_total_phi: int = 0
    csv_true_positives: int = 0
    csv_false_negatives: int = 0
    csv_false_positives: int = 0
    csv_true_negatives: int = 0
    csv_recall: float = 0.0

    # Macro-averaged CSV metrics (averaged per document)
    csv_macro_sensitivity: float = 0.0
    csv_macro_specificity: float = 0.0
    csv_macro_accuracy: float = 0.0
    csv_macro_precision: float = 0.0
    csv_macro_fnr: float = 0.0
    csv_macro_fpr: float = 0.0

    # Aggregate Image metrics (DICOM, PNG)
    image_total_regions: int = 0
    image_mean_iou: float = 0.0
    image_mean_coverage: float = 0.0
    image_iou_threshold_met: int = 0
    image_fully_covered: int = 0
    image_not_covered: int = 0

    # Aggregate PDF metrics (separate from images)
    pdf_total_regions: int = 0
    pdf_mean_iou: float = 0.0
    pdf_mean_coverage: float = 0.0
    pdf_iou_threshold_met: int = 0
    pdf_fully_covered: int = 0
    pdf_not_covered: int = 0

    # Aggregate HEA (text) metrics (micro-averaged / totals)
    hea_total_phi: int = 0
    hea_true_positives: int = 0
    hea_false_negatives: int = 0
    hea_false_positives: int = 0
    hea_true_negatives: int = 0
    hea_recall: float = 0.0

    # Macro-averaged HEA metrics (averaged per document)
    hea_macro_sensitivity: float = 0.0
    hea_macro_specificity: float = 0.0
    hea_macro_accuracy: float = 0.0
    hea_macro_precision: float = 0.0
    hea_macro_fnr: float = 0.0
    hea_macro_fpr: float = 0.0

    # Aggregate Filename metrics
    filename_total: int = 0
    filename_anonymized: int = 0
    filename_phi_leaked: int = 0

    # Aggregate Date Shift metrics (CSV + HEA combined)
    date_total: int = 0
    date_correctly_shifted: int = 0
    date_not_shifted: int = 0
    date_incorrectly_redacted: int = 0
    date_shift_failed: int = 0
    date_shift_rate: float = 0.0
    date_redaction_error_rate: float = 0.0

    # Combined macro-averaged metrics (across all document types)
    combined_macro_sensitivity: float = 0.0
    combined_macro_specificity: float = 0.0
    combined_macro_accuracy: float = 0.0
    combined_macro_precision: float = 0.0
    combined_macro_fnr: float = 0.0
    combined_macro_fpr: float = 0.0

    # Universal metrics (ALL document types: CSV, HEA, Images, PDFs, DICOM)
    # Coverage is treated as sensitivity for images/PDFs
    universal_macro_sensitivity: float = 0.0  # Includes image/PDF coverage as sensitivity
    universal_macro_specificity: float = 0.0
    universal_macro_accuracy: float = 0.0
    universal_macro_precision: float = 0.0
    universal_n_documents: int = 0  # Total documents included

    # Universal metrics WITH date shift evaluation
    # Date shift rate is treated as an additional accuracy component
    universal_with_dates_macro_sensitivity: float = 0.0
    universal_with_dates_macro_specificity: float = 0.0
    universal_with_dates_macro_accuracy: float = 0.0
    universal_with_dates_macro_precision: float = 0.0
    universal_with_dates_n_documents: int = 0


# ============================================================================
# Utility Functions
# ============================================================================

def extract_phi_from_text(text: str) -> Set[str]:
    """Extract all PHI values from text marked with <PER> tags"""
    pattern = r'<PER>(.*?)</PER>'
    return set(re.findall(pattern, text))


def is_year_only_or_range(date_str: str) -> bool:
    """
    Check if the date string is just a year (e.g., "2140") or a year range (e.g., "2011 - 2013").
    These formats are excluded from date shift evaluation because they are ambiguous
    and may represent non-date values.
    """
    date_str = date_str.strip()
    # Year only: exactly 4 digits
    if re.match(r'^\d{4}$', date_str):
        return True
    # Year range: e.g., "2011 - 2013", "2011-2013", "2011 – 2013"
    if re.match(r'^\d{4}\s*[-–]\s*\d{4}$', date_str):
        return True
    return False


def extract_dates_from_text(text: str) -> Set[str]:
    """Extract all date values from text marked with <DATE> tags"""
    pattern = r'<DATE>(.*?)</DATE>'
    return set(re.findall(pattern, text))


def extract_dates_with_positions(text: str) -> List[Tuple[str, int, int]]:
    """
    Extract all date values from text with their positions (after tag removal).

    Returns list of (date_value, start_pos, end_pos) tuples where positions
    refer to the text AFTER tags have been removed.
    """
    date_positions = []
    pattern = r'<DATE>(.*?)</DATE>'

    # We need to calculate positions after ALL tags (both PER and DATE) are removed
    # First, find all tags in order
    all_tags_pattern = r'<(?:PER|DATE)>(.*?)</(?:PER|DATE)>'

    current_pos = 0
    adjusted_pos = 0

    for match in re.finditer(all_tags_pattern, text):
        tag_value = match.group(1)
        match_start = match.start()
        match_end = match.end()

        # Add the length of text before this match
        adjusted_pos += (match_start - current_pos)

        # Check if this is a DATE tag
        if text[match_start:match_start+6] == '<DATE>':
            start_in_clean = adjusted_pos
            end_in_clean = adjusted_pos + len(tag_value)
            date_positions.append((tag_value, start_in_clean, end_in_clean))

        # Update positions
        adjusted_pos += len(tag_value)
        current_pos = match_end

    return date_positions


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


def extract_phi_from_annotated_filename(annotated_filename: str) -> List[str]:
    """Extract PHI values from annotated filename with tags like <PER>, <ID>, etc."""
    # Match any tag like <PER>value</PER>, <ID>value</ID>, etc.
    pattern = r'<(\w+)>([^<]+)</\1>'
    matches = re.findall(pattern, annotated_filename)
    return [value for tag, value in matches]


def load_label_filename_annotations(label_mapping_file: str) -> Dict[str, List[str]]:
    """Load filename annotations from labels and extract expected PHI values.

    Returns a dict mapping original_filename -> list of expected PHI values
    """
    phi_by_filename = {}

    if not os.path.exists(label_mapping_file):
        return phi_by_filename

    with open(label_mapping_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            original = row.get('original_filename', '')
            annotated = row.get('annotated_filename', '')

            if original and annotated:
                phi_values = extract_phi_from_annotated_filename(annotated)
                phi_by_filename[original] = phi_values

    return phi_by_filename


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

def tokenize_text(text: str) -> List[Tuple[str, int, int]]:
    """
    Tokenize text into words with their positions.
    Returns list of (token, start_pos, end_pos) tuples.
    """
    tokens = []
    pattern = r'\b\w+\b'
    for match in re.finditer(pattern, text):
        tokens.append((match.group(), match.start(), match.end()))
    return tokens


def detect_redaction_patterns(text: str) -> Set[str]:
    """
    Detect common redaction replacement patterns in text.
    Returns set of positions that appear to be redaction markers.
    """
    # Common redaction patterns
    patterns = [
        r'\[[\w_]+\]',           # [NAME], [DATE], [ID], etc.
        r'\{[\w_]+\}',           # {NAME}, {DATE}, etc.
        r'<[\w_]+>',             # <NAME>, <DATE>, etc.
        r'\*+',                  # ***, ****, etc.
        r'X{3,}',                # XXX, XXXX, etc.
        r'REDACTED',             # REDACTED
        r'ANONYMIZED',           # ANONYMIZED
        r'PHI_\w+',              # PHI_NAME, PHI_DATE, etc.
        r'PERSON_ID_\d+',        # PERSON_ID_123
    ]
    
    redaction_tokens = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            redaction_tokens.add((match.start(), match.end()))
    
    return redaction_tokens


def evaluate_csv_field(source_value: str, redacted_value: str, min_phi_length: int = 3) -> Tuple[int, int, int, int, List[str], List[str]]:
    """
    Evaluate redaction for a single CSV field using position-based comparison.

    This function:
    1. Extracts PHI values and their positions from the labeled source text
    2. Removes tags from source to get clean text
    3. Compares each PHI region in the clean text with the corresponding region in redacted text
    4. Only marks as FN if the PHI value still appears at its ORIGINAL position
    5. Detects over-redaction (false positives) by comparing non-PHI tokens

    Example:
        source_value:   "Patient is <PER>67</PER> years old, BP: 120/67"
        redacted_value: "Patient is [AGE] years old, BP: 120/67"

        - Extracts PHI "67" at position 11-13 (after tag removal)
        - Compares "67" with "[AGE]" at position 11-16 in redacted text
        - Different → TP (correctly redacted)
        - The "67" in "120/67" is ignored (different position)

    Returns: (true_positives, false_negatives, false_positives, true_negatives, phi_not_redacted, over_redacted)
    """
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    not_redacted = []
    over_redacted = []

    # Extract PHI with their positions
    phi_positions = extract_phi_with_positions(source_value)
    
    # Get clean source text (without tags)
    clean_source = remove_tags(source_value)
    
    # Create a set of positions that are PHI
    phi_position_ranges = set()
    for phi_value, start_pos, end_pos in phi_positions:
        for pos in range(start_pos, end_pos):
            phi_position_ranges.add(pos)

    # Evaluate PHI redaction (TP/FN)
    for phi_value, start_pos, end_pos in phi_positions:
        # Handle cases where redacted text might be longer/shorter
        if start_pos >= len(redacted_value):
            tp += 1
            continue

        # Check if the exact PHI value still appears at or near this position
        window_start = max(0, start_pos - len(phi_value))
        window_end = min(len(redacted_value), end_pos + len(phi_value))
        window_text = redacted_value[window_start:window_end]

        if phi_value in window_text:
            fn += 1
            not_redacted.append(f"{phi_value}@pos{start_pos}")
        else:
            tp += 1

    # Evaluate non-PHI (FP/TN) - token-based comparison
    # Tokenize the clean source text to get non-PHI tokens
    source_tokens = tokenize_text(clean_source)
    
    # Detect redaction markers in the redacted value
    redaction_markers = detect_redaction_patterns(redacted_value)
    
    for token, start_pos, end_pos in source_tokens:
        # Check if this token overlaps with any PHI position
        is_phi = any(pos in phi_position_ranges for pos in range(start_pos, end_pos))
        
        if is_phi:
            # This token is part of PHI, already handled above
            continue
        
        # This is a non-PHI token - check if it was incorrectly redacted
        # Look for the token in the redacted value
        if start_pos < len(redacted_value):
            # Check if this position now contains a redaction marker
            is_now_redacted = False
            for marker_start, marker_end in redaction_markers:
                # Check if marker overlaps with this token's expected position
                if (marker_start <= start_pos < marker_end) or (marker_start < end_pos <= marker_end):
                    is_now_redacted = True
                    break
            
            # Also check if the original token is still present in a window
            window_start = max(0, start_pos - 5)
            window_end = min(len(redacted_value), end_pos + 5)
            window_text = redacted_value[window_start:window_end] if window_end > window_start else ""
            
            if is_now_redacted or (token not in window_text and len(token) >= 2):
                # Token appears to be redacted (FP)
                fp += 1
                over_redacted.append(f"{token}@pos{start_pos}")
            else:
                # Token correctly left unchanged (TN)
                tn += 1
        else:
            # Position doesn't exist anymore, consider it potentially over-redacted
            # But only if it's a significant token
            if len(token) >= 3:
                fp += 1
                over_redacted.append(f"{token}@pos{start_pos}")

    return tp, fn, fp, tn, not_redacted, over_redacted


def is_redacted(value: str) -> bool:
    """Check if a value appears to be redacted (replaced with asterisks or redaction markers)"""
    if not value:
        return False
    # Check for common redaction patterns
    value_stripped = value.strip()
    # All asterisks
    if re.match(r'^\*+$', value_stripped):
        return True
    # Redaction markers
    if value_stripped.upper() in ['REDACTED', 'ANONYMIZED', '[REDACTED]', '[ANONYMIZED]']:
        return True
    # Pattern like [NAME], [DATE], etc.
    if re.match(r'^\[[\w_]+\]$', value_stripped):
        return True
    return False


def is_valid_shifted_date(original_date: str, result_value: str) -> bool:
    """
    Check if the result value is a valid time-shifted version of the original date.

    A valid shift means:
    1. The result is a valid date/datetime in a similar format
    2. The result is different from the original
    3. The result is NOT redacted (asterisks or redaction markers)
    """
    if not result_value or is_redacted(result_value):
        return False

    # Check if result contains failure markers
    if '[SHIFT_FAILED]' in result_value:
        return False

    # The date should be different from the original
    if result_value.strip() == original_date.strip():
        return False

    # Try to parse both dates to verify they're valid and different
    try:
        from anonymizer.tools.time_shift_tool import detect_format

        original_format = detect_format(original_date)
        result_format = detect_format(result_value)

        if original_format and result_format:
            # Both are valid dates - consider it shifted
            return True
    except ImportError:
        pass

    # If we can't parse the format but the value changed and isn't redacted,
    # assume it was shifted
    return result_value.strip() != original_date.strip()


def evaluate_date_shift_field(source_value: str, result_value: str) -> Tuple[int, int, int, int, List[str], List[str], List[str]]:
    """
    Evaluate date shifting for a single field.

    Args:
        source_value: The labeled source value (with <DATE> tags)
        result_value: The anonymized result value

    Returns:
        Tuple of (correctly_shifted, not_shifted, incorrectly_redacted, shift_failed,
                  dates_not_shifted, dates_redacted, dates_failed)
    """
    correctly_shifted = 0
    not_shifted = 0
    incorrectly_redacted = 0
    shift_failed = 0
    dates_not_shifted = []
    dates_redacted = []
    dates_failed = []

    # Extract dates with positions from the source
    date_positions = extract_dates_with_positions(source_value)

    if not date_positions:
        return 0, 0, 0, 0, [], [], []

    # Get clean source text (without tags)
    clean_source = remove_tags(source_value)

    for date_value, start_pos, end_pos in date_positions:
        # Skip year-only and year-range formats (too ambiguous for evaluation)
        if is_year_only_or_range(date_value):
            continue

        # Check what's at this position in the result
        if start_pos >= len(result_value):
            # Position doesn't exist - check if the whole value looks redacted
            if is_redacted(result_value):
                incorrectly_redacted += 1
                dates_redacted.append(f"{date_value}@pos{start_pos}")
            else:
                shift_failed += 1
                dates_failed.append(f"{date_value}@pos{start_pos}")
            continue

        # Extract the value at the corresponding position in the result
        # Since dates can vary in length after shifting, look for a date-like pattern
        window_start = max(0, start_pos - 5)
        window_end = min(len(result_value), end_pos + 10)
        window_text = result_value[window_start:window_end]

        # Check if the original date is still there (not shifted)
        if date_value in window_text:
            not_shifted += 1
            dates_not_shifted.append(f"{date_value}@pos{start_pos}")
        # Check if it was incorrectly redacted
        elif is_redacted(window_text) or re.match(r'^\*+', result_value[start_pos:end_pos+5]):
            incorrectly_redacted += 1
            dates_redacted.append(f"{date_value}@pos{start_pos}")
        # Check for shift failure markers
        elif '[SHIFT_FAILED]' in result_value:
            shift_failed += 1
            dates_failed.append(f"{date_value}@pos{start_pos}")
        else:
            # Date was changed and is not redacted - assume correctly shifted
            correctly_shifted += 1

    return correctly_shifted, not_shifted, incorrectly_redacted, shift_failed, dates_not_shifted, dates_redacted, dates_failed


def evaluate_csv_date_shifts(label_path: str, result_path: str) -> DateShiftEvaluationResult:
    """
    Evaluate date shifting for a single CSV file.

    Checks if dates marked with <DATE> tags were correctly time-shifted
    (not redacted, not unchanged).
    """
    filename = os.path.basename(label_path)
    result = DateShiftEvaluationResult(filename=filename)

    if not os.path.exists(label_path):
        return result

    if not os.path.exists(result_path):
        return result

    # Read both files
    with open(label_path, 'r', encoding='utf-8') as f:
        label_rows = list(csv.DictReader(f))

    with open(result_path, 'r', encoding='utf-8') as f:
        result_rows = list(csv.DictReader(f))

    # Compare row by row
    for label_row, result_row in zip(label_rows, result_rows):
        for field_name in label_row.keys():
            if field_name not in result_row:
                continue

            label_value = label_row[field_name] or ''
            result_value = result_row[field_name] or ''

            # Only evaluate fields that have DATE tags
            if '<DATE>' not in label_value:
                continue

            shifted, not_shifted, redacted, failed, dates_not_shifted, dates_redacted, dates_failed = \
                evaluate_date_shift_field(label_value, result_value)

            result.total_dates += shifted + not_shifted + redacted + failed
            result.correctly_shifted += shifted
            result.not_shifted += not_shifted
            result.incorrectly_redacted += redacted
            result.shift_failed += failed
            result.dates_not_shifted.extend(dates_not_shifted)
            result.dates_redacted.extend(dates_redacted)
            result.dates_failed.extend(dates_failed)

    return result


def evaluate_csv_file(label_path: str, result_path: str) -> CSVEvaluationResult:
    """
    Evaluate a single CSV file by comparing labeled PHI with anonymized output.
    Tracks TP, FN, FP, and TN for comprehensive metric calculation.
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
            
            # Evaluate all fields, not just those with PHI tags
            # This allows us to detect over-redaction (FP) in non-PHI fields too
            tp, fn, fp, tn, not_redacted, over_redacted = evaluate_csv_field(label_value, result_value)
            
            result.total_phi += tp + fn
            result.true_positives += tp
            result.false_negatives += fn
            result.false_positives += fp
            result.true_negatives += tn
            result.phi_not_redacted.extend(not_redacted)
            result.over_redacted.extend(over_redacted)
    
    return result


# ============================================================================
# HEA (Text) Evaluation
# ============================================================================

def evaluate_hea_file(label_path: str, result_path: str) -> HEAEvaluationResult:
    """
    Evaluate a single HEA text file by comparing labeled PHI with anonymized output.
    
    HEA files are ECG header files containing PHI marked with <PER> tags.
    The evaluation compares line by line to check if PHI was properly redacted.
    Tracks TP, FN, FP, and TN for comprehensive metric calculation.
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
    
    # Compare line by line - evaluate all lines for FP/TN tracking
    for i, label_line in enumerate(label_lines):
        # Get corresponding result line (if exists)
        result_line = result_lines[i] if i < len(result_lines) else ''
        
        # Use the same position-based evaluation as CSV fields
        tp, fn, fp, tn, not_redacted, over_redacted = evaluate_csv_field(label_line, result_line)
        
        result.total_phi += tp + fn
        result.true_positives += tp
        result.false_negatives += fn
        result.false_positives += fp
        result.true_negatives += tn
        result.phi_not_redacted.extend(not_redacted)
        result.over_redacted.extend(over_redacted)
    
    return result


def evaluate_hea_date_shifts(label_path: str, result_path: str) -> DateShiftEvaluationResult:
    """
    Evaluate date shifting for a single HEA (text) file.

    Checks if dates marked with <DATE> tags were correctly time-shifted
    (not redacted, not unchanged).
    """
    filename = os.path.basename(label_path)
    result = DateShiftEvaluationResult(filename=filename)

    if not os.path.exists(label_path):
        return result

    if not os.path.exists(result_path):
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
        # Get corresponding result line (if exists)
        result_line = result_lines[i] if i < len(result_lines) else ''

        # Only evaluate lines that have DATE tags
        if '<DATE>' not in label_line:
            continue

        shifted, not_shifted, redacted, failed, dates_not_shifted, dates_redacted, dates_failed = \
            evaluate_date_shift_field(label_line, result_line)

        result.total_dates += shifted + not_shifted + redacted + failed
        result.correctly_shifted += shifted
        result.not_shifted += not_shifted
        result.incorrectly_redacted += redacted
        result.shift_failed += failed
        result.dates_not_shifted.extend(dates_not_shifted)
        result.dates_redacted.extend(dates_redacted)
        result.dates_failed.extend(dates_failed)

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


def calculate_pixel_coverage_and_iou(redacted_img: np.ndarray, gt_box: Dict,
                                      black_threshold: int = 10) -> Tuple[float, float]:
    """
    Calculate both coverage and IoU for a ground truth PHI region using pixel-based analysis.

    This is more accurate than box-matching because:
    1. It doesn't depend on detecting redaction boxes via contours
    2. It directly measures if PHI regions are actually blacked out
    3. Works even when redactions span multiple fragmented areas

    Args:
        redacted_img: The redacted image as numpy array (BGR or grayscale)
        gt_box: Ground truth bounding box with 'x', 'y', 'width', 'height'
        black_threshold: Pixel values below this are considered black/redacted

    Returns:
        Tuple of (coverage, iou):
        - coverage: Float between 0.0 and 1.0 representing fraction of GT region that is black
        - iou: Float between 0.0 and 1.0 representing Intersection over Union
    """
    x, y, w, h = gt_box['x'], gt_box['y'], gt_box['width'], gt_box['height']

    # Ensure coordinates are within image bounds
    img_h, img_w = redacted_img.shape[:2]
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)

    if w <= 0 or h <= 0:
        return 0.0, 0.0

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

    # Calculate coverage (what fraction of GT is covered)
    coverage = black_pixels / total_pixels if total_pixels > 0 else 0.0

    # Calculate IoU (intersection over union)
    # Intersection = black pixels (already counted)
    # Union = GT area + redacted area - intersection
    # For pixel-based approach: Union = GT area (since we only have black pixels in GT region)
    # This gives us the same as coverage for pixel-based evaluation
    # For more accurate IoU, we would need to detect the actual redaction box

    # For pixel-based analysis, IoU ≈ coverage (conservative estimate)
    # because intersection = black_pixels and union ≥ total_pixels
    iou = coverage  # Conservative estimate

    return coverage, iou


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
    
    # Convert annotations to box format and calculate pixel-based coverage and IoU
    coverages = []
    ious = []
    for ann in file_annotations:
        gt_box = {
            'x': ann.x, 'y': ann.y,
            'width': ann.width, 'height': ann.height,
            'field': ann.field, 'text': ann.text
        }

        # Calculate pixel-based coverage and IoU
        coverage, iou = calculate_pixel_coverage_and_iou(redacted_img, gt_box)
        coverages.append(coverage)
        ious.append(iou)

        # Store details
        result.details.append({
            'field': ann.field,
            'text': ann.text,
            'iou': iou,
            'coverage': coverage,
            'status': 'covered' if coverage >= 0.95 else ('partial' if coverage >= 0.5 else 'missing')
        })

    # Calculate aggregate metrics
    result.mean_iou = sum(ious) / len(ious) if ious else 0.0
    result.mean_coverage = sum(coverages) / len(coverages) if coverages else 0.0
    result.iou_threshold_met = sum(1 for iou in ious if iou >= 0.5)
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
        result.iou_threshold_met = sum(1 for iou in ious if iou >= 0.5)
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

            # Also evaluate date shifts for this file
            date_result = evaluate_csv_date_shifts(label_path, result_path)
            if date_result.total_dates > 0:
                result.csv_date_results.append(date_result)

            status = "✓" if csv_result.false_negatives == 0 else "✗"
            print(f"  {status} {label_file}: {csv_result.true_positives}/{csv_result.total_phi} PHI redacted "
                  f"(recall: {csv_result.recall:.2%})")

            # Print date shift status if there are dates
            if date_result.total_dates > 0:
                date_status = "✓" if date_result.incorrectly_redacted == 0 and date_result.not_shifted == 0 else "✗"
                print(f"    {date_status} Dates: {date_result.correctly_shifted}/{date_result.total_dates} shifted "
                      f"({date_result.incorrectly_redacted} redacted, {date_result.not_shifted} unchanged)")

        # ---- CSV Filename Evaluation ----
        # Load expected PHI from label annotations
        label_phi_by_filename = load_label_filename_annotations(label_mapping_file)

        for mapping in result_mappings:
            expected_phi = label_phi_by_filename.get(mapping.original, [])
            fn_result = FilenameEvaluationResult(
                original=mapping.original,
                anonymized=mapping.anonymized,
                expected_phi=expected_phi
            )

            # Check if PHI leaked in anonymized filename
            for phi in expected_phi:
                if phi.lower().replace('_', ' ') in mapping.anonymized.lower().replace('_', ' '):
                    fn_result.phi_in_filename.append(phi)
                    fn_result.is_anonymized = False

            # Check for fallback anonymization: PHI existed but only _anonymized was added
            if expected_phi and fn_result.is_anonymized:
                # If original filename appears unchanged (just with _anonymized suffix), it's a fallback
                base_anonymized = mapping.anonymized.replace('_anonymized', '').replace('.csv', '')
                base_original = mapping.original.replace('.csv', '')
                if base_anonymized == base_original:
                    fn_result.is_anonymized = False
                    fn_result.phi_in_filename = expected_phi  # All PHI leaked

            result.filename_results.append(fn_result)

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
                  f"fully covered (IoU: {eval_result.mean_iou:.2%}, Coverage: {eval_result.mean_coverage:.2%})")
        
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

                # Also evaluate date shifts for HEA files
                hea_date_result = evaluate_hea_date_shifts(label_path, result_path)
                if hea_date_result.total_dates > 0:
                    result.hea_date_results.append(hea_date_result)

                status = "✓" if hea_result.false_negatives == 0 else "✗"
                print(f"  {status} {label_file}: {hea_result.true_positives}/{hea_result.total_phi} PHI redacted "
                      f"(recall: {hea_result.recall:.2%})")

                # Print date shift status if there are dates
                if hea_date_result.total_dates > 0:
                    date_status = "✓" if hea_date_result.incorrectly_redacted == 0 and hea_date_result.not_shifted == 0 else "✗"
                    print(f"    {date_status} Dates: {hea_date_result.correctly_shifted}/{hea_date_result.total_dates} shifted "
                          f"({hea_date_result.incorrectly_redacted} redacted, {hea_date_result.not_shifted} unchanged)")
        
        # Evaluate filename anonymization using PHI from label annotations
        # Build a mapping of original filename -> expected PHI from annotations
        label_phi_by_filename: Dict[str, List[str]] = {}
        for ann in annotations:
            if ann.filename not in label_phi_by_filename:
                label_phi_by_filename[ann.filename] = []

        # Also extract PHI from filename_with_phi_tags if available in the annotation file
        if os.path.exists(ann_file):
            with open(ann_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = row.get('filename', '')
                    annotated_fn = row.get('filename_with_phi_tags', '')
                    if filename and annotated_fn:
                        phi_values = extract_phi_from_annotated_filename(annotated_fn)
                        if filename not in label_phi_by_filename:
                            label_phi_by_filename[filename] = []
                        # Add unique PHI values
                        for phi in phi_values:
                            if phi not in label_phi_by_filename[filename]:
                                label_phi_by_filename[filename].append(phi)

        for mapping in result_mappings:
            # Use PHI from labels if available, otherwise fall back to result mapping
            expected_phi = label_phi_by_filename.get(mapping.original, mapping.phi_values)
            fn_result = FilenameEvaluationResult(
                original=mapping.original,
                anonymized=mapping.anonymized,
                expected_phi=expected_phi
            )

            # Check if PHI leaked in anonymized filename
            for phi in expected_phi:
                if phi.lower().replace('_', ' ') in mapping.anonymized.lower().replace('_', ' '):
                    fn_result.phi_in_filename.append(phi)
                    fn_result.is_anonymized = False

            # Check for fallback anonymization: PHI existed but only _anonymized was added
            if expected_phi and fn_result.is_anonymized:
                # Get base names without extension
                ext = os.path.splitext(mapping.original)[1]
                base_anonymized = mapping.anonymized.replace('_anonymized', '').replace(ext, '')
                base_original = mapping.original.replace(ext, '')
                if base_anonymized == base_original:
                    fn_result.is_anonymized = False
                    fn_result.phi_in_filename = expected_phi  # All PHI leaked

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
    # For images and PDFs, we need to calculate weighted average coverage and IoU
    image_coverage_weighted_sum = 0.0
    image_iou_weighted_sum = 0.0
    pdf_coverage_weighted_sum = 0.0
    pdf_iou_weighted_sum = 0.0

    # Lists for macro-averaging (per-document metrics)
    csv_sensitivities = []
    csv_specificities = []
    csv_accuracies = []
    csv_precisions = []
    csv_fnrs = []
    csv_fprs = []

    hea_sensitivities = []
    hea_specificities = []
    hea_accuracies = []
    hea_precisions = []
    hea_fnrs = []
    hea_fprs = []

    # Lists for universal metrics (all document types including images/PDFs)
    # For images/PDFs, we use coverage as a proxy for sensitivity
    image_coverages = []  # Per-image coverage (treated as sensitivity)
    pdf_coverages = []    # Per-PDF coverage (treated as sensitivity)

    # Lists for date shift metrics per document
    date_shift_rates = []  # Per-document date shift success rate

    for patient in overall.patient_results:
        # CSV metrics
        for csv_result in patient.csv_results:
            overall.csv_total_phi += csv_result.total_phi
            overall.csv_true_positives += csv_result.true_positives
            overall.csv_false_negatives += csv_result.false_negatives
            overall.csv_false_positives += csv_result.false_positives
            overall.csv_true_negatives += csv_result.true_negatives

            # Collect per-document metrics for macro-averaging
            # Only include documents that have PHI (for sensitivity/precision)
            # or have non-PHI data (for specificity)
            if csv_result.total_phi > 0:
                csv_sensitivities.append(csv_result.sensitivity)
                csv_precisions.append(csv_result.precision)
                csv_fnrs.append(csv_result.false_negative_rate)
            
            # Include all documents with any tokens for specificity, accuracy, FPR
            if csv_result.total_phi > 0 or csv_result.true_negatives > 0:
                csv_specificities.append(csv_result.specificity)
                csv_accuracies.append(csv_result.accuracy)
                csv_fprs.append(csv_result.false_positive_rate)

        # Image metrics (DICOM, PNG only)
        for img_result in patient.image_results:
            overall.image_total_regions += img_result.total_phi_regions
            overall.image_fully_covered += img_result.fully_covered
            overall.image_not_covered += img_result.not_covered
            overall.image_iou_threshold_met += img_result.iou_threshold_met
            # Accumulate weighted coverage and IoU (mean * number of regions)
            image_coverage_weighted_sum += img_result.mean_coverage * img_result.total_phi_regions
            image_iou_weighted_sum += img_result.mean_iou * img_result.total_phi_regions
            # Collect per-document coverage for universal metrics
            if img_result.total_phi_regions > 0:
                image_coverages.append(img_result.mean_coverage)

        # PDF metrics (separate)
        for pdf_result in patient.pdf_results:
            overall.pdf_total_regions += pdf_result.total_phi_regions
            overall.pdf_fully_covered += pdf_result.fully_covered
            overall.pdf_not_covered += pdf_result.not_covered
            overall.pdf_iou_threshold_met += pdf_result.iou_threshold_met
            # Accumulate weighted coverage and IoU (mean * number of regions)
            pdf_coverage_weighted_sum += pdf_result.mean_coverage * pdf_result.total_phi_regions
            pdf_iou_weighted_sum += pdf_result.mean_iou * pdf_result.total_phi_regions
            # Collect per-document coverage for universal metrics
            if pdf_result.total_phi_regions > 0:
                pdf_coverages.append(pdf_result.mean_coverage)

        # HEA (text) metrics
        for hea_result in patient.hea_results:
            overall.hea_total_phi += hea_result.total_phi
            overall.hea_true_positives += hea_result.true_positives
            overall.hea_false_negatives += hea_result.false_negatives
            overall.hea_false_positives += hea_result.false_positives
            overall.hea_true_negatives += hea_result.true_negatives

            # Collect per-document metrics for macro-averaging
            # Only include documents that have PHI (for sensitivity/precision)
            # or have non-PHI data (for specificity)
            if hea_result.total_phi > 0:
                hea_sensitivities.append(hea_result.sensitivity)
                hea_precisions.append(hea_result.precision)
                hea_fnrs.append(hea_result.false_negative_rate)
            
            # Include all documents with any tokens for specificity, accuracy, FPR
            if hea_result.total_phi > 0 or hea_result.true_negatives > 0:
                hea_specificities.append(hea_result.specificity)
                hea_accuracies.append(hea_result.accuracy)
                hea_fprs.append(hea_result.false_positive_rate)

        # Filename metrics
        for fn_result in patient.filename_results:
            overall.filename_total += 1
            if fn_result.is_anonymized:
                overall.filename_anonymized += 1
            else:
                overall.filename_phi_leaked += 1

        # Date shift metrics (from CSV files)
        for date_result in patient.csv_date_results:
            overall.date_total += date_result.total_dates
            overall.date_correctly_shifted += date_result.correctly_shifted
            overall.date_not_shifted += date_result.not_shifted
            overall.date_incorrectly_redacted += date_result.incorrectly_redacted
            overall.date_shift_failed += date_result.shift_failed
            # Collect per-document date shift rate for universal metrics with dates
            if date_result.total_dates > 0:
                date_shift_rates.append(date_result.shift_rate)

        # Date shift metrics (from HEA files)
        for date_result in patient.hea_date_results:
            overall.date_total += date_result.total_dates
            overall.date_correctly_shifted += date_result.correctly_shifted
            overall.date_not_shifted += date_result.not_shifted
            # Collect per-document date shift rate for universal metrics with dates
            if date_result.total_dates > 0:
                date_shift_rates.append(date_result.shift_rate)
            overall.date_incorrectly_redacted += date_result.incorrectly_redacted
            overall.date_shift_failed += date_result.shift_failed

    # Calculate micro-averaged rates
    if overall.csv_total_phi > 0:
        overall.csv_recall = overall.csv_true_positives / overall.csv_total_phi

    if overall.image_total_regions > 0:
        overall.image_mean_coverage = image_coverage_weighted_sum / overall.image_total_regions
        overall.image_mean_iou = image_iou_weighted_sum / overall.image_total_regions

    if overall.pdf_total_regions > 0:
        overall.pdf_mean_coverage = pdf_coverage_weighted_sum / overall.pdf_total_regions
        overall.pdf_mean_iou = pdf_iou_weighted_sum / overall.pdf_total_regions

    if overall.hea_total_phi > 0:
        overall.hea_recall = overall.hea_true_positives / overall.hea_total_phi

    # Calculate date shift rates
    if overall.date_total > 0:
        overall.date_shift_rate = overall.date_correctly_shifted / overall.date_total
        overall.date_redaction_error_rate = overall.date_incorrectly_redacted / overall.date_total

    # Calculate macro-averaged metrics for CSV
    if csv_sensitivities:
        overall.csv_macro_sensitivity = sum(csv_sensitivities) / len(csv_sensitivities)
        overall.csv_macro_specificity = sum(csv_specificities) / len(csv_specificities)
        overall.csv_macro_accuracy = sum(csv_accuracies) / len(csv_accuracies)
        overall.csv_macro_precision = sum(csv_precisions) / len(csv_precisions)
        overall.csv_macro_fnr = sum(csv_fnrs) / len(csv_fnrs)
        overall.csv_macro_fpr = sum(csv_fprs) / len(csv_fprs)

    # Calculate macro-averaged metrics for HEA
    if hea_sensitivities:
        overall.hea_macro_sensitivity = sum(hea_sensitivities) / len(hea_sensitivities)
        overall.hea_macro_specificity = sum(hea_specificities) / len(hea_specificities)
        overall.hea_macro_accuracy = sum(hea_accuracies) / len(hea_accuracies)
        overall.hea_macro_precision = sum(hea_precisions) / len(hea_precisions)
        overall.hea_macro_fnr = sum(hea_fnrs) / len(hea_fnrs)
        overall.hea_macro_fpr = sum(hea_fprs) / len(hea_fprs)

    # Calculate combined macro-averaged metrics (across all document types)
    all_sensitivities = csv_sensitivities + hea_sensitivities
    all_specificities = csv_specificities + hea_specificities
    all_accuracies = csv_accuracies + hea_accuracies
    all_precisions = csv_precisions + hea_precisions
    all_fnrs = csv_fnrs + hea_fnrs
    all_fprs = csv_fprs + hea_fprs

    if all_sensitivities:
        overall.combined_macro_sensitivity = sum(all_sensitivities) / len(all_sensitivities)
        overall.combined_macro_specificity = sum(all_specificities) / len(all_specificities)
        overall.combined_macro_accuracy = sum(all_accuracies) / len(all_accuracies)
        overall.combined_macro_precision = sum(all_precisions) / len(all_precisions)
        overall.combined_macro_fnr = sum(all_fnrs) / len(all_fnrs)
        overall.combined_macro_fpr = sum(all_fprs) / len(all_fprs)

    # Calculate UNIVERSAL macro-averaged metrics (ALL document types: CSV, HEA, Images, PDFs)
    # For images/PDFs, coverage is treated as sensitivity (how much PHI was covered/detected)
    # For images/PDFs, we don't have specificity/precision/accuracy, so we only include
    # sensitivity-like metrics from them
    universal_sensitivities = csv_sensitivities + hea_sensitivities + image_coverages + pdf_coverages
    # For specificity, accuracy, precision: only text-based documents have these
    universal_specificities = csv_specificities + hea_specificities
    universal_accuracies = csv_accuracies + hea_accuracies
    universal_precisions = csv_precisions + hea_precisions

    if universal_sensitivities:
        overall.universal_macro_sensitivity = sum(universal_sensitivities) / len(universal_sensitivities)
        overall.universal_n_documents = len(universal_sensitivities)
    if universal_specificities:
        overall.universal_macro_specificity = sum(universal_specificities) / len(universal_specificities)
    if universal_accuracies:
        overall.universal_macro_accuracy = sum(universal_accuracies) / len(universal_accuracies)
    if universal_precisions:
        overall.universal_macro_precision = sum(universal_precisions) / len(universal_precisions)

    # Calculate UNIVERSAL WITH DATES macro-averaged metrics
    # This includes date shift rates as additional "accuracy" measurements
    # Date shift rate = correctly_shifted / total_dates (how accurate was the date shifting)
    universal_with_dates_sensitivities = universal_sensitivities
    universal_with_dates_specificities = universal_specificities
    # For accuracy, we include both text document accuracies AND date shift rates
    universal_with_dates_accuracies = csv_accuracies + hea_accuracies + date_shift_rates
    universal_with_dates_precisions = universal_precisions

    if universal_with_dates_sensitivities:
        overall.universal_with_dates_macro_sensitivity = sum(universal_with_dates_sensitivities) / len(universal_with_dates_sensitivities)
        overall.universal_with_dates_n_documents = len(universal_with_dates_sensitivities) + len(date_shift_rates)
    if universal_with_dates_specificities:
        overall.universal_with_dates_macro_specificity = sum(universal_with_dates_specificities) / len(universal_with_dates_specificities)
    if universal_with_dates_accuracies:
        overall.universal_with_dates_macro_accuracy = sum(universal_with_dates_accuracies) / len(universal_with_dates_accuracies)
    if universal_with_dates_precisions:
        overall.universal_with_dates_macro_precision = sum(universal_with_dates_precisions) / len(universal_with_dates_precisions)

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
    print(f"   Over-redacted (FP):       {overall.csv_false_positives}")
    print(f"   Correctly kept (TN):      {overall.csv_true_negatives}")
    print(f"   Recall (micro):           {overall.csv_recall:.2%}")
    print(f"\n   Macro-averaged per document:")
    print(f"   ├─ Sensitivity:           {overall.csv_macro_sensitivity:.2%} (based on {len([p for p in overall.patient_results for c in p.csv_results if c.total_phi > 0])} docs with PHI)")
    print(f"   ├─ Specificity:           {overall.csv_macro_specificity:.2%}")
    print(f"   ├─ Accuracy:              {overall.csv_macro_accuracy:.2%}")
    print(f"   ├─ Precision:             {overall.csv_macro_precision:.2%} (based on {len([p for p in overall.patient_results for c in p.csv_results if c.total_phi > 0])} docs with PHI)")
    print(f"   ├─ False-Negative-Rate:   {overall.csv_macro_fnr:.2%} (based on {len([p for p in overall.patient_results for c in p.csv_results if c.total_phi > 0])} docs with PHI)")
    print(f"   └─ False-Positive-Rate:   {overall.csv_macro_fpr:.2%}")

    if overall.csv_false_negatives > 0:
        print(f"   ⚠️  WARNING: {overall.csv_false_negatives} PHI instances were NOT properly redacted!")
    else:
        print(f"   ✓ All CSV PHI successfully redacted!")

    print(f"\n🖼️  IMAGE EVALUATION (DICOM, PNG):")
    print(f"   Total PHI regions:        {overall.image_total_regions}")
    print(f"   Fully covered (≥95%):     {overall.image_fully_covered}")
    print(f"   IoU threshold met (≥0.5): {overall.image_iou_threshold_met}")
    print(f"   Not adequately covered:   {overall.image_not_covered}")
    print(f"   Mean IoU:                 {overall.image_mean_iou:.2%}")
    print(f"   Mean Coverage:            {overall.image_mean_coverage:.2%}")

    if overall.image_not_covered > 0:
        print(f"   ⚠️  WARNING: {overall.image_not_covered} PHI regions are not adequately covered!")
    else:
        print(f"   ✓ All image PHI regions properly redacted!")

    print(f"\n📄 PDF EVALUATION (ECG):")
    print(f"   Total PHI regions:        {overall.pdf_total_regions}")
    print(f"   Fully covered (≥95%):     {overall.pdf_fully_covered}")
    print(f"   IoU threshold met (≥0.5): {overall.pdf_iou_threshold_met}")
    print(f"   Not adequately covered:   {overall.pdf_not_covered}")
    print(f"   Mean IoU:                 {overall.pdf_mean_iou:.2%}")
    print(f"   Mean Coverage:            {overall.pdf_mean_coverage:.2%}")

    if overall.pdf_not_covered > 0:
        print(f"   ⚠️  WARNING: {overall.pdf_not_covered} PHI regions in PDFs are not adequately covered!")
    else:
        print(f"   ✓ All PDF PHI regions properly redacted!")

    print(f"\n📝 HEA EVALUATION (ECG Headers):")
    print(f"   Total PHI instances:      {overall.hea_total_phi}")
    print(f"   Correctly redacted (TP):  {overall.hea_true_positives}")
    print(f"   Not redacted (FN):        {overall.hea_false_negatives}")
    print(f"   Over-redacted (FP):       {overall.hea_false_positives}")
    print(f"   Correctly kept (TN):      {overall.hea_true_negatives}")
    print(f"   Recall (micro):           {overall.hea_recall:.2%}")
    print(f"\n   Macro-averaged per document:")
    print(f"   ├─ Sensitivity:           {overall.hea_macro_sensitivity:.2%}")
    print(f"   ├─ Specificity:           {overall.hea_macro_specificity:.2%}")
    print(f"   ├─ Accuracy:              {overall.hea_macro_accuracy:.2%}")
    print(f"   ├─ Precision:             {overall.hea_macro_precision:.2%}")
    print(f"   ├─ False-Negative-Rate:   {overall.hea_macro_fnr:.2%}")
    print(f"   └─ False-Positive-Rate:   {overall.hea_macro_fpr:.2%}")

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

    print(f"\n📅 DATE SHIFT EVALUATION:")
    print(f"   Total dates:              {overall.date_total}")
    print(f"   Correctly shifted:        {overall.date_correctly_shifted}")
    print(f"   Not shifted (unchanged):  {overall.date_not_shifted}")
    print(f"   Incorrectly redacted:     {overall.date_incorrectly_redacted}")
    print(f"   Shift failed:             {overall.date_shift_failed}")
    print(f"   Shift rate:               {overall.date_shift_rate:.2%}")
    print(f"   Redaction error rate:     {overall.date_redaction_error_rate:.2%}")

    if overall.date_incorrectly_redacted > 0:
        print(f"   ⚠️  WARNING: {overall.date_incorrectly_redacted} dates were incorrectly REDACTED instead of shifted!")
    if overall.date_not_shifted > 0:
        print(f"   ⚠️  WARNING: {overall.date_not_shifted} dates were NOT shifted (still original)!")
    if overall.date_incorrectly_redacted == 0 and overall.date_not_shifted == 0 and overall.date_total > 0:
        print(f"   ✓ All dates successfully time-shifted!")

    # Combined macro-averaged metrics (CSV + HEA only)
    print(f"\n{'─'*80}")
    print(f"📈 COMBINED MACRO-AVERAGED METRICS (CSV + HEA only):")
    print(f"   ├─ Sensitivity:           {overall.combined_macro_sensitivity:.2%}")
    print(f"   ├─ Specificity:           {overall.combined_macro_specificity:.2%}")
    print(f"   ├─ Accuracy:              {overall.combined_macro_accuracy:.2%}")
    print(f"   ├─ Precision:             {overall.combined_macro_precision:.2%}")
    print(f"   ├─ False-Negative-Rate:   {overall.combined_macro_fnr:.2%}")
    print(f"   └─ False-Positive-Rate:   {overall.combined_macro_fpr:.2%}")

    # Universal metrics (ALL document types)
    print(f"\n{'─'*80}")
    print(f"🌐 UNIVERSAL METRICS (CSV + HEA + Images + PDFs):")
    print(f"   Documents included:       {overall.universal_n_documents}")
    print(f"   ├─ Sensitivity*:          {overall.universal_macro_sensitivity:.2%}")
    print(f"   ├─ Specificity†:          {overall.universal_macro_specificity:.2%}")
    print(f"   ├─ Accuracy†:             {overall.universal_macro_accuracy:.2%}")
    print(f"   └─ Precision†:            {overall.universal_macro_precision:.2%}")
    print(f"   (* Image/PDF coverage treated as sensitivity)")
    print(f"   († Only from CSV/HEA documents)")

    # Universal metrics WITH date shift
    print(f"\n{'─'*80}")
    print(f"🌐📅 UNIVERSAL METRICS WITH DATE SHIFT (CSV + HEA + Images + PDFs + Dates):")
    print(f"   Documents/evaluations:    {overall.universal_with_dates_n_documents}")
    print(f"   ├─ Sensitivity*:          {overall.universal_with_dates_macro_sensitivity:.2%}")
    print(f"   ├─ Specificity†:          {overall.universal_with_dates_macro_specificity:.2%}")
    print(f"   ├─ Accuracy‡:             {overall.universal_with_dates_macro_accuracy:.2%}")
    print(f"   └─ Precision†:            {overall.universal_with_dates_macro_precision:.2%}")
    print(f"   (* Image/PDF coverage treated as sensitivity)")
    print(f"   († Only from CSV/HEA documents)")
    print(f"   (‡ Includes date shift rate as accuracy component)")

    # Overall assessment
    print(f"\n{'='*80}")
    total_issues = (overall.csv_false_negatives + overall.image_not_covered + overall.pdf_not_covered +
                    overall.hea_false_negatives + overall.filename_phi_leaked +
                    overall.date_incorrectly_redacted + overall.date_not_shifted)
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
            'false_positives': overall.csv_false_positives,
            'true_negatives': overall.csv_true_negatives,
            'recall_micro': overall.csv_recall,
            'macro_averaged': {
                'sensitivity': overall.csv_macro_sensitivity,
                'specificity': overall.csv_macro_specificity,
                'accuracy': overall.csv_macro_accuracy,
                'precision': overall.csv_macro_precision,
                'false_negative_rate': overall.csv_macro_fnr,
                'false_positive_rate': overall.csv_macro_fpr
            }
        },
        'image_metrics': {
            'total_regions': overall.image_total_regions,
            'fully_covered': overall.image_fully_covered,
            'iou_threshold_met': overall.image_iou_threshold_met,
            'not_covered': overall.image_not_covered,
            'mean_iou': overall.image_mean_iou,
            'mean_coverage': overall.image_mean_coverage
        },
        'pdf_metrics': {
            'total_regions': overall.pdf_total_regions,
            'fully_covered': overall.pdf_fully_covered,
            'iou_threshold_met': overall.pdf_iou_threshold_met,
            'not_covered': overall.pdf_not_covered,
            'mean_iou': overall.pdf_mean_iou,
            'mean_coverage': overall.pdf_mean_coverage
        },
        'hea_metrics': {
            'total_phi': overall.hea_total_phi,
            'true_positives': overall.hea_true_positives,
            'false_negatives': overall.hea_false_negatives,
            'false_positives': overall.hea_false_positives,
            'true_negatives': overall.hea_true_negatives,
            'recall_micro': overall.hea_recall,
            'macro_averaged': {
                'sensitivity': overall.hea_macro_sensitivity,
                'specificity': overall.hea_macro_specificity,
                'accuracy': overall.hea_macro_accuracy,
                'precision': overall.hea_macro_precision,
                'false_negative_rate': overall.hea_macro_fnr,
                'false_positive_rate': overall.hea_macro_fpr
            }
        },
        'filename_metrics': {
            'total': overall.filename_total,
            'anonymized': overall.filename_anonymized,
            'phi_leaked': overall.filename_phi_leaked
        },
        'date_shift_metrics': {
            'total_dates': overall.date_total,
            'correctly_shifted': overall.date_correctly_shifted,
            'not_shifted': overall.date_not_shifted,
            'incorrectly_redacted': overall.date_incorrectly_redacted,
            'shift_failed': overall.date_shift_failed,
            'shift_rate': overall.date_shift_rate,
            'redaction_error_rate': overall.date_redaction_error_rate
        },
        'combined_macro_averaged': {
            'sensitivity': overall.combined_macro_sensitivity,
            'specificity': overall.combined_macro_specificity,
            'accuracy': overall.combined_macro_accuracy,
            'precision': overall.combined_macro_precision,
            'false_negative_rate': overall.combined_macro_fnr,
            'false_positive_rate': overall.combined_macro_fpr
        },
        'universal_metrics': {
            'description': 'Macro-averaged across ALL document types (CSV, HEA, Images, PDFs). Image/PDF coverage is treated as sensitivity.',
            'n_documents': overall.universal_n_documents,
            'sensitivity': overall.universal_macro_sensitivity,
            'specificity': overall.universal_macro_specificity,
            'accuracy': overall.universal_macro_accuracy,
            'precision': overall.universal_macro_precision
        },
        'universal_with_dates_metrics': {
            'description': 'Macro-averaged across ALL document types including date shift evaluation. Date shift rate is included in accuracy calculation.',
            'n_documents': overall.universal_with_dates_n_documents,
            'sensitivity': overall.universal_with_dates_macro_sensitivity,
            'specificity': overall.universal_with_dates_macro_specificity,
            'accuracy': overall.universal_with_dates_macro_accuracy,
            'precision': overall.universal_with_dates_macro_precision
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
                # Basic counts
                writer.writerow([patient.original_folder, 'csv', csv_result.filename, 'true_positives', csv_result.true_positives])
                writer.writerow([patient.original_folder, 'csv', csv_result.filename, 'false_negatives', csv_result.false_negatives])
                writer.writerow([patient.original_folder, 'csv', csv_result.filename, 'false_positives', csv_result.false_positives])
                writer.writerow([patient.original_folder, 'csv', csv_result.filename, 'true_negatives', csv_result.true_negatives])
                # Per-document metrics
                writer.writerow([patient.original_folder, 'csv', csv_result.filename, 'sensitivity', f"{csv_result.sensitivity:.4f}"])
                writer.writerow([patient.original_folder, 'csv', csv_result.filename, 'specificity', f"{csv_result.specificity:.4f}"])
                writer.writerow([patient.original_folder, 'csv', csv_result.filename, 'accuracy', f"{csv_result.accuracy:.4f}"])
                writer.writerow([patient.original_folder, 'csv', csv_result.filename, 'precision', f"{csv_result.precision:.4f}"])
                writer.writerow([patient.original_folder, 'csv', csv_result.filename, 'false_negative_rate', f"{csv_result.false_negative_rate:.4f}"])
                writer.writerow([patient.original_folder, 'csv', csv_result.filename, 'false_positive_rate', f"{csv_result.false_positive_rate:.4f}"])

            for img_result in patient.image_results:
                writer.writerow([patient.original_folder, 'image', img_result.filename, 'mean_iou', f"{img_result.mean_iou:.4f}"])
                writer.writerow([patient.original_folder, 'image', img_result.filename, 'mean_coverage', f"{img_result.mean_coverage:.4f}"])
                writer.writerow([patient.original_folder, 'image', img_result.filename, 'iou_threshold_met', img_result.iou_threshold_met])
                writer.writerow([patient.original_folder, 'image', img_result.filename, 'fully_covered', img_result.fully_covered])
                writer.writerow([patient.original_folder, 'image', img_result.filename, 'not_covered', img_result.not_covered])

            for pdf_result in patient.pdf_results:
                writer.writerow([patient.original_folder, 'pdf', pdf_result.filename, 'mean_iou', f"{pdf_result.mean_iou:.4f}"])
                writer.writerow([patient.original_folder, 'pdf', pdf_result.filename, 'mean_coverage', f"{pdf_result.mean_coverage:.4f}"])
                writer.writerow([patient.original_folder, 'pdf', pdf_result.filename, 'iou_threshold_met', pdf_result.iou_threshold_met])
                writer.writerow([patient.original_folder, 'pdf', pdf_result.filename, 'fully_covered', pdf_result.fully_covered])
                writer.writerow([patient.original_folder, 'pdf', pdf_result.filename, 'not_covered', pdf_result.not_covered])

            for hea_result in patient.hea_results:
                # Basic counts
                writer.writerow([patient.original_folder, 'hea', hea_result.filename, 'true_positives', hea_result.true_positives])
                writer.writerow([patient.original_folder, 'hea', hea_result.filename, 'false_negatives', hea_result.false_negatives])
                writer.writerow([patient.original_folder, 'hea', hea_result.filename, 'false_positives', hea_result.false_positives])
                writer.writerow([patient.original_folder, 'hea', hea_result.filename, 'true_negatives', hea_result.true_negatives])
                # Per-document metrics
                writer.writerow([patient.original_folder, 'hea', hea_result.filename, 'sensitivity', f"{hea_result.sensitivity:.4f}"])
                writer.writerow([patient.original_folder, 'hea', hea_result.filename, 'specificity', f"{hea_result.specificity:.4f}"])
                writer.writerow([patient.original_folder, 'hea', hea_result.filename, 'accuracy', f"{hea_result.accuracy:.4f}"])
                writer.writerow([patient.original_folder, 'hea', hea_result.filename, 'precision', f"{hea_result.precision:.4f}"])
                writer.writerow([patient.original_folder, 'hea', hea_result.filename, 'false_negative_rate', f"{hea_result.false_negative_rate:.4f}"])
                writer.writerow([patient.original_folder, 'hea', hea_result.filename, 'false_positive_rate', f"{hea_result.false_positive_rate:.4f}"])

            for fn_result in patient.filename_results:
                writer.writerow([patient.original_folder, 'filename', fn_result.original, 'is_anonymized', fn_result.is_anonymized])

            # Date shift results from CSV files
            for date_result in patient.csv_date_results:
                writer.writerow([patient.original_folder, 'csv_date_shift', date_result.filename, 'total_dates', date_result.total_dates])
                writer.writerow([patient.original_folder, 'csv_date_shift', date_result.filename, 'correctly_shifted', date_result.correctly_shifted])
                writer.writerow([patient.original_folder, 'csv_date_shift', date_result.filename, 'not_shifted', date_result.not_shifted])
                writer.writerow([patient.original_folder, 'csv_date_shift', date_result.filename, 'incorrectly_redacted', date_result.incorrectly_redacted])
                writer.writerow([patient.original_folder, 'csv_date_shift', date_result.filename, 'shift_failed', date_result.shift_failed])
                writer.writerow([patient.original_folder, 'csv_date_shift', date_result.filename, 'shift_rate', f"{date_result.shift_rate:.4f}"])

            # Date shift results from HEA files
            for date_result in patient.hea_date_results:
                writer.writerow([patient.original_folder, 'hea_date_shift', date_result.filename, 'total_dates', date_result.total_dates])
                writer.writerow([patient.original_folder, 'hea_date_shift', date_result.filename, 'correctly_shifted', date_result.correctly_shifted])
                writer.writerow([patient.original_folder, 'hea_date_shift', date_result.filename, 'not_shifted', date_result.not_shifted])
                writer.writerow([patient.original_folder, 'hea_date_shift', date_result.filename, 'incorrectly_redacted', date_result.incorrectly_redacted])
                writer.writerow([patient.original_folder, 'hea_date_shift', date_result.filename, 'shift_failed', date_result.shift_failed])
                writer.writerow([patient.original_folder, 'hea_date_shift', date_result.filename, 'shift_rate', f"{date_result.shift_rate:.4f}"])

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
    total_issues = (overall.csv_false_negatives + overall.image_not_covered + overall.pdf_not_covered +
                    overall.hea_false_negatives + overall.filename_phi_leaked +
                    overall.date_incorrectly_redacted + overall.date_not_shifted)
    if total_issues > 0:
        exit(1)


if __name__ == '__main__':
    main()
