"""
Evaluation script for PII redaction quality using IoU (Intersection over Union) metrics.

This script compares original images with redacted versions to measure how well
the redactions cover the annotated PII regions (Schwärzungen).
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Tuple, Dict
import os


def load_ground_truth_boxes(csv_path: str, image_name: str) -> List[Dict]:
    """
    Load ground truth bounding boxes from CSV for a specific image.

    Args:
        csv_path: Path to CSV file with annotations
        image_name: Name of the image file

    Returns:
        List of dictionaries containing bounding box information
    """
    df = pd.read_csv(csv_path)
    image_boxes = df[df['filename'] == image_name]

    boxes = []
    for _, row in image_boxes.iterrows():
        boxes.append({
            'field': row['field'],
            'text': row['text'],
            'x': int(row['x']),
            'y': int(row['y']),
            'width': int(row['width']),
            'height': int(row['height'])
        })

    return boxes


def detect_redaction_boxes(original_img: np.ndarray, redacted_img: np.ndarray,
                          threshold: int = 50) -> List[Dict]:
    """
    Detect redaction regions by comparing original and redacted images.

    Args:
        original_img: Original image (BGR)
        redacted_img: Redacted image (BGR/BGRA)
        threshold: Threshold for detecting changes (0-255)

    Returns:
        List of detected redaction bounding boxes
    """
    # Convert to grayscale
    if len(original_img.shape) == 3:
        gray_original = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_original = original_img

    if len(redacted_img.shape) == 3 and redacted_img.shape[2] == 4:
        # RGBA to grayscale
        gray_redacted = cv2.cvtColor(redacted_img[:, :, :3], cv2.COLOR_BGR2GRAY)
    elif len(redacted_img.shape) == 3:
        gray_redacted = cv2.cvtColor(redacted_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_redacted = redacted_img

    # Compute absolute difference
    diff = cv2.absdiff(gray_original, gray_redacted)

    # Threshold to create binary mask
    _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Morphological operations to clean up and connect nearby regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract bounding boxes from contours
    redaction_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter out very small regions (likely noise)
        if w > 10 and h > 10:
            redaction_boxes.append({
                'x': x,
                'y': y,
                'width': w,
                'height': h
            })

    return redaction_boxes


def calculate_iou(box1: Dict, box2: Dict) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First bounding box {x, y, width, height}
        box2: Second bounding box {x, y, width, height}

    Returns:
        IoU score (0.0 to 1.0)
    """
    # Calculate coordinates
    x1_min, y1_min = box1['x'], box1['y']
    x1_max, y1_max = x1_min + box1['width'], y1_min + box1['height']

    x2_min, y2_min = box2['x'], box2['y']
    x2_max, y2_max = x2_min + box2['width'], y2_min + box2['height']

    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0

    intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Calculate union
    area1 = box1['width'] * box1['height']
    area2 = box2['width'] * box2['height']
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def match_boxes(ground_truth_boxes: List[Dict], redaction_boxes: List[Dict]) -> List[Tuple[Dict, Dict, float]]:
    """
    Match ground truth boxes with detected redaction boxes using IoU.

    Args:
        ground_truth_boxes: List of ground truth PII bounding boxes
        redaction_boxes: List of detected redaction bounding boxes

    Returns:
        List of tuples (gt_box, redaction_box, iou_score)
    """
    matches = []

    for gt_box in ground_truth_boxes:
        best_iou = 0.0
        best_match = None

        for red_box in redaction_boxes:
            iou = calculate_iou(gt_box, red_box)
            if iou > best_iou:
                best_iou = iou
                best_match = red_box

        matches.append((gt_box, best_match, best_iou))

    return matches


def calculate_coverage_ratio(gt_box: Dict, red_box: Dict) -> float:
    """
    Calculate what percentage of the ground truth box is covered by the redaction.
    This is more relevant than IoU for evaluating PII protection.

    Args:
        gt_box: Ground truth bounding box
        red_box: Redaction bounding box

    Returns:
        Coverage ratio (0.0 to 1.0), where 1.0 means 100% covered
    """
    if red_box is None:
        return 0.0

    # Calculate coordinates
    gt_x_min, gt_y_min = gt_box['x'], gt_box['y']
    gt_x_max, gt_y_max = gt_x_min + gt_box['width'], gt_y_min + gt_box['height']

    red_x_min, red_y_min = red_box['x'], red_box['y']
    red_x_max, red_y_max = red_x_min + red_box['width'], red_y_min + red_box['height']

    # Calculate intersection
    inter_x_min = max(gt_x_min, red_x_min)
    inter_y_min = max(gt_y_min, red_y_min)
    inter_x_max = min(gt_x_max, red_x_max)
    inter_y_max = min(gt_y_max, red_y_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0

    intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    gt_area = gt_box['width'] * gt_box['height']

    return intersection / gt_area if gt_area > 0 else 0.0


def calculate_precision_ratio(gt_box: Dict, red_box: Dict) -> float:
    """
    Calculate what percentage of the redaction actually covers PII.
    Lower values indicate more over-redaction.

    Args:
        gt_box: Ground truth bounding box
        red_box: Redaction bounding box

    Returns:
        Precision ratio (0.0 to 1.0)
    """
    if red_box is None:
        return 0.0

    # Calculate coordinates
    gt_x_min, gt_y_min = gt_box['x'], gt_box['y']
    gt_x_max, gt_y_max = gt_x_min + gt_box['width'], gt_y_min + gt_box['height']

    red_x_min, red_y_min = red_box['x'], red_box['y']
    red_x_max, red_y_max = red_x_min + red_box['width'], red_y_min + red_box['height']

    # Calculate intersection
    inter_x_min = max(gt_x_min, red_x_min)
    inter_y_min = max(gt_y_min, red_y_min)
    inter_x_max = min(gt_x_max, red_x_max)
    inter_y_max = min(gt_y_max, red_y_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0

    intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    red_area = red_box['width'] * red_box['height']

    return intersection / red_area if red_area > 0 else 0.0


def calculate_over_redaction_ratio(gt_box: Dict, red_box: Dict) -> float:
    """
    Calculate how much larger the redaction is compared to the ground truth.

    Args:
        gt_box: Ground truth bounding box
        red_box: Redaction bounding box

    Returns:
        Over-redaction ratio (e.g., 2.5 means redaction is 2.5x larger than needed)
    """
    if red_box is None:
        return 0.0

    gt_area = gt_box['width'] * gt_box['height']
    red_area = red_box['width'] * red_box['height']

    return red_area / gt_area if gt_area > 0 else 0.0


def calculate_metrics(matches: List[Tuple[Dict, Dict, float]], iou_threshold: float = 0.5) -> Dict:
    """
    Calculate evaluation metrics for redaction quality.
    Uses both IoU (traditional CV) and coverage-based metrics (PII-specific).

    Args:
        matches: List of (gt_box, redaction_box, iou_score) tuples
        iou_threshold: Minimum IoU to consider a match as "good"

    Returns:
        Dictionary with evaluation metrics
    """
    ious = [iou for _, _, iou in matches]

    # Calculate coverage ratios (more relevant for PII protection!)
    coverage_ratios = [calculate_coverage_ratio(gt, red) for gt, red, _ in matches]
    precision_ratios = [calculate_precision_ratio(gt, red) for gt, red, _ in matches]
    over_redaction_ratios = [calculate_over_redaction_ratio(gt, red) for gt, red, _ in matches]

    metrics = {
        # IoU metrics (traditional computer vision - penalizes over-redaction)
        'mean_iou': np.mean(ious) if ious else 0.0,
        'median_iou': np.median(ious) if ious else 0.0,
        'min_iou': np.min(ious) if ious else 0.0,
        'max_iou': np.max(ious) if ious else 0.0,
        'well_covered_iou': sum(1 for iou in ious if iou >= iou_threshold),
        'poorly_covered_iou': sum(1 for iou in ious if iou < iou_threshold),
        'coverage_rate_iou': sum(1 for iou in ious if iou >= iou_threshold) / len(ious) if ious else 0.0,

        # Coverage metrics (PII-specific - measures how well PII is protected)
        'mean_coverage': np.mean(coverage_ratios) if coverage_ratios else 0.0,
        'median_coverage': np.median(coverage_ratios) if coverage_ratios else 0.0,
        'min_coverage': np.min(coverage_ratios) if coverage_ratios else 0.0,
        'max_coverage': np.max(coverage_ratios) if coverage_ratios else 0.0,

        # Precision metrics (how much redaction is actually PII)
        'mean_precision': np.mean(precision_ratios) if precision_ratios else 0.0,
        'median_precision': np.median(precision_ratios) if precision_ratios else 0.0,

        # Over-redaction metrics (how much bigger redactions are than needed)
        'mean_over_redaction': np.mean(over_redaction_ratios) if over_redaction_ratios else 0.0,
        'median_over_redaction': np.median(over_redaction_ratios) if over_redaction_ratios else 0.0,
        'max_over_redaction': np.max(over_redaction_ratios) if over_redaction_ratios else 0.0,

        # Count metrics
        'total_pii_regions': len(matches),
        'fully_covered': sum(1 for cov in coverage_ratios if cov >= 0.95),  # 95%+ covered
        'partially_covered': sum(1 for cov in coverage_ratios if 0.5 <= cov < 0.95),
        'poorly_covered': sum(1 for cov in coverage_ratios if 0 < cov < 0.5),
        'not_covered': sum(1 for cov in coverage_ratios if cov == 0.0),

        # Overall rates
        'full_coverage_rate': sum(1 for cov in coverage_ratios if cov >= 0.95) / len(coverage_ratios) if coverage_ratios else 0.0,
        'any_coverage_rate': sum(1 for cov in coverage_ratios if cov > 0) / len(coverage_ratios) if coverage_ratios else 0.0,

        # Store raw data for detailed analysis
        'coverage_ratios': coverage_ratios,
        'precision_ratios': precision_ratios,
        'over_redaction_ratios': over_redaction_ratios
    }

    return metrics


def visualize_redaction_quality(original_img: np.ndarray, redacted_img: np.ndarray,
                                matches: List[Tuple[Dict, Dict, float]],
                                output_path: str, image_name: str):
    """
    Create visualization showing ground truth boxes vs detected redactions with IoU scores.

    Args:
        original_img: Original image
        redacted_img: Redacted image
        matches: List of (gt_box, redaction_box, iou_score) tuples
        output_path: Path to save visualization
        image_name: Name of the image being evaluated
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    # Original with ground truth boxes
    ax1 = axes[0]
    ax1.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original with Ground Truth PII', fontsize=14, fontweight='bold')
    for gt_box, _, iou in matches:
        rect = Rectangle((gt_box['x'], gt_box['y']), gt_box['width'], gt_box['height'],
                        linewidth=2, edgecolor='red', facecolor='none')
        ax1.add_patch(rect)
        ax1.text(gt_box['x'], gt_box['y'] - 5, gt_box['field'],
                color='red', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax1.axis('off')

    # Redacted image
    ax2 = axes[1]
    if redacted_img.shape[2] == 4:
        ax2.imshow(cv2.cvtColor(redacted_img[:, :, :3], cv2.COLOR_BGR2RGB))
    else:
        ax2.imshow(cv2.cvtColor(redacted_img, cv2.COLOR_BGR2RGB))
    ax2.set_title('Redacted Image (Schwärzungen)', fontsize=14, fontweight='bold')
    ax2.axis('off')

    # Overlay with Coverage scores (more relevant than IoU for PII!)
    ax3 = axes[2]
    if redacted_img.shape[2] == 4:
        ax3.imshow(cv2.cvtColor(redacted_img[:, :, :3], cv2.COLOR_BGR2RGB))
    else:
        ax3.imshow(cv2.cvtColor(redacted_img, cv2.COLOR_BGR2RGB))
    ax3.set_title('Redaction Quality Assessment (Coverage Scores)', fontsize=14, fontweight='bold')

    for gt_box, red_box, iou in matches:
        # Calculate coverage for better assessment
        coverage = calculate_coverage_ratio(gt_box, red_box)

        # Color code based on coverage quality
        if coverage >= 0.95:
            color = 'green'
        elif coverage >= 0.5:
            color = 'yellow'
        elif coverage > 0:
            color = 'orange'
        else:
            color = 'red'

        # Draw ground truth box
        rect_gt = Rectangle((gt_box['x'], gt_box['y']), gt_box['width'], gt_box['height'],
                           linewidth=2, edgecolor=color, facecolor='none', linestyle='--')
        ax3.add_patch(rect_gt)

        # Draw detected redaction box if exists
        if red_box:
            rect_red = Rectangle((red_box['x'], red_box['y']), red_box['width'], red_box['height'],
                               linewidth=2, edgecolor=color, facecolor='none')
            ax3.add_patch(rect_red)

        # Add coverage score label (with IoU for comparison)
        label = f"{gt_box['field']}\nCoverage: {coverage*100:.0f}%\nIoU: {iou:.2f}"
        ax3.text(gt_box['x'], gt_box['y'] - 5, label,
                color=color, fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax3.axis('off')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='green', label='Protected (Coverage ≥ 95%)'),
        Patch(facecolor='yellow', edgecolor='yellow', label='Partial (Coverage ≥ 50%)'),
        Patch(facecolor='orange', edgecolor='orange', label='Weak (Coverage > 0%)'),
        Patch(facecolor='red', edgecolor='red', label='Missing (Coverage = 0%)')
    ]
    ax3.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.suptitle(f'Redaction Quality Evaluation: {image_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.show()


def print_detailed_report(matches: List[Tuple[Dict, Dict, float]], metrics: Dict):
    """
    Print detailed evaluation report to console.

    Args:
        matches: List of (gt_box, redaction_box, iou_score) tuples
        metrics: Dictionary with calculated metrics
    """
    print("\n" + "="*80)
    print("REDACTION QUALITY EVALUATION REPORT")
    print("="*80)

    print("\n📊 OVERALL METRICS:")
    print(f"  • Total PII regions to redact: {metrics['total_pii_regions']}")
    print(f"  • Fully covered (≥95%):        {metrics['fully_covered']} ({metrics['full_coverage_rate']*100:.1f}%)")
    print(f"  • Partially covered (50-95%):  {metrics['partially_covered']}")
    print(f"  • Poorly covered (<50%):       {metrics['poorly_covered']}")
    print(f"  • Not covered (0%):            {metrics['not_covered']}")

    print(f"\n📈 COVERAGE STATISTICS (PII Protection):")
    print(f"  • Mean Coverage:   {metrics['mean_coverage']:.3f} ({metrics['mean_coverage']*100:.1f}%)")
    print(f"  • Median Coverage: {metrics['median_coverage']:.3f} ({metrics['median_coverage']*100:.1f}%)")
    print(f"  • Min Coverage:    {metrics['min_coverage']:.3f} ({metrics['min_coverage']*100:.1f}%)")
    print(f"  • Max Coverage:    {metrics['max_coverage']:.3f} ({metrics['max_coverage']*100:.1f}%)")

    print(f"\n📐 OVER-REDACTION STATISTICS:")
    print(f"  • Mean Over-redaction:   {metrics['mean_over_redaction']:.3f} ({metrics['mean_over_redaction']:.2f}x)")
    print(f"  • Median Over-redaction: {metrics['median_over_redaction']:.3f} ({metrics['median_over_redaction']:.2f}x)")
    print(f"  • Max Over-redaction:    {metrics['max_over_redaction']:.3f} ({metrics['max_over_redaction']:.2f}x)")
    print(f"  • Mean Precision:        {metrics['mean_precision']:.3f} ({metrics['mean_precision']*100:.1f}%)")

    print(f"\n📊 IoU STATISTICS (Traditional CV Metric):")
    print(f"  • Mean IoU:   {metrics['mean_iou']:.3f}")
    print(f"  • Median IoU: {metrics['median_iou']:.3f}")
    print(f"  • Min IoU:    {metrics['min_iou']:.3f}")
    print(f"  • Max IoU:    {metrics['max_iou']:.3f}")

    # Calculate coverage and over-redaction for each match
    coverage_data = []
    for gt_box, red_box, iou in matches:
        coverage = calculate_coverage_ratio(gt_box, red_box)
        over_redaction = calculate_over_redaction_ratio(gt_box, red_box)
        coverage_data.append((gt_box, red_box, iou, coverage, over_redaction))

    print("\n📋 DETAILED PII REGION ANALYSIS:")
    print(f"{'Field':<15} {'Text':<20} {'Coverage':<10} {'IoU':<8} {'Status':<12}")
    print("-" * 80)

    for gt_box, red_box, iou, coverage, _ in sorted(coverage_data, key=lambda x: x[3]):
        field = gt_box['field'][:14]
        text = gt_box['text'][:19]

        if coverage >= 0.95:
            status = "✓ Protected"
        elif coverage >= 0.5:
            status = "⚠ Partial"
        elif coverage > 0:
            status = "⚠ Weak"
        else:
            status = "✗ Missing"

        print(f"{field:<15} {text:<20} {coverage:.3f}    {iou:.3f}    {status:<12}")

    print("\n" + "="*80)

    # Performance assessment based on coverage (not IoU!)
    if metrics['full_coverage_rate'] >= 0.95:
        assessment = "EXCELLENT - All PII is well protected!"
    elif metrics['full_coverage_rate'] >= 0.8:
        assessment = "GOOD - Most PII regions are fully covered."
    elif metrics['mean_coverage'] >= 0.8:
        assessment = "MODERATE - PII is mostly covered but some regions are partial."
    elif metrics['mean_coverage'] >= 0.5:
        assessment = "FAIR - Significant PII exposure risk detected."
    else:
        assessment = "POOR - Many PII regions are not adequately redacted!"

    print(f"🎯 ASSESSMENT (Based on Coverage): {assessment}")

    # Additional context about over-redaction
    if metrics['mean_over_redaction'] > 5.0:
        print(f"⚠️  NOTE: Redactions are {metrics['mean_over_redaction']:.1f}x larger than necessary.")
        print(f"   This is safe for privacy but may obscure too much information.")
    elif metrics['mean_over_redaction'] > 2.0:
        print(f"ℹ️  NOTE: Redactions are {metrics['mean_over_redaction']:.1f}x larger than necessary (conservative but safe).")

    print("="*80 + "\n")

    print("💡 METRICS EXPLANATION:")
    print("  • Coverage: Fraction of PII area that is redacted (higher = better protection)")
    print("  • Over-redaction: How much larger redactions are vs. needed (higher = more conservative)")
    print("  • IoU: Traditional metric that penalizes both under- and over-redaction")
    print("  • For PII protection, Coverage is the most critical metric!")
    print("="*80 + "\n")


def evaluate_redaction(original_path: str, redacted_path: str, csv_path: str,
                      image_name: str, output_viz_path: str = None):
    """
    Main evaluation function to assess redaction quality.

    Args:
        original_path: Path to original image
        redacted_path: Path to redacted image
        csv_path: Path to CSV with ground truth annotations
        image_name: Name of the image in the CSV
        output_viz_path: Path to save visualization (optional)

    Returns:
        Dictionary with evaluation metrics
    """
    # Load images
    original_img = cv2.imread(original_path)
    redacted_img = cv2.imread(redacted_path, cv2.IMREAD_UNCHANGED)

    if original_img is None:
        raise FileNotFoundError(f"Could not load original image: {original_path}")
    if redacted_img is None:
        raise FileNotFoundError(f"Could not load redacted image: {redacted_path}")

    print(f"\n🔍 Evaluating redaction quality for: {image_name}")
    print(f"   Original image shape: {original_img.shape}")
    print(f"   Redacted image shape: {redacted_img.shape}")

    # Load ground truth boxes
    ground_truth_boxes = load_ground_truth_boxes(csv_path, image_name)
    print(f"   Found {len(ground_truth_boxes)} PII regions in ground truth")

    # Detect redaction boxes
    redaction_boxes = detect_redaction_boxes(original_img, redacted_img)
    print(f"   Detected {len(redaction_boxes)} redacted regions")

    # Match boxes and calculate IoU
    matches = match_boxes(ground_truth_boxes, redaction_boxes)

    # Calculate metrics
    metrics = calculate_metrics(matches)

    # Print detailed report
    print_detailed_report(matches, metrics)

    # Create visualization
    if output_viz_path is None:
        output_viz_path = f"redaction_evaluation_{os.path.splitext(image_name)[0]}.png"

    visualize_redaction_quality(original_img, redacted_img, matches, output_viz_path, image_name)

    return metrics


if __name__ == "__main__":
    # Example usage for Aaron Simmons image
    original_path = "data/primary/cxr_Aaron_Simmons.png"
    redacted_path = "data/output/anonymized_cxr_Aaron_Simmons_original.png"
    csv_path = "data/primary/labels/phi_annotations_cxr_header_jpg.csv"
    image_name = "cxr_Aaron_Simmons.png"

    metrics = evaluate_redaction(
        original_path=original_path,
        redacted_path=redacted_path,
        csv_path=csv_path,
        image_name=image_name,
        output_viz_path=f"data/evaluation_results/{os.path.splitext(image_name)[0]}_redaction_evaluation.png"
    )
