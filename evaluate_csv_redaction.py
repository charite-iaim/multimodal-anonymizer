#!/usr/bin/env python3
"""
CSV Redaction Evaluation Script

This script evaluates the quality of CSV file redaction by comparing:
1. A source file with XML tags (<PER>...</PER>) marking PHI that should be redacted
2. A redacted file where PHI should be replaced with asterisks or other markers

Metrics calculated:
- True Positives (TP): PHI correctly redacted
- False Negatives (FN): PHI not redacted (missed)
- False Positives (FP): Non-PHI incorrectly redacted
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
"""

import csv
import re
import argparse
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    true_positives: int = 0
    false_negatives: int = 0
    false_positives: int = 0
    total_phi_instances: int = 0

    @property
    def precision(self) -> float:
        """Calculate precision: TP / (TP + FP)"""
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator > 0 else 0.0

    @property
    def recall(self) -> float:
        """Calculate recall: TP / (TP + FN)"""
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """Calculate F1 score: 2 * (Precision * Recall) / (Precision + Recall)"""
        p = self.precision
        r = self.recall
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0


def extract_phi_segments(text: str) -> List[Tuple[str, int, int]]:
    """
    Extract PHI segments marked with XML tags from text.

    Args:
        text: Text containing XML tags like <PER>PHI content</PER>

    Returns:
        List of tuples (phi_content, start_pos, end_pos) where positions
        are relative to the text with tags removed
    """
    phi_segments = []
    pattern = r'<PER>(.*?)</PER>'

    # Track position offset as we remove tags
    offset = 0
    text_without_tags = text

    for match in re.finditer(pattern, text):
        phi_content = match.group(1)

        # Calculate position in text without tags
        start_in_original = match.start()
        start_without_tags = start_in_original - offset
        end_without_tags = start_without_tags + len(phi_content)

        phi_segments.append((phi_content, start_without_tags, end_without_tags))

        # Update offset: we're removing <PER> and </PER> tags (10 characters total)
        offset += len('<PER>') + len('</PER>')

    return phi_segments


def remove_tags(text: str) -> str:
    """Remove XML tags from text"""
    return re.sub(r'<PER>(.*?)</PER>', r'\1', text)


def is_redacted(text: str, start: int, end: int) -> bool:
    """
    Check if a segment of text is redacted (contains only asterisks or dashes).

    Args:
        text: The redacted text
        start: Start position of segment to check
        end: End position of segment to check

    Returns:
        True if segment is properly redacted, False otherwise
    """
    if start < 0 or end > len(text):
        return False

    segment = text[start:end]
    # Check if segment contains only redaction characters (*, -, or whitespace)
    return bool(segment) and all(c in '*- \t' for c in segment)


def evaluate_redaction(source_text: str, redacted_text: str) -> Tuple[EvaluationMetrics, List[Dict]]:
    """
    Evaluate redaction quality by comparing source with tagged PHI to redacted version.

    Args:
        source_text: Text with <PER> tags marking PHI
        redacted_text: Text with PHI redacted

    Returns:
        Tuple of (metrics, details_list) where details_list contains info about each PHI instance
    """
    metrics = EvaluationMetrics()
    details = []

    # Extract PHI segments and their positions
    phi_segments = extract_phi_segments(source_text)
    source_text_clean = remove_tags(source_text)

    metrics.total_phi_instances = len(phi_segments)

    for phi_content, start, end in phi_segments:
        # Check if this PHI was properly redacted
        redacted = is_redacted(redacted_text, start, end)

        detail = {
            'phi_content': phi_content,
            'start': start,
            'end': end,
            'expected': phi_content,
            'actual': redacted_text[start:end] if start < len(redacted_text) and end <= len(redacted_text) else "OUT_OF_BOUNDS",
            'status': 'TP' if redacted else 'FN'
        }

        if redacted:
            metrics.true_positives += 1
        else:
            metrics.false_negatives += 1

        details.append(detail)

    # Check for false positives: redacted content that wasn't marked as PHI
    # This is a simplified check - looks for sequences of asterisks that don't align with PHI
    redaction_pattern = r'\*{3,}'
    for match in re.finditer(redaction_pattern, redacted_text):
        start, end = match.span()

        # Check if this redaction aligns with any known PHI
        is_known_phi = any(
            abs(start - phi_start) < 5 and abs(end - phi_end) < 5
            for _, phi_start, phi_end in phi_segments
        )

        if not is_known_phi:
            metrics.false_positives += 1
            # Try to find what was in the original
            original_segment = source_text_clean[start:end] if start < len(source_text_clean) and end <= len(source_text_clean) else "UNKNOWN"
            details.append({
                'phi_content': None,
                'start': start,
                'end': end,
                'expected': original_segment,
                'actual': match.group(),
                'status': 'FP'
            })

    return metrics, details


def evaluate_csv_files(source_path: str, redacted_path: str) -> Dict:
    """
    Evaluate redaction quality for CSV files.

    Args:
        source_path: Path to CSV with PHI marked with <PER> tags
        redacted_path: Path to redacted CSV

    Returns:
        Dictionary containing evaluation results
    """
    print(f"Loading source file: {source_path}")
    print(f"Loading redacted file: {redacted_path}")
    print()

    # Read both CSV files
    with open(source_path, 'r', encoding='utf-8') as f:
        source_reader = csv.DictReader(f)
        source_rows = list(source_reader)

    with open(redacted_path, 'r', encoding='utf-8') as f:
        redacted_reader = csv.DictReader(f)
        redacted_rows = list(redacted_reader)

    if len(source_rows) != len(redacted_rows):
        print(f"WARNING: Row count mismatch - Source: {len(source_rows)}, Redacted: {len(redacted_rows)}")

    # Aggregate metrics across all rows
    overall_metrics = EvaluationMetrics()
    all_details = []

    # Evaluate each row
    for idx, (source_row, redacted_row) in enumerate(zip(source_rows, redacted_rows)):
        row_num = idx + 2  # +2 for header and 1-based indexing

        # Evaluate each field
        for field_name in source_row.keys():
            if field_name not in redacted_row:
                continue

            source_value = source_row[field_name]
            redacted_value = redacted_row[field_name]

            # Skip empty fields
            if not source_value or source_value.strip() == '':
                continue

            # Evaluate this field
            field_metrics, field_details = evaluate_redaction(source_value, redacted_value)

            # Add row and field context to details
            for detail in field_details:
                detail['row'] = row_num
                detail['field'] = field_name
                all_details.append(detail)

            # Accumulate metrics
            overall_metrics.true_positives += field_metrics.true_positives
            overall_metrics.false_negatives += field_metrics.false_negatives
            overall_metrics.false_positives += field_metrics.false_positives
            overall_metrics.total_phi_instances += field_metrics.total_phi_instances

    return {
        'metrics': overall_metrics,
        'details': all_details,
        'source_rows': len(source_rows),
        'redacted_rows': len(redacted_rows)
    }


def print_results(results: Dict, verbose: bool = False):
    """Print evaluation results in a readable format"""
    metrics = results['metrics']

    print("=" * 80)
    print("CSV REDACTION EVALUATION RESULTS")
    print("=" * 80)
    print()

    print(f"Total rows processed: {results['source_rows']}")
    print(f"Total PHI instances: {metrics.total_phi_instances}")
    print()

    print("METRICS:")
    print(f"  True Positives (correctly redacted):  {metrics.true_positives}")
    print(f"  False Negatives (missed PHI):         {metrics.false_negatives}")
    print(f"  False Positives (incorrect redaction): {metrics.false_positives}")
    print()

    print(f"  Precision: {metrics.precision:.2%}")
    print(f"  Recall:    {metrics.recall:.2%}")
    print(f"  F1 Score:  {metrics.f1_score:.2%}")
    print()

    if metrics.false_negatives > 0:
        print(f"⚠️  WARNING: {metrics.false_negatives} PHI instances were NOT properly redacted!")
        print()

    if verbose:
        print("=" * 80)
        print("DETAILED BREAKDOWN")
        print("=" * 80)
        print()

        # Group by status
        fn_details = [d for d in results['details'] if d['status'] == 'FN']
        fp_details = [d for d in results['details'] if d['status'] == 'FP']

        if fn_details:
            print(f"FALSE NEGATIVES (Missed PHI - {len(fn_details)} instances):")
            print("-" * 80)
            for detail in fn_details[:20]:  # Show first 20
                print(f"  Row {detail['row']}, Field '{detail['field']}':")
                print(f"    Expected to redact: '{detail['expected']}'")
                print(f"    Actually found:     '{detail['actual']}'")
                print()
            if len(fn_details) > 20:
                print(f"  ... and {len(fn_details) - 20} more")
            print()

        if fp_details:
            print(f"FALSE POSITIVES (Incorrect redactions - {len(fp_details)} instances):")
            print("-" * 80)
            for detail in fp_details[:20]:  # Show first 20
                print(f"  Row {detail['row']}, Field '{detail['field']}':")
                print(f"    Original text:  '{detail['expected']}'")
                print(f"    Redacted as:    '{detail['actual']}'")
                print()
            if len(fp_details) > 20:
                print(f"  ... and {len(fp_details) - 20} more")
            print()

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate CSV redaction quality by comparing tagged source with redacted output'
    )
    parser.add_argument(
        '--source',
        default='/Users/anja/Documents/Coding/mimiciv-anonymization-pipeline/data/primary/patient_10005749_20010003/csv/note_discharge_20010003_deanonymized_2.csv',
        help='Path to source CSV with <PER> tags marking PHI'
    )
    parser.add_argument(
        '--redacted',
        default='/Users/anja/Documents/Coding/mimiciv-anonymization-pipeline/data/output/anonymized_note_discharge_20010003_with_PHI.csv',
        help='Path to redacted CSV file'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed breakdown of errors'
    )

    args = parser.parse_args()

    try:
        results = evaluate_csv_files(args.source, args.redacted)
        print_results(results, verbose=args.verbose)

        # Exit with non-zero code if there are false negatives (security issue)
        if results['metrics'].false_negatives > 0:
            exit(1)

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == '__main__':
    main()
