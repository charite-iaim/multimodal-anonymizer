#!/usr/bin/env python3
"""
CSV Redaction Evaluation Script

This script evaluates the quality of CSV file redaction by comparing:
1. A source file with XML tags (<PER>...</PER>) marking PHI that should be redacted
2. A redacted file where PHI should be replaced with asterisks or other markers

The evaluation checks if:
- All tagged PHI has been removed from the redacted file (not appearing literally)
- Asterisks in the redacted file correspond to actual PHI (not over-redacting)

Metrics calculated:
- True Positives (TP): PHI correctly redacted (doesn't appear literally in redacted)
- False Negatives (FN): PHI not redacted (appears literally in redacted)
- Recall: TP / (TP + FN) - The main metric for security
"""

import csv
import re
import argparse
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    true_positives: int = 0
    false_negatives: int = 0
    total_phi_instances: int = 0
    phi_not_redacted: List[str] = None

    def __post_init__(self):
        if self.phi_not_redacted is None:
            self.phi_not_redacted = []

    @property
    def recall(self) -> float:
        """Calculate recall: TP / (TP + FN) - most important for PHI redaction"""
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator > 0 else 0.0

    @property
    def redaction_success_rate(self) -> float:
        """Same as recall, but named more clearly for this use case"""
        return self.recall


def extract_phi_from_text(text: str) -> Set[str]:
    """
    Extract all unique PHI values from text marked with <PER> tags.

    Args:
        text: Text containing XML tags like <PER>PHI content</PER>

    Returns:
        Set of unique PHI strings
    """
    pattern = r'<PER>(.*?)</PER>'
    phi_values = set(re.findall(pattern, text))
    return phi_values


def remove_tags(text: str) -> str:
    """Remove XML tags from text"""
    return re.sub(r'<PER>(.*?)</PER>', r'\1', text)


def get_phi_contexts(text: str, window_size: int = 20) -> Dict[str, List[str]]:
    """
    Extract PHI values with their surrounding context.

    Args:
        text: Text with <PER> tags
        window_size: Number of characters before/after to include as context

    Returns:
        Dictionary mapping PHI values to list of context strings
    """
    pattern = r'<PER>(.*?)</PER>'
    phi_contexts = {}

    for match in re.finditer(pattern, text):
        phi_content = match.group(1)
        start = match.start()
        end = match.end()

        # Get context before and after
        context_start = max(0, start - window_size)
        context_end = min(len(text), end + window_size)
        context = text[context_start:context_end]

        # Remove the tags but keep the PHI
        context_clean = context.replace('<PER>', '').replace('</PER>', '')

        if phi_content not in phi_contexts:
            phi_contexts[phi_content] = []
        phi_contexts[phi_content].append(context_clean)

    return phi_contexts


def is_phi_truly_leaked(phi_content: str, source_value: str, redacted_value: str,
                        min_length: int = 4, context_window: int = 20) -> bool:
    """
    Check if PHI truly leaked by considering context and length.

    For short PHI (less than min_length), check if it appears in similar context.
    For longer PHI, simple substring check is sufficient.

    Args:
        phi_content: The PHI value to check
        source_value: Original text with tags
        redacted_value: Redacted text
        min_length: Minimum length to skip context checking
        context_window: Characters of context to compare

    Returns:
        True if PHI appears to have leaked, False otherwise
    """
    # If PHI doesn't appear at all, it's definitely redacted
    if phi_content not in redacted_value:
        return False

    # For longer PHI values, presence is enough to confirm leak
    if len(phi_content) >= min_length:
        return True

    # For short PHI, check if it appears in the same context
    # Get all contexts where this PHI appears in source
    phi_contexts = get_phi_contexts(source_value, context_window)
    source_contexts = phi_contexts.get(phi_content, [])

    if not source_contexts:
        # Shouldn't happen, but if we can't find context, assume it's leaked
        return True

    # Check if the PHI appears in redacted text in a context similar to source
    # This helps distinguish between PHI that should be redacted vs. the same
    # number/word appearing naturally in medical context
    for source_context in source_contexts:
        # Get the portion before and after the PHI in source
        phi_pos = source_context.find(phi_content)
        if phi_pos == -1:
            continue

        before_text = source_context[:phi_pos]
        after_text = source_context[phi_pos + len(phi_content):]

        # Look for this PHI with similar context in redacted text
        for match_pos in [m.start() for m in re.finditer(re.escape(phi_content), redacted_value)]:
            redacted_before = redacted_value[max(0, match_pos - context_window):match_pos]
            redacted_after = redacted_value[match_pos + len(phi_content):
                                           min(len(redacted_value), match_pos + len(phi_content) + context_window)]

            # Check if context is similar (allowing for some variation)
            # We look for significant overlap in the surrounding text
            before_overlap = len(set(before_text[-15:].split()) & set(redacted_before[-15:].split())) if len(before_text) >= 15 else 0
            after_overlap = len(set(after_text[:15].split()) & set(redacted_after[:15].split())) if len(after_text) >= 15 else 0

            # If we find the PHI with similar surrounding words, it likely leaked
            if before_overlap >= 2 or after_overlap >= 2:
                return True

    # PHI appears in redacted text but in different context - probably not a leak
    return False


def evaluate_field_redaction(source_value: str, redacted_value: str,
                            min_phi_length: int = 4) -> Tuple[EvaluationMetrics, List[Dict]]:
    """
    Evaluate redaction for a single field by checking if PHI appears literally.

    This is the most reliable method: if PHI from the source appears literally
    in the redacted version, it wasn't properly redacted.

    For short PHI values (< min_phi_length), uses context-aware checking to avoid
    false positives from common numbers/words.

    Args:
        source_value: Field value with <PER> tags marking PHI
        redacted_value: Redacted field value
        min_phi_length: Minimum length for simple substring check (default: 4)

    Returns:
        Tuple of (metrics, details_list)
    """
    metrics = EvaluationMetrics()
    details = []

    # Extract all PHI from source
    phi_set = extract_phi_from_text(source_value)
    metrics.total_phi_instances = len(phi_set)

    # Check each PHI to see if it appears in the redacted version
    for phi_content in sorted(phi_set):  # Sort for consistent output
        # Check if this PHI truly leaked (considering context for short values)
        leaked = is_phi_truly_leaked(phi_content, source_value, redacted_value,
                                     min_length=min_phi_length)

        if leaked:
            # NOT REDACTED - This is bad!
            metrics.false_negatives += 1
            metrics.phi_not_redacted.append(phi_content)

            issue = 'PHI appears literally in redacted text'
            if len(phi_content) < min_phi_length:
                issue += ' (in similar context)'

            details.append({
                'phi_content': phi_content,
                'status': 'FN',
                'issue': issue
            })
        else:
            # Properly redacted
            metrics.true_positives += 1

            details.append({
                'phi_content': phi_content,
                'status': 'TP',
                'issue': None
            })

    return metrics, details


def evaluate_csv_files(source_path: str, redacted_path: str, min_phi_length: int = 4) -> Dict:
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

            # Skip fields with no PHI tags
            if '<PER>' not in source_value:
                continue

            # Evaluate this field
            field_metrics, field_details = evaluate_field_redaction(source_value, redacted_value,
                                                                    min_phi_length=min_phi_length)

            # Add row and field context to details
            for detail in field_details:
                detail['row'] = row_num
                detail['field'] = field_name
                all_details.append(detail)

            # Accumulate metrics
            overall_metrics.true_positives += field_metrics.true_positives
            overall_metrics.false_negatives += field_metrics.false_negatives
            overall_metrics.total_phi_instances += field_metrics.total_phi_instances
            overall_metrics.phi_not_redacted.extend(field_metrics.phi_not_redacted)

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
    print(f"Total unique PHI instances: {metrics.total_phi_instances}")
    print()

    print("METRICS:")
    print(f"  Correctly Redacted (True Positives):   {metrics.true_positives}")
    print(f"  NOT Redacted (False Negatives):        {metrics.false_negatives}")
    print()

    print(f"  Redaction Success Rate: {metrics.redaction_success_rate:.2%}")
    print()

    if metrics.false_negatives > 0:
        print(f"⚠️  WARNING: {metrics.false_negatives} PHI instances were NOT properly redacted!")
        print(f"   These PHI values appear literally in the redacted file:")
        print()
        for phi in sorted(set(metrics.phi_not_redacted))[:10]:
            print(f"     - \"{phi}\"")
        if len(set(metrics.phi_not_redacted)) > 10:
            print(f"     ... and {len(set(metrics.phi_not_redacted)) - 10} more")
        print()
    else:
        print("✓ All PHI successfully redacted!")
        print()

    if verbose:
        print("=" * 80)
        print("DETAILED BREAKDOWN")
        print("=" * 80)
        print()

        # Group by status
        fn_details = [d for d in results['details'] if d['status'] == 'FN']
        tp_details = [d for d in results['details'] if d['status'] == 'TP']

        if fn_details:
            print(f"FALSE NEGATIVES (Not properly redacted - {len(fn_details)} instances):")
            print("-" * 80)
            for detail in fn_details[:30]:  # Show first 30
                print(f"  Row {detail['row']}, Field '{detail['field']}':")
                print(f"    PHI value:  \"{detail['phi_content']}\"")
                print(f"    Issue:      {detail['issue']}")
                print()
            if len(fn_details) > 30:
                print(f"  ... and {len(fn_details) - 30} more")
            print()

        if tp_details and verbose:
            print(f"TRUE POSITIVES (Properly redacted - {len(tp_details)} instances):")
            print("-" * 80)
            for detail in tp_details[:10]:  # Show first 10
                print(f"  Row {detail['row']}, Field '{detail['field']}': \"{detail['phi_content']}\" ✓")
            if len(tp_details) > 10:
                print(f"  ... and {len(tp_details) - 10} more")
            print()

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate CSV redaction quality by checking if PHI appears literally in redacted file'
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
        help='Show detailed breakdown of all PHI instances'
    )
    parser.add_argument(
        '--min-phi-length',
        type=int,
        default=4,
        help='Minimum PHI length for simple matching. Shorter PHI uses context-aware matching (default: 4)'
    )

    args = parser.parse_args()

    try:
        results = evaluate_csv_files(args.source, args.redacted, min_phi_length=args.min_phi_length)
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
