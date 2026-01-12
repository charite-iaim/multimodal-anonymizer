#!/usr/bin/env python3
"""
Analyze anonymization failures in detail.
Shows exactly which PHI values were not properly redacted.

Usage:
    python analyze_failures.py <patient_folder_path> [labels_base_dir] [results_base_dir]

Example:
    python analyze_failures.py patient_10005749_20010003
    python analyze_failures.py /path/to/patient_10005749_20010003 ./data/primary/patient_records_labels ./data/results
"""

import os
import re
import csv
import sys
from collections import defaultdict
from typing import Set, Dict, List, Tuple, Optional


def extract_phi_from_text(text: str) -> Set[str]:
    """Extract all PHI values from text marked with <PER> tags"""
    pattern = r'<PER>(.*?)</PER>'
    return set(re.findall(pattern, text))


def analyze_csv_failures(label_path: str, result_path: str) -> Dict:
    """
    Analyze a single CSV file and return detailed failure information.
    """
    failures = {
        'filename': os.path.basename(label_path),
        'total_phi': 0,
        'redacted': 0,
        'not_redacted': 0,
        'details': []  # List of {row, field, phi_value, context}
    }
    
    if not os.path.exists(label_path) or not os.path.exists(result_path):
        return failures
    
    with open(label_path, 'r', encoding='utf-8') as f:
        label_rows = list(csv.DictReader(f))
    
    with open(result_path, 'r', encoding='utf-8') as f:
        result_rows = list(csv.DictReader(f))
    
    for row_idx, (label_row, result_row) in enumerate(zip(label_rows, result_rows)):
        for field_name in label_row.keys():
            if field_name not in result_row:
                continue
            
            label_value = label_row[field_name] or ''
            result_value = result_row[field_name] or ''
            
            if '<PER>' not in label_value:
                continue
            
            phi_set = extract_phi_from_text(label_value)
            failures['total_phi'] += len(phi_set)
            
            for phi in phi_set:
                if phi in result_value:
                    failures['not_redacted'] += 1
                    failures['details'].append({
                        'row': row_idx + 2,  # 1-based + header
                        'field': field_name,
                        'phi_value': phi,
                        'phi_length': len(phi),
                        'result_value_preview': result_value[:100] + '...' if len(result_value) > 100 else result_value
                    })
                else:
                    failures['redacted'] += 1
    
    return failures


def find_anonymized_folder(patient_folder_name: str, results_base_dir: str) -> Optional[str]:
    """
    Find the anonymized folder name for a given patient folder.
    Reads from folder_annotations.csv if available, otherwise looks for matching folders.
    """
    # Try to read folder mapping from CSV
    mapping_file = os.path.join(os.path.dirname(results_base_dir), 'folder_annotations.csv')
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('original_folder') == patient_folder_name:
                    return row.get('anonymized_folder')

    # Fallback: look for folders in results directory
    if os.path.exists(results_base_dir):
        folders = [f for f in os.listdir(results_base_dir)
                  if os.path.isdir(os.path.join(results_base_dir, f))]
        # Try to match by pattern or return first folder
        for folder in folders:
            if 'patient' in folder.lower():
                return folder

    return None


def discover_csv_files(patient_folder: str) -> List[Tuple[str, str]]:
    """
    Discover all CSV label files in the patient folder's annotations_csv directory.
    Returns list of (label_file_path, label_file_name) tuples.
    """
    annotations_dir = os.path.join(patient_folder, 'annotations_csv')
    if not os.path.exists(annotations_dir):
        print(f"⚠️  Annotations directory not found: {annotations_dir}")
        return []

    csv_files = []
    for filename in os.listdir(annotations_dir):
        if filename.endswith('.csv'):
            full_path = os.path.join(annotations_dir, filename)
            csv_files.append((full_path, filename))

    return sorted(csv_files)


def infer_result_filename(label_filename: str) -> str:
    """
    Infer the result filename from the label filename.
    Replaces the patient ID suffix with 'ID'.

    Example: hosp_emar_20130725.csv -> hosp_emar_ID.csv
    Special cases:
    - csv_filename_annotations.csv -> filename_anonymization.csv
    - hosp_patients.csv -> hosp_patients.csv (no change)
    """
    # Special case: csv_filename_annotations -> filename_anonymization
    if label_filename == 'csv_filename_annotations.csv':
        return 'filename_anonymization.csv'

    # Pattern: <prefix>_<patient_id>.csv -> <prefix>_ID.csv
    match = re.match(r'(.+)_\d+\.csv$', label_filename)
    if match:
        prefix = match.group(1)
        return f"{prefix}_ID.csv"

    # No pattern match, return as-is
    return label_filename


def main():
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python analyze_failures.py <patient_folder_path> [labels_base_dir] [results_base_dir]")
        print("\nExample:")
        print("  python analyze_failures.py patient_10005749_20010003")
        print("  python analyze_failures.py /path/to/patient_10005749_20010003 ./data/primary/patient_records_labels ./data/results")
        sys.exit(1)

    patient_folder_input = sys.argv[1]

    # Default directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_labels_dir = os.path.join(script_dir, "data/primary/patient_records_labels")
    default_results_dir = os.path.join(script_dir, "data/results")

    labels_base_dir = sys.argv[2] if len(sys.argv) > 2 else default_labels_dir
    results_base_dir = sys.argv[3] if len(sys.argv) > 3 else default_results_dir

    # Determine patient folder path
    if os.path.isabs(patient_folder_input):
        patient_folder_path = patient_folder_input
        patient_folder_name = os.path.basename(patient_folder_input)
    else:
        patient_folder_name = patient_folder_input
        patient_folder_path = os.path.join(labels_base_dir, patient_folder_name)

    # Check if patient folder exists
    if not os.path.exists(patient_folder_path):
        print(f"❌ Error: Patient folder not found: {patient_folder_path}")
        sys.exit(1)

    # Find anonymized folder
    anon_folder_name = find_anonymized_folder(patient_folder_name, results_base_dir)
    if not anon_folder_name:
        print(f"⚠️  Warning: Could not find anonymized folder for {patient_folder_name}")
        print(f"    Please ensure the results exist in: {results_base_dir}")
        sys.exit(1)

    anon_folder_path = os.path.join(results_base_dir, anon_folder_name)

    print("="*100)
    print("DETAILED ANONYMIZATION FAILURE ANALYSIS")
    print("="*100)
    print(f"\n📁 Patient Folder: {patient_folder_name}")
    print(f"📁 Anonymized Folder: {anon_folder_name}")
    print(f"📂 Labels Path: {patient_folder_path}")
    print(f"📂 Results Path: {anon_folder_path}")
    print("="*100)

    # Discover CSV files
    csv_files = discover_csv_files(patient_folder_path)
    if not csv_files:
        print(f"\n❌ No CSV label files found in {patient_folder_path}/annotations_csv")
        sys.exit(1)

    print(f"\n📊 Found {len(csv_files)} CSV label files")

    all_phi_values = defaultdict(int)  # PHI value -> count of failures
    phi_by_length = defaultdict(int)   # length -> count
    phi_by_field = defaultdict(int)    # field -> count
    total_files_analyzed = 0
    total_files_with_failures = 0

    for label_path, label_filename in csv_files:
        # Infer result filename
        result_filename = infer_result_filename(label_filename)
        result_path = os.path.join(anon_folder_path, 'csv', result_filename)

        if not os.path.exists(result_path):
            print(f"\n⚠️  Result file not found: {result_path}")
            continue

        total_files_analyzed += 1
        failures = analyze_csv_failures(label_path, result_path)

        if failures['not_redacted'] > 0:
            total_files_with_failures += 1
            print(f"\n{'='*80}")
            print(f"📄 FILE: {label_filename}")
            print(f"   Result: {result_filename}")
            print(f"   Total PHI: {failures['total_phi']}, Redacted: {failures['redacted']}, NOT Redacted: {failures['not_redacted']}")
            print(f"   Recall: {failures['redacted']/failures['total_phi']*100:.2f}%")
            print(f"{'='*80}")

            # Group by PHI value
            phi_counts = defaultdict(list)
            for detail in failures['details']:
                phi_counts[detail['phi_value']].append(detail)

            print(f"\n   Unique PHI values not redacted: {len(phi_counts)}")
            print(f"   {'PHI Value':<30} {'Length':<8} {'Occurrences':<12} {'Field(s)'}")
            print(f"   {'-'*80}")

            for phi_value, occurrences in sorted(phi_counts.items(), key=lambda x: -len(x[1]))[:20]:
                fields = set(occ['field'] for occ in occurrences)
                fields_str = ', '.join(list(fields)[:3])
                if len(fields) > 3:
                    fields_str += f" (+{len(fields)-3} more)"

                print(f"   {phi_value:<30} {len(phi_value):<8} {len(occurrences):<12} {fields_str}")

                all_phi_values[phi_value] += len(occurrences)
                phi_by_length[len(phi_value)] += len(occurrences)
                for f in fields:
                    phi_by_field[f] += len(occurrences)

            if len(phi_counts) > 20:
                print(f"   ... and {len(phi_counts) - 20} more unique PHI values")
        elif failures['total_phi'] > 0:
            print(f"\n✅ {label_filename}: All {failures['total_phi']} PHI values redacted successfully!")
    
    # Summary analysis
    print("\n" + "="*100)
    print("SUMMARY ANALYSIS")
    print("="*100)

    print(f"\n📈 FILES ANALYZED:")
    print(f"   Total CSV files analyzed: {total_files_analyzed}")
    print(f"   Files with failures: {total_files_with_failures}")
    print(f"   Files with perfect redaction: {total_files_analyzed - total_files_with_failures}")

    if not all_phi_values:
        print("\n🎉 No PHI redaction failures found! All files passed successfully.")
        return

    print("\n📊 TOP 20 MOST COMMON UNREDACTED PHI VALUES:")
    print(f"   {'PHI Value':<40} {'Total Occurrences':<20}")
    print(f"   {'-'*60}")
    for phi, count in sorted(all_phi_values.items(), key=lambda x: -x[1])[:20]:
        print(f"   {phi:<40} {count:<20}")
    
    print("\n📏 FAILURES BY PHI LENGTH:")
    print(f"   {'Length':<10} {'Count':<15} {'Percentage':<15}")
    print(f"   {'-'*40}")
    total_failures = sum(phi_by_length.values())
    for length in sorted(phi_by_length.keys()):
        count = phi_by_length[length]
        pct = count / total_failures * 100 if total_failures > 0 else 0
        print(f"   {length:<10} {count:<15} {pct:.1f}%")
    
    print("\n📁 FAILURES BY FIELD:")
    print(f"   {'Field':<30} {'Count':<15} {'Percentage':<15}")
    print(f"   {'-'*60}")
    for field, count in sorted(phi_by_field.items(), key=lambda x: -x[1])[:15]:
        pct = count / total_failures * 100 if total_failures > 0 else 0
        print(f"   {field:<30} {count:<15} {pct:.1f}%")
    
    # Analyze patterns
    print("\n🔍 PATTERN ANALYSIS:")
    
    # Check for short values
    short_values = [phi for phi in all_phi_values.keys() if len(phi) <= 3]
    if short_values:
        print(f"\n   ⚠️  Short PHI values (≤3 chars) that may be difficult to detect:")
        for phi in short_values[:10]:
            print(f"      '{phi}' (length: {len(phi)}, occurrences: {all_phi_values[phi]})")
    
    # Check for numeric values
    numeric_values = [phi for phi in all_phi_values.keys() if phi.replace('-', '').replace(':', '').replace(' ', '').isdigit()]
    if numeric_values:
        print(f"\n   📝 Numeric PHI values (IDs, dates, times):")
        for phi in sorted(numeric_values, key=lambda x: -all_phi_values[x])[:10]:
            print(f"      '{phi}' (occurrences: {all_phi_values[phi]})")
    
    # Check for date/time patterns
    datetime_pattern = re.compile(r'\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}')
    datetime_values = [phi for phi in all_phi_values.keys() if datetime_pattern.search(phi)]
    if datetime_values:
        print(f"\n   📅 Date/Time PHI values:")
        for phi in sorted(datetime_values, key=lambda x: -all_phi_values[x])[:10]:
            print(f"      '{phi}' (occurrences: {all_phi_values[phi]})")


if __name__ == '__main__':
    main()
