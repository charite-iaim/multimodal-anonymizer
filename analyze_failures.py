#!/usr/bin/env python3
"""
Analyze anonymization failures in detail.
Shows exactly which PHI values were not properly redacted.
"""

import os
import re
import csv
from collections import defaultdict
from typing import Set, Dict, List, Tuple


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


def main():
    labels_dir = "/Users/julian_anja/Documents/coding/mimiciv-anonymization-pipeline/data/primary/patient_records_labels"
    results_dir = "/Users/julian_anja/Documents/coding/mimiciv-anonymization-pipeline/data/results"
    
    # Folder mapping
    folder_mapping = {
        'patient_10005749_20010003': 'patient_ID_ID_001',
        'patient_10045929_20130725': 'patient_ID_ID_002',
    }
    
    # Files with known issues
    problem_files = [
        ('patient_10045929_20130725', 'icu_chartevents_20130725.csv', 'icu_chartevents_ID.csv'),
        ('patient_10045929_20130725', 'hosp_emar_20130725.csv', 'hosp_emar_ID.csv'),
        ('patient_10005749_20010003', 'hosp_labevents_20010003.csv', 'hosp_labevents_ID.csv'),
        ('patient_10045929_20130725', 'hosp_labevents_20130725.csv', 'hosp_labevents_ID.csv'),
        ('patient_10045929_20130725', 'icu_outputevents_20130725.csv', 'icu_outputevents_ID.csv'),
        ('patient_10045929_20130725', 'icu_ingredientevents_20130725.csv', 'icu_ingredientevents_ID.csv'),
        ('patient_10005749_20010003', 'hosp_prescriptions_20010003.csv', 'hosp_prescriptions_ID.csv'),
        ('patient_10045929_20130725', 'note_radiology_20130725.csv', 'note_radiology_ID.csv'),
        ('patient_10045929_20130725', 'icu_procedureevents_20130725.csv', 'icu_procedureevents_ID.csv'),
    ]
    
    print("="*100)
    print("DETAILED ANONYMIZATION FAILURE ANALYSIS")
    print("="*100)
    
    all_phi_values = defaultdict(int)  # PHI value -> count of failures
    phi_by_length = defaultdict(int)   # length -> count
    phi_by_field = defaultdict(int)    # field -> count
    
    for patient_folder, label_file, result_file in problem_files:
        anon_folder = folder_mapping.get(patient_folder, patient_folder)
        
        label_path = os.path.join(labels_dir, patient_folder, 'annotations_csv', label_file)
        result_path = os.path.join(results_dir, anon_folder, 'csv', result_file)
        
        if not os.path.exists(label_path):
            print(f"\n⚠️  Label file not found: {label_path}")
            continue
        if not os.path.exists(result_path):
            print(f"\n⚠️  Result file not found: {result_path}")
            continue
        
        failures = analyze_csv_failures(label_path, result_path)
        
        if failures['not_redacted'] > 0:
            print(f"\n{'='*80}")
            print(f"📄 FILE: {label_file}")
            print(f"   Patient: {patient_folder}")
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
    
    # Summary analysis
    print("\n" + "="*100)
    print("SUMMARY ANALYSIS")
    print("="*100)
    
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
