#!/usr/bin/env python3
"""
Script to remove <PER></PER>, <DATE></DATE>, and <AGE></AGE> tags from a CSV file while preserving the content.
Can process single files or entire directories recursively.
"""
import re
import argparse
import csv
import os
import shutil
from pathlib import Path


def remove_tags(text):
    """Remove <PER></PER>, <DATE></DATE>, and <AGE></AGE> tags but keep the content inside."""
    if not isinstance(text, str):
        return text
    # Remove all three types of tags
    text = re.sub(r'<PER>(.*?)</PER>', r'\1', text)
    text = re.sub(r'<DATE>(.*?)</DATE>', r'\1', text)
    text = re.sub(r'<AGE>(.*?)</AGE>', r'\1', text)
    return text


def process_csv(input_file, output_file):
    """Process CSV file to remove PER, DATE, and AGE tags from all cells."""
    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        rows = list(reader)

    # Process each cell
    processed_rows = []
    for row in rows:
        processed_row = [remove_tags(cell) for cell in row]
        processed_rows.append(processed_row)

    # Write to output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(processed_rows)

    return len(processed_rows)


def process_text_file(input_file, output_file):
    """Process text file (like .hea) to remove tags."""
    with open(input_file, 'r', encoding='utf-8') as infile:
        content = infile.read()

    # Remove tags from content
    processed_content = remove_tags(content)

    # Write to output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write(processed_content)

    return len(content)


def process_directory(input_dir, output_dir):
    """Process entire directory recursively, processing CSVs, .hea files, and copying other files."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Files to skip (neither process nor copy)
    skip_files = {
        'folder_annotations.csv',
        'csv_filename_annotations.csv',
        'phi_annotations_cxr.csv',
        'phi_annotations_ecg.csv'
    }

    csv_count = 0
    hea_count = 0
    copied_count = 0
    skipped_count = 0

    # Walk through all files in input directory
    for root, dirs, files in os.walk(input_path):
        # Calculate relative path from input directory
        rel_path = Path(root).relative_to(input_path)
        current_output_dir = output_path / rel_path

        # Create subdirectories in output
        current_output_dir.mkdir(parents=True, exist_ok=True)

        for file in files:
            # Skip files in the skip list
            if file in skip_files:
                skipped_count += 1
                print(f"Skipped: {Path(root).relative_to(input_path) / file}")
                continue

            input_file = Path(root) / file
            output_file = current_output_dir / file

            if file.lower().endswith('.csv'):
                # Process CSV file
                try:
                    rows = process_csv(str(input_file), str(output_file))
                    csv_count += 1
                    print(f"Processed CSV: {input_file.relative_to(input_path)} ({rows} rows)")
                except Exception as e:
                    print(f"Error processing {input_file}: {e}")
            elif file.lower().endswith('.hea'):
                # Process .hea file
                try:
                    chars = process_text_file(str(input_file), str(output_file))
                    hea_count += 1
                    print(f"Processed HEA: {input_file.relative_to(input_path)} ({chars} chars)")
                except Exception as e:
                    print(f"Error processing {input_file}: {e}")
            else:
                # Copy non-CSV/non-HEA files
                try:
                    shutil.copy2(str(input_file), str(output_file))
                    copied_count += 1
                    print(f"Copied: {input_file.relative_to(input_path)}")
                except Exception as e:
                    print(f"Error copying {input_file}: {e}")

    print(f"\nSummary:")
    print(f"  CSV files processed: {csv_count}")
    print(f"  HEA files processed: {hea_count}")
    print(f"  Other files copied: {copied_count}")
    print(f"  Files skipped: {skipped_count}")
    print(f"  Output directory: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove <PER></PER>, <DATE></DATE>, and <AGE></AGE> tags from CSV and .hea files while preserving content. "
                    "Can process single files or entire directories recursively."
    )
    parser.add_argument("input", help="Input CSV/.hea file or directory path")
    parser.add_argument("-o", "--output", help="Output file or directory path (default: input_notags.csv/.hea or input_notags/)")

    args = parser.parse_args()

    input_path = Path(args.input)

    # Check if input is a directory or file
    if input_path.is_dir():
        # Process directory
        if not args.output:
            args.output = str(input_path) + '_notags'

        print(f"Processing directory: {input_path}")
        print(f"Output directory: {args.output}\n")
        process_directory(str(input_path), args.output)

    elif input_path.is_file():
        # Process single file
        if not args.output:
            if str(input_path).lower().endswith('.csv'):
                args.output = str(input_path).replace('.csv', '_notags.csv')
            elif str(input_path).lower().endswith('.hea'):
                args.output = str(input_path).replace('.hea', '_notags.hea')
            else:
                args.output = str(input_path) + '_notags'

        print(f"Processing file: {input_path}")

        if str(input_path).lower().endswith('.csv'):
            rows = process_csv(str(input_path), args.output)
            print(f"Processed {rows} rows")
        elif str(input_path).lower().endswith('.hea'):
            chars = process_text_file(str(input_path), args.output)
            print(f"Processed {chars} characters")
        else:
            print(f"Warning: Unknown file type, treating as text file")
            chars = process_text_file(str(input_path), args.output)
            print(f"Processed {chars} characters")

        print(f"Output saved to: {args.output}")

    else:
        print(f"Error: {input_path} is neither a file nor a directory")
        exit(1)
