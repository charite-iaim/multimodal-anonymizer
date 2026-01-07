#!/usr/bin/env python3
"""
Script to remove <PER></PER> tags from a CSV file while preserving the content.
"""
import re
import argparse
import csv


def remove_per_tags(text):
    """Remove <PER></PER> tags but keep the content inside."""
    if not isinstance(text, str):
        return text
    return re.sub(r'<PER>(.*?)</PER>', r'\1', text)


def process_csv(input_file, output_file):
    """Process CSV file to remove PER tags from all cells."""
    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        rows = list(reader)
    
    # Process each cell
    processed_rows = []
    for row in rows:
        processed_row = [remove_per_tags(cell) for cell in row]
        processed_rows.append(processed_row)
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(processed_rows)
    
    print(f"Processed {len(processed_rows)} rows")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove <PER></PER> tags from CSV file while preserving content"
    )
    parser.add_argument("input", help="Input CSV file path")
    parser.add_argument("-o", "--output", help="Output CSV file path (default: input_notags.csv)")
    
    args = parser.parse_args()
    
    if not args.output:
        args.output = args.input.replace('.csv', '_notags.csv')
    
    process_csv(args.input, args.output)
