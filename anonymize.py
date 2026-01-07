#!/usr/bin/env python3
"""
Main script for anonymizing files using LLM-based processors.
"""

import argparse
from pathlib import Path
from typing import List

from anonymizer import AnonymizerConfig, PNGProcessor, PNGOCRProcessor, CSVProcessor


def get_processor(file_path: Path, config: AnonymizerConfig, use_ocr: bool = False):
    """
    Get appropriate processor for the file type.

    Args:
        file_path: Path to the file
        config: Anonymizer configuration
        use_ocr: If True, use OCR-based processor for images; otherwise use vision-based processor

    Returns:
        FileProcessor instance or None
    """
    if use_ocr:
        processors = [
            PNGOCRProcessor(config),
            CSVProcessor(config),
        ]
    else:
        processors = [
            PNGProcessor(config),
            CSVProcessor(config),
        ]

    for processor in processors:
        if processor.can_process(file_path):
            return processor

    return None


def process_file(input_path: Path, output_dir: Path, config: AnonymizerConfig, use_ocr: bool = False) -> bool:
    """
    Process a single file.

    Args:
        input_path: Path to input file
        output_dir: Directory for output
        config: Anonymizer configuration
        use_ocr: If True, use OCR-based processor

    Returns:
        True if successful
    """
    processor = get_processor(input_path, config, use_ocr)

    if processor is None:
        print(f"No processor available for: {input_path.name}")
        return False

    # Create output path with same filename
    output_path = output_dir / f"anonymized_{input_path.name}"

    try:
        processor.anonymize(input_path, output_path)
        return True
    except Exception as e:
        import traceback
        print(f"Error processing {input_path.name}: {e}")
        traceback.print_exc()
        return False


def process_directory(input_dir: Path, output_dir: Path, config: AnonymizerConfig, use_ocr: bool = False):
    """
    Process all supported files in a directory.

    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        config: Anonymizer configuration
        use_ocr: If True, use OCR-based processor
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all files
    files = [f for f in input_dir.iterdir() if f.is_file()]

    if not files:
        print(f"No files found in {input_dir}")
        return

    print(f"Found {len(files)} files to process\n")

    successful = 0
    failed = 0

    for file_path in files:
        print(f"\n{'='*60}")
        if process_file(file_path, output_dir, config, use_ocr):
            successful += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"Processing complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Anonymize files using LLM-based detection and redaction"
    )
    parser.add_argument(
        "input", type=str, help="Input file or directory path"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="data/output",
        help="Output directory (default: data/output)"
    )
    parser.add_argument(
        "--mode", "-m", type=str, choices=["vision", "ocr"], default="vision",
        help="Processing mode: 'vision' for direct LLM vision analysis, 'ocr' for OCR + LLM classification (default: vision)"
    )

    args = parser.parse_args()

    # Create config
    config = AnonymizerConfig(
        output_dir=args.output,
    )

    input_path = Path(args.input)
    output_dir = Path(args.output)
    use_ocr = (args.mode == "ocr")

    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return

    print(f"Using processing mode: {args.mode}")
    print()

    if input_path.is_file():
        output_dir.mkdir(parents=True, exist_ok=True)
        process_file(input_path, output_dir, config, use_ocr)
    elif input_path.is_dir():
        process_directory(input_path, output_dir, config, use_ocr)
    else:
        print(f"Error: Invalid input path: {input_path}")


if __name__ == "__main__":
    main()
