#!/usr/bin/env python3
"""
Main script for anonymizing files using LLM-based processors.
"""

import argparse
from pathlib import Path
from typing import List

from anonymizer import (
    AnonymizerConfig,
    PNGProcessor,
    PNGOCRProcessor,
    PDFOCRProcessor,
    CSVProcessor,
    DICOMProcessor,
    FileTypeDetector,
    DataType,
)


def get_processor(
    file_path: Path,
    config: AnonymizerConfig,
    use_ocr: bool = False,
    use_llm_detection: bool = False
):
    """
    Get appropriate processor for the file type.

    Args:
        file_path: Path to the file
        config: Anonymizer configuration
        use_ocr: If True, use OCR-based processor for images; otherwise use vision-based processor
        use_llm_detection: If True, use multimodal LLM to detect file type and choose processor

    Returns:
        FileProcessor instance or None
    """
    if use_llm_detection:
        # Use LLM to detect file type and determine processor
        detector = FileTypeDetector(config)
        detection_result = detector.detect_file_type(file_path)

        # Map detected type to processor
        if detection_result.data_type == DataType.TEXT:
            # Text data -> use CSV processor
            processor = CSVProcessor(config)
            if processor.can_process(file_path):
                print(f"Using CSV processor based on LLM detection")
                return processor

        elif detection_result.data_type == DataType.IMAGE:
            # Image data -> use OCR or vision processor based on suggestion
            if detection_result.suggested_processor == "ocr" or use_ocr:
                processor = PNGOCRProcessor(config)
            else:
                processor = PNGProcessor(config)

            if processor.can_process(file_path):
                print(f"Using {detection_result.suggested_processor} processor based on LLM detection")
                return processor

        # If LLM detection didn't work or type is unknown, fall back to extension-based matching
        print(f"LLM detected type '{detection_result.data_type}' but no suitable processor found, falling back to extension-based matching")

    # Original extension-based processor selection
    if use_ocr:
        processors = [
            DICOMProcessor(config),
            PNGOCRProcessor(config),
            PDFOCRProcessor(config),
            CSVProcessor(config),
        ]
    else:
        processors = [
            DICOMProcessor(config),
            PNGProcessor(config),
            PDFOCRProcessor(config),
            CSVProcessor(config),
        ]

    for processor in processors:
        if processor.can_process(file_path):
            return processor

    return None


def process_file(
    input_path: Path,
    output_dir: Path,
    config: AnonymizerConfig,
    use_ocr: bool = False,
    use_llm_detection: bool = False
) -> bool:
    """
    Process a single file.

    Args:
        input_path: Path to input file
        output_dir: Directory for output
        config: Anonymizer configuration
        use_ocr: If True, use OCR-based processor
        use_llm_detection: If True, use multimodal LLM to detect file type

    Returns:
        True if successful
    """
    processor = get_processor(input_path, config, use_ocr, use_llm_detection)

    if processor is None:
        print(f"No processor available for: {input_path.name}")
        return False

    # Create separate output folder for this file
    file_stem = input_path.stem  # filename without extension
    file_output_dir = output_dir / file_stem
    file_output_dir.mkdir(parents=True, exist_ok=True)

    # Create output path with same filename
    output_path = file_output_dir / f"anonymized_{input_path.name}"

    try:
        processor.anonymize(input_path, output_path)
        print(f"Output saved to: {file_output_dir}")
        return True
    except Exception as e:
        import traceback
        print(f"Error processing {input_path.name}: {e}")
        traceback.print_exc()
        return False


def process_directory(
    input_dir: Path,
    output_dir: Path,
    config: AnonymizerConfig,
    use_ocr: bool = False,
    use_llm_detection: bool = False
):
    """
    Process all supported files in a directory.

    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        config: Anonymizer configuration
        use_ocr: If True, use OCR-based processor
        use_llm_detection: If True, use multimodal LLM to detect file type
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
        if process_file(file_path, output_dir, config, use_ocr, use_llm_detection):
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
    parser.add_argument(
        "--auto-detect", "-a", action="store_true",
        help="Use multimodal LLM to automatically detect file type and select appropriate processor"
    )

    args = parser.parse_args()

    # Create config
    config = AnonymizerConfig(
        output_dir=args.output,
    )

    input_path = Path(args.input)
    output_dir = Path(args.output)
    use_ocr = (args.mode == "ocr")
    use_llm_detection = args.auto_detect

    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return

    if use_llm_detection:
        print(f"Using automatic file type detection with multimodal LLM")
    else:
        print(f"Using processing mode: {args.mode}")
    print()

    if input_path.is_file():
        process_file(input_path, output_dir, config, use_ocr, use_llm_detection)
    elif input_path.is_dir():
        process_directory(input_path, output_dir, config, use_ocr, use_llm_detection)
    else:
        print(f"Error: Invalid input path: {input_path}")


if __name__ == "__main__":
    main()
