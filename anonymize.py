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
    TextProcessor,
    DICOMProcessor,
    FileTypeDetector,
    DataType,
)
from anonymizer.filename_anonymizer import FilenameAnonymizer


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
            # Text data -> use suggested processor
            if detection_result.suggested_processor == "text":
                processor = TextProcessor(config)
                if processor.can_process(file_path):
                    print(f"Using Text processor based on LLM detection")
                    return processor
            elif detection_result.suggested_processor == "csv":
                processor = CSVProcessor(config)
                if processor.can_process(file_path):
                    print(f"Using CSV processor based on LLM detection")
                    return processor

            # Fallback: try both processors
            text_processor = TextProcessor(config)
            if text_processor.can_process(file_path):
                print(f"Using Text processor (fallback)")
                return text_processor

            csv_processor = CSVProcessor(config)
            if csv_processor.can_process(file_path):
                print(f"Using CSV processor (fallback)")
                return csv_processor

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
            TextProcessor(config),
            CSVProcessor(config),
        ]
    else:
        processors = [
            DICOMProcessor(config),
            PNGProcessor(config),
            PDFOCRProcessor(config),
            TextProcessor(config),
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
    use_llm_detection: bool = False,
    preserve_structure: bool = False,
    relative_path: Path = None,
    filename_anonymizer: FilenameAnonymizer = None,
    anonymize_paths: bool = True
) -> bool:
    """
    Process a single file.

    Args:
        input_path: Path to input file
        output_dir: Directory for output
        config: Anonymizer configuration
        use_ocr: If True, use OCR-based processor
        use_llm_detection: If True, use multimodal LLM to detect file type
        preserve_structure: If True, preserve directory structure in output
        relative_path: Relative path from input root (used when preserve_structure=True)
        filename_anonymizer: Optional FilenameAnonymizer instance for anonymizing filenames
        anonymize_paths: If True, automatically anonymize filename (default: True)

    Returns:
        True if successful
    """
    processor = get_processor(input_path, config, use_ocr, use_llm_detection)

    if processor is None:
        print(f"No processor available for: {input_path.name}")
        return False

    # Automatically create filename anonymizer if needed and enabled
    if anonymize_paths and filename_anonymizer is None:
        filename_anonymizer = FilenameAnonymizer(config, output_dir=output_dir)

    # Anonymize filename if enabled
    if anonymize_paths and filename_anonymizer:
        print(f"Anonymizing filename: {input_path.name}")
        anonymization_result = filename_anonymizer.anonymize_filename(
            input_path.name,
            is_directory=False
        )
        anonymized_filename = anonymization_result.anonymized_filename
        print(f"Anonymized filename: {anonymized_filename}")
        if anonymization_result.phi_detections:
            print(f"  Found {len(anonymization_result.phi_detections)} PHI values in filename:")
            for detection in anonymization_result.phi_detections:
                print(f"    - {detection.original_value} ({detection.category})")
    else:
        anonymized_filename = f"anonymized_{input_path.name}"
        anonymization_result = None

    if preserve_structure and relative_path:
        # Preserve exact directory structure
        file_output_dir = output_dir / relative_path.parent
        file_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = file_output_dir / anonymized_filename

        # Record file mapping for CSV export
        if anonymize_paths and filename_anonymizer and anonymization_result:
            # Get folder path for grouping (e.g., "patient_ID_ID/csv")
            folder_path = str(relative_path.parent) if relative_path.parent != Path('.') else ""
            filename_anonymizer.add_file_mapping(
                folder_path=folder_path,
                original_filename=input_path.name,
                anonymized_filename=anonymized_filename,
                phi_detections=anonymization_result.phi_detections
            )
    else:
        # Create separate output folder for this file (original behavior)
        file_stem = input_path.stem  # filename without extension
        file_output_dir = output_dir / file_stem
        file_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = file_output_dir / anonymized_filename

        # Record file mapping for CSV export
        if anonymize_paths and filename_anonymizer and anonymization_result:
            filename_anonymizer.add_file_mapping(
                folder_path=file_stem,  # Use file stem as folder path for standalone files
                original_filename=input_path.name,
                anonymized_filename=anonymized_filename,
                phi_detections=anonymization_result.phi_detections
            )

    try:
        processor.anonymize(input_path, output_path)
        print(f"Output saved to: {output_path}")

        # Save CSV mappings if this is a standalone file processing
        if anonymize_paths and filename_anonymizer and not preserve_structure:
            filename_anonymizer.save_all_mappings(output_dir=output_dir)

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
    use_llm_detection: bool = False,
    recursive: bool = False,
    preserve_structure: bool = False,
    skip_hidden: bool = True,
    anonymize_paths: bool = True
):
    """
    Process all supported files in a directory.

    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        config: Anonymizer configuration
        use_ocr: If True, use OCR-based processor
        use_llm_detection: If True, use multimodal LLM to detect file type
        recursive: If True, process subdirectories recursively
        preserve_structure: If True, preserve directory structure in output
        skip_hidden: If True, skip hidden files and directories (starting with '.')
        anonymize_paths: If True, anonymize file and folder names
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not recursive:
        # Original behavior: process only files in current directory
        files = [f for f in input_dir.iterdir() if f.is_file()]

        if skip_hidden:
            files = [f for f in files if not f.name.startswith('.')]

        if not files:
            print(f"No files found in {input_dir}")
            return

        print(f"Found {len(files)} files to process\n")

        # Create filename anonymizer if needed
        filename_anonymizer = FilenameAnonymizer(config, output_dir=output_dir) if anonymize_paths else None

        successful = 0
        failed = 0

        for file_path in files:
            print(f"\n{'='*60}")
            if process_file(file_path, output_dir, config, use_ocr, use_llm_detection,
                          filename_anonymizer=filename_anonymizer, anonymize_paths=anonymize_paths):
                successful += 1
            else:
                failed += 1

        print(f"\n{'='*60}")
        print(f"Processing complete:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")

        # Save CSV mappings if anonymization was enabled
        if anonymize_paths and filename_anonymizer:
            filename_anonymizer.save_all_mappings(output_dir=output_dir)
    else:
        # Recursive processing with structure preservation
        process_directory_recursive(
            input_dir=input_dir,
            output_dir=output_dir,
            config=config,
            use_ocr=use_ocr,
            use_llm_detection=use_llm_detection,
            preserve_structure=preserve_structure,
            skip_hidden=skip_hidden,
            anonymize_paths=anonymize_paths
        )


def process_directory_recursive(
    input_dir: Path,
    output_dir: Path,
    config: AnonymizerConfig,
    use_ocr: bool = False,
    use_llm_detection: bool = False,
    preserve_structure: bool = True,
    skip_hidden: bool = True,
    anonymize_paths: bool = True,
    _root_dir: Path = None,
    _stats: dict = None,
    _filename_anonymizer: FilenameAnonymizer = None,
    _folder_mapping: dict = None
):
    """
    Recursively process all files in a directory tree, preserving structure.

    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        config: Anonymizer configuration
        use_ocr: If True, use OCR-based processor
        use_llm_detection: If True, use multimodal LLM to detect file type
        preserve_structure: If True, preserve directory structure in output
        skip_hidden: If True, skip hidden files and directories
        anonymize_paths: If True, anonymize file and folder names
        _root_dir: Internal parameter for tracking root directory
        _stats: Internal parameter for tracking statistics
        _filename_anonymizer: Internal parameter for filename anonymization
        _folder_mapping: Internal parameter for tracking folder name mappings
    """
    # Initialize on first call
    if _root_dir is None:
        _root_dir = input_dir
        _stats = {"successful": 0, "failed": 0, "skipped": 0}
        _folder_mapping = {}
        if anonymize_paths:
            _filename_anonymizer = FilenameAnonymizer(config, output_dir=output_dir)
        print(f"Starting recursive directory processing...")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Structure preservation: {preserve_structure}")
        print(f"Skip hidden files: {skip_hidden}")
        print(f"Anonymize paths: {anonymize_paths}\n")

    try:
        items = sorted(input_dir.iterdir())
    except PermissionError:
        print(f"Permission denied: {input_dir}")
        return _stats

    for item in items:
        # Skip hidden files/directories if requested
        if skip_hidden and item.name.startswith('.'):
            continue

        if item.is_file():
            # Calculate relative path from root
            original_relative_path = item.relative_to(_root_dir)

            # Build anonymized relative path by replacing folder names
            if anonymize_paths and _folder_mapping:
                anonymized_parts = []
                for part in original_relative_path.parts[:-1]:  # All parts except filename
                    anonymized_parts.append(_folder_mapping.get(part, part))
                # Filename will be anonymized by process_file
                anonymized_relative_path = Path(*anonymized_parts) if anonymized_parts else Path()
            else:
                anonymized_relative_path = original_relative_path.parent

            print(f"\n{'='*60}")
            print(f"Processing: {original_relative_path}")

            success = process_file(
                input_path=item,
                output_dir=output_dir,
                config=config,
                use_ocr=use_ocr,
                use_llm_detection=use_llm_detection,
                preserve_structure=preserve_structure,
                relative_path=original_relative_path if not anonymize_paths else anonymized_relative_path / item.name,
                filename_anonymizer=_filename_anonymizer,
                anonymize_paths=anonymize_paths
            )

            if success:
                _stats["successful"] += 1
            else:
                _stats["failed"] += 1

        elif item.is_dir():
            # Recursively process subdirectory
            original_relative_dir = item.relative_to(_root_dir)
            print(f"\nEntering directory: {original_relative_dir}/")

            # Anonymize folder name if enabled
            if anonymize_paths:
                print(f"Anonymizing folder name: {item.name}")
                anonymization_result = _filename_anonymizer.anonymize_filename(item.name, is_directory=True)
                anonymized_folder_name = anonymization_result.anonymized_filename
                print(f"Anonymized folder name: {anonymized_folder_name}")
                if anonymization_result.phi_detections:
                    print(f"  Found {len(anonymization_result.phi_detections)} PHI values in folder name:")
                    for detection in anonymization_result.phi_detections:
                        print(f"    - {detection.original_value} ({detection.category})")

                # Store folder mapping for building paths
                _folder_mapping[item.name] = anonymized_folder_name

                # Build anonymized relative path for this directory
                anonymized_parts = []
                for part in original_relative_dir.parts:
                    anonymized_parts.append(_folder_mapping.get(part, part))
                anonymized_relative_dir = Path(*anonymized_parts)

                # Record folder mapping for CSV export
                _filename_anonymizer.add_folder_mapping(
                    original_foldername=item.name,
                    anonymized_foldername=anonymized_folder_name,
                    phi_detections=anonymization_result.phi_detections
                )
            else:
                anonymized_relative_dir = original_relative_dir

            process_directory_recursive(
                input_dir=item,
                output_dir=output_dir,
                config=config,
                use_ocr=use_ocr,
                use_llm_detection=use_llm_detection,
                preserve_structure=preserve_structure,
                skip_hidden=skip_hidden,
                anonymize_paths=anonymize_paths,
                _root_dir=_root_dir,
                _stats=_stats,
                _filename_anonymizer=_filename_anonymizer,
                _folder_mapping=_folder_mapping
            )

    # Print summary and save mappings only on initial call
    if input_dir == _root_dir:
        print(f"\n{'='*60}")
        print(f"Processing complete:")
        print(f"  Successful: {_stats['successful']}")
        print(f"  Failed: {_stats['failed']}")
        print(f"  Total processed: {_stats['successful'] + _stats['failed']}")

        # Save CSV mappings if anonymization was enabled
        if anonymize_paths and _filename_anonymizer:
            _filename_anonymizer.save_all_mappings(output_dir=output_dir)

    return _stats


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
        "--mode", "-m", type=str, choices=["vision", "ocr"], default="ocr",
        help="Processing mode: 'vision' for direct LLM vision analysis, 'ocr' for OCR + LLM classification (default: vision)"
    )
    parser.add_argument(
        "--auto-detect", "-a", action="store_true",
        help="Use multimodal LLM to automatically detect file type and select appropriate processor"
    )
    parser.add_argument(
        "--recursive", "-r", action="store_true",
        help="Process directories recursively, including all subdirectories"
    )
    parser.add_argument(
        "--preserve-structure", "-p", action="store_true",
        help="Preserve the exact directory structure in the output (recommended with --recursive)"
    )
    parser.add_argument(
        "--include-hidden", action="store_true",
        help="Include hidden files and directories (starting with '.'). By default, hidden files are skipped."
    )
    parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Save debug files (JSON metadata, intermediate PNG files from DICOM, etc.). By default, only anonymized files are saved."
    )
    parser.add_argument(
        "--no-anonymize-paths", action="store_true",
        help="Disable automatic filename and folder name anonymization. By default, paths ARE anonymized."
    )

    args = parser.parse_args()

    # Invert the flag - default is to anonymize paths
    anonymize_paths = not args.no_anonymize_paths

    # Create config
    config = AnonymizerConfig(
        output_dir=args.output,
        save_debug_files=args.debug,
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

    if args.recursive:
        print(f"Recursive processing: enabled")
        print(f"Structure preservation: {'enabled' if args.preserve_structure else 'disabled'}")
        print(f"Hidden files: {'included' if args.include_hidden else 'skipped'}")

    if args.debug:
        print(f"Debug mode: enabled (JSON metadata and intermediate files will be saved)")
    else:
        print(f"Debug mode: disabled (only anonymized files will be saved)")

    if anonymize_paths:
        print(f"Path anonymization: enabled (filenames and folders will be anonymized)")
    else:
        print(f"Path anonymization: disabled (only file contents will be anonymized)")
    print()

    if input_path.is_file():
        # For single file, process with automatic filename anonymization
        process_file(input_path, output_dir, config, use_ocr, use_llm_detection,
                    anonymize_paths=anonymize_paths)
    elif input_path.is_dir():
        process_directory(
            input_path,
            output_dir,
            config,
            use_ocr,
            use_llm_detection,
            recursive=args.recursive,
            preserve_structure=args.preserve_structure,
            skip_hidden=not args.include_hidden,
            anonymize_paths=anonymize_paths
        )
    else:
        print(f"Error: Invalid input path: {input_path}")


if __name__ == "__main__":
    main()
