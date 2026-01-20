#!/usr/bin/env python3
"""
Main script for anonymizing files using agentic/vision-based LLM processors.

This script uses:
- AgenticCSVProcessor for CSV files (tool-calling approach)
- AgenticTextProcessor for text files (tool-calling approach)
- DICOMVisionOCRProcessor for DICOM images (Vision LLM + OCR)
- PDFVisionOCRProcessor for PDF files (Vision LLM + OCR)
- PNGVisionOCRProcessor for PNG/JPG images (Vision LLM + OCR)
"""

import argparse
from pathlib import Path
from typing import List

from anonymizer import (
    AnonymizerConfig,
    FileTypeDetector,
    DataType,
)
from anonymizer.processors.agentic_csv_processor import AgenticCSVProcessor
from anonymizer.processors.agentic_text_processor import AgenticTextProcessor
from anonymizer.processors.dicom_vision_ocr_processor import DICOMVisionOCRProcessor
from anonymizer.processors.pdf_vision_ocr_processor import PDFVisionOCRProcessor
from anonymizer.processors.png_vision_ocr_processor import PNGVisionOCRProcessor
from anonymizer.filename_anonymizer import FilenameAnonymizer
from anonymizer.processing_tracker import ProcessingTracker
from anonymizer.parallel_processor import (
    ParallelFileProcessor,
    collect_files_for_processing
)


def generate_patient_time_offset() -> int:
    """
    Generate a random time offset for a patient.

    Returns a random offset between 1-3 years (positive or negative),
    in days. This ensures each patient gets a unique but consistent
    time shift for all their files.

    Returns:
        int: Time offset in days (between 365-1095 or -1095 to -365)
    """
    import random
    # Random offset between 1-3 years (365-1095 days)
    # This gives us a truly random value, e.g., 1 year 2 months, 2 years 7 months, etc.
    days = random.randint(365, 1095)
    # Random sign (positive or negative)
    if random.random() < 0.5:
        days = -days
    return days


def get_processor(
    file_path: Path,
    config: AnonymizerConfig,
    use_llm_detection: bool = False,
    time_offset_days: int = None
):
    """
    Get appropriate agentic processor for the file type.

    Args:
        file_path: Path to the file
        config: Anonymizer configuration
        use_llm_detection: If True, use multimodal LLM to detect file type and choose processor
        time_offset_days: Optional time offset in days for date shifting. If None, processor
                          will generate its own random offset.

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
                processor = AgenticTextProcessor(config, time_offset_days=time_offset_days)
                if processor.can_process(file_path):
                    print(f"Using Agentic Text processor based on LLM detection")
                    return processor
            elif detection_result.suggested_processor == "csv":
                processor = AgenticCSVProcessor(config, time_offset_days=time_offset_days)
                if processor.can_process(file_path):
                    print(f"Using Agentic CSV processor based on LLM detection")
                    return processor

            # Fallback: try both processors
            text_processor = AgenticTextProcessor(config, time_offset_days=time_offset_days)
            if text_processor.can_process(file_path):
                print(f"Using Agentic Text processor (fallback)")
                return text_processor

            csv_processor = AgenticCSVProcessor(config, time_offset_days=time_offset_days)
            if csv_processor.can_process(file_path):
                print(f"Using Agentic CSV processor (fallback)")
                return csv_processor

        elif detection_result.data_type == DataType.IMAGE:
            # Image data -> use Vision OCR processor
            processor = PNGVisionOCRProcessor(config)
            if processor.can_process(file_path):
                print(f"Using Vision+OCR processor based on LLM detection")
                return processor

        # If LLM detection didn't work or type is unknown, fall back to extension-based matching
        print(f"LLM detected type '{detection_result.data_type}' but no suitable processor found, falling back to extension-based matching")

    # Original extension-based processor selection with agentic processors
    processors = [
        DICOMVisionOCRProcessor(config),
        PNGVisionOCRProcessor(config),
        PDFVisionOCRProcessor(config),
        AgenticTextProcessor(config, time_offset_days=time_offset_days),
        AgenticCSVProcessor(config, time_offset_days=time_offset_days),
    ]

    for processor in processors:
        if processor.can_process(file_path):
            return processor

    return None


def process_file(
    input_path: Path,
    output_dir: Path,
    config: AnonymizerConfig,
    use_llm_detection: bool = False,
    preserve_structure: bool = False,
    relative_path: Path = None,
    filename_anonymizer: FilenameAnonymizer = None,
    anonymize_paths: bool = True,
    tracker: ProcessingTracker = None,
    time_offset_days: int = None
) -> bool:
    """
    Process a single file.

    Args:
        input_path: Path to input file
        output_dir: Directory for output
        config: Anonymizer configuration
        use_llm_detection: If True, use multimodal LLM to detect file type
        preserve_structure: If True, preserve directory structure in output
        relative_path: Relative path from input root (used when preserve_structure=True)
        filename_anonymizer: Optional FilenameAnonymizer instance for anonymizing filenames
        anonymize_paths: If True, automatically anonymize filename (default: True)
        tracker: Optional ProcessingTracker instance for tracking processed files
        time_offset_days: Patient-specific time offset in days for date shifting.
                          If None, processor will generate its own random offset.

    Returns:
        True if successful
    """
    # Check if file has already been processed
    if tracker and tracker.is_file_processed(input_path):
        print(f"Skipping already processed file: {input_path.name}")
        return True

    processor = get_processor(input_path, config, use_llm_detection, time_offset_days=time_offset_days)

    if processor is None:
        print(f"No processor available for: {input_path.name}")
        return False

    # Automatically create filename anonymizer if needed and enabled
    if anonymize_paths and filename_anonymizer is None:
        filename_anonymizer = FilenameAnonymizer(config, output_dir=output_dir)

    # Determine folder path for file uniqueness
    if preserve_structure and relative_path:
        folder_path = str(relative_path.parent) if relative_path.parent != Path('.') else ""
    else:
        folder_path = input_path.stem  # Use file stem as folder path for standalone files

    # Anonymize filename if enabled
    if anonymize_paths and filename_anonymizer:
        print(f"Anonymizing filename: {input_path.name}")
        anonymization_result = filename_anonymizer.anonymize_filename(
            input_path.name,
            is_directory=False,
            folder_path=folder_path
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
                folder_path=folder_path,
                original_filename=input_path.name,
                anonymized_filename=anonymized_filename,
                phi_detections=anonymization_result.phi_detections
            )

    try:
        processor.anonymize(input_path, output_path)
        print(f"Output saved to: {output_path}")

        # Mark file as processed in tracker
        if tracker:
            tracker.mark_file_processed(input_path, output_path, success=True)

        # Save CSV mappings if this is a standalone file processing
        if anonymize_paths and filename_anonymizer and not preserve_structure:
            filename_anonymizer.save_all_mappings(output_dir=output_dir)

        return True
    except Exception as e:
        import traceback
        print(f"Error processing {input_path.name}: {e}")
        traceback.print_exc()

        # Mark file as failed in tracker
        if tracker:
            tracker.mark_file_processed(input_path, output_path, success=False)

        return False


def process_directory(
    input_dir: Path,
    output_dir: Path,
    config: AnonymizerConfig,
    use_llm_detection: bool = False,
    recursive: bool = False,
    preserve_structure: bool = False,
    skip_hidden: bool = True,
    anonymize_paths: bool = True,
    tracker: ProcessingTracker = None,
    parallel: bool = True,
    num_workers: int = None
):
    """
    Process all supported files in a directory.

    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        config: Anonymizer configuration
        use_llm_detection: If True, use multimodal LLM to detect file type
        recursive: If True, process subdirectories recursively
        preserve_structure: If True, preserve directory structure in output
        skip_hidden: If True, skip hidden files and directories (starting with '.')
        anonymize_paths: If True, anonymize file and folder names
        tracker: Optional ProcessingTracker instance for tracking processed files
        parallel: If True, use parallel processing
        num_workers: Number of parallel workers
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
        skipped = 0

        for file_path in files:
            print(f"\n{'='*60}")

            # Check if file was already processed
            if tracker and tracker.is_file_processed(file_path):
                print(f"Skipping already processed file: {file_path.name}")
                skipped += 1
                continue

            if process_file(file_path, output_dir, config, use_llm_detection,
                          filename_anonymizer=filename_anonymizer, anonymize_paths=anonymize_paths,
                          tracker=tracker):
                successful += 1
            else:
                failed += 1

        print(f"\n{'='*60}")
        print(f"Processing complete:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Skipped: {skipped}")

        # Save CSV mappings if anonymization was enabled
        if anonymize_paths and filename_anonymizer:
            filename_anonymizer.save_all_mappings(output_dir=output_dir)

        # Save tracker data
        if tracker:
            tracker.save()
    else:
        # Recursive processing with structure preservation
        process_directory_recursive(
            input_dir=input_dir,
            output_dir=output_dir,
            config=config,
            use_llm_detection=use_llm_detection,
            preserve_structure=preserve_structure,
            skip_hidden=skip_hidden,
            anonymize_paths=anonymize_paths,
            tracker=tracker,
            parallel=parallel,
            num_workers=num_workers
        )


def _process_directory_parallel(
    input_dir: Path,
    output_dir: Path,
    config: AnonymizerConfig,
    use_llm_detection: bool = False,
    preserve_structure: bool = True,
    skip_hidden: bool = True,
    anonymize_paths: bool = True,
    tracker: ProcessingTracker = None,
    num_workers: int = None
) -> dict:
    """
    Process directory using parallel processing.

    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        config: Anonymizer configuration
        use_llm_detection: If True, use multimodal LLM to detect file type
        preserve_structure: If True, preserve directory structure in output
        skip_hidden: If True, skip hidden files and directories
        anonymize_paths: If True, anonymize file and folder names
        tracker: Optional ProcessingTracker instance
        num_workers: Number of parallel workers

    Returns:
        Statistics dictionary
    """
    import time

    start_time = time.time()

    # Collect all files first
    print("Collecting files...")
    all_files = collect_files_for_processing(
        input_dir=input_dir,
        skip_hidden=skip_hidden,
        recursive=True
    )

    if not all_files:
        print("No files found to process")
        return {"successful": 0, "failed": 0, "skipped": 0}

    # Filter out already processed files if tracker is enabled
    files_to_process = []
    skipped = 0

    for file_path in all_files:
        if tracker and tracker.is_file_processed(file_path):
            skipped += 1
        else:
            files_to_process.append(file_path)

    if skipped > 0:
        print(f"Skipping {skipped} already processed files")

    if not files_to_process:
        print("All files already processed")
        return {"successful": 0, "failed": 0, "skipped": skipped}

    print(f"Found {len(files_to_process)} files to process")

    # Initialize filename anonymizer
    filename_anonymizer = None
    folder_mapping = {}

    if anonymize_paths:
        filename_anonymizer = FilenameAnonymizer(config, output_dir=output_dir)

        # Pre-anonymize all folder names (skip already anonymized ones)
        print("Anonymizing folder names...")
        unique_folders = set()
        for file_path in files_to_process:
            relative_path = file_path.relative_to(input_dir)
            # Get all folder names in the path (excluding the filename)
            for part in relative_path.parent.parts:
                unique_folders.add(part)

        already_anonymized_folders = 0
        newly_anonymized_folders = 0
        for folder_name in sorted(unique_folders):
            # Check if folder was already anonymized
            existing_mapping = filename_anonymizer.get_existing_folder_mapping(folder_name)
            if existing_mapping:
                folder_mapping[folder_name] = existing_mapping
                already_anonymized_folders += 1
            else:
                result = filename_anonymizer.anonymize_filename(folder_name, is_directory=True)
                folder_mapping[folder_name] = result.anonymized_filename
                filename_anonymizer.add_folder_mapping(
                    original_foldername=folder_name,
                    anonymized_foldername=result.anonymized_filename,
                    phi_detections=result.phi_detections
                )
                newly_anonymized_folders += 1

        if already_anonymized_folders > 0:
            print(f"  Skipped {already_anonymized_folders} already anonymized folders")
        if newly_anonymized_folders > 0:
            print(f"  Anonymized {newly_anonymized_folders} new folders")

    # Generate patient-specific time offsets
    # Each top-level folder (patient folder) gets a unique random time offset
    print("Generating patient-specific time offsets (1-3 years)...")
    patient_time_offsets = {}
    for file_path in files_to_process:
        relative_path = file_path.relative_to(input_dir)
        # Get the top-level folder (patient folder)
        if relative_path.parent.parts:
            patient_folder = relative_path.parent.parts[0]
        else:
            # File is directly in input_dir, use filename stem as "patient"
            patient_folder = file_path.stem

        if patient_folder not in patient_time_offsets:
            patient_time_offsets[patient_folder] = generate_patient_time_offset()
            print(f"  Patient '{patient_folder}': {patient_time_offsets[patient_folder]} days ({patient_time_offsets[patient_folder] // 365} years)")

    print(f"Generated time offsets for {len(patient_time_offsets)} patients")

    # Create parallel processor
    parallel_processor = ParallelFileProcessor(
        config=config,
        num_workers=num_workers,
        use_ocr=False,  # Not used for agentic processors
        use_llm_detection=use_llm_detection,
        preserve_structure=preserve_structure,
        anonymize_paths=anonymize_paths,
        processor_module='anonymize_agentic',
        max_retries=3  # Initial retry count per file
    )

    # Create jobs for all files
    jobs = []
    job_lookup = {}  # Map file path to job data for retries
    total_files = len(files_to_process)
    print(f"Anonymizing {total_files} filenames (this may take a while with LLM-based anonymization)...")

    already_anonymized_files = 0
    newly_anonymized_files = 0

    for i, file_path in enumerate(files_to_process):
        relative_path = file_path.relative_to(input_dir)

        # Build anonymized relative path (all folder parts, not the filename)
        if anonymize_paths and folder_mapping:
            anonymized_parts = []
            for part in relative_path.parent.parts:  # All folder parts
                anonymized_parts.append(folder_mapping.get(part, part))
            anonymized_relative_path = Path(*anonymized_parts) if anonymized_parts else Path('.')
        else:
            anonymized_relative_path = relative_path.parent

        # Anonymize filename
        if anonymize_paths:
            # Use original folder_path for filename uniqueness
            original_folder_path = str(relative_path.parent) if relative_path.parent != Path('.') else ""
            # Use anonymized folder_path for CSV output location
            anonymized_folder_path = str(anonymized_relative_path) if anonymized_relative_path != Path('.') else ""

            # Check if file was already anonymized
            existing_file_mapping = filename_anonymizer.get_existing_file_mapping(
                anonymized_folder_path, file_path.name
            )
            if existing_file_mapping:
                anonymized_filename = existing_file_mapping
                already_anonymized_files += 1
                print(f"  [{i+1}/{total_files}] Already anonymized: {file_path.name} -> {anonymized_filename}")
            else:
                print(f"  [{i+1}/{total_files}] Anonymizing: {file_path.name}", end=" ", flush=True)
                anonymization_result = filename_anonymizer.anonymize_filename(
                    file_path.name,
                    is_directory=False,
                    folder_path=original_folder_path
                )
                anonymized_filename = anonymization_result.anonymized_filename
                print(f"-> {anonymized_filename}")
                newly_anonymized_files += 1

                # Record file mapping using anonymized folder path for correct output location
                filename_anonymizer.add_file_mapping(
                    folder_path=anonymized_folder_path,
                    original_filename=file_path.name,
                    anonymized_filename=anonymized_filename,
                    phi_detections=anonymization_result.phi_detections
                )
        else:
            anonymized_filename = f"anonymized_{file_path.name}"

        # Get patient-specific time offset
        if relative_path.parent.parts:
            patient_folder = relative_path.parent.parts[0]
        else:
            patient_folder = file_path.stem
        time_offset = patient_time_offsets.get(patient_folder)

        job = parallel_processor.create_job(
            input_path=file_path,
            output_dir=output_dir,
            relative_path=relative_path,
            anonymized_relative_path=anonymized_relative_path,
            anonymized_filename=anonymized_filename,
            folder_path=str(relative_path.parent) if relative_path.parent != Path('.') else "",
            time_offset_days=time_offset
        )
        jobs.append(job)
        # Also store job data for potential retry
        job_lookup[str(file_path)] = job

    # Print filename anonymization summary
    if anonymize_paths:
        print(f"\nFilename anonymization summary:")
        print(f"  Already anonymized: {already_anonymized_files}")
        print(f"  Newly anonymized: {newly_anonymized_files}")

    # Process all files in parallel
    results = parallel_processor.process_files_parallel(jobs, show_progress=True)

    # Update tracker with results and collect failed files
    successful = 0
    failed = 0
    retryable_failed = []

    for result in results:
        if result.success:
            successful += 1
            if tracker:
                tracker.mark_file_processed(result.input_path, result.output_path, success=True)
        else:
            failed += 1
            if tracker:
                tracker.mark_file_processed(
                    result.input_path, 
                    result.output_path, 
                    success=False,
                    error=result.error,
                    is_retryable=result.is_retryable_error,
                    retries_attempted=result.retries_attempted
                )
            if result.is_retryable_error:
                retryable_failed.append(result.input_path)
            print(f"\nError processing {result.input_path.name}:")
            if result.error:
                # Print first few lines of error
                error_lines = result.error.split('\n')[:3]
                for line in error_lines:
                    print(f"  {line}")

    # Retry loop for retryable failures
    max_global_retries = 3
    retry_round = 0
    
    while retryable_failed and retry_round < max_global_retries:
        retry_round += 1
        print(f"\n{'='*60}")
        print(f"RETRY ROUND {retry_round}/{max_global_retries}: {len(retryable_failed)} files to retry")
        print(f"{'='*60}")
        
        # Wait a bit before retrying to allow rate limits to reset
        wait_time = 30 * retry_round  # Progressive wait: 30s, 60s, 90s
        print(f"Waiting {wait_time} seconds before retry...")
        time.sleep(wait_time)
        
        # Create retry jobs with increased max_retries
        retry_jobs = []
        for file_path in retryable_failed:
            job_data = job_lookup.get(str(file_path))
            if job_data:
                # Increase max_retries for this retry round
                job_data['max_retries'] = 5
                retry_jobs.append(job_data)
        
        # Clear failed status from tracker to allow reprocessing
        for file_path in retryable_failed:
            if tracker:
                tracker.clear_file(file_path)
        
        # Process retries
        retry_results = parallel_processor.process_files_parallel(retry_jobs, show_progress=True)
        
        # Update counts and tracker
        new_retryable_failed = []
        for result in retry_results:
            if result.success:
                successful += 1
                failed -= 1
                if tracker:
                    tracker.mark_file_processed(result.input_path, result.output_path, success=True)
                print(f"  ✓ Successfully processed on retry: {result.input_path.name}")
            else:
                if tracker:
                    tracker.mark_file_processed(
                        result.input_path, 
                        result.output_path, 
                        success=False,
                        error=result.error,
                        is_retryable=result.is_retryable_error,
                        retries_attempted=result.retries_attempted
                    )
                if result.is_retryable_error:
                    new_retryable_failed.append(result.input_path)
                else:
                    print(f"  ✗ Non-retryable error for: {result.input_path.name}")
        
        retryable_failed = new_retryable_failed
        
        if not retryable_failed:
            print(f"\n✓ All retryable files have been successfully processed!")
            break
    
    if retryable_failed:
        print(f"\n{'='*60}")
        print(f"WARNING: {len(retryable_failed)} files still failed after {max_global_retries} retry rounds:")
        for fp in retryable_failed:
            print(f"  - {fp.name}")
        print(f"\nRun the command again to retry these files.")

    # Save mappings
    if anonymize_paths and filename_anonymizer:
        filename_anonymizer.save_all_mappings(output_dir=output_dir)

    # Save tracker
    if tracker:
        tracker.save()
        tracker.print_stats()

    elapsed_time = time.time() - start_time

    # Print summary
    print(f"\n{'='*60}")
    print(f"Processing complete in {elapsed_time:.2f} seconds")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Total processed: {successful + failed}")
    if len(files_to_process) > 0:
        print(f"  Average time per file: {elapsed_time / len(files_to_process):.2f}s")

    # Final verification message
    if failed == 0:
        print(f"\n✅ SUCCESS: All {successful} files have been processed successfully!")
    elif len(retryable_failed) == 0 and failed > 0:
        print(f"\n⚠️  COMPLETED WITH ERRORS: {successful} succeeded, {failed} failed (non-retryable)")
    else:
        print(f"\n⚠️  INCOMPLETE: {successful} succeeded, {len(retryable_failed)} retryable failures remaining")
        print(f"   Run with --retry-failed to retry failed files")

    return {"successful": successful, "failed": failed, "skipped": skipped, "retryable_remaining": len(retryable_failed)}


def process_directory_recursive(
    input_dir: Path,
    output_dir: Path,
    config: AnonymizerConfig,
    use_llm_detection: bool = False,
    preserve_structure: bool = True,
    skip_hidden: bool = True,
    anonymize_paths: bool = True,
    tracker: ProcessingTracker = None,
    _root_dir: Path = None,
    _stats: dict = None,
    _filename_anonymizer: FilenameAnonymizer = None,
    _folder_mapping: dict = None,
    _patient_time_offsets: dict = None,
    parallel: bool = True,
    num_workers: int = None
):
    """
    Recursively process all files in a directory tree, preserving structure.

    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        config: Anonymizer configuration
        use_llm_detection: If True, use multimodal LLM to detect file type
        preserve_structure: If True, preserve directory structure in output
        skip_hidden: If True, skip hidden files and directories
        anonymize_paths: If True, anonymize file and folder names
        tracker: Optional ProcessingTracker instance for tracking processed files
        _root_dir: Internal parameter for tracking root directory
        _stats: Internal parameter for tracking statistics
        _filename_anonymizer: Internal parameter for filename anonymization
        _folder_mapping: Internal parameter for tracking folder name mappings
        _patient_time_offsets: Internal parameter for tracking patient-specific time offsets
        parallel: If True, use parallel processing
        num_workers: Number of parallel workers
    """
    # Initialize on first call
    is_root_call = _root_dir is None
    if is_root_call:
        _root_dir = input_dir
        _stats = {"successful": 0, "failed": 0, "skipped": 0}
        _folder_mapping = {}
        _patient_time_offsets = {}
        if anonymize_paths:
            _filename_anonymizer = FilenameAnonymizer(config, output_dir=output_dir)
        print(f"Starting recursive directory processing...")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Structure preservation: {preserve_structure}")
        print(f"Skip hidden files: {skip_hidden}")
        print(f"Anonymize paths: {anonymize_paths}")
        if tracker:
            print(f"Tracking: enabled")

        # Use parallel processing if enabled
        if parallel:
            print(f"Parallel processing: enabled")
            return _process_directory_parallel(
                input_dir=input_dir,
                output_dir=output_dir,
                config=config,
                use_llm_detection=use_llm_detection,
                preserve_structure=preserve_structure,
                skip_hidden=skip_hidden,
                anonymize_paths=anonymize_paths,
                tracker=tracker,
                num_workers=num_workers
            )

        print()

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

            # Check if file was already processed
            if tracker and tracker.is_file_processed(item):
                print(f"Skipping already processed file: {item.name}")
                _stats["skipped"] += 1
                continue

            # Get patient-specific time offset (top-level folder is the patient)
            if original_relative_path.parent.parts:
                patient_folder = original_relative_path.parent.parts[0]
            else:
                patient_folder = item.stem

            if patient_folder not in _patient_time_offsets:
                _patient_time_offsets[patient_folder] = generate_patient_time_offset()
                print(f"  Generated time offset for patient '{patient_folder}': {_patient_time_offsets[patient_folder]} days ({_patient_time_offsets[patient_folder] // 365} years)")

            time_offset = _patient_time_offsets[patient_folder]

            success = process_file(
                input_path=item,
                output_dir=output_dir,
                config=config,
                use_llm_detection=use_llm_detection,
                preserve_structure=preserve_structure,
                relative_path=original_relative_path if not anonymize_paths else anonymized_relative_path / item.name,
                filename_anonymizer=_filename_anonymizer,
                anonymize_paths=anonymize_paths,
                tracker=tracker,
                time_offset_days=time_offset
            )

            if success:
                _stats["successful"] += 1
            else:
                _stats["failed"] += 1

            # Save tracker after each file to ensure progress is not lost
            if tracker:
                tracker.save()

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
                use_llm_detection=use_llm_detection,
                preserve_structure=preserve_structure,
                skip_hidden=skip_hidden,
                anonymize_paths=anonymize_paths,
                tracker=tracker,
                _root_dir=_root_dir,
                _stats=_stats,
                _filename_anonymizer=_filename_anonymizer,
                _folder_mapping=_folder_mapping,
                _patient_time_offsets=_patient_time_offsets
            )

    # Print summary and save mappings only on initial call
    if is_root_call:
        print(f"\n{'='*60}")
        print(f"Processing complete:")
        print(f"  Successful: {_stats['successful']}")
        print(f"  Failed: {_stats['failed']}")
        print(f"  Skipped: {_stats['skipped']}")
        print(f"  Total processed: {_stats['successful'] + _stats['failed']}")

        # Save CSV mappings if anonymization was enabled
        if anonymize_paths and _filename_anonymizer:
            _filename_anonymizer.save_all_mappings(output_dir=output_dir)

        # Save tracker data (final save)
        if tracker:
            tracker.save()
            tracker.print_stats()

    return _stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Anonymize files using agentic/vision-based LLM processors"
    )
    parser.add_argument(
        "input", type=str, help="Input file or directory path"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="data/output-agentic",
        help="Output directory (default: data/output-agentic)"
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
    parser.add_argument(
        "--tracking-file", "-t", type=str, default=None,
        help="Path to tracking file (JSON) for skipping already processed files. If not specified, tracking is disabled."
    )
    parser.add_argument(
        "--no-hash", action="store_true",
        help="Disable file hash computation in tracking (faster but won't detect file modifications)"
    )
    parser.add_argument(
        "--clear-tracking", action="store_true",
        help="Clear all tracking data before processing"
    )
    parser.add_argument(
        "--no-parallel", action="store_true",
        help="Disable parallel processing (process files sequentially). By default, parallel processing is enabled."
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=None,
        help="Number of parallel workers (default: CPU count - 1)"
    )
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="Only retry previously failed files from the tracking file (requires --tracking-file)"
    )
    parser.add_argument(
        "--max-retries", type=int, default=3,
        help="Maximum number of retries per file for transient errors (default: 3)"
    )
    parser.add_argument(
        "--retry-rounds", type=int, default=3,
        help="Maximum number of global retry rounds for failed files at the end (default: 3)"
    )
    parser.add_argument(
        "--provider", type=str, choices=["azure", "fireworks"], default=None,
        help="LLM provider to use (azure or fireworks). Overrides LLM_PROVIDER environment variable."
    )

    args = parser.parse_args()

    # Invert the flag - default is to anonymize paths
    anonymize_paths = not args.no_anonymize_paths
    # Invert the flag - default is to use parallel processing
    parallel = not args.no_parallel

    # Create config with optional provider override
    config_kwargs = {
        "output_dir": args.output,
        "save_debug_files": args.debug,
    }
    if args.provider:
        config_kwargs["llm_provider"] = args.provider

    config = AnonymizerConfig(**config_kwargs)

    input_path = Path(args.input)
    output_dir = Path(args.output)
    use_llm_detection = args.auto_detect
    num_workers = args.workers

    # Initialize tracker if tracking file is specified
    tracker = None
    if args.tracking_file:
        tracking_file_path = Path(args.tracking_file)
        compute_hashes = not args.no_hash
        tracker = ProcessingTracker(tracking_file_path, compute_hashes=compute_hashes)

        if args.clear_tracking:
            print("Clearing all tracking data...")
            tracker.clear_all()
            tracker.save()
            print("Tracking data cleared.\n")

    # Handle --retry-failed mode
    if args.retry_failed:
        if not tracker:
            print("Error: --retry-failed requires --tracking-file to be specified")
            return
        
        failed_files = tracker.get_failed_files()
        if not failed_files:
            print("No failed files found in tracking data.")
            return
        
        print(f"\n{'='*60}")
        print(f"RETRY MODE: Processing {len(failed_files)} previously failed files")
        print(f"{'='*60}\n")
        
        # Clear failed status to allow reprocessing
        tracker.clear_failed_files()
        tracker.save()
        
        # Show the files to be retried
        for fp in failed_files[:10]:
            print(f"  - {fp.name}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")
        print()

    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return

    print(f"LLM Provider: {config.llm_provider}")
    if config.llm_provider == "azure":
        print(f"  Model: {config.azure_deployment_name}")
    else:
        print(f"  Model: {config.fireworks_model}")
        print(f"  Vision Model: {config.fireworks_vision_model}")
    print()
    print("Using AGENTIC/VISION-BASED processors:")
    print("  - AgenticCSVProcessor (tool-calling approach)")
    print("  - AgenticTextProcessor (tool-calling approach)")
    print("  - DICOMVisionOCRProcessor (Vision LLM + OCR)")
    print("  - PDFVisionOCRProcessor (Vision LLM + OCR)")
    print("  - PNGVisionOCRProcessor (Vision LLM + OCR)")
    print()

    if use_llm_detection:
        print(f"Using automatic file type detection with multimodal LLM")

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

    if tracker:
        print(f"Tracking: enabled (tracking file: {args.tracking_file})")
        print(f"  Hash computation: {'disabled' if args.no_hash else 'enabled'}")
        # Show existing tracking stats if available
        stats = tracker.get_stats()
        if stats['total_files'] > 0 or stats['total_directories'] > 0:
            print(f"  Already tracked: {stats['completed_files']} files, {stats['completed_directories']} directories")
    else:
        print(f"Tracking: disabled")

    if parallel:
        import multiprocessing as mp
        worker_count = num_workers if num_workers else max(1, mp.cpu_count() - 1)
        print(f"Parallel processing: enabled (using {worker_count} workers)")
    else:
        print(f"Parallel processing: disabled")

    print()

    if input_path.is_file():
        # For single file, process with automatic filename anonymization
        process_file(input_path, output_dir, config, use_llm_detection,
                    anonymize_paths=anonymize_paths, tracker=tracker)
    elif input_path.is_dir():
        process_directory(
            input_path,
            output_dir,
            config,
            use_llm_detection,
            recursive=args.recursive,
            preserve_structure=args.preserve_structure,
            skip_hidden=not args.include_hidden,
            anonymize_paths=anonymize_paths,
            tracker=tracker,
            parallel=parallel,
            num_workers=num_workers
        )
    else:
        print(f"Error: Invalid input path: {input_path}")


if __name__ == "__main__":
    main()
