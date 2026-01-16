#!/usr/bin/env python3
"""
Parallel file processing using multiprocessing for faster anonymization.
"""

import multiprocessing as mp
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import time


@dataclass
class ProcessingJob:
    """Represents a single file processing job."""
    input_path: Path
    output_dir: Path
    relative_path: Optional[Path] = None
    anonymized_relative_path: Optional[Path] = None
    time_offset_days: Optional[int] = None  # Patient-specific time offset


@dataclass
class ProcessingResult:
    """Result of processing a single file."""
    input_path: Path
    output_path: Optional[Path]
    success: bool
    error: Optional[str] = None
    processing_time: float = 0.0


def _process_file_worker(job_data: Dict[str, Any]) -> ProcessingResult:
    """
    Worker function to process a single file.
    This runs in a separate process.

    Args:
        job_data: Dictionary containing all necessary data for processing

    Returns:
        ProcessingResult with outcome
    """
    from anonymizer import AnonymizerConfig
    from anonymizer.filename_anonymizer import FilenameAnonymizer
    import traceback

    start_time = time.time()

    try:
        # Extract job data
        input_path = Path(job_data['input_path'])
        output_dir = Path(job_data['output_dir'])
        config_dict = job_data['config']
        use_ocr = job_data['use_ocr']
        use_llm_detection = job_data['use_llm_detection']
        preserve_structure = job_data['preserve_structure']
        relative_path = Path(job_data['relative_path']) if job_data['relative_path'] else None
        anonymized_relative_path = Path(job_data['anonymized_relative_path']) if job_data['anonymized_relative_path'] else None
        anonymize_paths = job_data['anonymize_paths']
        anonymized_filename = job_data.get('anonymized_filename')
        folder_path = job_data.get('folder_path', '')
        time_offset_days = job_data.get('time_offset_days')  # Patient-specific time offset

        # Recreate config from dict
        config = AnonymizerConfig(
            output_dir=config_dict['output_dir'],
            save_debug_files=config_dict['save_debug_files']
        )

        # Import get_processor from the appropriate module
        processor_module = job_data.get('processor_module', 'anonymize')
        if processor_module == 'anonymize_agentic':
            from anonymize_agentic import get_processor
            processor = get_processor(input_path, config, use_llm_detection, time_offset_days=time_offset_days)
        else:
            from anonymize import get_processor
            processor = get_processor(input_path, config, use_ocr, use_llm_detection)

        if processor is None:
            return ProcessingResult(
                input_path=input_path,
                output_path=None,
                success=False,
                error=f"No processor available for: {input_path.name}",
                processing_time=time.time() - start_time
            )

        # Determine output path
        if preserve_structure and anonymized_relative_path:
            file_output_dir = output_dir / anonymized_relative_path
            file_output_dir.mkdir(parents=True, exist_ok=True)
            output_path = file_output_dir / anonymized_filename
        else:
            # Create separate output folder for this file
            file_stem = input_path.stem
            file_output_dir = output_dir / file_stem
            file_output_dir.mkdir(parents=True, exist_ok=True)
            output_path = file_output_dir / anonymized_filename

        # Process the file
        processor.anonymize(input_path, output_path)

        processing_time = time.time() - start_time

        return ProcessingResult(
            input_path=input_path,
            output_path=output_path,
            success=True,
            processing_time=processing_time
        )

    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"{str(e)}\n{traceback.format_exc()}"

        return ProcessingResult(
            input_path=Path(job_data['input_path']),
            output_path=None,
            success=False,
            error=error_msg,
            processing_time=processing_time
        )


class ParallelFileProcessor:
    """
    Manages parallel processing of files using multiprocessing.
    """

    def __init__(
        self,
        config: Any,
        num_workers: Optional[int] = None,
        use_ocr: bool = False,
        use_llm_detection: bool = False,
        preserve_structure: bool = True,
        anonymize_paths: bool = True,
        processor_module: str = 'anonymize'
    ):
        """
        Initialize parallel processor.

        Args:
            config: AnonymizerConfig instance
            num_workers: Number of worker processes (default: CPU count - 1)
            use_ocr: If True, use OCR-based processor
            use_llm_detection: If True, use multimodal LLM to detect file type
            preserve_structure: If True, preserve directory structure in output
            anonymize_paths: If True, anonymize file and folder names
            processor_module: Module to import get_processor from ('anonymize' or 'anonymize_agentic')
        """
        self.config = config
        self.use_ocr = use_ocr
        self.use_llm_detection = use_llm_detection
        self.preserve_structure = preserve_structure
        self.anonymize_paths = anonymize_paths
        self.processor_module = processor_module

        # Determine number of workers
        if num_workers is None:
            # Use CPU count - 1, minimum 1
            cpu_count = mp.cpu_count()
            self.num_workers = max(1, cpu_count - 1)
        else:
            self.num_workers = max(1, num_workers)

    def process_files_parallel(
        self,
        jobs: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[ProcessingResult]:
        """
        Process multiple files in parallel.

        Args:
            jobs: List of job dictionaries containing processing parameters
            show_progress: If True, show progress updates

        Returns:
            List of ProcessingResult objects
        """
        if not jobs:
            return []

        total_jobs = len(jobs)

        if show_progress:
            print(f"\nProcessing {total_jobs} files using {self.num_workers} workers...")
            print(f"{'='*60}")

        results = []
        completed = 0

        # Create process pool
        with mp.Pool(processes=self.num_workers) as pool:
            # Submit all jobs and get iterator
            for result in pool.imap_unordered(_process_file_worker, jobs):
                completed += 1
                results.append(result)

                if show_progress:
                    # Show progress
                    if result.success:
                        status = "✓"
                        msg = f"Completed: {result.input_path.name}"
                    else:
                        status = "✗"
                        msg = f"Failed: {result.input_path.name}"
                        if result.error:
                            # Show first line of error only
                            error_line = result.error.split('\n')[0]
                            msg += f" - {error_line[:60]}"

                    print(f"[{completed}/{total_jobs}] {status} {msg}")

        return results

    def create_job(
        self,
        input_path: Path,
        output_dir: Path,
        relative_path: Optional[Path] = None,
        anonymized_relative_path: Optional[Path] = None,
        anonymized_filename: Optional[str] = None,
        folder_path: str = '',
        time_offset_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create a job dictionary for parallel processing.

        Args:
            input_path: Path to input file
            output_dir: Directory for output
            relative_path: Relative path from input root
            anonymized_relative_path: Anonymized relative path
            anonymized_filename: Pre-computed anonymized filename
            folder_path: Folder path for uniqueness
            time_offset_days: Patient-specific time offset in days for date shifting

        Returns:
            Job dictionary
        """
        # Convert config to dictionary for pickling
        config_dict = {
            'output_dir': str(self.config.output_dir),
            'save_debug_files': self.config.save_debug_files
        }

        return {
            'input_path': str(input_path),
            'output_dir': str(output_dir),
            'config': config_dict,
            'use_ocr': self.use_ocr,
            'use_llm_detection': self.use_llm_detection,
            'preserve_structure': self.preserve_structure,
            'relative_path': str(relative_path) if relative_path else None,
            'anonymized_relative_path': str(anonymized_relative_path) if anonymized_relative_path else None,
            'anonymize_paths': self.anonymize_paths,
            'anonymized_filename': anonymized_filename,
            'folder_path': folder_path,
            'processor_module': self.processor_module,
            'time_offset_days': time_offset_days
        }


def collect_files_for_processing(
    input_dir: Path,
    skip_hidden: bool = True,
    recursive: bool = True
) -> List[Path]:
    """
    Collect all files to be processed from a directory.

    Args:
        input_dir: Input directory path
        skip_hidden: If True, skip hidden files and directories
        recursive: If True, process subdirectories recursively

    Returns:
        List of file paths to process
    """
    files = []

    if recursive:
        # Use rglob for recursive search
        pattern = '**/*'
        for item in input_dir.rglob('*'):
            if item.is_file():
                # Skip hidden files if requested
                if skip_hidden and any(part.startswith('.') for part in item.parts):
                    continue
                files.append(item)
    else:
        # Only files in current directory
        for item in input_dir.iterdir():
            if item.is_file():
                if skip_hidden and item.name.startswith('.'):
                    continue
                files.append(item)

    return sorted(files)
