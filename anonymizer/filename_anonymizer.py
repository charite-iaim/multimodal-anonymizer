"""
Filename and folder name anonymizer with PII detection and reversible mapping.
"""

import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from .config import AnonymizerConfig


class PHIDetection(BaseModel):
    """A detected PHI value in a filename."""
    original_value: str = Field(description="Original PHI value (e.g., '10045929', 'John_Doe')")
    category: str = Field(description="Category of PHI (PERSON, DATE, ID, LOCATION, etc.)")


class FilenameAnonymizationResult(BaseModel):
    """Result of filename PII detection and anonymization."""
    original_filename: str = Field(description="Original filename")
    anonymized_filename: str = Field(description="Anonymized filename with generic placeholders (e.g., hosp_admissions_ID.csv)")
    phi_detections: List[PHIDetection] = Field(
        default_factory=list,
        description="List of PHI values detected and their categories"
    )


class FileMapping:
    """Simple file mapping for CSV export."""
    def __init__(self, original_name: str, anonymized_name: str, phi_values: str, phi_categories: str):
        self.original_name = original_name
        self.anonymized_name = anonymized_name
        self.phi_values = phi_values
        self.phi_categories = phi_categories


class FolderMapping:
    """Simple folder mapping for CSV export."""
    def __init__(self, original_name: str, anonymized_name: str, phi_values: str, phi_categories: str):
        self.original_name = original_name
        self.anonymized_name = anonymized_name
        self.phi_values = phi_values
        self.phi_categories = phi_categories


class FilenameAnonymizer:
    """
    Anonymizes filenames and folder names by detecting PII and replacing with sequential IDs.
    Maintains a reversible mapping for evaluation purposes.

    Context-aware: Ensures that files within the same patient folder use consistent IDs.
    """

    def __init__(self, config: AnonymizerConfig, output_dir: Path = None):
        """Initialize filename anonymizer.

        Args:
            config: Anonymizer configuration
            output_dir: Output directory for tracking files (optional, for immediate tracking)
        """
        self.config = config
        self.output_dir = output_dir

        # Initialize LLM for PII detection in filenames
        self.llm = AzureChatOpenAI(
            azure_deployment=config.azure_deployment_name,
            azure_endpoint=config.azure_endpoint,
            api_key=config.azure_api_key,
            api_version=config.azure_api_version,
            temperature=0.0,  # Use deterministic output for consistency
        ).with_structured_output(FilenameAnonymizationResult)

        # Store file mappings by folder for CSV export
        # Key: folder path (e.g., "patient_ID_ID/csv"), Value: list of FileMappings
        self.file_mappings_by_folder: Dict[str, List[FileMapping]] = defaultdict(list)

        # Store folder mappings for CSV export
        self.folder_mappings: List[FolderMapping] = []

        # Counter for folder name sequences
        # Key: anonymized folder name (e.g., "patient_ID_ID"), Value: counter
        self.folder_counters: Dict[str, int] = defaultdict(int)

        # Track which CSV files have been initialized with headers
        self._initialized_csv_files: set = set()

        # Initialize folder mapping CSV if output_dir is provided
        if self.output_dir:
            self._initialize_folder_csv()

    def _initialize_folder_csv(self) -> None:
        """Initialize the folder mapping CSV file with headers."""
        if not self.output_dir:
            return

        csv_path = self.output_dir / "folder_anonymization.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Only write header if file doesn't exist
        if not csv_path.exists():
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['original_foldername', 'anonymized_foldername', 'phi_values', 'phi_categories'])

    def _initialize_file_csv(self, folder_path: str) -> None:
        """Initialize a file mapping CSV file with headers for a specific folder.

        Args:
            folder_path: Path to the folder (e.g., "patient_ID_ID_001/csv")
        """
        if not self.output_dir:
            return

        csv_path = self.output_dir / folder_path / "filename_anonymization.csv"

        # Check if already initialized
        if str(csv_path) in self._initialized_csv_files:
            return

        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Only write header if file doesn't exist
        if not csv_path.exists():
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['original_filename', 'anonymized_filename', 'phi_values', 'phi_categories'])

        self._initialized_csv_files.add(str(csv_path))

    def get_sequential_folder_name(self, base_anonymized_name: str) -> str:
        """
        Get a sequential folder name with a counter suffix.

        Args:
            base_anonymized_name: Base anonymized name (e.g., "patient_ID_ID")

        Returns:
            Sequential folder name (e.g., "patient_ID_ID_001", "patient_ID_ID_002")
        """
        self.folder_counters[base_anonymized_name] += 1
        counter = self.folder_counters[base_anonymized_name]
        return f"{base_anonymized_name}_{counter:03d}"

    def anonymize_filename(self, filename: str, is_directory: bool = False) -> FilenameAnonymizationResult:
        """
        Detect PII in filename and replace with generic category placeholders.
        For directories, adds a sequential counter suffix (e.g., patient_ID_ID_001).

        Args:
            filename: Original filename (without path)
            is_directory: Whether this is a directory name

        Returns:
            FilenameAnonymizationResult with original filename, anonymized filename, and PHI detections
        """
        # Prepare the prompt for PII detection in filename
        prompt = f"""You are a PHI (Protected Health Information) detection expert specializing in medical record filenames. Analyze the following {'folder' if is_directory else 'file'} name and create a simple anonymized version.

{'Folder' if is_directory else 'File'} name: {filename}

Task: Replace ALL PHI (Protected Health Information) with generic category placeholders.

Common PHI categories:
- ID: Replace all patient IDs, medical record numbers, admission IDs with "ID"
  * Numeric sequences of 6+ digits are typically IDs (e.g., "20010003", "10005749")
- PERSON: Replace all names with "PERSON"
- DATE: Replace all dates with "DATE"
- LOCATION: Replace locations with "LOCATION"

Rules:
- Replace PHI values with ONLY the category name (e.g., "ID", "PERSON", "DATE")
- Keep file extensions (.csv, .pdf, etc.)
- Keep underscores, hyphens, and separators
- Keep medical terms like "hosp", "ecg", "admissions", etc.
- List all detected PHI values in phi_detections

Examples:

Input: "ecg_Noah_Rhodes_PID-183667_20130725.pdf"
Output:
- anonymized_filename: "ecg_PERSON_ID_DATE.pdf"
- phi_detections: [
    {{"original_value": "Noah_Rhodes", "category": "PERSON"}},
    {{"original_value": "PID-183667", "category": "ID"}},
    {{"original_value": "20130725", "category": "DATE"}}
  ]

Input: "hosp_admissions_20130725.csv"
Output:
- anonymized_filename: "hosp_admissions_ID.csv"
- phi_detections: [{{"original_value": "20130725", "category": "ID"}}]

Input: "patient_10005749_20010003"
Output:
- anonymized_filename: "patient_ID_ID"
- phi_detections: [
    {{"original_value": "10005749", "category": "ID"}},
    {{"original_value": "20010003", "category": "ID"}}
  ]

If no PHI: return original filename with empty phi_detections list.
"""

        try:
            # Call LLM to detect PII in filename and get anonymized version
            message = HumanMessage(content=prompt)
            result = self.llm.invoke([message])

            # For directories containing "ID", add sequential counter suffix
            if is_directory and "ID" in result.anonymized_filename:
                base_anonymized_name = result.anonymized_filename
                sequential_name = self.get_sequential_folder_name(base_anonymized_name)
                anonymized_filename = sequential_name
            else:
                anonymized_filename = result.anonymized_filename

            return FilenameAnonymizationResult(
                original_filename=result.original_filename,
                anonymized_filename=anonymized_filename,
                phi_detections=result.phi_detections
            )

        except Exception as e:
            print(f"Warning: Failed to anonymize filename '{filename}': {e}")
            print("Using fallback: prepending 'anonymized_' to filename")
            # Fallback: just add prefix
            fallback_name = f"anonymized_{filename}"

            # For directories containing "ID", still add sequential counter even in fallback
            if is_directory and "ID" in fallback_name:
                fallback_name = self.get_sequential_folder_name(fallback_name)

            return FilenameAnonymizationResult(
                original_filename=filename,
                anonymized_filename=fallback_name,
                phi_detections=[]
            )

    def add_file_mapping(
        self,
        folder_path: str,
        original_filename: str,
        anonymized_filename: str,
        phi_detections: Optional[List[PHIDetection]] = None
    ) -> None:
        """
        Add a file mapping and immediately write to CSV if output_dir is set.
        Only writes to CSV if the filename actually changed.

        Args:
            folder_path: Path to the folder containing the file (e.g., "patient_ID_ID/csv")
            original_filename: Original filename
            anonymized_filename: Anonymized filename
            phi_detections: PHI values detected in the filename
        """
        # Only add mapping if filename actually changed
        if original_filename == anonymized_filename:
            return

        detections = phi_detections or []
        phi_values = "; ".join([d.original_value for d in detections])
        phi_categories = "; ".join([d.category for d in detections])

        mapping = FileMapping(
            original_name=original_filename,
            anonymized_name=anonymized_filename,
            phi_values=phi_values,
            phi_categories=phi_categories
        )
        self.file_mappings_by_folder[folder_path].append(mapping)

        # Immediately write to CSV if output_dir is set
        if self.output_dir:
            self._append_file_mapping_to_csv(folder_path, mapping)

    def _append_file_mapping_to_csv(self, folder_path: str, mapping: FileMapping) -> None:
        """
        Append a file mapping to the CSV file immediately.

        Args:
            folder_path: Path to the folder containing the file
            mapping: File mapping to append
        """
        if not self.output_dir:
            return

        # Initialize CSV file if needed
        self._initialize_file_csv(folder_path)

        csv_path = self.output_dir / folder_path / "filename_anonymization.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Append the mapping
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                mapping.original_name,
                mapping.anonymized_name,
                mapping.phi_values,
                mapping.phi_categories
            ])

    def _append_folder_mapping_to_csv(self, mapping: FolderMapping) -> None:
        """
        Append a folder mapping to the CSV file immediately.

        Args:
            mapping: Folder mapping to append
        """
        if not self.output_dir:
            return

        csv_path = self.output_dir / "folder_anonymization.csv"

        # Append the mapping
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                mapping.original_name,
                mapping.anonymized_name,
                mapping.phi_values,
                mapping.phi_categories
            ])

    def add_folder_mapping(
        self,
        original_foldername: str,
        anonymized_foldername: str,
        phi_detections: Optional[List[PHIDetection]] = None
    ) -> None:
        """
        Add a folder mapping and immediately write to CSV if output_dir is set.
        Only writes to CSV if the folder name actually changed.

        Args:
            original_foldername: Original folder name
            anonymized_foldername: Anonymized folder name
            phi_detections: PHI values detected in the folder name
        """
        # Only add mapping if folder name actually changed
        if original_foldername == anonymized_foldername:
            return

        detections = phi_detections or []
        phi_values = "; ".join([d.original_value for d in detections])
        phi_categories = "; ".join([d.category for d in detections])

        mapping = FolderMapping(
            original_name=original_foldername,
            anonymized_name=anonymized_foldername,
            phi_values=phi_values,
            phi_categories=phi_categories
        )
        self.folder_mappings.append(mapping)

        # Immediately write to CSV if output_dir is set
        if self.output_dir:
            self._append_folder_mapping_to_csv(mapping)

    def save_file_mappings_csv(self, output_dir: Path) -> None:
        """
        Save file mappings to CSV files (one per folder).

        Args:
            output_dir: Root output directory where CSV files will be saved
        """
        for folder_path, mappings in self.file_mappings_by_folder.items():
            # Create CSV in the folder where the files are
            csv_path = output_dir / folder_path / "filename_anonymization.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)

            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['original_filename', 'anonymized_filename', 'phi_values', 'phi_categories'])

                for mapping in mappings:
                    writer.writerow([
                        mapping.original_name,
                        mapping.anonymized_name,
                        mapping.phi_values,
                        mapping.phi_categories
                    ])

            print(f"Saved filename mappings to: {csv_path} ({len(mappings)} files)")

    def save_folder_mappings_csv(self, output_dir: Path) -> None:
        """
        Save folder mappings to a CSV file in the root output directory.

        Args:
            output_dir: Root output directory where CSV file will be saved
        """
        if not self.folder_mappings:
            return

        csv_path = output_dir / "folder_anonymization.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['original_foldername', 'anonymized_foldername', 'phi_values', 'phi_categories'])

            for mapping in self.folder_mappings:
                writer.writerow([
                    mapping.original_name,
                    mapping.anonymized_name,
                    mapping.phi_values,
                    mapping.phi_categories
                ])

        print(f"Saved folder mappings to: {csv_path} ({len(self.folder_mappings)} folders)")

    def save_all_mappings(self, output_dir: Path) -> None:
        """
        Save all mappings (files and folders) to CSV files.

        Args:
            output_dir: Root output directory
        """
        self.save_file_mappings_csv(output_dir)
        self.save_folder_mappings_csv(output_dir)

    def reset_counters(self) -> None:
        """Reset all mappings and counters (useful for testing or separate runs)."""
        self.file_mappings_by_folder.clear()
        self.folder_mappings.clear()
        self.folder_counters.clear()
