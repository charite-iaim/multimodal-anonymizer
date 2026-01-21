"""
Filename and folder name anonymizer with PII detection and reversible mapping.
"""

import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from .config import AnonymizerConfig
from .llm_factory import create_chat_llm


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
        # Note: max_tokens needs to be high enough to accommodate reasoning models
        # which use reasoning_tokens within the max_tokens budget (e.g., GLM-4 can use
        # ~1800 reasoning tokens, so we need at least 2500+ for a complete response)
        self.llm = create_chat_llm(
            config=config,
            temperature=0.0,  # Use deterministic output for consistency
            max_tokens=4096,  # Higher limit to accommodate reasoning models
            structured_output=FilenameAnonymizationResult,
        )

        # Store file mappings by folder for CSV export
        # Key: folder path (e.g., "patient_ID_ID/csv"), Value: list of FileMappings
        self.file_mappings_by_folder: Dict[str, List[FileMapping]] = defaultdict(list)

        # Store folder mappings for CSV export
        self.folder_mappings: List[FolderMapping] = []

        # Counter for folder name sequences
        # Key: anonymized folder name (e.g., "patient_ID_ID"), Value: counter
        self.folder_counters: Dict[str, int] = defaultdict(int)

        # Counter for file name sequences within folders
        # Key: (folder_path, base_filename) tuple, Value: counter
        # This ensures files like "ID.hea" get unique names like "ID_001.hea", "ID_002.hea"
        self.file_counters: Dict[tuple, int] = defaultdict(int)

        # Track which CSV files have been initialized with headers
        self._initialized_csv_files: set = set()

        # Track already anonymized mappings (loaded from existing CSVs)
        # Key: original folder name, Value: anonymized folder name
        self._existing_folder_mappings: Dict[str, str] = {}
        # Key: (folder_path, original_filename), Value: anonymized filename
        self._existing_file_mappings: Dict[tuple, str] = {}

        # Initialize folder mapping CSV if output_dir is provided and load existing mappings
        if self.output_dir:
            self._initialize_folder_csv()
            self._load_existing_mappings()

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

    def _load_existing_mappings(self) -> None:
        """Load existing mappings from CSV files to prevent duplicate anonymization."""
        if not self.output_dir:
            return

        # Load folder mappings
        folder_csv_path = self.output_dir / "folder_anonymization.csv"
        if folder_csv_path.exists():
            try:
                with open(folder_csv_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        original = row.get('original_foldername', '')
                        anonymized = row.get('anonymized_foldername', '')
                        if original and anonymized:
                            self._existing_folder_mappings[original] = anonymized
                            # Also update folder counters to continue from where we left off
                            # Extract base name and counter from anonymized name (e.g., "patient_ID_ID_001")
                            self._update_folder_counter_from_anonymized(anonymized)
                if self._existing_folder_mappings:
                    print(f"Loaded {len(self._existing_folder_mappings)} existing folder mappings from {folder_csv_path}")
            except Exception as e:
                print(f"Warning: Could not load folder mappings from {folder_csv_path}: {e}")

        # Load file mappings from all subdirectories
        self._load_file_mappings_recursive(self.output_dir)

    def _update_folder_counter_from_anonymized(self, anonymized_name: str) -> None:
        """
        Update folder counters based on an existing anonymized folder name.

        Args:
            anonymized_name: Anonymized folder name (e.g., "patient_ID_ID_001")
        """
        # Try to extract base name and counter (e.g., "patient_ID_ID_001" -> base="patient_ID_ID", counter=1)
        import re
        match = re.match(r'^(.+)_(\d{3})$', anonymized_name)
        if match:
            base_name = match.group(1)
            counter = int(match.group(2))
            # Update counter to at least this value
            if self.folder_counters[base_name] < counter:
                self.folder_counters[base_name] = counter

    def _update_file_counter_from_anonymized(self, folder_path: str, anonymized_name: str) -> None:
        """
        Update file counters based on an existing anonymized filename.

        Args:
            folder_path: Path to the folder containing the file
            anonymized_name: Anonymized filename (e.g., "ID_0001.hea")
        """
        import re
        # Extract base name and counter (e.g., "ID_0001.hea" -> base="ID.hea", counter=1)
        # Handle filenames with extension
        if '.' in anonymized_name:
            last_dot = anonymized_name.rfind('.')
            stem = anonymized_name[:last_dot]
            ext = anonymized_name[last_dot:]
        else:
            stem = anonymized_name
            ext = ''

        # Try to extract counter from stem (e.g., "ID_0001" -> base="ID", counter=1)
        match = re.match(r'^(.+)_(\d{4})$', stem)
        if match:
            base_stem = match.group(1)
            counter = int(match.group(2))
            base_filename = f"{base_stem}{ext}"
            key = (folder_path, base_filename)
            # Update counter to at least this value
            if self.file_counters[key] < counter:
                self.file_counters[key] = counter

    def _load_file_mappings_recursive(self, directory: Path) -> None:
        """
        Recursively load file mappings from filename_anonymization.csv files.

        Args:
            directory: Directory to search for CSV files
        """
        if not directory.exists():
            return

        file_mapping_count = 0

        for csv_path in directory.rglob("filename_anonymization.csv"):
            try:
                # Get folder path relative to output_dir
                folder_path = str(csv_path.parent.relative_to(self.output_dir))
                if folder_path == '.':
                    folder_path = ''

                with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        original = row.get('original_filename', '')
                        anonymized = row.get('anonymized_filename', '')
                        if original and anonymized:
                            key = (folder_path, original)
                            self._existing_file_mappings[key] = anonymized
                            # Update file counters
                            self._update_file_counter_from_anonymized(folder_path, anonymized)
                            file_mapping_count += 1

                # Mark this CSV as initialized
                self._initialized_csv_files.add(str(csv_path))
            except Exception as e:
                print(f"Warning: Could not load file mappings from {csv_path}: {e}")

        if file_mapping_count > 0:
            print(f"Loaded {file_mapping_count} existing file mappings")

    def is_folder_already_anonymized(self, original_folder_name: str) -> bool:
        """
        Check if a folder has already been anonymized.

        Args:
            original_folder_name: Original folder name to check

        Returns:
            True if folder was already anonymized
        """
        return original_folder_name in self._existing_folder_mappings

    def get_existing_folder_mapping(self, original_folder_name: str) -> Optional[str]:
        """
        Get the existing anonymized folder name for an original folder.

        Args:
            original_folder_name: Original folder name

        Returns:
            Anonymized folder name if exists, None otherwise
        """
        return self._existing_folder_mappings.get(original_folder_name)

    def is_file_already_anonymized(self, folder_path: str, original_filename: str) -> bool:
        """
        Check if a file has already been anonymized.

        Args:
            folder_path: Path to the folder containing the file
            original_filename: Original filename to check

        Returns:
            True if file was already anonymized
        """
        key = (folder_path, original_filename)
        return key in self._existing_file_mappings

    def get_existing_file_mapping(self, folder_path: str, original_filename: str) -> Optional[str]:
        """
        Get the existing anonymized filename for an original file.

        Args:
            folder_path: Path to the folder containing the file
            original_filename: Original filename

        Returns:
            Anonymized filename if exists, None otherwise
        """
        key = (folder_path, original_filename)
        return self._existing_file_mappings.get(key)

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

    def get_sequential_filename(self, folder_path: str, base_anonymized_name: str) -> str:
        """
        Get a sequential filename with a counter suffix to ensure uniqueness within a folder.

        Args:
            folder_path: Path to the folder containing the file (e.g., "patient_ID_ID_001/csv")
            base_anonymized_name: Base anonymized filename (e.g., "ID.hea", "ID.pdf")

        Returns:
            Sequential filename (e.g., "ID_001.hea", "ID_002.hea")
        """
        # Split filename into stem and extension
        if '.' in base_anonymized_name:
            # Find the last dot for extension
            last_dot = base_anonymized_name.rfind('.')
            stem = base_anonymized_name[:last_dot]
            ext = base_anonymized_name[last_dot:]
        else:
            stem = base_anonymized_name
            ext = ''

        # Create a unique key for this folder + filename combination
        key = (folder_path, base_anonymized_name)
        self.file_counters[key] += 1
        counter = self.file_counters[key]

        return f"{stem}_{counter:04d}{ext}"

    def anonymize_filename(self, filename: str, is_directory: bool = False, folder_path: str = "") -> FilenameAnonymizationResult:
        """
        Detect PII in filename and replace with generic category placeholders.
        For directories, adds a sequential counter suffix (e.g., patient_ID_ID_001).
        For files, adds a sequential counter to ensure uniqueness within the folder.

        Args:
            filename: Original filename (without path)
            is_directory: Whether this is a directory name
            folder_path: Path to the folder containing the file (used for file uniqueness)

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
            elif not is_directory and folder_path:
                # For files within a folder, always add sequential number to ensure uniqueness
                base_anonymized_name = result.anonymized_filename
                anonymized_filename = self.get_sequential_filename(folder_path, base_anonymized_name)
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
            elif not is_directory and folder_path:
                # For files, add sequential number even in fallback
                fallback_name = self.get_sequential_filename(folder_path, fallback_name)

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
        Only writes to CSV if the filename actually changed and wasn't already in existing mappings.

        Args:
            folder_path: Path to the folder containing the file (e.g., "patient_ID_ID/csv")
            original_filename: Original filename
            anonymized_filename: Anonymized filename
            phi_detections: PHI values detected in the filename
        """
        # Only add mapping if filename actually changed
        if original_filename == anonymized_filename:
            return

        # Skip if already in existing mappings (loaded from CSV)
        key = (folder_path, original_filename)
        if key in self._existing_file_mappings:
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
        Only writes to CSV if the folder name actually changed and wasn't already in existing mappings.

        Args:
            original_foldername: Original folder name
            anonymized_foldername: Anonymized folder name
            phi_detections: PHI values detected in the folder name
        """
        # Only add mapping if folder name actually changed
        if original_foldername == anonymized_foldername:
            return

        # Skip if already in existing mappings (loaded from CSV)
        if original_foldername in self._existing_folder_mappings:
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

        Note: When output_dir is set during init, mappings are written immediately via
        _append_file_mapping_to_csv. This method is kept for backwards compatibility
        and for cases where output_dir is not set during init.

        Args:
            output_dir: Root output directory where CSV files will be saved
        """
        # If output_dir was set during init, mappings were already written incrementally
        # Only print summary in this case
        if self.output_dir == output_dir:
            total_new_mappings = sum(len(m) for m in self.file_mappings_by_folder.values())
            if total_new_mappings > 0:
                print(f"Filename mappings already saved incrementally ({total_new_mappings} new mappings)")
            return

        # Legacy path: write all mappings at once (only used if output_dir was not set during init)
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

        Note: When output_dir is set during init, mappings are written immediately via
        _append_folder_mapping_to_csv. This method is kept for backwards compatibility
        and for cases where output_dir is not set during init.

        Args:
            output_dir: Root output directory where CSV file will be saved
        """
        if not self.folder_mappings:
            return

        # If output_dir was set during init, mappings were already written incrementally
        # Only print summary in this case
        if self.output_dir == output_dir:
            print(f"Folder mappings already saved incrementally ({len(self.folder_mappings)} new mappings)")
            return

        # Legacy path: write all mappings at once (only used if output_dir was not set during init)
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
        self.file_counters.clear()
