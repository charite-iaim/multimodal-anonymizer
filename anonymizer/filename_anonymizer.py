"""
Filename and folder name anonymizer with PII detection and reversible mapping.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from .config import AnonymizerConfig


class FilenameSegment(BaseModel):
    """A segment of a filename that contains PII."""
    original_text: str = Field(description="Original text segment containing PII")
    anonymized_text: str = Field(description="Anonymized replacement text (e.g., PER_001)")
    phi_category: str = Field(description="Category of PHI (PERSON, DATE, ID, LOCATION, etc.)")
    start_position: int = Field(description="Start position in original filename")
    end_position: int = Field(description="End position in original filename")


class FilenameAnonymizationResult(BaseModel):
    """Result of filename PII detection and anonymization."""
    original_filename: str = Field(description="Original filename")
    anonymized_filename: str = Field(description="Anonymized filename with PII replaced")
    segments: List[FilenameSegment] = Field(
        default_factory=list,
        description="List of PII segments that were anonymized"
    )


class PathMapping(BaseModel):
    """Mapping between original and anonymized paths with annotations."""
    original_path: str = Field(description="Original relative path")
    anonymized_path: str = Field(description="Anonymized relative path")
    is_directory: bool = Field(description="Whether this is a directory mapping")
    filename_segments: List[FilenameSegment] = Field(
        default_factory=list,
        description="PII segments found in filename/folder name"
    )
    timestamp: str = Field(description="When this mapping was created")


class PathMappingCollection(BaseModel):
    """Collection of all path mappings for a processing run."""
    metadata: Dict[str, str] = Field(description="Metadata about the anonymization run")
    mappings: List[PathMapping] = Field(
        default_factory=list,
        description="List of all path mappings"
    )


class FilenameAnonymizer:
    """
    Anonymizes filenames and folder names by detecting PII and replacing with sequential IDs.
    Maintains a reversible mapping for evaluation purposes.
    """

    def __init__(self, config: AnonymizerConfig):
        """Initialize filename anonymizer."""
        self.config = config

        # Initialize LLM for PII detection in filenames
        self.llm = AzureChatOpenAI(
            azure_deployment=config.azure_deployment_name,
            azure_endpoint=config.azure_endpoint,
            api_key=config.azure_api_key,
            api_version=config.azure_api_version,
            temperature=0.0,  # Use deterministic output for consistency
        ).with_structured_output(FilenameAnonymizationResult)

        # Category counters for sequential ID generation
        self.category_counters: Dict[str, int] = defaultdict(int)

        # Store all mappings for JSON output
        self.mappings: List[PathMapping] = []

    def anonymize_filename(self, filename: str, is_directory: bool = False) -> FilenameAnonymizationResult:
        """
        Detect PII in filename and replace with sequential IDs per category.

        Args:
            filename: Original filename (without path)
            is_directory: Whether this is a directory name

        Returns:
            FilenameAnonymizationResult with original filename, anonymized filename, and segments
        """
        # Prepare the prompt for PII detection in filename
        prompt = f"""You are a PHI (Protected Health Information) detection expert. Analyze the following {'folder' if is_directory else 'file'} name and identify ALL PHI elements.

{'Folder' if is_directory else 'File'} name: {filename}

Common PHI categories in filenames:
- PERSON: Patient names, doctor names (e.g., "John_Doe", "DrSmith")
- ID: Patient IDs, medical record numbers, account numbers (e.g., "PID-12345", "MRN123456")
- DATE: Dates of birth, admission dates, dates in YYYYMMDD format (e.g., "19850615", "20230725")
- LOCATION: Hospital names, room numbers, addresses (e.g., "RoomA", "NYC")
- PHONE: Phone numbers or parts of phone numbers
- EMAIL: Email addresses or parts of email
- OTHER: Any other identifiable information

For each PHI element found:
1. Extract the exact text from the filename
2. Identify its category
3. Mark its start and end position in the string
4. In the anonymized_filename, replace ONLY the PHI elements while preserving:
   - File extension
   - Underscores, hyphens, and other separators
   - Non-PHI descriptive words (e.g., "ecg", "report", "summary")

Example:
Input: "ecg_Noah_Rhodes_PID-183667_20130725.pdf"
Output:
- segments:
  * "Noah_Rhodes" (PERSON, positions 4-15)
  * "PID-183667" (ID, positions 17-27)
  * "20130725" (DATE, positions 29-37)
- anonymized_filename: "ecg_PERSON_ID_DATE.pdf" (we'll replace PERSON/ID/DATE with sequential IDs in post-processing)

If no PHI is detected, return the original filename unchanged with an empty segments list.

IMPORTANT:
- Be thorough - medical filenames often contain multiple PHI elements
- Preserve the file structure and extension
- Mark exact positions for reversible mapping
"""

        try:
            # Call LLM to detect PII in filename
            message = HumanMessage(content=prompt)
            result = self.llm.invoke([message])

            # Post-process: replace category placeholders with sequential IDs
            anonymized_filename = result.anonymized_filename
            updated_segments = []

            # Sort segments by position to process in order
            for segment in sorted(result.segments, key=lambda s: s.start_position):
                # Generate sequential ID for this category
                category = segment.phi_category.upper()
                self.category_counters[category] += 1
                sequential_id = f"{category}-{self.category_counters[category]:03d}"

                # Update the segment with the sequential ID
                updated_segment = FilenameSegment(
                    original_text=segment.original_text,
                    anonymized_text=sequential_id,
                    phi_category=category,
                    start_position=segment.start_position,
                    end_position=segment.end_position
                )
                updated_segments.append(updated_segment)

            # Rebuild anonymized filename with sequential IDs
            # Replace from end to start to preserve positions
            anonymized_filename = result.original_filename
            for segment in reversed(updated_segments):
                anonymized_filename = (
                    anonymized_filename[:segment.start_position] +
                    segment.anonymized_text +
                    anonymized_filename[segment.end_position:]
                )

            return FilenameAnonymizationResult(
                original_filename=result.original_filename,
                anonymized_filename=anonymized_filename,
                segments=updated_segments
            )

        except Exception as e:
            print(f"Warning: Failed to anonymize filename '{filename}': {e}")
            print("Using fallback: prepending 'anonymized_' to filename")
            # Fallback: just add prefix
            return FilenameAnonymizationResult(
                original_filename=filename,
                anonymized_filename=f"anonymized_{filename}",
                segments=[]
            )

    def add_mapping(
        self,
        original_path: Path,
        anonymized_path: Path,
        is_directory: bool = False,
        segments: Optional[List[FilenameSegment]] = None
    ) -> None:
        """
        Add a path mapping to the collection.

        Args:
            original_path: Original path (relative to input root)
            anonymized_path: Anonymized path (relative to output root)
            is_directory: Whether this is a directory
            segments: PII segments found in the filename/folder name
        """
        mapping = PathMapping(
            original_path=str(original_path),
            anonymized_path=str(anonymized_path),
            is_directory=is_directory,
            filename_segments=segments or [],
            timestamp=datetime.now().isoformat()
        )
        self.mappings.append(mapping)

    def save_mappings(self, output_path: Path, input_root: str, output_root: str) -> None:
        """
        Save all path mappings to a JSON file.

        Args:
            output_path: Path where to save the mapping JSON file
            input_root: Root directory of input files
            output_root: Root directory of output files
        """
        collection = PathMappingCollection(
            metadata={
                "timestamp": datetime.now().isoformat(),
                "input_root": input_root,
                "output_root": output_root,
                "total_mappings": str(len(self.mappings)),
                "files_anonymized": str(sum(1 for m in self.mappings if not m.is_directory)),
                "folders_anonymized": str(sum(1 for m in self.mappings if m.is_directory)),
                "total_phi_segments": str(sum(len(m.filename_segments) for m in self.mappings))
            },
            mappings=self.mappings
        )

        # Save to JSON with pretty formatting
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                collection.model_dump(mode='json'),
                f,
                indent=2,
                ensure_ascii=False
            )
        print(f"\nSaved path mappings to: {output_path}")
        print(f"Total mappings: {len(self.mappings)}")
        print(f"Files: {collection.metadata['files_anonymized']}, "
              f"Folders: {collection.metadata['folders_anonymized']}, "
              f"PHI segments: {collection.metadata['total_phi_segments']}")

    def reset_counters(self) -> None:
        """Reset category counters (useful for testing or separate runs)."""
        self.category_counters.clear()
        self.mappings.clear()
