"""
Processing tracker for tracking anonymized files and directories.

This module provides functionality to track which files and directories have already been
anonymized, allowing the pipeline to skip already-processed items on subsequent runs.
"""

import json
from pathlib import Path
from typing import Dict, Set, Optional
from datetime import datetime
import hashlib


class ProcessingTracker:
    """
    Tracks which files and directories have been successfully anonymized.

    The tracker stores:
    - File paths that have been processed
    - Directory paths that have been fully processed
    - Timestamps of when processing occurred
    - File hashes to detect modifications
    - Processing metadata (mode, output location, etc.)
    """

    def __init__(self, tracking_file: Path, compute_hashes: bool = True):
        """
        Initialize the processing tracker.

        Args:
            tracking_file: Path to the JSON file for tracking data
            compute_hashes: If True, compute and store file hashes to detect changes
        """
        self.tracking_file = tracking_file
        self.compute_hashes = compute_hashes
        self.data = self._load()

    def _load(self) -> Dict:
        """Load tracking data from file."""
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load tracking file {self.tracking_file}: {e}")
                print("Starting with empty tracking data.")
                return self._create_empty_data()
        return self._create_empty_data()

    def _create_empty_data(self) -> Dict:
        """Create empty tracking data structure."""
        return {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "files": {},  # path -> {timestamp, hash, output_path, status}
            "directories": {},  # path -> {timestamp, file_count, status}
        }

    def save(self):
        """Save tracking data to file."""
        self.data["last_updated"] = datetime.now().isoformat()
        self.tracking_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(self.tracking_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Warning: Could not save tracking file {self.tracking_file}: {e}")

    def _compute_file_hash(self, file_path: Path) -> Optional[str]:
        """
        Compute SHA-256 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            Hex string of the file hash, or None if error
        """
        if not self.compute_hashes:
            return None

        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                # Read file in chunks to handle large files
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except IOError as e:
            print(f"Warning: Could not compute hash for {file_path}: {e}")
            return None

    def is_file_processed(self, file_path: Path) -> bool:
        """
        Check if a file has already been processed.

        Args:
            file_path: Path to check

        Returns:
            True if file has been processed and hasn't changed
        """
        path_str = str(file_path.absolute())

        if path_str not in self.data["files"]:
            return False

        file_info = self.data["files"][path_str]

        # Check if file still exists
        if not file_path.exists():
            return False

        # If we have a hash, check if file has been modified
        if self.compute_hashes and "hash" in file_info:
            current_hash = self._compute_file_hash(file_path)
            if current_hash != file_info["hash"]:
                # File has been modified
                return False

        # Check status
        return file_info.get("status") == "completed"

    def is_directory_processed(self, dir_path: Path) -> bool:
        """
        Check if a directory has already been fully processed.

        Args:
            dir_path: Path to check

        Returns:
            True if directory has been fully processed
        """
        path_str = str(dir_path.absolute())

        if path_str not in self.data["directories"]:
            return False

        dir_info = self.data["directories"][path_str]
        return dir_info.get("status") == "completed"

    def mark_file_processed(
        self,
        file_path: Path,
        output_path: Path,
        success: bool = True
    ):
        """
        Mark a file as processed.

        Args:
            file_path: Path to the input file
            output_path: Path to the output file
            success: Whether processing was successful
        """
        path_str = str(file_path.absolute())

        file_hash = self._compute_file_hash(file_path) if self.compute_hashes else None

        self.data["files"][path_str] = {
            "timestamp": datetime.now().isoformat(),
            "output_path": str(output_path.absolute()) if output_path else None,
            "status": "completed" if success else "failed",
            "hash": file_hash,
        }

    def mark_directory_processed(
        self,
        dir_path: Path,
        file_count: int,
        success: bool = True
    ):
        """
        Mark a directory as fully processed.

        Args:
            dir_path: Path to the directory
            file_count: Number of files processed in this directory
            success: Whether processing was successful
        """
        path_str = str(dir_path.absolute())

        self.data["directories"][path_str] = {
            "timestamp": datetime.now().isoformat(),
            "file_count": file_count,
            "status": "completed" if success else "failed",
        }

    def clear_file(self, file_path: Path):
        """
        Remove a file from tracking (force reprocessing).

        Args:
            file_path: Path to the file
        """
        path_str = str(file_path.absolute())
        if path_str in self.data["files"]:
            del self.data["files"][path_str]

    def clear_directory(self, dir_path: Path):
        """
        Remove a directory from tracking (force reprocessing).

        Args:
            dir_path: Path to the directory
        """
        path_str = str(dir_path.absolute())
        if path_str in self.data["directories"]:
            del self.data["directories"][path_str]

    def clear_all(self):
        """Clear all tracking data."""
        self.data = self._create_empty_data()

    def get_stats(self) -> Dict:
        """
        Get statistics about processed files and directories.

        Returns:
            Dictionary with statistics
        """
        total_files = len(self.data["files"])
        completed_files = sum(
            1 for f in self.data["files"].values()
            if f.get("status") == "completed"
        )
        failed_files = sum(
            1 for f in self.data["files"].values()
            if f.get("status") == "failed"
        )

        total_dirs = len(self.data["directories"])
        completed_dirs = sum(
            1 for d in self.data["directories"].values()
            if d.get("status") == "completed"
        )

        return {
            "total_files": total_files,
            "completed_files": completed_files,
            "failed_files": failed_files,
            "total_directories": total_dirs,
            "completed_directories": completed_dirs,
            "last_updated": self.data.get("last_updated"),
        }

    def print_stats(self):
        """Print statistics about processed files and directories."""
        stats = self.get_stats()
        print(f"\nProcessing Tracker Statistics:")
        print(f"  Files tracked: {stats['total_files']}")
        print(f"    - Completed: {stats['completed_files']}")
        print(f"    - Failed: {stats['failed_files']}")
        print(f"  Directories tracked: {stats['total_directories']}")
        print(f"    - Completed: {stats['completed_directories']}")
        print(f"  Last updated: {stats['last_updated']}")
