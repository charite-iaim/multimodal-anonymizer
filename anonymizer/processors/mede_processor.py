"""
CT/MRI Medical Image Processor using mede library.

This processor is OPTIONAL and requires additional setup:
- Python 3.11 (due to PyTorch compatibility constraints)
- A separate virtual environment with mede installed

See docs/MEDE_SETUP.md for installation instructions.

For 3D DICOM volumes (CT/MRI scans composed of multiple DICOM slices):
- Name the folder with suffix "_extended_3d_image"
- Example: patient001_ct_extended_3d_image/
- All DICOM files in the folder will be processed as a single 3D volume
"""

import subprocess
import json
import logging
from pathlib import Path
from typing import Any, Optional, Dict, List

from ..base_processor import FileProcessor
from ..config import AnonymizerConfig

logger = logging.getLogger(__name__)

# Path to the Python 3.11 virtual environment for mede
# Users should create this environment following docs/MEDE_SETUP.md
MEDE_VENV_PATH = Path(__file__).parent.parent.parent / ".venv311"
MEDE_PYTHON_PATH = MEDE_VENV_PATH / "bin" / "python"
MEDE_RUNNER_PATH = Path(__file__).parent / "mede_runner.py"

# Supported file extensions for CT/MRI processing
CT_MRI_EXTENSIONS = ['.nii', '.nii.gz', '.nrrd', '.mha', '.mhd']

# Folder naming convention for 3D DICOM volumes
EXTENDED_3D_IMAGE_SUFFIX = "_extended_3d_image"


def is_mede_available() -> bool:
    """
    Check if the mede processing environment is available.

    Returns:
        True if mede is properly configured and can be used.
    """
    if not MEDE_PYTHON_PATH.exists():
        return False

    if not MEDE_RUNNER_PATH.exists():
        return False

    # Verify mede is actually importable in the venv
    try:
        result = subprocess.run(
            [str(MEDE_PYTHON_PATH), "-c", "import mede; print('ok')"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0 and "ok" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logger.debug(f"mede availability check failed: {e}")
        return False


def is_extended_3d_image_folder(folder_name: str) -> bool:
    """
    Check if a folder name indicates it contains 3D DICOM volume data.

    The convention is to suffix the folder name with "_extended_3d_image".
    Example: "patient001_ct_extended_3d_image"

    Args:
        folder_name: The name of the folder (not full path)

    Returns:
        True if the folder should be treated as a 3D DICOM volume
    """
    return folder_name.lower().endswith(EXTENDED_3D_IMAGE_SUFFIX.lower())


def find_3d_image_folders_in_path(rel_path: str) -> List[str]:
    """
    Find any folders in a relative path that are marked as 3D image folders.

    Args:
        rel_path: Relative path like "patient_data_extended_3d_image/slice001.dcm"

    Returns:
        List of folder names that match the 3D image convention
    """
    path_parts = Path(rel_path).parts
    return [part for part in path_parts[:-1] if is_extended_3d_image_folder(part)]


def get_mede_status() -> Dict[str, Any]:
    """
    Get detailed status about mede availability.

    Returns:
        Dictionary with status information for debugging/display.
    """
    status = {
        "available": False,
        "venv_exists": MEDE_VENV_PATH.exists(),
        "python_exists": MEDE_PYTHON_PATH.exists(),
        "runner_exists": MEDE_RUNNER_PATH.exists(),
        "mede_importable": False,
        "supported_extensions": CT_MRI_EXTENSIONS,
        "setup_instructions": "See docs/MEDE_SETUP.md for installation instructions."
    }

    if status["python_exists"] and status["runner_exists"]:
        try:
            result = subprocess.run(
                [str(MEDE_PYTHON_PATH), "-c", "import mede; print(mede.__version__)"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                status["mede_importable"] = True
                status["mede_version"] = result.stdout.strip()
                status["available"] = True
        except Exception as e:
            status["error"] = str(e)

    return status


class MedeProcessor(FileProcessor):
    """
    Processor for CT/MRI medical images using the mede library.

    This processor runs in a separate Python 3.11 environment due to
    PyTorch version constraints in the mede library.

    Supported formats:
        - NIfTI (.nii, .nii.gz)
        - NRRD (.nrrd)
        - MetaImage (.mha, .mhd)
    """

    def __init__(
        self,
        config: AnonymizerConfig,
        timeout: int = 3600,
        **kwargs
    ):
        """
        Initialize the mede processor.

        Args:
            config: Anonymizer configuration
            timeout: Maximum time in seconds to wait for processing (default: 60 minutes)
                    Large 3D volumes can take 20-40 minutes to process.
            **kwargs: Additional arguments (ignored, for compatibility)
        """
        super().__init__(config)
        self.timeout = timeout
        self._check_availability()

    def _check_availability(self) -> None:
        """Check if mede is available and raise a clear error if not."""
        if not is_mede_available():
            status = get_mede_status()
            raise RuntimeError(
                "CT/MRI processing requires additional setup.\n\n"
                f"Status:\n"
                f"  - Virtual environment exists: {status['venv_exists']}\n"
                f"  - Python 3.11 found: {status['python_exists']}\n"
                f"  - Runner script found: {status['runner_exists']}\n"
                f"  - mede importable: {status['mede_importable']}\n\n"
                f"Please follow the instructions in docs/MEDE_SETUP.md to enable CT/MRI processing."
            )

    def can_process(self, file_path: Path) -> bool:
        """
        Check if this processor can handle the given file or directory.

        Args:
            file_path: Path to the file or directory

        Returns:
            True if this processor can handle the input
        """
        # Check if it's a directory (3D DICOM folder)
        if file_path.is_dir():
            return is_extended_3d_image_folder(file_path.name)

        suffix = file_path.suffix.lower()
        name = file_path.name.lower()

        # Handle .nii.gz (compound extension)
        if name.endswith('.nii.gz'):
            return True

        return suffix in CT_MRI_EXTENSIONS

    def extract_content(self, file_path: Path) -> Any:
        """
        Extract content from file for processing.

        For medical images, this returns metadata about the file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with file metadata
        """
        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size if file_path.exists() else 0
        }

    def anonymize(self, input_path: Path, output_path: Path) -> None:
        """
        Anonymize the CT/MRI file or 3D DICOM folder using mede.

        Args:
            input_path: Path to input file or directory (for 3D DICOM folders)
            output_path: Path to save anonymized output (file or directory)
        """
        is_directory = input_path.is_dir()
        
        if is_directory:
            logger.info(f"Processing 3D DICOM folder with mede: {input_path}")
            # For directories, output_path is also a directory
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            logger.info(f"Processing CT/MRI file with mede: {input_path}")
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare arguments for the runner script
        args = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "is_directory": is_directory,
        }

        try:
            result = subprocess.run(
                [
                    str(MEDE_PYTHON_PATH),
                    str(MEDE_RUNNER_PATH),
                    json.dumps(args)
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown error"
                raise RuntimeError(f"mede processing failed: {error_msg}")

            # Parse result
            try:
                output = json.loads(result.stdout)
                if output.get("status") == "error":
                    raise RuntimeError(f"mede processing error: {output.get('error', 'Unknown error')}")

                if is_directory:
                    logger.info(f"Successfully processed 3D DICOM folder: {output_path}")
                else:
                    logger.info(f"Successfully processed CT/MRI file: {output_path}")

            except json.JSONDecodeError:
                # If output isn't JSON, check if output was created
                if is_directory:
                    if not output_path.exists() or not any(output_path.iterdir()):
                        raise RuntimeError(f"mede processing failed: {result.stdout}")
                else:
                    if not output_path.exists():
                        raise RuntimeError(f"mede processing failed: {result.stdout}")
                
                if is_directory:
                    logger.info(f"Successfully processed 3D DICOM folder: {output_path}")
                else:
                    logger.info(f"Successfully processed CT/MRI file: {output_path}")

        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"CT/MRI processing timed out after {self.timeout} seconds. "
                "Consider increasing the timeout for large files."
            )


def process_ct_mri_file(
    input_path: str,
    output_path: str,
    timeout: int = 300
) -> Dict[str, Any]:
    """
    Convenience function to process a CT/MRI file.

    This function can be used without instantiating the full processor.

    Args:
        input_path: Path to input file
        output_path: Path to save anonymized file
        timeout: Maximum processing time in seconds

    Returns:
        Dictionary with processing result

    Raises:
        RuntimeError: If mede is not available or processing fails
    """
    if not is_mede_available():
        raise RuntimeError(
            "CT/MRI processing requires additional setup. "
            "See docs/MEDE_SETUP.md for installation instructions."
        )

    args = {
        "input_path": input_path,
        "output_path": output_path,
    }

    result = subprocess.run(
        [
            str(MEDE_PYTHON_PATH),
            str(MEDE_RUNNER_PATH),
            json.dumps(args)
        ],
        capture_output=True,
        text=True,
        timeout=timeout
    )

    if result.returncode != 0:
        raise RuntimeError(f"mede processing failed: {result.stderr or result.stdout}")

    return json.loads(result.stdout)
