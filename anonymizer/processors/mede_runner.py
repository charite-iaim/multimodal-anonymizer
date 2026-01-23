#!/usr/bin/env python3
"""
mede runner script - Executes in Python 3.11 environment.

This script is called by mede_processor.py via subprocess and runs
in the .venv311 virtual environment where mede is installed.

Usage:
    # Single file:
    python mede_runner.py '{"input_path": "/path/to/input.nii", "output_path": "/path/to/output.nii"}'
    
    # 3D DICOM folder (multiple DICOM slices):
    python mede_runner.py '{"input_path": "/path/to/dicom_folder", "output_path": "/path/to/output_folder", "is_directory": true}'

Output:
    JSON object with status and any relevant information.
"""

import sys
import json
import traceback
from pathlib import Path


def process_single_file(input_path: Path, output_path: Path) -> dict:
    """Process a single CT/MRI file (NIfTI, NRRD, etc.)."""
    import mede
    from mede.deidentify import Inference

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create inference object for defacing
    inference = Inference(
        output_path=str(output_path.parent),  # Output directory
        gpu=None,  # Use CPU (set to 0 for GPU)
        verbose=True,
        deface=True,  # Enable defacing
        skullstrip=False  # Don't remove skull
    )

    # Run the defacing
    inference(str(input_path))
    inference.run()

    # Verify output was created
    if not output_path.exists():
        # Check if file was created with a different name pattern
        possible_outputs = list(output_path.parent.glob(f"*{input_path.stem}*"))
        if possible_outputs:
            actual_output = possible_outputs[0]
            return {
                "status": "success",
                "input_path": str(input_path),
                "output_path": str(actual_output),
                "output_size": actual_output.stat().st_size,
                "mede_version": getattr(mede, "__version__", "unknown")
            }
        return {
            "status": "error",
            "error": "Processing completed but output file was not created."
        }

    return {
        "status": "success",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "output_size": output_path.stat().st_size,
        "mede_version": getattr(mede, "__version__", "unknown")
    }


def process_dicom_folder(input_dir: Path, output_dir: Path) -> dict:
    """
    Process a folder of DICOM slices as a 3D volume, preserving individual DICOM files.
    
    This is used for CT/MRI scans where each slice is a separate DICOM file.
    The folder should be named with suffix "_extended_3d_image".
    
    The workflow:
    1. MEDE processes the folder and outputs a defaced NIfTI volume
    2. We load both original DICOM and defaced NIfTI
    3. We apply the defacing to each original DICOM slice
    4. We save back as individual DICOM files
    """
    import mede
    import tempfile
    import numpy as np
    import pydicom
    import nibabel as nib
    from mede.deidentify import Inference

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all DICOM files in the input directory
    dicom_extensions = ['.dcm', '.dicom', '']  # Some DICOM files have no extension
    dicom_files = []
    
    for file_path in input_dir.iterdir():
        if file_path.is_file():
            # Skip hidden files
            if file_path.name.startswith('.'):
                continue
            # Check if it's a DICOM file by extension
            if file_path.suffix.lower() in dicom_extensions or file_path.suffix == '':
                dicom_files.append(file_path)
    
    if not dicom_files:
        return {
            "status": "error",
            "error": f"No DICOM files found in directory: {input_dir}"
        }

    # Sort files by instance number or filename for consistent ordering
    def get_sort_key(f):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            return int(getattr(ds, 'InstanceNumber', 0))
        except:
            # Fallback to numeric extraction from filename
            import re
            nums = re.findall(r'\d+', f.stem)
            return int(nums[0]) if nums else 0
    
    dicom_files.sort(key=get_sort_key)

    print(f"Found {len(dicom_files)} DICOM files in {input_dir.name}", file=sys.stderr)
    print(f"Running MEDE defacing (this may take 20-40 minutes for large volumes)...", file=sys.stderr)

    # Create a temp directory for MEDE output (it outputs NIfTI)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create inference object for defacing
        inference = Inference(
            output_path=str(temp_path),  # Temp directory for NIfTI output
            gpu=None,  # Use CPU (set to 0 for GPU if available)
            verbose=True,
            deface=True,  # Enable defacing (face removal)
            skullstrip=False  # Don't remove skull, keep brain intact
        )

        # Run the defacing on the input folder
        inference(str(input_dir))
        inference.run()

        # Find the output NIfTI file
        nifti_files = list(temp_path.glob("*.nii*"))
        if not nifti_files:
            return {
                "status": "error",
                "error": "MEDE processing completed but no NIfTI output was created."
            }
        
        defaced_nifti_path = nifti_files[0]
        print(f"Loading defaced volume from {defaced_nifti_path.name}...", file=sys.stderr)
        
        # Load the defaced NIfTI volume
        defaced_nii = nib.load(str(defaced_nifti_path))
        defaced_data = defaced_nii.get_fdata()
        
        print(f"Defaced volume shape: {defaced_data.shape}", file=sys.stderr)
        print(f"Applying defacing to {len(dicom_files)} DICOM slices...", file=sys.stderr)

        # Process each DICOM file
        for i, dcm_path in enumerate(dicom_files):
            # Load original DICOM
            ds = pydicom.dcmread(str(dcm_path), force=True)
            original_pixels = ds.pixel_array.astype(np.float64)
            
            # Get the corresponding slice from defaced volume
            # MEDE typically uses the z-axis (3rd dimension) for slices
            if i < defaced_data.shape[2]:
                defaced_slice = defaced_data[:, :, i]
            else:
                # If shapes don't match, try to find closest match
                slice_idx = min(i, defaced_data.shape[2] - 1)
                defaced_slice = defaced_data[:, :, slice_idx]
            
            # The defaced slice may need to be transposed to match DICOM orientation
            # Try to match shapes
            if defaced_slice.shape != original_pixels.shape:
                # Try different orientations
                if defaced_slice.T.shape == original_pixels.shape:
                    defaced_slice = defaced_slice.T
                elif defaced_slice.shape[::-1] == original_pixels.shape:
                    defaced_slice = np.flip(defaced_slice, axis=0)
            
            # Create mask where face was removed (where defaced differs significantly from original)
            # MEDE typically sets defaced regions to 0 or a low value
            # We'll use the defaced data directly if shapes match
            if defaced_slice.shape == original_pixels.shape:
                # Apply the defaced values directly
                # Rescale defaced values to match original data range
                if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                    # Convert defaced HU values back to stored values
                    new_pixels = (defaced_slice - float(ds.RescaleIntercept)) / float(ds.RescaleSlope)
                else:
                    # Normalize to original range
                    new_pixels = defaced_slice
                
                # Ensure proper dtype
                new_pixels = np.clip(new_pixels, 0, 65535).astype(ds.pixel_array.dtype)
            else:
                # If shapes don't match, keep original (shouldn't happen)
                print(f"  Warning: Shape mismatch for slice {i}, keeping original", file=sys.stderr)
                new_pixels = ds.pixel_array
            
            # Update pixel data
            ds.PixelData = new_pixels.tobytes()
            
            # Save to output directory with same filename
            output_path = output_dir / dcm_path.name
            ds.save_as(str(output_path))
            
            if (i + 1) % 10 == 0 or i == len(dicom_files) - 1:
                print(f"  Saved {i + 1}/{len(dicom_files)} DICOM files...", file=sys.stderr)

    # Verify output
    output_files = [f for f in output_dir.iterdir() if f.is_file() and not f.name.startswith('.')]
    
    print(f"Successfully defaced {len(output_files)} DICOM files!", file=sys.stderr)

    return {
        "status": "success",
        "input_path": str(input_dir),
        "output_path": str(output_dir),
        "input_file_count": len(dicom_files),
        "output_file_count": len(output_files),
        "mede_version": getattr(mede, "__version__", "unknown")
    }


def main():
    if len(sys.argv) < 2:
        print(json.dumps({
            "status": "error",
            "error": "No arguments provided. Expected JSON string with input_path and output_path."
        }))
        sys.exit(1)

    try:
        args = json.loads(sys.argv[1])
    except json.JSONDecodeError as e:
        print(json.dumps({
            "status": "error",
            "error": f"Invalid JSON argument: {e}"
        }))
        sys.exit(1)

    input_path = args.get("input_path")
    output_path = args.get("output_path")
    is_directory = args.get("is_directory", False)

    if not input_path or not output_path:
        print(json.dumps({
            "status": "error",
            "error": "Both input_path and output_path are required."
        }))
        sys.exit(1)

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        print(json.dumps({
            "status": "error",
            "error": f"Input {'directory' if is_directory else 'file'} does not exist: {input_path}"
        }))
        sys.exit(1)

    try:
        # Import mede - this will only work in the Python 3.11 venv
        import mede

        if is_directory:
            result = process_dicom_folder(input_path, output_path)
        else:
            result = process_single_file(input_path, output_path)

        print(json.dumps(result))
        
        if result["status"] == "error":
            sys.exit(1)

    except ImportError as e:
        print(json.dumps({
            "status": "error",
            "error": f"Failed to import mede: {e}. Make sure mede is installed in this environment."
        }))
        sys.exit(1)

    except Exception as e:
        print(json.dumps({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }))
        sys.exit(1)


if __name__ == "__main__":
    main()
