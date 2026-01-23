# CT/MRI Processing Setup (Optional)

This document explains how to enable CT/MRI medical image processing using the `mede` library.

## Overview

CT/MRI processing is an **optional feature** that requires additional setup because:

1. The `mede` library requires **Python 3.11** (due to PyTorch version constraints)
2. It has large dependencies (PyTorch, medical imaging libraries) that would bloat the main installation

If you don't need to process NIfTI (`.nii`, `.nii.gz`), NRRD (`.nrrd`), or MetaImage (`.mha`, `.mhd`) files, or 3D DICOM volumes, you can skip this setup entirely.

## 3D DICOM Volume Processing

### The Problem

Standard DICOM files (2D medical images like X-rays, ECGs) are processed individually with our regular DICOM processor. However, **3D volumes** (CT scans, MRI scans) are stored as multiple DICOM files - one file per "slice" of the 3D image. These need to be processed together as a single volume to properly de-identify them.

### The Solution: Folder Naming Convention

To indicate that a folder contains 3D DICOM slices that should be processed together:

**Name the folder with the suffix `_extended_3d_image`**

#### Examples

```
✅ patient001_ct_extended_3d_image/
   ├── slice_001.dcm
   ├── slice_002.dcm
   ├── slice_003.dcm
   └── ... (more slices)

✅ brain_mri_extended_3d_image/
   ├── IMG0001
   ├── IMG0002
   └── ...

✅ data/john_doe_chest_extended_3d_image/
   └── (DICOM files)
```

#### Important Notes

- Each `_extended_3d_image` folder should contain data for **one patient only**
- All DICOM files in the folder will be processed as a single 3D volume
- The folder name itself will be anonymized (PHI will be detected and replaced)
- Files NOT in `_extended_3d_image` folders are processed individually as 2D images

## Prerequisites

- Python 3.11 installed on your system
- Sufficient disk space (~5GB for PyTorch and dependencies)
- GPU recommended for faster processing (but not required)

### Installing Python 3.11

**macOS (Homebrew):**
```bash
brew install python@3.11
```

**macOS (pyenv):**
```bash
pyenv install 3.11
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv
```

**Windows:**
Download from [python.org](https://www.python.org/downloads/) or use pyenv-win.

## Setup Instructions

### 1. Navigate to the Project Root

```bash
cd /path/to/mimiciv-anonymization-pipeline-feature
```

### 2. Create the Python 3.11 Virtual Environment

The mede processor expects the virtual environment at `.venv311` in the project root.

```bash
# Create the virtual environment
python3.11 -m venv .venv311

# Activate it
source .venv311/bin/activate  # Linux/macOS
# or
.venv311\Scripts\activate     # Windows
```

### 3. Install the mede Library

```bash
# Ensure pip is up to date
pip install --upgrade pip

# Install mede and its dependencies
pip install mede
```

**Note:** This installation may take several minutes as it downloads PyTorch and other large dependencies.

### 4. Verify Installation

```bash
# Test that mede is properly installed
python -c "import mede; print(f'mede version: {mede.__version__}')"
```

### 5. Deactivate and Test

```bash
# Deactivate the mede environment
deactivate

# Return to the main environment and restart the application
source .venv/bin/activate  # Activate main venv
# Start your application as usual
```

## Verification

Once set up, you can verify CT/MRI processing is available:

### Via API

```bash
curl http://localhost:8000/api/features
```

You should see:
```json
{
  "ct_mri_processing": {
    "available": true,
    "name": "CT/MRI Processing",
    ...
  }
}
```

### Via Frontend

When CT/MRI processing is available, the supported formats list will include "NIfTI, NRRD, MHA".

## Supported File Formats

| Format | Extensions | Description |
|--------|------------|-------------|
| NIfTI | `.nii`, `.nii.gz` | Neuroimaging Informatics Technology Initiative |
| NRRD | `.nrrd` | Nearly Raw Raster Data |
| MetaImage | `.mha`, `.mhd` | ITK MetaImage format |

## Troubleshooting

### "CT/MRI processing requires additional setup"

This error means the `.venv311` environment is not properly configured. Verify:

1. The `.venv311` directory exists in the project root
2. Python 3.11 is installed in that environment
3. The `mede` package is installed

Check status via API:
```bash
curl http://localhost:8000/api/features | jq '.ct_mri_processing'
```

### mede import fails

If `import mede` fails, try reinstalling with verbose output:

```bash
source .venv311/bin/activate
pip install --verbose mede
```

Common issues:
- Incompatible Python version (must be 3.11.x)
- Missing system libraries (check mede documentation)
- Insufficient disk space

### GPU Not Detected

If you have a CUDA-capable GPU but mede isn't using it:

1. Install the CUDA-enabled PyTorch in the `.venv311` environment:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

2. Verify GPU availability:
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## Architecture

```
mimiciv-anonymization-pipeline-feature/
├── .venv/                    # Main Python environment (3.13+)
├── .venv311/                 # mede Python environment (3.11)
│   └── bin/
│       └── python           # Used by mede_processor.py
├── anonymizer/
│   └── processors/
│       ├── mede_processor.py    # Calls .venv311/bin/python
│       └── mede_runner.py       # Runs in .venv311 environment
└── docs/
    └── MEDE_SETUP.md        # This file
```

The `mede_processor.py` spawns a subprocess using the Python 3.11 interpreter to run `mede_runner.py`, which performs the actual CT/MRI anonymization. This architecture allows the main application to remain on a modern Python version while supporting mede's specific requirements.

## Uninstalling

To remove CT/MRI processing support:

```bash
rm -rf .venv311
```

The application will automatically detect that CT/MRI processing is unavailable and gracefully degrade.
