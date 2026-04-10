<div align="center">
  <h1 style="border-bottom: none; display: inline-block; vertical-align: middle; margin-left: 10px;">Multimodal Anonymizer</h1>
</div>

<p align="center">
  A multimodal de-identification system that uses a agentic LLM-based processing to automatically detect and redact Personally Identifiable Information from medical files
</p>

---

The pipeline uses any LLM with an OpenAI-compatible API with vision capabilities to automatically identify and redact personally identifiable information. All files are processed entirely on your machine, nothing is stored permanently. You can use the browser-based interface for easy file and folder upload, or the command-line script for batch processing.

## Features
- **Multimodal Support:** Images (PNG, JPG, JPEG, DICOM), Videos (MP4, AVI, MOV, MKV, DICOM), Text (TXT, DOCX), PDFs, Tables (CSV, Excel), Metadata (DICOM, HEA), and Audio (WAV, MP3)
- **Face Redaction & Defacing:** Automatically detects and blurs faces in images and PDFs; uses a defacing model to anonymize 3D DICOM scans
- **Handwriting Redaction:** Specialized models ensure handwritten PII is detected and redacted
- **Filename Anonymization:** Filenames are anonymized alongside file contents
- **Date Shifting:** Applies a random but consistent date shift across all files in a folder to preserve sequential relationships
- **Customizable Prompts:** Control exactly which personal information is redacted for each file type
- **Optional Pseudonymization:** Generate CSV mapping files that track original-to-anonymized filenames

## User Interface

### 1. LLM Configuration
Connect to any OpenAI-compatible LLM (e.g. Ollama, LM Studio, vLLM, LocalAI).

<div align="center">
<img src="frontend/src/assets/screenshots/LLM-Configuration.png" width="700">
</div>

### 2. Anonymization Settings and File Upload
Configure processing options and select files or folders or use drag and drop.

<div align="center">
<img src="frontend/src/assets/screenshots/Anonymization-Settings.png" width="700">
</div>

### 3. Prompt Customization (optional)
Customize the prompts for each file type to customize the anonymization of personal information.

<div align="center">
<img src="frontend/src/assets/screenshots/Prompt-Customization.png" width="700">
</div>

### 4. Start File Processing

<div align="center">
<img src="frontend/src/assets/screenshots/File-Processing.png" width="700">
</div>

### 5. Download Results

<div align="center">
<img src="frontend/src/assets/screenshots/File-Download.png" width="700">
</div>

## How to UseTable of Contents

- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
  - [macOS / Linux](#macos--linux-setup)
  - [Windows](#windows-setup)
- [LLM Configuration](#llm-configuration)
- [Running the Application](#running-the-application)
- [Command-Line Usage](#command-line-usage)
- [Supported File Types](#supported-file-types)
- [Troubleshooting](#troubleshooting)

## Prerequisites

| Software | Purpose |
|----------|---------|
| **Python 3.13** | Backend server and anonymization engine |
| **Node.js v24.x** and **npm v11.x** | Frontend development server |
| **Git LFS** | Downloading pre-trained ML models |
| **FFmpeg** | Audio and video processing |
| **Poppler** | PDF to image conversion |

## Setup Instructions

### macOS / Linux Setup

#### 1. Install System Dependencies

**macOS (Homebrew):**

```bash
brew install python@3.13 ffmpeg poppler git-lfs
```

**Ubuntu / Debian:**

```bash
sudo apt update
sudo apt install python3.13 python3.13-venv ffmpeg poppler-utils git-lfs
```

#### 2. Install Node.js with nvm

```bash
# Install nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash

# Reload shell configuration (or restart your terminal)
. "$HOME/.nvm/nvm.sh"

# Install Node.js
nvm install 24

# Verify installation
node -v  # Should print v24.x.x
npm -v   # Should print 11.x.x
```

#### 3. Clone the Repository and Download Models

```bash
git clone <repository-url>
cd multimodal-anonymizer

# Initialize Git LFS and pull model files
git lfs install
git lfs pull
```

#### 4. Set Up Python Virtual Environment

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### 5. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

---

### Windows Setup

#### 1. Install System Dependencies

**Option A: Using winget (recommended)**

```powershell
winget install Python.Python.3.13
winget install GnuWin32.Poppler
winget install Gyan.FFmpeg
winget install GitHub.GitLFS
```

**Option B: Manual installation**

- **Python 3.13**: Download from [python.org](https://www.python.org/downloads/). During installation, check **"Add Python to PATH"**.
- **FFmpeg**: Download from [ffmpeg.org](https://ffmpeg.org/download.html). Extract and add the `bin` folder to your system PATH.
- **Poppler**: Download from [poppler releases](https://github.com/oschwartz10612/poppler-windows/releases). Extract and add the `bin` folder to your system PATH.
- **Git LFS**: Download from [git-lfs.com](https://git-lfs.com/).

#### 2. Install Node.js with nvm-windows

Download and install [nvm-windows](https://github.com/coreybutler/nvm-windows/releases) (run the installer `.exe`).

```powershell
# Open a new PowerShell or Command Prompt window after installing nvm-windows
nvm install 24
nvm use 24

# Verify installation
node -v
npm -v
```

#### 3. Clone the Repository and Download Models

```powershell
git clone <repository-url>
cd multimodal-anonymizer

git lfs install
git lfs pull
```

#### 4. Set Up Python Virtual Environment

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

#### 5. Install Frontend Dependencies

```powershell
cd frontend
npm install
cd ..
```

## LLM Configuration

The pipeline requires an LLM for anonymization. You can use either a local LLM server or a cloud provider (OpenRouter). Configuration differs depending on whether you use the **web frontend** or the **command-line script**.

---

### Configuring the Web Frontend

The frontend configuration form connects to any **OpenAI-compatible API** — this includes both local servers and cloud providers like OpenRouter.

**Local LLM example (e.g., Ollama):**

| Field | Value |
|-------|-------|
| LLM Endpoint URL | `http://localhost:11434/v1` |
| Model Name | `qwen3.5:9b` |
| API Key | *(leave empty)* |

**OpenRouter example:**

| Field | Value |
|-------|-------|
| LLM Endpoint URL | `https://openrouter.ai/api/v1` |
| Model Name | `anthropic/claude-sonnet-4` |
| API Key | Your OpenRouter API key |

The frontend will test the connection before proceeding to file upload.

---

### Configuring the Command-Line Script (ONLY necessary if you want to run the anonymization in the CLI)

The CLI script reads its configuration from a `.env` file in the project root.

**Local LLM `.env`:**

```env
LLM_PROVIDER=local
LOCAL_BASE_URL=http://localhost:11434/v1
LOCAL_MODEL=qwen3.5:9b
LOCAL_VISION_MODEL=qwen3.5:9b
LOCAL_API_KEY=                # Leave empty if not required
LOCAL_THINKING=false          # Set to true for reasoning models
```

**OpenRouter `.env`:**

```env
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your-api-key-here
OPENROUTER_MODEL=anthropic/claude-sonnet-4
OPENROUTER_VISION_MODEL=anthropic/claude-sonnet-4
OPENROUTER_BASE_URL=base-url-here
```

## Running the Application

### Start the Backend Server

**macOS / Linux:**

```bash
source .venv/bin/activate
cd backend
python -m uvicorn app.main:app
```

**Windows:**

```powershell
.venv\Scripts\activate
cd backend
python -m uvicorn app.main:app
```

The backend will be available at `http://localhost:8000`.

### Start the Frontend Development Server

Open a **second terminal**:

```bash
cd frontend
npm run
```

The frontend will be available at `http://localhost:5173`.

## Command-Line Usage

For batch processing without the web UI:

```bash
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
python anonymize.py --input data/input
```

Input files are read from the path given via `--input` and results are written to `data/output-agentic/` by default.

### Parameters

| Parameter | Short | Default | Description |
|---|---|---|---|
| `--input` | | *(required)* | Input file or directory path |
| `--output` | `-o` | `data/output-agentic` | Output directory |
| `--auto-detect` | `-a` | off | Use multimodal LLM to automatically detect file type and select the appropriate processor |
| `--recursive` | `-r` | off | Process directories recursively, including all subdirectories |
| `--preserve-structure` | `-p` | off | Preserve the exact directory structure in the output (recommended with `--recursive`) |
| `--include-hidden` | | off | Include hidden files and directories (starting with `.`) |
| `--debug` | `-d` | off | Save debug files (JSON metadata, intermediate PNGs from DICOM, etc.) |
| `--no-anonymize-paths` | | off | Disable automatic filename and folder name anonymization |
| `--skip-existing-output` | | off | Skip processing when the expected output file already exists |
| `--tracking-file` | `-t` | disabled | Path to a tracking JSON file for skipping already-processed files |
| `--no-hash` | | off | Disable file hash computation in tracking |
| `--clear-tracking` | | off | Clear all tracking data before processing |
| `--no-parallel` | | off | Disable parallel processing |
| `--workers` | `-w` | `1` | Number of parallel workers |
| `--retry-failed` | | off | Only retry previously failed files from the tracking file (requires `--tracking-file`) |
| `--max-retries` | | `3` | Maximum number of retries per file for transient errors |
| `--retry-rounds` | | `3` | Maximum number of global retry rounds for failed files at the end |
| `--provider` | | env `LLM_PROVIDER` | LLM provider: `openrouter` or `local` |
| `--local-base-url` | | env `LOCAL_BASE_URL` | Base URL for a local LLM server |
| `--local-model` | | env `LOCAL_MODEL` | Model name for the local LLM |
| `--local-vision-model` | | env `LOCAL_VISION_MODEL` | Vision model name for the local LLM (if different from `--local-model`) |
| `--prompt-config` | | `default` | Prompt configuration to use (`default` or `custom`) |

### Examples

```bash
# Process a single file
python anonymize.py --input data/input/report.pdf

# Process a directory recursively, keeping folder structure
python anonymize.py --input data/input --recursive --preserve-structure

# Use 4 parallel workers with a tracking file to resume interrupted runs
python anonymize.py --input data/input -r -p -w 4 -t tracking.json

# Use a local LLM instead of OpenRouter
python anonymize.py --input data/input --provider local --local-base-url http://localhost:1234/v1 --local-model my-model
```

### Prompt Customization

The LLM prompts used during anonymization can be customized by editing [anonymizer/custom_prompts.py](anonymizer/custom_prompts.py). This file contains a `CustomPromptConfig` class that inherits from `PromptConfig` and lets you override prompts for each processing phase:

- **`column_detection_prompt`** — PII column detection for tabular data
- **`text_anonymization_prompt`** — PII redaction in TXT and DOCX files
- **`csv_anonymization_prompt`** — PII redaction in tabular data
- **`image_anonymization_prompt`** — PII detection in images (used with OCR results)
- **`image_verification_prompt`** — Verification that redacted images have no remaining PII
- **`text_verification_prompt`** — Verification of anonymized text against the original
- **`pdf_anonymization_prompt`** — PII detection in PDF documents

To use custom prompts, pass `--prompt-config custom`:

```bash
python anonymize.py --input data/input --prompt-config custom
```

## Troubleshooting

### SSL Certificate Errors (macOS)

The backend includes an automatic SSL fix for macOS. If you still see certificate errors, run:

```bash
pip install --upgrade certifi
```

### Git LFS: Models Not Downloaded

If model files appear as small pointer files instead of large binaries:

```bash
git lfs install
git lfs pull
```

### FFmpeg / Poppler Not Found

Ensure both are on your system PATH:

```bash
ffmpeg -version
pdfinfo -v       # Poppler check (macOS/Linux)
pdftoppm -h      # Poppler check (Windows)
```

### Windows: `activate` Script Not Recognized

Use the full path to activate the virtual environment:

```powershell
.venv\Scripts\Activate.ps1
```

If you get an execution policy error in PowerShell:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Port Already in Use

If port 8000 or 5173 is occupied, the servers will fail to start. Kill the existing process or specify a different port:

```bash
# Backend on a different port
python -m uvicorn app.main:app --port 8001

# Frontend on a different port
npm run -- --port 5174
```

## Citation

If you use our tool in your work, please cite us with the following BibTeX entry TODO BibTex Citation
This repository is part of the paper TODO: paperlink
