from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import os
import sys
from pathlib import Path
import tempfile
import shutil
import uuid
import zipfile
import asyncio
import json
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()

# Add parent directory to path to import anonymizer package
sys.path.append(str(Path(__file__).parent.parent.parent))

from anonymizer.processors.png_vision_ocr_processor import PNGVisionOCRProcessor
from anonymizer.processors.dicom_vision_ocr_processor import DICOMVisionOCRProcessor
from anonymizer.processors.pdf_vision_ocr_processor import PDFVisionOCRProcessor
from anonymizer.processors.video_vision_ocr_processor import VideoVisionOCRProcessor
from anonymizer.processors.agentic_csv_processor import AgenticCSVProcessor
from anonymizer.processors.agentic_excel_processor import AgenticExcelProcessor
from anonymizer.processors.agentic_text_processor import AgenticTextProcessor
from anonymizer.processors.agentic_docx_processor import AgenticDocxProcessor
from anonymizer.processors.agentic_audio_processor import AgenticAudioProcessor
from anonymizer.processors.mede_processor import (
    MedeProcessor,
    is_mede_available,
    get_mede_status,
    CT_MRI_EXTENSIONS,
    EXTENDED_3D_IMAGE_SUFFIX,
    is_extended_3d_image_folder,
    find_3d_image_folders_in_path,
)
from anonymizer.config import AnonymizerConfig
from anonymizer.filename_anonymizer import FilenameAnonymizer
from anonymizer.prompt_config import PromptConfig, DEFAULT_PROMPT_CONFIG, get_prompt_descriptions

app = FastAPI(title="PHI Anonymization API", version="1.0.0")

# Global exception handler to log errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"[ERROR] Unhandled exception: {exc}")
    print(f"[ERROR] Traceback: {traceback.format_exc()}")
    raise exc

# Middleware to log all incoming requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"[REQUEST] {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        print(f"[ERROR] Request failed: {e}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        raise

# Configure CORS for local development and potential deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",  # Vite default port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "https://*.github.io",  # GitHub Pages domains
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],  # Expose all headers for SSE
)

# Global config instance
config: Optional[AnonymizerConfig] = None

# Global prompt configuration (customizable via API)
prompt_config: PromptConfig = DEFAULT_PROMPT_CONFIG

# Video processing mode: False = first-frame-only (default), True = all-frames
video_process_all_frames: bool = False

# Temporary directory for processing files
TEMP_DIR = Path(tempfile.gettempdir()) / "phi_anonymization"
TEMP_DIR.mkdir(exist_ok=True)

# Progress tracking for jobs
job_progress: Dict[str, Dict] = {}


class CustomLLMConfig(BaseModel):
    llm_url: str = Field(..., description="Custom LLM endpoint URL (OpenAI-compatible)")
    api_key: Optional[str] = Field(default=None, description="Optional API key for custom LLM")


class ProcessingStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, error
    message: Optional[str] = None
    download_url: Optional[str] = None


@app.get("/")
async def root():
    return {
        "message": "PHI Anonymization API",
        "version": "1.0.0",
        "status": "running",
        "config_loaded": config is not None
    }


@app.get("/api/test-mede")
async def test_mede():
    """Test endpoint to check mede availability."""
    return {
        "mede_available": is_mede_available(),
        "mede_status": get_mede_status()
    }


@app.post("/api/config/dev")
async def use_dev_config():
    """Use development configuration from environment variables."""
    global config

    try:
        # Try to load from environment variables
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

        if not all([azure_endpoint, azure_api_key, azure_deployment]):
            raise HTTPException(
                status_code=400,
                detail="Development credentials not found in environment variables. "
                       "Please set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT_NAME"
            )

        config = AnonymizerConfig(
            llm_provider="azure",
            azure_endpoint=azure_endpoint,
            azure_api_key=azure_api_key,
            azure_deployment_name=azure_deployment,
            azure_api_version=os.getenv("AZURE_API_VERSION", "2024-08-01-preview"),
            model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        )

        return {
            "status": "success",
            "message": "Using development configuration",
            "mode": "dev"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {str(e)}")


@app.post("/api/config/custom")
async def set_custom_llm(llm_config: CustomLLMConfig):
    """Configure a custom LLM endpoint (OpenAI-compatible API)."""
    global config

    try:
        # For custom LLM, we'll use OpenAI-compatible endpoint
        # This will require modifying the AnonymizerConfig to support custom endpoints
        # For now, we'll use the URL as the azure_endpoint
        config = AnonymizerConfig(
            azure_endpoint=llm_config.llm_url,
            azure_api_key=llm_config.api_key or "dummy-key",
            azure_deployment_name="custom",  # Not used for custom endpoints
            azure_api_version="2024-08-01-preview",
            model_name="gpt-4o-mini",
        )

        return {
            "status": "success",
            "message": "Custom LLM configuration updated successfully",
            "mode": "custom"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {str(e)}")


@app.get("/api/config/status")
async def get_config_status():
    """Check if LLM configuration is set."""
    return {
        "configured": config is not None,
        "model_name": config.model_name if config else None
    }


# ==================== Feature Availability Endpoints ====================


@app.get("/api/features")
async def get_features():
    """
    Get available features and their status.

    Some features require additional setup (e.g., CT/MRI processing requires
    a separate Python 3.11 environment with mede installed).
    """
    mede_status = get_mede_status()

    return {
        "ct_mri_processing": {
            "available": mede_status["available"],
            "name": "CT/MRI Processing",
            "description": "Anonymize CT and MRI medical images (NIfTI, NRRD, MetaImage formats)",
            "supported_extensions": mede_status["supported_extensions"],
            "setup_required": not mede_status["available"],
            "setup_instructions": mede_status["setup_instructions"] if not mede_status["available"] else None,
            "details": mede_status if not mede_status["available"] else None,
            "folder_convention": {
                "suffix": EXTENDED_3D_IMAGE_SUFFIX,
                "description": "Name folders with suffix '_extended_3d_image' to process DICOM slices as 3D volumes",
                "example": "patient001_ct_extended_3d_image/"
            }
        },
        # Standard features (always available)
        "dicom_processing": {
            "available": True,
            "name": "DICOM Processing",
            "description": "Anonymize DICOM medical images",
            "supported_extensions": [".dcm", ".dicom"],
        },
        "image_processing": {
            "available": True,
            "name": "Image Processing",
            "description": "Anonymize PNG and JPEG images",
            "supported_extensions": [".png", ".jpg", ".jpeg"],
        },
        "document_processing": {
            "available": True,
            "name": "Document Processing",
            "description": "Anonymize PDF, Word, Excel, and text documents",
            "supported_extensions": [".pdf", ".docx", ".xlsx", ".xls", ".txt", ".hea", ".csv"],
        },
        "video_processing": {
            "available": True,
            "name": "Video Processing",
            "description": "Anonymize video files",
            "supported_extensions": [".mp4", ".avi", ".mov", ".mkv"],
        },
        "audio_processing": {
            "available": True,
            "name": "Audio Processing",
            "description": "Anonymize audio files",
            "supported_extensions": [".wav", ".mp3"],
        },
    }


# ==================== Prompt Configuration Endpoints ====================


class PromptConfigUpdate(BaseModel):
    """Request model for updating prompt configuration."""
    column_detection_prompt: Optional[str] = None
    csv_anonymization_prompt: Optional[str] = None
    text_anonymization_prompt: Optional[str] = None
    csv_verification_prompt: Optional[str] = None
    text_verification_prompt: Optional[str] = None
    image_anonymization_prompt: Optional[str] = None
    image_verification_prompt: Optional[str] = None
    pdf_anonymization_prompt: Optional[str] = None
    pdf_verification_prompt: Optional[str] = None
    additional_instructions: Optional[str] = None


@app.get("/api/prompts")
async def get_prompts():
    """Get current prompt configuration."""
    return {
        "prompts": prompt_config.to_dict(),
        "descriptions": get_prompt_descriptions()
    }


@app.get("/api/prompts/defaults")
async def get_default_prompts():
    """Get default prompt configuration (for reset functionality)."""
    return {
        "prompts": DEFAULT_PROMPT_CONFIG.to_dict(),
        "descriptions": get_prompt_descriptions()
    }


@app.post("/api/prompts")
async def update_prompts(update: PromptConfigUpdate):
    """Update prompt configuration."""
    global prompt_config

    try:
        # Get current config as dict
        current = prompt_config.to_dict()

        # Update only provided fields
        if update.column_detection_prompt is not None:
            current["column_detection_prompt"] = update.column_detection_prompt
        if update.csv_anonymization_prompt is not None:
            current["csv_anonymization_prompt"] = update.csv_anonymization_prompt
        if update.text_anonymization_prompt is not None:
            current["text_anonymization_prompt"] = update.text_anonymization_prompt
        if update.csv_verification_prompt is not None:
            current["csv_verification_prompt"] = update.csv_verification_prompt
        if update.text_verification_prompt is not None:
            current["text_verification_prompt"] = update.text_verification_prompt
        if update.image_anonymization_prompt is not None:
            current["image_anonymization_prompt"] = update.image_anonymization_prompt
        if update.image_verification_prompt is not None:
            current["image_verification_prompt"] = update.image_verification_prompt
        if update.pdf_anonymization_prompt is not None:
            current["pdf_anonymization_prompt"] = update.pdf_anonymization_prompt
        if update.pdf_verification_prompt is not None:
            current["pdf_verification_prompt"] = update.pdf_verification_prompt
        if update.additional_instructions is not None:
            current["additional_instructions"] = update.additional_instructions

        # Create new config
        prompt_config = PromptConfig.from_dict(current)

        return {
            "status": "success",
            "message": "Prompt configuration updated",
            "prompts": prompt_config.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid prompt configuration: {str(e)}")


@app.post("/api/prompts/reset")
async def reset_prompts():
    """Reset prompts to default values."""
    global prompt_config
    prompt_config = PromptConfig()
    return {
        "status": "success",
        "message": "Prompts reset to defaults",
        "prompts": prompt_config.to_dict()
    }


@app.get("/api/video-settings")
async def get_video_settings():
    """Get current video processing settings."""
    return {
        "process_all_frames": video_process_all_frames
    }


class VideoSettingsUpdate(BaseModel):
    process_all_frames: bool = Field(..., description="If true, process every frame. If false, detect on first frame only.")


@app.post("/api/video-settings")
async def update_video_settings(update: VideoSettingsUpdate):
    """Update video processing settings."""
    global video_process_all_frames
    video_process_all_frames = update.process_all_frames
    return {
        "status": "success",
        "process_all_frames": video_process_all_frames
    }


@app.get("/api/progress/{job_id}")
async def get_progress(job_id: str):
    """SSE endpoint for real-time progress updates."""
    async def event_generator():
        while True:
            if job_id in job_progress:
                progress = job_progress[job_id]
                yield f"data: {json.dumps(progress)}\n\n"

                if progress.get("status") in ["completed", "error"]:
                    # Clean up progress tracking
                    del job_progress[job_id]
                    break

            await asyncio.sleep(0.3)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/api/process")
async def process_file(
    file: UploadFile = File(...)
):
    """
    Process an uploaded file for PHI anonymization using agentic processors.

    Args:
        file: The file to process (PNG, JPG, CSV, TXT, DICOM, PDF)
    """
    if config is None:
        raise HTTPException(
            status_code=400,
            detail="LLM configuration not set. Please configure the API endpoint first."
        )

    # Generate unique job ID
    job_id = str(uuid.uuid4())

    # Create job-specific directory
    job_dir = TEMP_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    input_path = job_dir / "input" / file.filename
    input_path.parent.mkdir(exist_ok=True)
    output_dir = job_dir / "output"
    output_dir.mkdir(exist_ok=True)

    try:
        # Save uploaded file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Anonymize filename using LLM
        filename_anonymizer = FilenameAnonymizer(config)
        filename_result = filename_anonymizer.anonymize_filename(file.filename, is_directory=False, folder_path=job_id)
        output_filename = filename_result.anonymized_filename
        output_path = output_dir / output_filename

        # Store filename mapping for later retrieval
        filename_mapping = {
            "original_filename": file.filename,
            "anonymized_filename": output_filename,
            "phi_detections": [
                {
                    "original_value": detection.original_value,
                    "category": detection.category
                }
                for detection in filename_result.phi_detections
            ]
        }

        # Determine processor based on file extension (always use agentic/vision processors)
        processor = None
        file_extension = input_path.suffix.lower()

        if file_extension in ['.dcm', '.dicom']:
            processor = DICOMVisionOCRProcessor(config, save_intermediate=True, prompt_config=prompt_config)
        elif file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
            processor = VideoVisionOCRProcessor(config, save_intermediate=True, prompt_config=prompt_config, process_all_frames=video_process_all_frames)
        elif file_extension == '.pdf':
            processor = PDFVisionOCRProcessor(config, prompt_config=prompt_config)
        elif file_extension in ['.png', '.jpg', '.jpeg']:
            processor = PNGVisionOCRProcessor(config, prompt_config=prompt_config)
        elif file_extension in ['.txt', '.hea']:
            processor = AgenticTextProcessor(config, prompt_config=prompt_config)
        elif file_extension == '.csv':
            processor = AgenticCSVProcessor(config, prompt_config=prompt_config)
        elif file_extension in ['.xlsx', '.xls']:
            processor = AgenticExcelProcessor(config, prompt_config=prompt_config)
        elif file_extension == '.docx':
            processor = AgenticDocxProcessor(config, prompt_config=prompt_config)
        elif file_extension in ['.wav', '.mp3']:
            processor = AgenticAudioProcessor(config, prompt_config=prompt_config)
        elif file_extension in CT_MRI_EXTENSIONS or input_path.name.lower().endswith('.nii.gz'):
            # CT/MRI file - check if mede is available
            if not is_mede_available():
                raise HTTPException(
                    status_code=400,
                    detail="CT/MRI processing requires additional setup. See docs/MEDE_SETUP.md for instructions."
                )
            processor = MedeProcessor(config)
        else:
            # Fallback: try DICOM processor for files without extension (common in MIMIC)
            test_processor = DICOMVisionOCRProcessor(config, save_intermediate=True, prompt_config=prompt_config)
            if test_processor.can_process(input_path):
                processor = test_processor

        if processor is None:
            raise HTTPException(
                status_code=400,
                detail="No suitable processor found for this file type"
            )

        # Process the file
        processor.anonymize(input_path, output_path)

        # Check if the output file was created
        if not output_path.exists():
            raise HTTPException(
                status_code=500,
                detail="Anonymization completed but no output file was generated"
            )

        output_file = output_path

        # Check if there's a debug video for multi-frame DICOMs
        video_url = None
        intermediate_dir = output_dir / "intermediate"
        if intermediate_dir.exists():
            # Look for MP4 files
            video_files = list(intermediate_dir.glob("*.mp4"))
            if video_files:
                video_file = video_files[0]
                # Copy video to output dir for easier access
                video_output = output_dir / video_file.name
                shutil.copy2(video_file, video_output)
                video_url = f"/api/download/{job_id}/{video_output.name}"

        response_data = {
            "status": "success",
            "job_id": job_id,
            "message": "File processed successfully",
            "download_url": f"/api/download/{job_id}/{output_file.name}",
            "filename_mapping": filename_mapping
        }

        if video_url:
            response_data["video_url"] = video_url
            response_data["message"] = "Multi-frame DICOM processed successfully. Debug video available."

        return response_data

    except HTTPException:
        # Clean up on known errors
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    except Exception as e:
        # Clean up on unexpected errors
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/api/process-folder")
async def process_folder(
    files: List[UploadFile] = File(...),
    paths: List[str] = Form(...),
    job_id: Optional[str] = Form(default=None)
):
    """
    Process multiple uploaded files (folder) for PHI anonymization using agentic processors.

    Args:
        files: List of files to process
        paths: List of relative paths corresponding to each file
        job_id: Optional job ID for progress tracking (generated if not provided)
    """
    print(f"[DEBUG] process_folder called with {len(files)} files")
    print(f"[DEBUG] paths: {paths[:5]}{'...' if len(paths) > 5 else ''}")
    
    if config is None:
        raise HTTPException(
            status_code=400,
            detail="LLM configuration not set. Please configure the API endpoint first."
        )

    # Use provided job ID or generate a new one
    if not job_id:
        job_id = str(uuid.uuid4())
    total_files = len(files)

    # Initialize progress tracking
    job_progress[job_id] = {
        "status": "processing",
        "current": 0,
        "total": total_files,
        "current_file": ""
    }

    # Create job-specific directory
    job_dir = TEMP_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    input_dir = job_dir / "input"
    input_dir.mkdir(exist_ok=True)
    output_dir = job_dir / "output"
    output_dir.mkdir(exist_ok=True)

    file_results = []
    files_processed = 0
    files_failed = 0

    # Initialize filename anonymizer
    filename_anonymizer = FilenameAnonymizer(config)

    # Dictionary to store folder name mappings (original -> anonymized)
    folder_mapping = {}

    try:
        # Save all uploaded files first
        saved_files = []
        for idx, (file, rel_path) in enumerate(zip(files, paths)):
            # Update progress during file saving phase
            job_progress[job_id] = {
                "status": "processing",
                "current": idx + 1,
                "total": total_files,
                "current_file": f"Saving: {rel_path}"
            }

            # Create directory structure if needed
            file_input_path = input_dir / rel_path
            file_input_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_input_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            saved_files.append((file_input_path, rel_path))

        # Pre-anonymize all folder names
        unique_folders = set()
        for _, rel_path in saved_files:
            # Get all folder names in the path (excluding the filename)
            for part in Path(rel_path).parent.parts:
                unique_folders.add(part)

        # Anonymize each unique folder name
        for folder_name in sorted(unique_folders):
            # Check if folder was already anonymized
            existing_mapping = filename_anonymizer.get_existing_folder_mapping(folder_name)
            if existing_mapping:
                folder_mapping[folder_name] = existing_mapping
            else:
                folder_result = filename_anonymizer.anonymize_filename(folder_name, is_directory=True)
                folder_mapping[folder_name] = folder_result.anonymized_filename
                filename_anonymizer.add_folder_mapping(
                    original_foldername=folder_name,
                    anonymized_foldername=folder_result.anonymized_filename,
                    phi_detections=folder_result.phi_detections
                )

        # Identify 3D DICOM folders (folders ending with _extended_3d_image)
        # and group files by their parent 3D folder
        extended_3d_folders = {}  # folder_path -> list of (input_path, rel_path)
        regular_files = []  # files not in _extended_3d_image folders
        
        print(f"[DEBUG] Processing {len(saved_files)} saved files")
        
        for input_path, rel_path in saved_files:
            # Check if any parent folder is an extended 3D image folder
            path_parts = Path(rel_path).parts
            found_3d_folder = None
            
            print(f"[DEBUG] Checking file: {rel_path}, parts: {path_parts}")
            
            for i, part in enumerate(path_parts[:-1]):  # Exclude filename
                if is_extended_3d_image_folder(part):
                    # Get the full relative path up to and including the 3D folder
                    found_3d_folder = str(Path(*path_parts[:i+1]))
                    print(f"[DEBUG] Found 3D folder: {found_3d_folder}")
                    break
            
            if found_3d_folder:
                if found_3d_folder not in extended_3d_folders:
                    extended_3d_folders[found_3d_folder] = []
                extended_3d_folders[found_3d_folder].append((input_path, rel_path))
            else:
                regular_files.append((input_path, rel_path))
        
        print(f"[DEBUG] Found {len(extended_3d_folders)} 3D folders, {len(regular_files)} regular files")

        # Process 3D DICOM folders first (as whole folders with mede)
        processed_3d_folders = set()
        print(f"[DEBUG] Processing {len(extended_3d_folders)} 3D folders")
        
        for folder_rel_path, folder_files in extended_3d_folders.items():
            print(f"[DEBUG] Processing 3D folder: {folder_rel_path} with {len(folder_files)} files")
            
            # Update progress
            job_progress[job_id] = {
                "status": "processing",
                "current": files_processed + files_failed + 1,
                "total": total_files,
                "current_file": f"Processing 3D volume: {folder_rel_path}"
            }
            
            try:
                # Check if mede is available
                mede_available = is_mede_available()
                print(f"[DEBUG] mede available: {mede_available}")
                
                if not mede_available:
                    print(f"[DEBUG] mede not available, marking {len(folder_files)} files as error")
                    for _, rel_path in folder_files:
                        file_results.append({
                            "original_path": rel_path,
                            "status": "error",
                            "error": "3D DICOM processing requires mede setup. See docs/MEDE_SETUP.md"
                        })
                        files_failed += 1
                    continue
                
                # Get the input folder path
                input_folder = input_dir / folder_rel_path
                
                # Build anonymized folder path
                folder_path_obj = Path(folder_rel_path)
                anonymized_folder_parts = []
                for part in folder_path_obj.parts:
                    anonymized_folder_parts.append(folder_mapping.get(part, part))
                anonymized_folder_rel = str(Path(*anonymized_folder_parts)) if anonymized_folder_parts else folder_rel_path
                
                output_folder = output_dir / anonymized_folder_rel
                
                # Process the entire folder as a 3D volume
                processor = MedeProcessor(config)
                processor.anonymize(input_folder, output_folder)
                
                # Mark all files in the folder as processed
                for _, rel_path in folder_files:
                    file_results.append({
                        "original_path": rel_path,
                        "anonymized_filename": f"(3D volume: {anonymized_folder_rel})",
                        "status": "success"
                    })
                    files_processed += 1
                
                processed_3d_folders.add(folder_rel_path)
                
            except Exception as e:
                for _, rel_path in folder_files:
                    file_results.append({
                        "original_path": rel_path,
                        "status": "error",
                        "error": str(e)
                    })
                    files_failed += 1

        # Process regular files (not in 3D folders)
        for idx, (input_path, rel_path) in enumerate(regular_files):
            # Update progress (account for already processed 3D folder files)
            current_progress = files_processed + files_failed + idx + 1
            job_progress[job_id] = {
                "status": "processing",
                "current": current_progress,
                "total": total_files,
                "current_file": f"Processing: {rel_path}"
            }
            try:
                # Build anonymized relative path by replacing folder names
                rel_path_obj = Path(rel_path)
                anonymized_parts = []
                for part in rel_path_obj.parent.parts:
                    anonymized_parts.append(folder_mapping.get(part, part))
                anonymized_rel_dir = Path(*anonymized_parts) if anonymized_parts else Path('.')

                # Use anonymized folder path for CSV storage
                anonymized_folder_path = str(anonymized_rel_dir) if anonymized_rel_dir != Path('.') else ""

                # Anonymize filename
                filename_result = filename_anonymizer.anonymize_filename(
                    input_path.name,
                    is_directory=False,
                    folder_path=str(Path(rel_path).parent) if '/' in rel_path else job_id
                )
                output_filename = filename_result.anonymized_filename

                # Preserve anonymized directory structure in output
                rel_output_dir = output_dir / anonymized_rel_dir
                rel_output_dir.mkdir(parents=True, exist_ok=True)
                output_path = rel_output_dir / output_filename

                # Record file mapping using anonymized folder path
                filename_anonymizer.add_file_mapping(
                    folder_path=anonymized_folder_path,
                    original_filename=input_path.name,
                    anonymized_filename=output_filename,
                    phi_detections=filename_result.phi_detections
                )

                # Determine processor based on file extension (always use agentic/vision processors)
                processor = None
                file_extension = input_path.suffix.lower()

                if file_extension in ['.dcm', '.dicom']:
                    processor = DICOMVisionOCRProcessor(config, save_intermediate=False, prompt_config=prompt_config)
                elif file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
                    processor = VideoVisionOCRProcessor(config, save_intermediate=False, prompt_config=prompt_config, process_all_frames=video_process_all_frames)
                elif file_extension == '.pdf':
                    processor = PDFVisionOCRProcessor(config, prompt_config=prompt_config)
                elif file_extension in ['.png', '.jpg', '.jpeg']:
                    processor = PNGVisionOCRProcessor(config, prompt_config=prompt_config)
                elif file_extension in ['.txt', '.hea']:
                    processor = AgenticTextProcessor(config, prompt_config=prompt_config)
                elif file_extension == '.csv':
                    processor = AgenticCSVProcessor(config, prompt_config=prompt_config)
                elif file_extension in ['.xlsx', '.xls']:
                    processor = AgenticExcelProcessor(config, prompt_config=prompt_config)
                elif file_extension == '.docx':
                    processor = AgenticDocxProcessor(config, prompt_config=prompt_config)
                elif file_extension in ['.wav', '.mp3']:
                    processor = AgenticAudioProcessor(config, prompt_config=prompt_config)
                elif file_extension in CT_MRI_EXTENSIONS or input_path.name.lower().endswith('.nii.gz'):
                    # CT/MRI file - check if mede is available
                    if not is_mede_available():
                        file_results.append({
                            "original_path": rel_path,
                            "status": "error",
                            "error": "CT/MRI processing requires additional setup. See docs/MEDE_SETUP.md"
                        })
                        files_failed += 1
                        continue
                    processor = MedeProcessor(config)
                else:
                    # Fallback: try DICOM processor for files without extension (common in MIMIC)
                    test_processor = DICOMVisionOCRProcessor(config, save_intermediate=False, prompt_config=prompt_config)
                    if test_processor.can_process(input_path):
                        processor = test_processor

                if processor is None:
                    file_results.append({
                        "original_path": rel_path,
                        "status": "error",
                        "error": "No suitable processor found"
                    })
                    files_failed += 1
                    continue

                # Process the file
                processor.anonymize(input_path, output_path)

                if output_path.exists():
                    file_results.append({
                        "original_path": rel_path,
                        "anonymized_filename": output_filename,
                        "status": "success"
                    })
                    files_processed += 1
                else:
                    file_results.append({
                        "original_path": rel_path,
                        "status": "error",
                        "error": "Output file not generated"
                    })
                    files_failed += 1

            except Exception as e:
                file_results.append({
                    "original_path": rel_path,
                    "status": "error",
                    "error": str(e)
                })
                files_failed += 1

        # Save filename anonymizer mappings (CSV files) to output directory
        filename_anonymizer.save_all_mappings(output_dir=output_dir)

        # Create ZIP archive of all output files
        zip_filename = f"anonymized_{job_id[:8]}.zip"
        zip_path = job_dir / zip_filename

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files_in_dir in os.walk(output_dir):
                for file in files_in_dir:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(output_dir)
                    zipf.write(file_path, arcname)

        # Mark progress as completed
        job_progress[job_id] = {
            "status": "completed",
            "current": total_files,
            "total": total_files,
            "current_file": ""
        }

        return {
            "status": "success",
            "job_id": job_id,
            "message": f"Processed {files_processed} files ({files_failed} failed)",
            "files_processed": files_processed,
            "files_failed": files_failed,
            "file_results": file_results,
            "download_url": f"/api/download-zip/{job_id}/{zip_filename}",
            "is_batch": True
        }

    except Exception as e:
        # Mark progress as error
        if job_id in job_progress:
            job_progress[job_id] = {
                "status": "error",
                "current": 0,
                "total": total_files,
                "error": str(e)
            }
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/api/download-zip/{job_id}/{filename}")
async def download_zip(job_id: str, filename: str):
    """Download a ZIP archive of anonymized files."""
    file_path = TEMP_DIR / job_id / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="ZIP file not found")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/zip"
    )


@app.get("/api/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    """Download an anonymized file."""
    file_path = TEMP_DIR / job_id / "output" / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )


@app.delete("/api/cleanup/{job_id}")
async def cleanup_job(job_id: str):
    """Clean up temporary files for a job."""
    job_dir = TEMP_DIR / job_id

    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
        return {"status": "success", "message": "Job files cleaned up"}

    return {"status": "success", "message": "No files to clean up"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
