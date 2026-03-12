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
import ssl
import certifi

# Fix SSL certificate verification on macOS
# This is necessary because Python on macOS doesn't always have proper SSL certificates
try:
    import urllib.request
    # Create SSL context with certifi's certificates
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    # Set as default for urllib
    urllib.request.install_opener(
        urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
    )
except Exception as e:
    print(f"[WARNING] Could not configure SSL certificates: {e}")

# Load environment variables
load_dotenv()

# Add parent directory to path to import anonymizer package
sys.path.append(str(Path(__file__).parent.parent.parent))

from anonymizer.processors.png_vision_ocr_processor import PNGVisionOCRProcessor
from anonymizer.processors.dicom_vision_ocr_processor import DICOMVisionOCRProcessor, get_dicom_info
from anonymizer.processors.pdf_vision_ocr_processor import PDFVisionOCRProcessor
from anonymizer.processors.video_vision_ocr_processor import VideoVisionOCRProcessor
from anonymizer.processors.agentic_csv_processor import AgenticCSVProcessor
from anonymizer.processors.agentic_excel_processor import AgenticExcelProcessor
from anonymizer.processors.agentic_text_processor import AgenticTextProcessor
from anonymizer.processors.agentic_docx_processor import AgenticDocxProcessor
from anonymizer.processors.agentic_audio_processor import AgenticAudioProcessor
from anonymizer.config import AnonymizerConfig
from anonymizer.filename_anonymizer import FilenameAnonymizer
from anonymizer.prompt_config import PromptConfig, DEFAULT_PROMPT_CONFIG, get_prompt_descriptions, get_template_variables, validate_all_prompts

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

# Mapping files setting: True = save CSV mapping files (default), False = don't save
save_mapping_files: bool = True

# Temporary directory for processing files
TEMP_DIR = Path(tempfile.gettempdir()) / "phi_anonymization"
TEMP_DIR.mkdir(exist_ok=True)

# Progress tracking for jobs
job_progress: Dict[str, Dict] = {}


class CustomLLMConfig(BaseModel):
    llm_url: str = Field(..., description="Custom LLM endpoint URL (OpenAI-compatible)")
    model_name: str = Field(default="llama3.2", description="Model name as known by the local server")
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


@app.post("/api/config/custom")
async def set_custom_llm(llm_config: CustomLLMConfig):
    """
    Configure a custom local LLM endpoint (OpenAI-compatible API).

    Works with any local LLM server:
    - Ollama: http://localhost:11434/v1
    - LM Studio: http://localhost:1234/v1
    - vLLM: http://localhost:8000/v1
    - LocalAI: http://localhost:8080/v1
    - Any other OpenAI-compatible server

    Validates the connection by sending a test message to the LLM.
    """
    global config

    try:
        test_config = AnonymizerConfig(
            llm_provider="local",
            local_base_url=llm_config.llm_url,
            local_model=llm_config.model_name,
            local_api_key=llm_config.api_key,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {str(e)}")

    # Test the connection by sending a simple message to the LLM
    try:
        from anonymizer.llm_factory import create_chat_llm
        llm = create_chat_llm(test_config, temperature=0.0, timeout=15, max_tokens=50)
        response = await asyncio.to_thread(llm.invoke, "Reply with OK.")
        if not response or not response.content:
            raise HTTPException(
                status_code=502,
                detail=f"LLM at {llm_config.llm_url} returned an empty response. "
                       f"Please check that the model '{llm_config.model_name}' is available."
            )
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        # Provide user-friendly error messages for common failures
        if "Connection refused" in error_msg or "ConnectError" in error_msg:
            raise HTTPException(
                status_code=502,
                detail=f"Could not connect to LLM server at {llm_config.llm_url}. "
                       f"Make sure the server is running and the URL is correct."
            )
        elif "404" in error_msg or "Not Found" in error_msg:
            raise HTTPException(
                status_code=502,
                detail=f"Model '{llm_config.model_name}' not found at {llm_config.llm_url}. "
                       f"Check that the model name is correct and the model is downloaded/loaded."
            )
        elif "401" in error_msg or "Unauthorized" in error_msg or "403" in error_msg:
            raise HTTPException(
                status_code=502,
                detail=f"Authentication failed for {llm_config.llm_url}. "
                       f"Please provide a valid API key."
            )
        elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            raise HTTPException(
                status_code=504,
                detail=f"Connection to {llm_config.llm_url} timed out. "
                       f"The server may be starting up or the model may be loading. Please try again."
            )
        else:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to connect to LLM: {error_msg}"
            )

    # Connection test passed — save the config
    config = test_config

    return {
        "status": "success",
        "message": f"Connected to local LLM at {llm_config.llm_url} using model '{llm_config.model_name}'",
        "mode": "custom",
        "model": llm_config.model_name
    }


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
    """
    return {
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
    dicom_metadata_anonymization_prompt: Optional[str] = None
    additional_instructions: Optional[str] = None


@app.get("/api/prompts")
async def get_prompts():
    """Get current prompt configuration."""
    return {
        "prompts": prompt_config.to_dict(),
        "descriptions": get_prompt_descriptions(),
        "template_variables": get_template_variables(),
    }


@app.get("/api/prompts/defaults")
async def get_default_prompts():
    """Get default prompt configuration (for reset functionality)."""
    return {
        "prompts": DEFAULT_PROMPT_CONFIG.to_dict(),
        "descriptions": get_prompt_descriptions(),
        "template_variables": get_template_variables(),
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
        if update.dicom_metadata_anonymization_prompt is not None:
            current["dicom_metadata_anonymization_prompt"] = update.dicom_metadata_anonymization_prompt
        if update.additional_instructions is not None:
            current["additional_instructions"] = update.additional_instructions

        # Validate that required template variables are still present
        validation_errors = validate_all_prompts(current)
        if validation_errors:
            error_messages = []
            for field_name, missing_vars in validation_errors.items():
                error_messages.append(
                    f"{field_name}: missing required placeholder(s) {', '.join(missing_vars)}"
                )
            raise HTTPException(
                status_code=400,
                detail="Some prompts are missing required placeholders that are needed at runtime. "
                       "Please restore them:\n" + "\n".join(error_messages)
            )

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


@app.post("/api/file-info")
async def get_file_info(file: UploadFile = File(...)):
    """
    Get information about an uploaded file, including whether it's a video.

    This is useful for checking DICOM files before processing to determine
    if they are multi-frame (video) DICOMs, allowing the user to choose
    frame-by-frame processing.

    Returns:
        File information including:
        - is_video: True if the file contains multiple frames
        - frame_count: Number of frames (1 for single-frame files)
        - file_type: Detected file type (dicom, video, image, etc.)
        - Additional metadata for DICOM files (dimensions, modality, etc.)
    """
    # Create temporary file to analyze
    job_id = str(uuid.uuid4())
    job_dir = TEMP_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    input_path = job_dir / file.filename

    try:
        # Save uploaded file temporarily
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_extension = input_path.suffix.lower()

        # Check if it's a DICOM file
        if file_extension in ['.dcm', '.dicom']:
            is_dicom = True
        else:
            # Check for DICOM magic bytes for files without extension
            is_dicom = False
            try:
                with open(input_path, 'rb') as f:
                    f.seek(128)
                    magic = f.read(4)
                    if magic == b'DICM':
                        is_dicom = True
            except (IOError, OSError):
                pass

        if is_dicom:
            try:
                info = get_dicom_info(input_path)
                return {
                    "file_type": "dicom",
                    "is_video": info['is_video'],
                    "frame_count": info['frame_count'],
                    "dimensions": info['dimensions'],
                    "is_color": info['is_color'],
                    "modality": info['modality'],
                    "bits_stored": info['bits_stored'],
                    "supports_frame_by_frame": info['is_video'],  # Can use frame-by-frame if it's a video
                }
            except Exception as e:
                return {
                    "file_type": "dicom",
                    "is_video": False,
                    "frame_count": 1,
                    "error": f"Could not read DICOM pixel data: {str(e)}",
                    "supports_frame_by_frame": False,
                }

        elif file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
            # For regular video files, get frame count
            import cv2
            cap = cv2.VideoCapture(str(input_path))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            return {
                "file_type": "video",
                "is_video": True,
                "frame_count": frame_count,
                "dimensions": (width, height),
                "fps": fps,
                "supports_frame_by_frame": True,
            }

        elif file_extension in ['.png', '.jpg', '.jpeg']:
            return {
                "file_type": "image",
                "is_video": False,
                "frame_count": 1,
                "supports_frame_by_frame": False,
            }

        elif file_extension == '.pdf':
            return {
                "file_type": "pdf",
                "is_video": False,
                "frame_count": 1,
                "supports_frame_by_frame": False,
            }

        else:
            return {
                "file_type": "unknown",
                "is_video": False,
                "frame_count": 1,
                "supports_frame_by_frame": False,
            }

    finally:
        # Clean up temporary files
        shutil.rmtree(job_dir, ignore_errors=True)


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


@app.get("/api/mapping-files-settings")
async def get_mapping_files_settings():
    """Get current mapping files settings."""
    return {
        "save_mapping_files": save_mapping_files
    }


class MappingFilesSettingsUpdate(BaseModel):
    save_mapping_files: bool = Field(..., description="If true, save filename_anonymization.csv and folder_anonymization.csv files. If false, don't save.")


@app.post("/api/mapping-files-settings")
async def update_mapping_files_settings(update: MappingFilesSettingsUpdate):
    """Update mapping files settings."""
    global save_mapping_files
    save_mapping_files = update.save_mapping_files
    return {
        "status": "success",
        "save_mapping_files": save_mapping_files
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

        # Anonymize filename using LLM (note: single file processing doesn't save mappings to CSV)
        filename_anonymizer = FilenameAnonymizer(config, save_mappings=False)
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
            processor = DICOMVisionOCRProcessor(config, save_intermediate=True, prompt_config=prompt_config, process_all_frames=video_process_all_frames)
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
        else:
            # Fallback: try DICOM processor for files without extension (common in MIMIC)
            test_processor = DICOMVisionOCRProcessor(config, save_intermediate=True, prompt_config=prompt_config, process_all_frames=video_process_all_frames)
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

        # Check for processor warnings (e.g. verification agent errors)
        processor_warnings = getattr(processor, 'warnings', [])

        response_data = {
            "status": "success",
            "job_id": job_id,
            "message": "File processed successfully",
            "download_url": f"/api/download/{job_id}/{output_file.name}",
            "filename_mapping": filename_mapping
        }

        if processor_warnings:
            response_data["warnings"] = processor_warnings
            response_data["message"] = "File processed with warnings"

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

    # Initialize filename anonymizer with save_mappings option from global setting
    filename_anonymizer = FilenameAnonymizer(config, output_dir=output_dir, save_mappings=save_mapping_files)

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

        # Process all files
        for idx, (input_path, rel_path) in enumerate(saved_files):
            # Update progress
            job_progress[job_id] = {
                "status": "processing",
                "current": idx + 1,
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
                    processor = DICOMVisionOCRProcessor(config, save_intermediate=False, prompt_config=prompt_config, process_all_frames=video_process_all_frames)
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
                else:
                    # Fallback: try DICOM processor for files without extension (common in MIMIC)
                    test_processor = DICOMVisionOCRProcessor(config, save_intermediate=False, prompt_config=prompt_config, process_all_frames=video_process_all_frames)
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

                # Check for processor warnings (e.g. verification agent errors)
                processor_warnings = getattr(processor, 'warnings', [])

                if output_path.exists():
                    file_result = {
                        "original_path": rel_path,
                        "anonymized_filename": output_filename,
                        "status": "warning" if processor_warnings else "success",
                    }
                    if processor_warnings:
                        file_result["warnings"] = processor_warnings
                    file_results.append(file_result)
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
