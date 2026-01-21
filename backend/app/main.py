from fastapi import FastAPI, File, UploadFile, HTTPException, Form
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

# Load environment variables
load_dotenv()

# Add parent directory to path to import anonymizer package
sys.path.append(str(Path(__file__).parent.parent.parent))

from anonymizer.processors.png_processor import PNGProcessor
from anonymizer.processors.png_ocr_processor import PNGOCRProcessor
from anonymizer.processors.png_vision_ocr_processor import PNGVisionOCRProcessor
from anonymizer.processors.csv_processor import CSVProcessor
from anonymizer.processors.text_processor import TextProcessor
from anonymizer.processors.dicom_processor import DICOMProcessor
from anonymizer.processors.dicom_vision_ocr_processor import DICOMVisionOCRProcessor
from anonymizer.processors.pdf_ocr_processor import PDFOCRProcessor
from anonymizer.processors.pdf_vision_ocr_processor import PDFVisionOCRProcessor
from anonymizer.processors.agentic_csv_processor import AgenticCSVProcessor
from anonymizer.processors.agentic_text_processor import AgenticTextProcessor
from anonymizer.config import AnonymizerConfig
from anonymizer.filename_anonymizer import FilenameAnonymizer
from anonymizer.prompt_config import PromptConfig, DEFAULT_PROMPT_CONFIG, get_prompt_descriptions

app = FastAPI(title="PHI Anonymization API", version="1.0.0")

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
)

# Global config instance
config: Optional[AnonymizerConfig] = None

# Global prompt configuration (customizable via API)
prompt_config: PromptConfig = DEFAULT_PROMPT_CONFIG

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
    file: UploadFile = File(...),
    use_agentic: bool = Form(default=False)  # Use agentic processors for CSV/text
):
    """
    Process an uploaded file for PHI anonymization.

    Args:
        file: The file to process (PNG, JPG, CSV, TXT, DICOM, PDF)
        use_agentic: If True, use agentic processors (tool-calling) for CSV and text files
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

        # Determine processor based on file extension
        processor = None
        file_extension = input_path.suffix.lower()

        # Map file extensions to processors
        # When use_agentic is True, use Vision+OCR processors for images/PDFs/DICOMs
        if file_extension in ['.dcm', '.dicom']:
            processor = DICOMVisionOCRProcessor(config, save_intermediate=True, prompt_config=prompt_config) if use_agentic else DICOMProcessor(config, save_intermediate=True)
        elif file_extension == '.pdf':
            processor = PDFVisionOCRProcessor(config, prompt_config=prompt_config) if use_agentic else PDFOCRProcessor(config, prompt_config=prompt_config)
        elif file_extension in ['.png', '.jpg', '.jpeg']:
            processor = PNGVisionOCRProcessor(config, prompt_config=prompt_config) if use_agentic else PNGOCRProcessor(config, prompt_config=prompt_config)
        elif file_extension in ['.txt', '.hea']:
            processor = AgenticTextProcessor(config, prompt_config=prompt_config) if use_agentic else TextProcessor(config)
        elif file_extension == '.csv':
            processor = AgenticCSVProcessor(config, prompt_config=prompt_config) if use_agentic else CSVProcessor(config)
        else:
            # Fallback: try DICOM processor for files without extension (common in MIMIC)
            if use_agentic:
                test_processor = DICOMVisionOCRProcessor(config, save_intermediate=True, prompt_config=prompt_config)
            else:
                test_processor = DICOMProcessor(config, save_intermediate=True)
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
        if isinstance(processor, DICOMProcessor):
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
    use_agentic: bool = Form(default=False),
    job_id: Optional[str] = Form(default=None)
):
    """
    Process multiple uploaded files (folder) for PHI anonymization.

    Args:
        files: List of files to process
        paths: List of relative paths corresponding to each file
        use_agentic: If True, use agentic processors for CSV and text files
        job_id: Optional job ID for progress tracking (generated if not provided)
    """
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

    try:
        # Save all uploaded files first
        saved_files = []
        for file, rel_path in zip(files, paths):
            # Create directory structure if needed
            file_input_path = input_dir / rel_path
            file_input_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_input_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            saved_files.append((file_input_path, rel_path))

        # Process each file
        for idx, (input_path, rel_path) in enumerate(saved_files):
            # Update progress
            job_progress[job_id] = {
                "status": "processing",
                "current": idx + 1,
                "total": total_files,
                "current_file": rel_path
            }
            try:
                # Anonymize filename
                filename_result = filename_anonymizer.anonymize_filename(
                    input_path.name,
                    is_directory=False,
                    folder_path=str(Path(rel_path).parent) if '/' in rel_path else job_id
                )
                output_filename = filename_result.anonymized_filename

                # Preserve directory structure in output
                rel_output_dir = output_dir / Path(rel_path).parent
                rel_output_dir.mkdir(parents=True, exist_ok=True)
                output_path = rel_output_dir / output_filename

                # Determine processor based on file extension
                processor = None
                file_extension = input_path.suffix.lower()

                # Map file extensions to processors
                # When use_agentic is True, use Vision+OCR processors for images/PDFs/DICOMs
                if file_extension in ['.dcm', '.dicom']:
                    processor = DICOMVisionOCRProcessor(config, save_intermediate=False, prompt_config=prompt_config) if use_agentic else DICOMProcessor(config, save_intermediate=False)
                elif file_extension == '.pdf':
                    processor = PDFVisionOCRProcessor(config, prompt_config=prompt_config) if use_agentic else PDFOCRProcessor(config, prompt_config=prompt_config)
                elif file_extension in ['.png', '.jpg', '.jpeg']:
                    processor = PNGVisionOCRProcessor(config, prompt_config=prompt_config) if use_agentic else PNGOCRProcessor(config, prompt_config=prompt_config)
                elif file_extension in ['.txt', '.hea']:
                    processor = AgenticTextProcessor(config, prompt_config=prompt_config) if use_agentic else TextProcessor(config)
                elif file_extension == '.csv':
                    processor = AgenticCSVProcessor(config, prompt_config=prompt_config) if use_agentic else CSVProcessor(config)
                else:
                    # Fallback: try DICOM processor for files without extension (common in MIMIC)
                    if use_agentic:
                        test_processor = DICOMVisionOCRProcessor(config, save_intermediate=False, prompt_config=prompt_config)
                    else:
                        test_processor = DICOMProcessor(config, save_intermediate=False)
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
