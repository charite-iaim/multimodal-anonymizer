from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import os
import sys
from pathlib import Path
import tempfile
import shutil
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path to import anonymizer package
sys.path.append(str(Path(__file__).parent.parent.parent))

from anonymizer.file_type_detector import FileTypeDetector, DataType
from anonymizer.processors.png_processor import PNGProcessor
from anonymizer.processors.png_ocr_processor import PNGOCRProcessor
from anonymizer.processors.csv_processor import CSVProcessor
from anonymizer.processors.text_processor import TextProcessor
from anonymizer.processors.dicom_processor import DICOMProcessor
from anonymizer.processors.pdf_ocr_processor import PDFOCRProcessor
from anonymizer.config import AnonymizerConfig
from anonymizer.filename_anonymizer import FilenameAnonymizer

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

# Temporary directory for processing files
TEMP_DIR = Path(tempfile.gettempdir()) / "phi_anonymization"
TEMP_DIR.mkdir(exist_ok=True)


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


@app.post("/api/process")
async def process_file(
    file: UploadFile = File(...),
    mode: str = Form(default="auto")  # auto, vision, ocr
):
    """
    Process an uploaded file for PHI anonymization.

    Args:
        file: The file to process (PNG, JPG, CSV, TXT, DICOM, PDF)
        mode: Processing mode - 'auto' (LLM detection)
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

        # Determine processor
        processor = None

        if mode == "auto":
            # Use LLM-based file type detection
            detector = FileTypeDetector(config)
            detection_result = detector.detect_file_type(input_path)

            if detection_result.data_type == DataType.UNKNOWN:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {detection_result.reasoning}"
                )

            # Get processor based on suggestion
            if detection_result.suggested_processor == "vision":
                processor = PNGProcessor(config)
            elif detection_result.suggested_processor == "ocr":
                processor = PNGOCRProcessor(config)
            elif detection_result.suggested_processor == "text":
                processor = TextProcessor(config)
            elif detection_result.suggested_processor == "csv":
                processor = CSVProcessor(config)
            elif detection_result.suggested_processor == "dicom":
                processor = DICOMProcessor(config, save_intermediate=True)
            elif detection_result.suggested_processor == "pdf_ocr":
                processor = PDFOCRProcessor(config)

        # If LLM detection didn't work, try extension-based detection
        if processor is None or not processor.can_process(input_path):
            # Try all processors to find a match
            for proc_class in [DICOMProcessor, PDFOCRProcessor, PNGProcessor, PNGOCRProcessor, TextProcessor, CSVProcessor]:
                if proc_class == DICOMProcessor:
                    test_processor = proc_class(config, save_intermediate=True)
                else:
                    test_processor = proc_class(config)

                if test_processor.can_process(input_path):
                    processor = test_processor
                    break

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
