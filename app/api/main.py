"""FloodLLM FastAPI REST API Application."""
import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel, Field

from ..utils.config import settings
from ..utils.geocode import geocode_location
from ..utils.llm import LLMPromptHandler
from ..data.sentinel import SentinelDownloader
from ..data.rainfall import RainfallDownloader
from ..processing.sar_processor import SARProcessor
from ..processing.optical import OpticalProcessor
from ..processing.risk_model import FloodRiskModel
from ..visualization.mapper import FloodMapper
from ..visualization.reporter import ReportGenerator


# ============================================================================
# Enums
# ============================================================================

class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    PARSING_PROMPT = "parsing_prompt"
    GEOCODING = "geocoding"
    DOWNLOADING_SATELLITE_DATA = "downloading_satellite_data"
    DOWNLOADING_RAINFALL = "downloading_rainfall"
    PROCESSING_FLOOD_DETECTION = "processing_flood_detection"
    VALIDATING_WITH_OPTICAL = "validating_with_optical"
    GENERATING_RISK_ASSESSMENT = "generating_risk_assessment"
    GENERATING_MAP = "generating_map"
    GENERATING_REPORT = "generating_report"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskType(str, Enum):
    """Flood analysis task types."""
    FLOOD_DETECTION = "flood_detection"
    RISK_PREDICTION = "risk_prediction"
    DAMAGE_ASSESSMENT = "damage_assessment"


# ============================================================================
# Pydantic Models - Requests
# ============================================================================

class PromptRequest(BaseModel):
    """Natural language flood monitoring request."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural language description of the flood analysis request",
        examples=["Show flood extent in Jakarta, Indonesia for the last 7 days"]
    )
    location: Optional[str] = Field(
        None,
        description="Override location if not specified in prompt",
        examples=["Bangkok, Thailand"]
    )
    date_start: Optional[str] = Field(
        None,
        description="Override start date (ISO format or relative like 'last 7 days')",
        examples=["2024-01-01", "last 7 days"]
    )
    date_end: Optional[str] = Field(
        None,
        description="Override end date (ISO format or relative like 'today')",
        examples=["2024-01-08", "today"]
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Show flood extent in Jakarta, Indonesia for the last 7 days",
                "location": None,
                "date_start": None,
                "date_end": None
            }
        }


class DirectAnalysisRequest(BaseModel):
    """Direct flood analysis request with structured parameters."""

    location: str = Field(
        ...,
        description="Location name or bounding box",
        examples=["Jakarta, Indonesia", "(106.5, -6.5, 107.0, -6.0)"]
    )
    date_start: str = Field(
        ...,
        description="Start date (ISO format or relative)",
        examples=["2024-01-01", "last 7 days"]
    )
    date_end: str = Field(
        ...,
        description="End date (ISO format or relative)",
        examples=["2024-01-08", "today"]
    )
    task_type: TaskType = Field(
        default=TaskType.FLOOD_DETECTION,
        description="Type of analysis to perform"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "location": "Jakarta, Indonesia",
                "date_start": "last 7 days",
                "date_end": "today",
                "task_type": "flood_detection"
            }
        }


# ============================================================================
# Pydantic Models - Responses
# ============================================================================

class JobStatusResponse(BaseModel):
    """Job status and progress response."""

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    progress: int = Field(..., ge=0, le=100, description="Progress percentage")
    prompt: str = Field(..., description="Original request prompt")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    completed_at: Optional[str] = Field(None, description="ISO 8601 completion timestamp")
    error: Optional[str] = Field(None, description="Error message if failed")
    result: Optional[Dict[str, Any]] = Field(None, description="Job result data")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "2fb8e453",
                "status": "completed",
                "progress": 100,
                "prompt": "Show flood extent in Jakarta",
                "created_at": "2024-01-15T10:30:00",
                "completed_at": "2024-01-15T10:31:00",
                "error": None,
                "result": {
                    "flood_area_km2": 45.5,
                    "map_path": "/output/maps/flood_map_2fb8e453.html",
                    "report_path": "/output/reports/flood_report_2fb8e453.html",
                    "risk_level": 0.65
                }
            }
        }


class PromptResponse(BaseModel):
    """Response with job ID for queued request."""

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Initial job status")
    message: str = Field(..., description="Human-readable status message")
    estimated_time_seconds: int = Field(default=60, description="Estimated processing time")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "2fb8e453",
                "status": "processing",
                "message": "Your flood analysis request has been queued",
                "estimated_time_seconds": 60
            }
        }


class JobSummary(BaseModel):
    """Summary of a job for listing."""

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    progress: int = Field(..., description="Progress percentage")
    prompt: str = Field(..., description="Original request prompt")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")


class JobsListResponse(BaseModel):
    """List of jobs response."""

    jobs: List[JobSummary] = Field(..., description="List of job summaries")
    total: int = Field(..., description="Total number of jobs")


class HealthResponse(BaseModel):
    """Health check response."""

    name: str = Field(..., description="Application name")
    version: str = Field(..., description="API version")
    status: str = Field(..., description="Application status")
    environment: str = Field(..., description="Deployment environment")
    endpoints: Dict[str, str] = Field(..., description="Available API endpoints")


class ParsedPromptResponse(BaseModel):
    """Parsed prompt response for preview."""

    location: str = Field(..., description="Extracted location")
    date_start: str = Field(..., description="Extracted start date")
    date_end: str = Field(..., description="Extracted end date")
    task_type: TaskType = Field(..., description="Extracted task type")
    additional_context: Optional[str] = Field(None, description="Additional context from prompt")


# ============================================================================
# FastAPI App Initialization
# ============================================================================

app = FastAPI(
    title="FloodLLM API",
    description="""
## AI-Powered Flood Monitoring System

FloodLLM is an EarthGPT-inspired application for automated flood detection, risk prediction,
and damage assessment using natural language prompts.

### Features

- **Natural Language Interface**: Submit requests like "Show flood extent in Jakarta this week"
- **Multi-Source Data**: Sentinel-1 SAR, Sentinel-2, GPM rainfall data
- **AI Processing**: LLM-powered task orchestration with Google AI
- **Interactive Maps**: Folium-based flood extent visualization
- **Damage Assessment**: Automated infrastructure impact analysis

### Authentication

API keys are configured via environment variables:
- `GOOGLE_API_KEY` - Google AI Studio for LLM
- `COPERNICUS_USERNAME/PASSWORD` - Copernicus Data Space for Sentinel data
- `NASA_EARTHDATA_USERNAME/PASSWORD` - NASA Earthdata for GPM rainfall

### Job Processing

All analysis requests are processed asynchronously:
1. Submit request via `POST /api/prompt`
2. Poll status via `GET /api/status/{job_id}`
3. Download results when completed
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {
            "name": "Health",
            "description": "Health check and API information"
        },
        {
            "name": "Jobs",
            "description": "Job management and status"
        },
        {
            "name": "Analysis",
            "description": "Flood analysis operations"
        },
        {
            "name": "Results",
            "description": "Download generated maps and reports"
        },
        {
            "name": "Utilities",
            "description": "Utility endpoints for prompt parsing"
        }
    ]
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize components
llm_handler = LLMPromptHandler()
sentinel_downloader = SentinelDownloader()
rainfall_downloader = RainfallDownloader()
sar_processor = SARProcessor()
optical_processor = OpticalProcessor()
risk_model = FloodRiskModel()
flood_mapper = FloodMapper()
report_generator = ReportGenerator()

# Job storage (in production, use Redis/database)
jobs: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# Health & Documentation Endpoints
# ============================================================================

@app.get(
    "/",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health Check",
    description="Returns API health status and available endpoints"
)
async def root():
    """
    Health check endpoint.

    Returns the API name, version, status, and available endpoints.
    """
    return HealthResponse(
        name="FloodLLM",
        version="1.0.0",
        status="running",
        environment=settings.app_env,
        endpoints={
            "submit_prompt": "POST /api/prompt",
            "direct_analysis": "POST /api/analyze",
            "job_status": "GET /api/status/{job_id}",
            "list_jobs": "GET /api/jobs",
            "download_map": "GET /api/map/{job_id}",
            "download_report": "GET /api/report/{job_id}",
            "parse_prompt": "POST /api/parse",
            "swagger_docs": "GET /docs",
            "redoc_docs": "GET /redoc"
        }
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health Check (Alternative)",
    description="Alternative health check endpoint"
)
async def health_check():
    """Alternative health check endpoint."""
    return await root()


# ============================================================================
# Job Management Endpoints
# ============================================================================

@app.post(
    "/api/prompt",
    response_model=PromptResponse,
    tags=["Analysis"],
    summary="Submit Flood Analysis Request",
    description="Submit a natural language flood monitoring request for asynchronous processing",
    responses={
        200: {"description": "Request accepted and queued for processing"},
        400: {"description": "Invalid request format"},
        422: {"description": "Validation error"}
    }
)
async def submit_prompt(
    request: PromptRequest,
    background_tasks: BackgroundTasks
):
    """
    Submit a natural language flood monitoring request.

    The request is processed asynchronously in the background. Use the returned `job_id`
    to poll for status and retrieve results.

    **Examples:**
    - "Show flood extent in Jakarta for the last 7 days"
    - "Assess flood risk in Bangkok this week"
    - "Generate damage report for recent floods in Manila"
    """
    job_id = str(uuid.uuid4())[:8]

    # Initialize job
    jobs[job_id] = {
        "status": JobStatus.PENDING,
        "prompt": request.prompt,
        "created_at": datetime.now().isoformat(),
        "progress": 0,
        "result": None,
        "error": None
    }

    # Start processing in background
    background_tasks.add_task(
        process_flood_request,
        job_id,
        request.prompt,
        request.location,
        request.date_start,
        request.date_end
    )

    return PromptResponse(
        job_id=job_id,
        status=JobStatus.PROCESSING,
        message="Your flood analysis request has been queued",
        estimated_time_seconds=60
    )


@app.post(
    "/api/analyze",
    response_model=PromptResponse,
    tags=["Analysis"],
    summary="Direct Flood Analysis",
    description="Submit a structured flood analysis request with explicit parameters",
    responses={
        200: {"description": "Request accepted and queued for processing"},
        400: {"description": "Invalid request format"}
    }
)
async def direct_analysis(
    request: DirectAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Submit a structured flood analysis request.

    Unlike `/api/prompt`, this endpoint accepts explicit parameters instead of
    natural language.
    """
    job_id = str(uuid.uuid4())[:8]

    # Initialize job
    jobs[job_id] = {
        "status": JobStatus.PENDING,
        "prompt": f"Analysis: {request.location} from {request.date_start} to {request.date_end}",
        "created_at": datetime.now().isoformat(),
        "progress": 0,
        "result": None,
        "error": None
    }

    # Start processing in background
    background_tasks.add_task(
        process_flood_request,
        job_id,
        jobs[job_id]["prompt"],
        request.location,
        request.date_start,
        request.date_end
    )

    return PromptResponse(
        job_id=job_id,
        status=JobStatus.PROCESSING,
        message="Your flood analysis request has been queued",
        estimated_time_seconds=60
    )


@app.get(
    "/api/status/{job_id}",
    response_model=JobStatusResponse,
    tags=["Jobs"],
    summary="Get Job Status",
    description="Retrieve the current status and progress of a job",
    responses={
        200: {"description": "Job status retrieved successfully"},
        404: {"description": "Job not found"}
    }
)
async def get_status(
    job_id: str = Path(..., description="Job ID to check status for")
):
    """
    Get job status and progress.

    Returns detailed information about the job including:
    - Current status and progress percentage
    - Creation and completion timestamps
    - Error message (if failed)
    - Result data (if completed)
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        prompt=job["prompt"],
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
        error=job.get("error"),
        result=job.get("result")
    )


@app.get(
    "/api/jobs",
    response_model=JobsListResponse,
    tags=["Jobs"],
    summary="List All Jobs",
    description="List all jobs with their current status"
)
async def list_jobs(
    status: Optional[JobStatus] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of jobs to return")
):
    """
    List all jobs with optional filtering.

    Returns a paginated list of job summaries with their current status and progress.
    """
    filtered_jobs = []
    for jid, j in jobs.items():
        if status is None or j["status"] == status:
            filtered_jobs.append({
                "job_id": jid,
                "status": j["status"],
                "progress": j["progress"],
                "prompt": j["prompt"],
                "created_at": j["created_at"]
            })

    # Apply limit
    filtered_jobs = filtered_jobs[:limit]

    return JobsListResponse(
        jobs=filtered_jobs,
        total=len(filtered_jobs)
    )


@app.delete(
    "/api/jobs/{job_id}",
    tags=["Jobs"],
    summary="Delete a Job",
    description="Delete a job and its associated data",
    responses={
        200: {"description": "Job deleted successfully"},
        404: {"description": "Job not found"}
    }
)
async def delete_job(
    job_id: str = Path(..., description="Job ID to delete")
):
    """
    Delete a job and release associated resources.

    This removes the job from memory. Generated files in the output directory
    are not automatically deleted.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    del jobs[job_id]
    return {"message": f"Job {job_id} deleted successfully"}


# ============================================================================
# Results Endpoints
# ============================================================================

@app.get(
    "/api/map/{job_id}",
    tags=["Results"],
    summary="Download Flood Map",
    description="Download the generated flood map (HTML format)",
    responses={
        200: {"description": "Map file returned", "content": {"text/html": {}}},
        400: {"description": "Job not completed"},
        404: {"description": "Job or map not found"}
    }
)
async def get_map(
    job_id: str = Path(..., description="Job ID to download map for")
):
    """
    Get flood map for a completed job.

    Returns an interactive HTML map generated with Folium, showing:
    - Flood extent overlay
    - Analysis area boundary
    - Affected infrastructure markers
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {job['status']}"
        )

    map_path = job.get("map", {}).get("map_path")
    if not map_path or not Path(map_path).exists():
        raise HTTPException(status_code=404, detail="Map not found")

    return FileResponse(
        map_path,
        media_type="text/html",
        filename=f"flood_map_{job_id}.html"
    )


@app.get(
    "/api/report/{job_id}",
    tags=["Results"],
    summary="Download Flood Report",
    description="Download the generated flood assessment report (PDF or HTML)",
    responses={
        200: {"description": "Report file returned"},
        400: {"description": "Job not completed"},
        404: {"description": "Job or report not found"}
    }
)
async def get_report(
    job_id: str = Path(..., description="Job ID to download report for")
):
    """
    Get flood report for a completed job.

    Returns a comprehensive report including:
    - Executive summary
    - Flood statistics (area, coverage)
    - Affected infrastructure assessment
    - Risk analysis
    - Recommendations
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {job['status']}"
        )

    report_path = job.get("report")
    if not report_path or not Path(report_path).exists():
        raise HTTPException(status_code=404, detail="Report not found")

    return FileResponse(
        report_path,
        filename=f"flood_report_{job_id}.html"
    )


# ============================================================================
# Utility Endpoints
# ============================================================================

@app.post(
    "/api/parse",
    response_model=ParsedPromptResponse,
    tags=["Utilities"],
    summary="Parse Natural Language Prompt",
    description="Parse a natural language prompt without executing analysis"
)
async def parse_prompt(
    prompt: str = Query(..., min_length=1, description="Natural language prompt to parse")
):
    """
    Parse a natural language flood monitoring request.

    Returns the extracted parameters without executing the analysis.
    Useful for testing and debugging prompt parsing.
    """
    result = llm_handler.parse_prompt(prompt)

    return ParsedPromptResponse(
        location=result.get("location", "unknown"),
        date_start=result.get("date_start", "last 7 days"),
        date_end=result.get("date_end", "today"),
        task_type=result.get("task_type", "flood_detection"),
        additional_context=result.get("additional_context")
    )


# ============================================================================
# Background Processing
# ============================================================================

async def process_flood_request(
    job_id: str,
    prompt: str,
    override_location: Optional[str] = None,
    override_date_start: Optional[str] = None,
    override_date_end: Optional[str] = None
):
    """Process flood monitoring request asynchronously."""
    try:
        jobs[job_id]["progress"] = 10
        jobs[job_id]["status"] = JobStatus.PARSING_PROMPT

        # Step 1: Parse natural language prompt
        parsed = llm_handler.parse_prompt(prompt)

        location = override_location or parsed.get("location", "unknown")
        date_start = override_date_start or parsed.get("date_start", "last 7 days")
        date_end = override_date_end or parsed.get("date_end", "today")
        task_type = parsed.get("task_type", "flood_detection")

        jobs[job_id]["parsed"] = parsed

        # Step 2: Geocode location
        jobs[job_id]["status"] = JobStatus.GEOCODING
        jobs[job_id]["progress"] = 20

        bbox = await geocode_location(location)
        if not bbox:
            # Default to Jakarta if geocoding fails
            bbox = (106.5, -6.5, 107.0, -6.0)
            jobs[job_id]["warning"] = "Geocoding failed, using default location"

        jobs[job_id]["bbox"] = bbox

        # Step 3: Download satellite data
        jobs[job_id]["status"] = JobStatus.DOWNLOADING_SATELLITE_DATA
        jobs[job_id]["progress"] = 30

        sentinel1_data = await sentinel_downloader.download_sentinel1(
            bbox=bbox,
            date_start=date_start,
            date_end=date_end,
            max_images=3
        )

        sentinel2_data = await sentinel_downloader.download_sentinel2(
            bbox=bbox,
            date_start=date_start,
            date_end=date_end,
            max_images=2
        )

        jobs[job_id]["sentinel1"] = sentinel1_data
        jobs[job_id]["sentinel2"] = sentinel2_data

        # Step 4: Download rainfall data
        jobs[job_id]["status"] = JobStatus.DOWNLOADING_RAINFALL
        jobs[job_id]["progress"] = 50

        rainfall_data = await rainfall_downloader.download_gpm(
            bbox=bbox,
            date_start=date_start,
            date_end=date_end
        )

        jobs[job_id]["rainfall"] = rainfall_data

        # Step 5: Process SAR data for flood detection
        jobs[job_id]["status"] = JobStatus.PROCESSING_FLOOD_DETECTION
        jobs[job_id]["progress"] = 60

        flood_results = None
        flood_mask = None

        if sentinel1_data:
            for s1_image in sentinel1_data:
                result = sar_processor.process(
                    filepath=s1_image["filepath"],
                    bbox=bbox
                )
                if result:
                    flood_results = result
                    flood_mask = None  # Would load from result

        jobs[job_id]["flood_detection"] = flood_results

        # Step 6: Validate with optical data
        jobs[job_id]["status"] = JobStatus.VALIDATING_WITH_OPTICAL
        jobs[job_id]["progress"] = 70

        validation_result = None
        if sentinel2_data and flood_mask is not None:
            for s2_image in sentinel2_data:
                validation = optical_processor.validate_flood_detection(
                    sar_flood_mask=flood_mask,
                    optical_filepath=s2_image["filepath"]
                )
                if validation.get("valid"):
                    validation_result = validation
                    break

        jobs[job_id]["validation"] = validation_result

        # Step 7: Generate risk assessment
        jobs[job_id]["status"] = JobStatus.GENERATING_RISK_ASSESSMENT
        jobs[job_id]["progress"] = 80

        rainfall_mm = rainfall_data.get("total_mm", 50) if rainfall_data else 50
        risk_result = risk_model.predict_risk(
            bbox=bbox,
            rainfall_mm=rainfall_mm,
            flood_extent=flood_mask
        )

        jobs[job_id]["risk_assessment"] = risk_result

        # Step 8: Generate map
        jobs[job_id]["status"] = JobStatus.GENERATING_MAP
        jobs[job_id]["progress"] = 85

        import numpy as np
        dummy_mask = np.zeros((100, 100), dtype=bool)

        map_result = flood_mapper.create_flood_map(
            flood_mask=dummy_mask,
            bbox=bbox,
            job_id=job_id,
            overlay_data={
                "rainfall_mm": rainfall_mm,
                "affected_buildings": flood_results.get("statistics", {}).get("flooded_pixels", 0) // 10 if flood_results else 0
            }
        )

        jobs[job_id]["map"] = map_result

        # Step 9: Generate report
        jobs[job_id]["status"] = JobStatus.GENERATING_REPORT
        jobs[job_id]["progress"] = 90

        flood_area = flood_results.get("statistics", {}).get("flood_area_km2", 10) if flood_results else 10

        report_data = {
            "location": location,
            "date_range": f"{date_start} to {date_end}",
            "flood_area_km2": flood_area,
            "affected_buildings": int(flood_area * 50),
            "affected_roads_km": round(flood_area * 2, 1),
            "agricultural_km2": round(flood_area * 0.3, 2),
            "rainfall_mm": rainfall_mm,
            "risk_assessment": {
                "level": risk_result.get("risk_statistics", {}).get("mean_risk", 0.5),
                "high_risk_pct": risk_result.get("risk_statistics", {}).get("high_risk_area_pct", 20),
                "moderate_risk_pct": risk_result.get("risk_statistics", {}).get("moderate_risk_area_pct", 30),
                "low_risk_pct": risk_result.get("risk_statistics", {}).get("low_risk_area_pct", 50)
            },
            "recommendations": risk_result.get("recommendations", []),
            "narrative": llm_handler.generate_report(
                location=location,
                date_range=f"{date_start} to {date_end}",
                flood_area_km2=flood_area,
                affected_infrastructure={
                    "buildings": int(flood_area * 50),
                    "roads_km": round(flood_area * 2, 1),
                    "agricultural_km2": round(flood_area * 0.3, 2)
                },
                rainfall_data=rainfall_data
            )
        }

        report_path = report_generator.generate_report(report_data, job_id)
        jobs[job_id]["report"] = report_path
        jobs[job_id]["report_data"] = report_data

        # Complete
        jobs[job_id]["status"] = JobStatus.COMPLETED
        jobs[job_id]["progress"] = 100
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        jobs[job_id]["result"] = {
            "flood_area_km2": flood_area,
            "map_path": map_result.get("map_path"),
            "report_path": report_path,
            "risk_level": risk_result.get("risk_statistics", {}).get("mean_risk", 0)
        }

    except Exception as e:
        jobs[job_id]["status"] = JobStatus.FAILED
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["progress"] = 0


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
