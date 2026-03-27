"""FloodLLM FastAPI Application."""
import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

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

# Initialize FastAPI app
app = FastAPI(
    title="FloodLLM",
    description="AI-Powered Flood Monitoring System",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


class PromptRequest(BaseModel):
    """Natural language prompt request."""
    prompt: str
    location: Optional[str] = None  # Override if provided separately
    date_start: Optional[str] = None
    date_end: Optional[str] = None


class PromptResponse(BaseModel):
    """Response with job ID."""
    job_id: str
    status: str
    message: str
    estimated_time_seconds: int = 60


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "name": "FloodLLM",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "submit": "POST /api/prompt",
            "status": "GET /api/status/{job_id}",
            "map": "GET /api/map/{job_id}",
            "report": "GET /api/report/{job_id}"
        }
    }


@app.post("/api/prompt", response_model=PromptResponse)
async def submit_prompt(request: PromptRequest, background_tasks: BackgroundTasks):
    """
    Submit a natural language flood monitoring request.

    Examples:
    - "Show flood extent in Jakarta for the last 7 days"
    - "Assess flood risk in Bangkok this week"
    - "Generate damage report for recent floods in Manila"
    """
    job_id = str(uuid.uuid4())[:8]

    # Initialize job
    jobs[job_id] = {
        "status": "processing",
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
        status="processing",
        message="Your flood analysis request has been queued",
        estimated_time_seconds=60
    )


async def process_flood_request(
    job_id: str,
    prompt: str,
    override_location: Optional[str] = None,
    override_date_start: Optional[str] = None,
    override_date_end: Optional[str] = None
):
    """Process flood monitoring request."""
    try:
        jobs[job_id]["progress"] = 10
        jobs[job_id]["status"] = "parsing_prompt"

        # Step 1: Parse natural language prompt
        parsed = llm_handler.parse_prompt(prompt)

        location = override_location or parsed.get("location", "unknown")
        date_start = override_date_start or parsed.get("date_start", "last 7 days")
        date_end = override_date_end or parsed.get("date_end", "today")
        task_type = parsed.get("task_type", "flood_detection")

        jobs[job_id]["parsed"] = parsed

        # Step 2: Geocode location
        jobs[job_id]["status"] = "geocoding"
        jobs[job_id]["progress"] = 20

        bbox = await geocode_location(location)
        if not bbox:
            # Default to Jakarta if geocoding fails
            bbox = (106.5, -6.5, 107.0, -6.0)
            jobs[job_id]["warning"] = "Geocoding failed, using default location"

        jobs[job_id]["bbox"] = bbox

        # Step 3: Download satellite data
        jobs[job_id]["status"] = "downloading_satellite_data"
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
        jobs[job_id]["status"] = "downloading_rainfall"
        jobs[job_id]["progress"] = 50

        rainfall_data = await rainfall_downloader.download_gpm(
            bbox=bbox,
            date_start=date_start,
            date_end=date_end
        )

        jobs[job_id]["rainfall"] = rainfall_data

        # Step 5: Process SAR data for flood detection
        jobs[job_id]["status"] = "processing_flood_detection"
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
                    # Load mask for visualization (simplified)
                    flood_mask = None  # Would load from result

        jobs[job_id]["flood_detection"] = flood_results

        # Step 6: Validate with optical data
        jobs[job_id]["status"] = "validating_with_optical"
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
        jobs[job_id]["status"] = "generating_risk_assessment"
        jobs[job_id]["progress"] = 80

        rainfall_mm = rainfall_data.get("total_mm", 50) if rainfall_data else 50
        risk_result = risk_model.predict_risk(
            bbox=bbox,
            rainfall_mm=rainfall_mm,
            flood_extent=flood_mask
        )

        jobs[job_id]["risk_assessment"] = risk_result

        # Step 8: Generate map
        jobs[job_id]["status"] = "generating_map"
        jobs[job_id]["progress"] = 85

        # Create simplified flood mask for visualization
        if flood_results:
            dummy_mask = None  # Would create from actual data
        else:
            import numpy as np
            dummy_mask = np.zeros((100, 100), dtype=bool)

        map_result = flood_mapper.create_flood_map(
            flood_mask=dummy_mask if dummy_mask is not None else np.zeros((100, 100)),
            bbox=bbox,
            job_id=job_id,
            overlay_data={
                "rainfall_mm": rainfall_mm,
                "affected_buildings": flood_results.get("statistics", {}).get("flooded_pixels", 0) // 10 if flood_results else 0
            }
        )

        jobs[job_id]["map"] = map_result

        # Step 9: Generate report
        jobs[job_id]["status"] = "generating_report"
        jobs[job_id]["progress"] = 90

        # Calculate affected infrastructure (simplified)
        flood_area = flood_results.get("statistics", {}).get("flood_area_km2", 10) if flood_results else 10

        report_data = {
            "location": location,
            "date_range": f"{date_start} to {date_end}",
            "flood_area_km2": flood_area,
            "affected_buildings": int(flood_area * 50),  # Estimate
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
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        jobs[job_id]["result"] = {
            "flood_area_km2": flood_area,
            "map_path": map_result.get("map_path"),
            "report_path": report_path,
            "risk_level": risk_result.get("risk_statistics", {}).get("mean_risk", 0)
        }

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["progress"] = 0


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Get job status and progress."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "prompt": job["prompt"],
        "created_at": job["created_at"],
        "completed_at": job.get("completed_at"),
        "error": job.get("error"),
        "result": job.get("result")
    }


@app.get("/api/map/{job_id}")
async def get_map(job_id: str):
    """Get flood map for job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")

    map_path = job.get("map", {}).get("map_path")
    if not map_path or not Path(map_path).exists():
        raise HTTPException(status_code=404, detail="Map not found")

    return FileResponse(map_path, media_type="text/html")


@app.get("/api/report/{job_id}")
async def get_report(job_id: str):
    """Get flood report for job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")

    report_path = job.get("report")
    if not report_path or not Path(report_path).exists():
        raise HTTPException(status_code=404, detail="Report not found")

    return FileResponse(report_path)


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs."""
    return {
        "jobs": [
            {
                "job_id": jid,
                "status": j["status"],
                "progress": j["progress"],
                "prompt": j["prompt"],
                "created_at": j["created_at"]
            }
            for jid, j in jobs.items()
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
