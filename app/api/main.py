"""FloodLLM FastAPI Application."""
import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import google.generativeai as genai
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ..utils.config import settings
from ..utils.geocode import geocode_location
from ..utils.llm import LLMPromptHandler, get_parsing_messages, SYSTEM_PROMPT
from ..data.sentinel import SentinelDownloader
from ..data.rainfall import RainfallDownloader
from ..processing.sar_processor import SARProcessor, detect_water_sar
from ..processing.optical import OpticalProcessor, calculate_ndwi_and_mask
from ..processing.risk_model import FloodRiskModel
from ..visualization.mapper import FloodMapper
from ..visualization.reporter import ReportGenerator, calculate_area, generate_flood_report

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

        # Step 1: Parse via get_parsing_messages -> Gemini -> JSON
        messages = get_parsing_messages(prompt)
        parsed = {}
        if llm_handler.model:
            try:
                parse_model = genai.GenerativeModel(
                    "gemini-2.0-flash-exp",
                    system_instruction=messages[0]["content"]
                )
                response = parse_model.generate_content(messages[1]["content"])
                parsed = json.loads(response.text)
            except Exception as e:
                print(f"Gemini structured parse error: {e}")
                parsed = llm_handler._simple_parse(prompt)
        else:
            parsed = llm_handler._simple_parse(prompt)

        location = override_location or parsed.get("location_name") or parsed.get("location", "unknown")
        bbox = parsed.get("bbox") or None
        date_start = override_date_start or parsed.get("start_date") or parsed.get("date_start", "last 7 days")
        date_end = override_date_end or parsed.get("end_date") or parsed.get("date_end", "today")

        jobs[job_id]["parsed"] = parsed

        # Step 2: Geocode if bbox not in parsed result
        jobs[job_id]["status"] = "geocoding"
        jobs[job_id]["progress"] = 20

        if not bbox:
            bbox = await geocode_location(location)
        if not bbox:
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

        # Step 5: Process SAR data with detect_water_sar -> calculate_area
        jobs[job_id]["status"] = "processing_flood_detection"
        jobs[job_id]["progress"] = 60

        water_mask = None
        flood_area_ha = 0.0

        if sentinel1_data:
            for s1_image in sentinel1_data:
                try:
                    water_mask, flood_profile = detect_water_sar(s1_image["filepath"])
                    flood_area_ha = calculate_area(water_mask, flood_profile["transform"])
                    break
                except Exception as e:
                    print(f"SAR processing error for {s1_image['filepath']}: {e}")

        flood_area_km2 = flood_area_ha / 100.0
        jobs[job_id]["flood_detection"] = {"flood_area_ha": flood_area_ha, "flood_area_km2": flood_area_km2}

        # Step 6: Validate with optical via calculate_ndwi_and_mask
        jobs[job_id]["status"] = "validating_with_optical"
        jobs[job_id]["progress"] = 70

        ndwi_mask = None
        if sentinel2_data:
            for s2_image in sentinel2_data:
                try:
                    ndwi_mask, _ = calculate_ndwi_and_mask(s2_image["filepath"])
                    break
                except Exception as e:
                    print(f"Optical processing error for {s2_image['filepath']}: {e}")

        jobs[job_id]["validation"] = {"ndwi_mask_computed": ndwi_mask is not None}

        # Step 7: Generate risk assessment
        jobs[job_id]["status"] = "generating_risk_assessment"
        jobs[job_id]["progress"] = 80

        rainfall_mm = rainfall_data.get("total_mm", 50) if rainfall_data else 50
        risk_result = risk_model.predict_risk(
            bbox=bbox,
            rainfall_mm=rainfall_mm,
            flood_extent=water_mask
        )

        jobs[job_id]["risk_assessment"] = risk_result

        # Step 8: Generate map
        jobs[job_id]["status"] = "generating_map"
        jobs[job_id]["progress"] = 85

        display_mask = water_mask if water_mask is not None else np.zeros((100, 100), dtype=bool)
        map_result = flood_mapper.create_flood_map(
            flood_mask=display_mask,
            bbox=bbox,
            job_id=job_id,
            overlay_data={
                "rainfall_mm": rainfall_mm,
                "affected_buildings": int(flood_area_km2 * 50)
            }
        )

        jobs[job_id]["map"] = map_result

        # Step 9: Generate HTML report via generate_flood_report
        jobs[job_id]["status"] = "generating_report"
        jobs[job_id]["progress"] = 90

        report_html_path = str(settings.output_dir / "reports" / f"flood_report_{job_id}.html")
        generate_flood_report(location, date_start, date_end, flood_area_ha, report_html_path)
        jobs[job_id]["report"] = report_html_path

        # Complete
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        jobs[job_id]["result"] = {
            "flood_area_km2": flood_area_km2,
            "flood_area_ha": flood_area_ha,
            "map_path": map_result.get("map_path") if map_result else None,
            "report_path": report_html_path,
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
