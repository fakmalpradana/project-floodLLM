"""FloodLLM FastAPI Application."""
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import google.generativeai as genai
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..utils.config import settings
from ..utils.geocode import geocode_location
from ..utils.llm import LLMPromptHandler, get_parsing_messages, SYSTEM_PROMPT
from ..data.sentinel import SentinelDownloader
from ..data.rainfall import RainfallDownloader
from ..processing.sar_processor import SARProcessor, detect_water_sar
from ..processing.optical import OpticalProcessor, calculate_ndwi_and_mask
from ..processing.risk_model import FloodRiskModel
from ..processing.change_detection import ChangeDetector
from ..processing.vector_generator import VectorGenerator
from ..visualization.vector_map import VectorFloodMap
from ..visualization.satellite_report import SatelliteFloodReport
from ..visualization.reporter import calculate_area

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
change_detector = ChangeDetector()
vector_generator = VectorGenerator()
vector_map = VectorFloodMap()
satellite_reporter = SatelliteFloodReport()

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

        rainfall_mm = rainfall_data.get("total_mm", 50) if rainfall_data else 50

        # Step 7: Change detection (S1 + S2 fusion)
        jobs[job_id]["status"] = "generating_risk_assessment"
        jobs[job_id]["progress"] = 72

        change_result = change_detector.compute_flood_change(
            baseline_ndwi=None,
            flood_ndwi=ndwi_mask,
            bbox=tuple(bbox)
        )
        severity = change_detector.compute_flood_severity(
            flood_area_km2=change_result.get("new_flood_area_km2", flood_area_km2)
        )
        jobs[job_id]["change_detection"] = change_result
        jobs[job_id]["severity"] = severity

        # Step 8a: Flood extent vector layer
        jobs[job_id]["progress"] = 78
        analysis_date = datetime.now().strftime("%Y-%m-%d")
        flood_extent_result = vector_generator.generate_flood_extent_vector(
            flood_mask=water_mask,
            bbox=tuple(bbox),
            job_id=job_id,
            source="Sentinel-1 SAR + Sentinel-2 Optical",
            confidence=change_result.get("fusion", {}).get("confidence", "MEDIUM"),
            date_detected=analysis_date
        )
        # Use vector-derived area when available
        if flood_extent_result.get("total_area_km2", 0) > 0:
            flood_area_km2 = flood_extent_result["total_area_km2"]
            flood_area_ha = flood_area_km2 * 100

        # Step 8b: Risk zone vector layer
        jobs[job_id]["progress"] = 82
        risk_zones_result = vector_generator.generate_risk_zones(
            bbox=tuple(bbox),
            risk_map=None,
            flood_extent_geojson=flood_extent_result.get("geojson"),
            job_id=job_id
        )

        # Step 8c: Impact buffer zones vector layer
        jobs[job_id]["progress"] = 85
        impact_zones_result = vector_generator.generate_impact_zones(
            flood_extent_geojson=flood_extent_result.get("geojson"),
            job_id=job_id,
            date_analysis=analysis_date
        )

        # Step 8d: District-level statistics vector layer
        jobs[job_id]["progress"] = 88
        districts_result = vector_generator.generate_district_statistics(
            flood_extent_geojson=flood_extent_result.get("geojson"),
            risk_zones_geojson=risk_zones_result.get("geojson"),
            bbox=tuple(bbox),
            job_id=job_id
        )

        # Step 9: Generate interactive vector map (4 layers)
        jobs[job_id]["status"] = "generating_map"
        jobs[job_id]["progress"] = 91

        analysis_stats = {
            "flood_area_km2": flood_extent_result.get("total_area_km2", flood_area_km2),
            "high_risk_km2": risk_zones_result.get("high_risk_km2", 0),
            "medium_risk_km2": risk_zones_result.get("medium_risk_km2", 0),
            "low_risk_km2": risk_zones_result.get("low_risk_km2", 0),
            "total_population_exposed": districts_result.get("total_population_exposed", 0),
            "districts_affected_count": districts_result.get("district_count_affected", 0),
            "confidence": change_result.get("fusion", {}).get("confidence", "HIGH"),
        }
        map_result = vector_map.create_vector_map(
            job_id=job_id,
            bbox=tuple(bbox),
            flood_extent_geojson=flood_extent_result.get("geojson"),
            risk_zones_geojson=risk_zones_result.get("geojson"),
            impact_zones_geojson=impact_zones_result.get("geojson"),
            districts_geojson=districts_result.get("geojson"),
            analysis_stats=analysis_stats,
            title=f"Flood Risk & Impact Assessment — {location}",
            analysis_period=f"{date_start} to {date_end}"
        )
        jobs[job_id]["map"] = map_result

        # Step 10: Generate comprehensive satellite analysis report
        jobs[job_id]["status"] = "generating_report"
        jobs[job_id]["progress"] = 95

        report_path = satellite_reporter.generate(
            job_id=job_id,
            location=location,
            analysis_period=f"{date_start} to {date_end}",
            bbox=tuple(bbox),
            flood_extent_result=flood_extent_result,
            risk_zones_result=risk_zones_result,
            impact_zones_result=impact_zones_result,
            districts_result=districts_result,
            change_detection_result=change_result,
            map_path=map_result.get("map_path"),
            rainfall_mm=rainfall_mm
        )
        jobs[job_id]["report"] = report_path

        # Risk summary from FloodRiskModel (still used for risk_level scalar)
        risk_result = risk_model.predict_risk(
            bbox=tuple(bbox),
            rainfall_mm=rainfall_mm,
            flood_extent=water_mask
        )

        # Complete
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        jobs[job_id]["result"] = {
            "flood_area_km2": round(flood_area_km2, 3),
            "flood_area_ha": round(flood_area_ha, 1),
            "map_path": map_result.get("map_path"),
            "report_path": report_path,
            "risk_level": risk_result.get("risk_statistics", {}).get("mean_risk", 0),
            "severity": severity.get("severity", "UNKNOWN"),
            "flood_extent_features": flood_extent_result.get("feature_count", 0),
            "risk_zones": {
                "high_km2": risk_zones_result.get("high_risk_km2", 0),
                "medium_km2": risk_zones_result.get("medium_risk_km2", 0),
                "low_km2": risk_zones_result.get("low_risk_km2", 0),
            },
            "population_exposed": districts_result.get("total_population_exposed", 0),
            "districts_affected": districts_result.get("district_count_affected", 0),
            "vector_layers": {
                "flood_extent": flood_extent_result.get("path"),
                "risk_zones": risk_zones_result.get("path"),
                "impact_zones": impact_zones_result.get("path"),
                "districts": districts_result.get("path"),
            }
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
        "result": job.get("result"),
        "parsed": job.get("parsed"),
        # expose vector layer file paths if available
        "vector_layers": job.get("result", {}).get("vector_layers") if job.get("result") else None,
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
