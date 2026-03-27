# FloodLLM API Reference

Complete API documentation for the FloodLLM FastAPI backend.

## Base URL

```
Development: http://localhost:8000
Production:  https://your-deployment-url.run.app
```

## Authentication

Currently, the API does not require authentication. For production deployments, it is recommended to add API key authentication or OAuth2.

---

## Endpoints

### `GET /` - Health Check

Returns the API status and available endpoints.

**Response:**
```json
{
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
```

---

### `POST /api/prompt` - Submit Flood Analysis Request

Submit a natural language flood monitoring request. The API processes the request asynchronously and returns a job ID for tracking.

**Request Body:**
```json
{
  "prompt": "Show flood extent in Jakarta, Indonesia for the last 7 days",
  "location": "Jakarta, Indonesia (optional, overrides prompt)",
  "date_start": "2024-01-01 (optional)",
  "date_end": "2024-01-08 (optional)"
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompt` | string | Yes | Natural language description of the flood analysis request |
| `location` | string | No | Override location extracted from prompt |
| `date_start` | string | No | Override start date (YYYY-MM-DD or relative like "last 7 days") |
| `date_end` | string | No | Override end date (YYYY-MM-DD or "today") |

**Response (200 OK):**
```json
{
  "job_id": "a1b2c3d4",
  "status": "processing",
  "message": "Your flood analysis request has been queued",
  "estimated_time_seconds": 60
}
```

**Example (cURL):**
```bash
curl -X POST http://localhost:8000/api/prompt \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Show flood extent in Jakarta, Indonesia for the last 7 days"
  }'
```

**Example (Python):**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/prompt",
    json={"prompt": "Show flood extent in Jakarta for the last 7 days"}
)
job_id = response.json()["job_id"]
print(f"Job ID: {job_id}")
```

---

### `GET /api/status/{job_id}` - Check Processing Status

Check the status and progress of a flood analysis job.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string | Job ID returned from POST /api/prompt |

**Response (200 OK):**
```json
{
  "job_id": "a1b2c3d4",
  "status": "completed",
  "progress": 100,
  "prompt": "Show flood extent in Jakarta, Indonesia for the last 7 days",
  "created_at": "2024-01-15T10:30:00",
  "completed_at": "2024-01-15T10:31:30",
  "error": null,
  "result": {
    "flood_area_km2": 45.5,
    "map_path": "/output/maps/flood_map_a1b2c3d4.html",
    "report_path": "/output/reports/flood_report_a1b2c3d4.pdf",
    "risk_level": 0.65
  }
}
```

**Status Values:**

| Status | Description |
|--------|-------------|
| `processing` | Job is being processed |
| `parsing_prompt` | Parsing natural language prompt |
| `geocoding` | Converting location to coordinates |
| `downloading_satellite_data` | Fetching Sentinel-1/2 imagery |
| `downloading_rainfall` | Fetching GPM rainfall data |
| `processing_flood_detection` | Running SAR flood detection |
| `validating_with_optical` | Validating with Sentinel-2 |
| `generating_risk_assessment` | Computing risk scores |
| `generating_map` | Creating flood map |
| `generating_report` | Generating PDF/HTML report |
| `completed` | Job finished successfully |
| `failed` | Job failed (check `error` field) |

**Response (404 Not Found):**
```json
{
  "detail": "Job not found"
}
```

**Example (cURL):**
```bash
curl http://localhost:8000/api/status/a1b2c3d4
```

**Example (Python):**
```python
import time

def wait_for_completion(job_id, poll_interval=5):
    while True:
        response = requests.get(f"http://localhost:8000/api/status/{job_id}")
        status = response.json()["status"]

        if status == "completed":
            return response.json()["result"]
        elif status == "failed":
            raise Exception(response.json()["error"])

        print(f"Status: {status} ({response.json()['progress']}%)")
        time.sleep(poll_interval)

result = wait_for_completion(job_id)
```

---

### `GET /api/map/{job_id}` - Download Flood Map

Download the interactive flood map generated for a completed job.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string | Job ID of completed analysis |

**Response (200 OK):**
- Content-Type: `text/html`
- Body: HTML file containing interactive Folium map

**Response (400 Bad Request):**
```json
{
  "detail": "Job not completed"
}
```

**Response (404 Not Found):**
```json
{
  "detail": "Map not found"
}
```

**Example (cURL):**
```bash
curl -O http://localhost:8000/api/map/a1b2c3d4
```

**Example (Python):**
```python
response = requests.get(f"http://localhost:8000/api/map/{job_id}")
with open(f"flood_map_{job_id}.html", "wb") as f:
    f.write(response.content)
```

---

### `GET /api/report/{job_id}` - Download Assessment Report

Download the PDF/HTML flood assessment report for a completed job.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string | Job ID of completed analysis |

**Response (200 OK):**
- Content-Type: `application/pdf` or `text/html`
- Body: Report file

**Response (400 Bad Request):**
```json
{
  "detail": "Job not completed"
}
```

**Response (404 Not Found):**
```json
{
  "detail": "Report not found"
}
```

**Example (cURL):**
```bash
curl -O http://localhost:8000/api/report/a1b2c3d4
```

**Example (Python):**
```python
response = requests.get(f"http://localhost:8000/api/report/{job_id}")
with open(f"flood_report_{job_id}.pdf", "wb") as f:
    f.write(response.content)
```

---

### `GET /api/jobs` - List All Jobs

List all submitted jobs with their status.

**Response (200 OK):**
```json
{
  "jobs": [
    {
      "job_id": "a1b2c3d4",
      "status": "completed",
      "progress": 100,
      "prompt": "Show flood extent in Jakarta",
      "created_at": "2024-01-15T10:30:00"
    },
    {
      "job_id": "e5f6g7h8",
      "status": "processing",
      "progress": 45,
      "prompt": "Assess flood risk in Bangkok",
      "created_at": "2024-01-15T10:35:00"
    }
  ]
}
```

---

## Error Handling

All errors return a JSON response with a `detail` field:

```json
{
  "detail": "Error message describing what went wrong"
}
```

**Common HTTP Status Codes:**

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request (job not completed) |
| 404 | Not Found (job/map/report not found) |
| 500 | Internal Server Error |

---

## Rate Limiting

Currently, there is no rate limiting implemented. For production deployments, consider adding:
- Request throttling per IP/client
- Concurrent job limits
- Queue management with Redis/Celery

---

## OpenAPI/Swagger Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`
