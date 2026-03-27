# FloodLLM Architecture

This document describes the system architecture and data flow of the FloodLLM flood monitoring system.

## System Overview

FloodLLM is an AI-powered flood monitoring system that processes natural language requests and generates comprehensive flood analysis reports using satellite imagery.

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐     ┌──────────────┐
│  User Input │ ──► │  LLM Parser  │ ──► │  Data Pipeline  │ ──► │  Processing │ ──► │ Visualization│
│  (Natural   │     │  (Gemini)    │     │  (Satellite +   │     │  (SAR +     │     │  (Maps +     │
│   Language) │     │              │     │   Rainfall)     │     │   Optical)  │     │   Reports)   │
└─────────────┘     └──────────────┘     └─────────────────┘     └──────────────┘     └──────────────┘
```

---

## Component Architecture

### 1. Input Layer

**Interfaces:**
- **CLI** (`cli.py`): Command-line interface for direct analysis
- **REST API** (`app/api/main.py`): FastAPI backend for async job processing

**Responsibilities:**
- Accept natural language prompts
- Queue jobs for processing
- Return job IDs for status tracking

---

### 2. LLM Parser

**Module:** `app/utils/llm.py` (`LLMPromptHandler`)

**Responsibilities:**
- Parse natural language prompts using Google Gemini
- Extract structured query parameters:
  - Location (place name)
  - Date range (start/end)
  - Task type (flood_detection, risk_prediction, damage_assessment)
  - Additional context

**Flow:**
```
"Show flood extent in Jakarta for the last 7 days"
         │
         ▼
┌─────────────────────────────────────────┐
│  LLM Prompt (Gemini)                    │
│  - Extract location: "Jakarta"          │
│  - Extract dates: "last 7 days"         │
│  - Extract task: "flood_detection"      │
└─────────────────────────────────────────┘
         │
         ▼
{
  "location": "Jakarta, Indonesia",
  "date_start": "last 7 days",
  "date_end": "today",
  "task_type": "flood_detection"
}
```

**Fallback:** When LLM is unavailable, regex-based parsing extracts location and dates.

---

### 3. Geocoding

**Module:** `app/utils/geocode.py`

**Responsibilities:**
- Convert location names to bounding boxes
- Return coordinates in format: `(min_lon, min_lat, max_lon, max_lat)`

**Example:**
```
"Jakarta, Indonesia" → (106.5, -6.5, 107.0, -6.0)
```

---

### 4. Data Pipeline

#### 4.1 Satellite Data Download

**Module:** `app/data/sentinel.py` (`SentinelDownloader`)

**Sentinel-1 (SAR):**
- Product: COPERNICUS/S1_GRD
- Polarization: VV, VH
- Use: Flood detection (cloud-penetrating radar)
- Resolution: ~10m

**Sentinel-2 (Optical):**
- Product: COPERNICUS/S2_SR_HARMONIZED
- Bands: B2, B3, B4, B8, B11
- Use: Validation (NDWI-based water detection)
- Resolution: 10-60m

**Data Sources:**
- Google Earth Engine (primary)
- Copernicus Data Space API (fallback)

#### 4.2 Rainfall Data Download

**Module:** `app/data/rainfall.py` (`RainfallDownloader`)

**Product:** NASA GPM (Global Precipitation Measurement)
- Format: NetCDF
- Resolution: 0.1 degrees
- Use: Context for flood causation

---

### 5. Processing Pipeline

#### 5.1 SAR Processor

**Module:** `app/processing/sar_processor.py` (`SARProcessor`)

**Algorithm:**
```
┌─────────────────┐
│ Sentinel-1 GRD  │
│ (GeoTIFF)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Read VV/VH      │
│ Convert to dB   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Otsu Threshold  │
│ (Automatic)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Binary Water    │
│ Mask            │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Morphological   │
│ Operations      │
│ - Fill holes    │
│ - Smooth edges  │
│ - Remove noise  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Flood Statistics│
│ - Area (km²)    │
│ - Pixel count   │
│ - Coverage %    │
└─────────────────┘
```

**Key Methods:**
- `_calculate_otsu_threshold()`: Bimodal threshold for water/land separation
- `_post_process_mask()`: Clean up binary mask
- `_calculate_flood_stats()`: Compute flood area and coverage

#### 5.2 Optical Processor

**Module:** `app/processing/optical.py` (`OpticalProcessor`)

**Responsibilities:**
- Validate SAR flood detection using Sentinel-2
- Compute NDWI (Normalized Difference Water Index)
- Cross-reference flood masks

#### 5.3 Risk Model

**Module:** `app/processing/risk_model.py` (`FloodRiskModel`)

**Responsibilities:**
- Combine flood extent with rainfall data
- Load GeoJSON layers (buildings, roads, agriculture)
- Compute risk scores per land use type

---

### 6. Visualization

#### 6.1 Flood Mapper

**Module:** `app/visualization/mapper.py` (`FloodMapper`)

**Output:** Interactive HTML map (Folium)

**Features:**
- Flood extent overlay (blue polygon)
- Analysis area boundary (red dashed)
- Rainfall markers (color-coded)
- Affected infrastructure markers
- Fullscreen toggle

#### 6.2 Report Generator

**Module:** `app/visualization/reporter.py` (`ReportGenerator`)

**Output:** HTML or PDF report

**Sections:**
- Executive summary
- Flood statistics
- Affected infrastructure
- Risk assessment
- LLM-generated narrative
- Recommendations

---

## Data Flow

### Complete Pipeline

```
1. User submits: "Show floods in Jakarta last week"
         │
         ▼
2. POST /api/prompt → job_id: "abc123"
         │
         ▼
3. Background task starts
         │
         ├──► 3a. Parse prompt (LLM)
         │       → location: "Jakarta", dates: "last 7 days"
         │
         ├──► 3b. Geocode location
         │       → bbox: (106.5, -6.5, 107.0, -6.0)
         │
         ├──► 3c. Download Sentinel-1
         │       → 3 GeoTIFF files
         │
         ├──► 3d. Download Sentinel-2
         │       → 2 GeoTIFF files (validation)
         │
         ├──► 3e. Download GPM rainfall
         │       → NetCDF file
         │
         ├──► 3f. Process SAR data
         │       → Flood mask + statistics
         │
         ├──► 3g. Validate with optical
         │       → Confidence score
         │
         ├──► 3h. Generate risk assessment
         │       → Risk scores by land use
         │
         ├──► 3i. Generate flood map
         │       → HTML interactive map
         │
         └──► 3j. Generate report
                 → PDF/HTML report
         │
         ▼
4. Job status: "completed"
         │
         ▼
5. GET /api/map/abc123 → Download map
   GET /api/report/abc123 → Download report
```

---

## Module Dependencies

```
flood-llm/
├── cli.py                      # CLI entry point
│
├── app/
│   ├── __init__.py
│   ├── api/
│   │   └── main.py             # FastAPI app
│   │
│   ├── utils/
│   │   ├── config.py           # Settings (Pydantic)
│   │   ├── geocode.py          # Location → bbox
│   │   └── llm.py              # LLM integration
│   │
│   ├── data/
│   │   ├── sentinel.py         # Satellite download
│   │   └── rainfall.py         # Rainfall download
│   │
│   ├── processing/
│   │   ├── sar_processor.py    # SAR flood detection
│   │   ├── optical.py          # Optical validation
│   │   └── risk_model.py       # Risk assessment
│   │
│   └── visualization/
│       ├── mapper.py           # Folium maps
│       └── reporter.py         # Report generation
│
└── tests/
    ├── test_end_to_end.py      # E2E tests
    └── test_flood_detection.py # Unit tests
```

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| **API** | FastAPI, Uvicorn |
| **LLM** | Google Gemini (gemini-2.0-flash-exp) |
| **Geospatial** | GDAL, rasterio, geopandas, shapely |
| **Satellite** | Earth Engine, Copernicus API |
| **Processing** | NumPy, SciPy, scikit-image |
| **Visualization** | Folium, reportlab, Jinja2 |
| **CLI** | Click |
| **Config** | Pydantic Settings, python-dotenv |

---

## Job State Machine

```
                    ┌─────────────┐
                    │  processing │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ parsing_prompt│  │downloading_   │  │processing_    │
└───────┬───────┘  │satellite_data │  │flood_detection│
        │          └───────┬───────┘  └───────┬───────┘
        │                  │                  │
        │                  ▼                  │
        │          ┌───────────────┐          │
        │          │downloading_   │          │
        │          │rainfall       │          │
        │          └───────┬───────┘          │
        │                  │                  │
        │                  ▼                  │
        │          ┌───────────────┐          │
        │          │validating_    │          │
        │          │with_optical   │          │
        │          └───────┬───────┘          │
        │                  │                  │
        │                  ▼                  │
        │          ┌───────────────┐          │
        │          │generating_    │          │
        │          │risk_assessment│          │
        │          └───────┬───────┘          │
        │                  │                  │
        │                  ▼                  │
        │          ┌───────────────┐          │
        │          │generating_map │          │
        │          └───────┬───────┘          │
        │                  │                  │
        │                  ▼                  │
        │          ┌───────────────┐          │
        └─────────►│generating_    │◄─────────┘
                   │report         │
                   └───────┬───────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  completed  │
                    └─────────────┘
```

---

## Scalability Considerations

### Current Limitations

1. **In-memory job storage**: Jobs are stored in a Python dict; they are lost on restart.
2. **Single-worker processing**: Background tasks run in the same process.
3. **No rate limiting**: API can be overwhelmed by concurrent requests.

### Production Recommendations

1. **Job Queue**: Use Redis + Celery for distributed task processing.
2. **Database**: Store jobs in PostgreSQL with status tracking.
3. **Caching**: Cache geocoding results and satellite data.
4. **Horizontal Scaling**: Deploy on Cloud Run with auto-scaling.
5. **Monitoring**: Add Prometheus/Grafana for metrics.
