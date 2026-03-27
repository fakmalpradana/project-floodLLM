# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FloodLLM is an EarthGPT-inspired application for automated flood detection, risk prediction, and damage assessment using natural language prompts. Built as an MVP that can run locally on M4 MacBook Air or deploy to GCP.

## Quick Start

```bash
# Create environment
conda env create -f environment.yml
conda activate flood-llm

# Copy and configure API keys
cp .env.example .env
# Edit .env with your API keys

# Run API server
python -m app.api.main
# or
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

# Run CLI
python cli.py <command>

# Run tests
python -m tests.test_end_to_end
```

## Architecture

### Data Pipeline

```
User Prompt → LLM Parser → Geocoding → Data Download → Processing → Visualization
                ↓              ↓           ↓              ↓            ↓
           (Gemini)      (bbox)    (Sentinel-1/2,   (SAR +       (Folium +
                                        GPM)        NDWI)         PDF)
```

### Core Modules

| Module | Responsibility |
|--------|----------------|
| `app/api/main.py` | FastAPI server with async background job processing |
| `app/utils/llm.py` | `LLMPromptHandler` - parses prompts using Google Gemini |
| `app/utils/geocode.py` | Location name → bounding box conversion |
| `app/data/sentinel.py` | `SentinelDownloader` - downloads Sentinel-1/2 via Earth Engine API |
| `app/data/rainfall.py` | `RainfallDownloader` - downloads GPM precipitation data |
| `app/processing/sar_processor.py` | `SARProcessor` - Otsu thresholding for water detection |
| `app/processing/optical.py` | `OpticalProcessor` - NDWI validation of SAR results |
| `app/processing/risk_model.py` | `FloodRiskModel` - risk scoring by land use |
| `app/visualization/mapper.py` | `FloodMapper` - interactive Folium maps |
| `app/visualization/reporter.py` | `ReportGenerator` - PDF/HTML reports |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/api/prompt` | POST | Submit prompt, returns `job_id` |
| `/api/status/{job_id}` | GET | Check job status/progress |
| `/api/map/{job_id}` | GET | Download flood map (HTML) |
| `/api/report/{job_id}` | GET | Download report |

### CLI Commands

```bash
python cli.py analyze -l "Jakarta, Indonesia" -s "last 7 days"  # Run analysis
python cli.py parse "Show flood extent in Jakarta"              # Parse prompt
python cli.py status                                            # Check system
python cli.py test                                              # Run tests
```

## Configuration

### Required API Keys (`.env`)

```bash
GOOGLE_API_KEY=              # Google AI Studio for LLM
COPERNICUS_USERNAME=         # Copernicus Data Space for Sentinel
COPERNICUS_PASSWORD=
NASA_EARTHDATA_USERNAME=     # NASA Earthdata for GPM
NASA_EARTHDATA_PASSWORD=
```

### Key Parameters (`app/utils/config.py`)

```python
water_threshold_vv = -17.0    # dB threshold for water in SAR
default_buffer_km = 50.0      # Default search radius
cloud_cover_max = 20.0        # Max cloud cover for optical
```

## Development Notes

- **Job storage**: In-memory dict (lost on restart) - see architecture docs for production recommendations
- **Background tasks**: Run in same process via FastAPI `BackgroundTasks`
- **SAR processing**: Uses Otsu's method for automatic thresholding; fallback to fixed threshold at -17 dB
- **Geocoding fallback**: Jakarta coordinates (106.5, -6.5, 107.0, -6.0) if geocoding fails
- **Validation**: `notebooks/validation.ipynb` for IoU metrics against ground truth

## Documentation

See `docs/` directory for detailed guides:
- `docs/ARCHITECTURE.md` - System design and data flow diagrams
- `docs/API_REFERENCE.md` - FastAPI endpoint documentation
- `docs/DEPLOYMENT.md` - GCP deployment guide
- `docs/VALIDATION.md` - Testing and IoU metrics
- `docs/USER_GUIDE.md` - CLI and API usage
