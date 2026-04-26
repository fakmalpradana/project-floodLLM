# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FloodLLM is an EarthGPT-inspired application for automated flood detection, risk prediction, and damage assessment using natural language prompts. Supports English and Bahasa Indonesia. Runs locally on M4 MacBook Air or deploys to GCP Cloud Run.

## Commands

```bash
# Environment setup
conda env create -f environment.yml && conda activate flood-llm

# Backend
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

# Frontend (separate terminal)
cd frontend && npm install && npm run dev

# CLI
python cli.py analyze -l "Jakarta, Indonesia" -s "last 7 days"
python cli.py parse "Show flood extent in Jakarta"
python cli.py status

# Tests
python -m pytest tests/test_end_to_end.py -v
python -m pytest tests/test_flood_detection.py -v
python -m tests.test_end_to_end   # direct runner
```

## Architecture

### Processing Pipeline (11 steps, async background task)

```
Prompt → LLM Parse → Geocode → Download S1 → Download S2 → Download Rainfall
                                    ↓              ↓              ↓
                              SAR Process → NDWI Validate → Risk Model
                                    ↓
                          Vector Generation (4 GIS layers)
                                    ↓
                          Map (Folium HTML) + Report (HTML)
```

All steps run as `BackgroundTasks` in FastAPI. Job state is tracked in an **in-memory dict** (lost on restart — use Redis/DB for production).

### Key Modules

| Module | Responsibility |
|--------|----------------|
| `app/api/main.py` | FastAPI server, 11-step pipeline orchestration, job state machine |
| `app/utils/llm.py` | Gemini 2.0 Flash — parses prompts into `{location_name, start_date, end_date}` |
| `app/utils/geocode.py` | Nominatim → bounding box; falls back to 50km buffer around point |
| `app/data/sentinel.py` | Sentinel-1/2 via Earth Engine API (primary) + Copernicus fallback |
| `app/data/rainfall.py` | NASA GPM precipitation download |
| `app/processing/sar_processor.py` | Otsu thresholding on VV band → binary water mask |
| `app/processing/optical.py` | NDWI = (Green−NIR)/(Green+NIR) > 0.3 → water validation |
| `app/processing/change_detection.py` | S1+S2 fusion for confidence scoring |
| `app/processing/vector_generator.py` | Raster masks → 4 GeoJSON layers |
| `app/processing/risk_model.py` | Risk scoring by land use (HIGH/MEDIUM/LOW) |
| `app/visualization/vector_map.py` | Multi-layer Folium HTML with toggles |
| `app/visualization/satellite_report.py` | Comprehensive HTML analysis report |
| `frontend/src/App.jsx` | React 19 + Tailwind CSS UI; polls `/api/status` every 2s |

### Job State Machine

`processing → parsing_prompt → geocoding → downloading_sentinel1 → downloading_sentinel2 → downloading_rainfall → processing_sar → validating_optical → generating_vectors → creating_map → generating_report → completed / failed`

### Vector Output Layers (render order, bottom → top)

1. `districts` — administrative boundaries with population/infrastructure stats
2. `impact_zones` — 500m / 1000m / 2000m buffer rings
3. `risk_zones` — HIGH (red) / MEDIUM (yellow) / LOW (green)
4. `flood_extent` — SAR+optical fusion polygons (blue)

Layer render order matters for Folium visibility.

### Frontend

React 19 + Vite + Tailwind CSS 4. `VITE_API_URL` in `frontend/.env` points to the backend (empty = local proxy). Job history persists in `localStorage` and resumes polling on reload.

## Configuration

Required in `.env` (copy from `.env.example`):

```bash
GOOGLE_API_KEY=              # Gemini LLM (Google AI Studio)
COPERNICUS_USERNAME=         # Sentinel data
COPERNICUS_PASSWORD=
NASA_EARTHDATA_USERNAME=     # GPM rainfall
NASA_EARTHDATA_PASSWORD=
```

Key thresholds in `app/utils/config.py` (Pydantic BaseSettings):
- `water_threshold_vv = -17.0 dB` — SAR fixed fallback threshold
- `cloud_cover_max = 20%`
- `default_buffer_km = 50`

## Non-Obvious Patterns

**LLM schema split**: `llm.py` returns `{location_name, start_date, end_date}` but `main.py:170-187` also handles legacy `_simple_parse()` keys `{location, date_start, date_end}` — both must stay supported.

**Simulation fallback**: `vector_generator.py` has `_simulate_flood_extent()` that generates realistic Jakarta patterns when no real satellite data is available — enables demo mode without credentials.

**SAR dual thresholding**: Otsu (automatic, preferred) falls back to fixed -15 dB when the algorithm fails on edge cases.

**Coordinate order**: Always `(min_lon, min_lat, max_lon, max_lat)` throughout — not `(lat, lon)`.

**Bahasa Indonesia support**: `llm.py` system prompt handles Indonesian date expressions ("N tahun/bulan/minggu/hari kebelakang", "kemarin", "minggu lalu") and location prefixes ("di", "untuk", "daerah").

**Output directories**: Maps, reports, vector data, and flood masks are written under `output/` (gitignored at repo root).

## Documentation

- `docs/ARCHITECTURE.md` — System design and data flow diagrams
- `docs/API_REFERENCE.md` — FastAPI endpoint specs
- `docs/DEPLOYMENT.md` — GCP Cloud Run / Dockerfile / GitHub Actions CI/CD
- `docs/VALIDATION.md` — IoU metrics and accuracy benchmarks
- `docs/USER_GUIDE.md` — CLI and API usage
