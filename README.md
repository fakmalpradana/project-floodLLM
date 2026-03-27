# FloodLLM - AI-Powered Flood Monitoring System

An EarthGPT-inspired application for automated flood detection, risk prediction, and damage assessment using natural language prompts.

## Features

- 🌍 **Natural Language Interface**: "Show flood extent in Jakarta this week"
- 🛰️ **Multi-Source Data**: Sentinel-1 SAR, Sentinel-2, GPM rainfall, river gauges
- 🤖 **AI Processing**: LLM-powered task orchestration with Google AI (Gemini/Claude)
- 🗺️ **Interactive Maps**: Folium-based flood extent visualization
- 📊 **Damage Assessment**: Automated infrastructure impact analysis

## Architecture

```
User Prompt → LLM Parser → Data Pipeline → Flood Detection → Maps + Reports
```

## Quick Start

### 1. Create Conda Environment

```bash
cd /Users/akmal/Desktop/2_smt2/01_komgeo/project-floodLLM
conda env create -f environment.yml
conda activate flood-llm
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env with your API keys
```

**Required API Keys:**
- [Google AI Studio](https://aistudio.google.com/apikey) - For LLM
- [Copernicus Data Space](https://dataspace.copernicus.org/) - For Sentinel data
- [NASA Earthdata](https://urs.earthdata.nasa.gov/) - For GPM/CHIRPS

### 3. Run the Application

```bash
# Development mode
python -m app.api.main

# Or with uvicorn
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. API Endpoints

- `POST /api/prompt` - Submit natural language query
- `GET /api/status/{job_id}` - Check processing status
- `GET /api/map/{job_id}` - Download flood map
- `GET /api/report/{job_id}` - Download PDF report

## Example Usage

```bash
curl -X POST http://localhost:8000/api/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Show flood extent in Jakarta, Indonesia for the last 7 days"}'
```

## Project Structure

```
flood-llm/
├── app/
│   ├── api/              # FastAPI endpoints
│   ├── data/             # Data downloaders
│   ├── processing/       # Flood detection algorithms
│   ├── visualization/    # Map and report generation
│   └── utils/            # Utilities
├── tests/
├── notebooks/            # Validation notebooks
├── data/                 # Local data storage
└── output/               # Generated maps and reports
```

## Validation

Run the validation suite:

```bash
python -m tests.test_end_to_end
python -m notebooks.validation
```

## Documentation

For detailed guides, see the [`docs/`](docs/) directory:

- **[User Guide](docs/USER_GUIDE.md)** - CLI usage and workflows
- **[API Reference](docs/API_REFERENCE.md)** - FastAPI endpoint documentation
- **[Architecture](docs/ARCHITECTURE.md)** - System design and data flow
- **[Deployment](docs/DEPLOYMENT.md)** - GCP deployment guide
- **[Validation](docs/VALIDATION.md)** - Testing and IoU metrics

## License

MIT License
