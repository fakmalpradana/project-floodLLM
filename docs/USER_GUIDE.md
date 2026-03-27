# FloodLLM User Guide

This guide covers how to use the FloodLLM command-line interface (CLI) for flood monitoring and analysis.

## Installation

### 1. Create Conda Environment

```bash
cd project-floodLLM
conda env create -f environment.yml
conda activate flood-llm
```

### 2. Configure API Keys

Copy the example environment file and configure your API keys:

```bash
cp .env.example .env
nano .env  # or use your preferred editor
```

**Required API Keys:**

| Service | Purpose | Get Key |
|---------|---------|---------|
| `GOOGLE_API_KEY` | LLM prompt parsing and report generation | [Google AI Studio](https://aistudio.google.com/apikey) |
| `COPERNICUS_USERNAME` / `PASSWORD` | Sentinel-1/2 satellite data | [Copernicus Data Space](https://dataspace.copernicus.org/) |
| `NASA_EARTHDATA_USERNAME` / `PASSWORD` | GPM rainfall data | [NASA Earthdata](https://urs.earthdata.nasa.gov/) |

---

## CLI Commands

### `flood-llm analyze` - Run Flood Analysis

Run a complete flood analysis for a specified location and time period.

**Usage:**
```bash
flood-llm analyze --location <LOCATION> [OPTIONS]
```

**Options:**

| Option | Short | Required | Default | Description |
|--------|-------|----------|---------|-------------|
| `--location` | `-l` | Yes | - | Location name (e.g., "Jakarta, Indonesia") |
| `--start` | `-s` | No | "last 7 days" | Start date or relative date |
| `--end` | `-e` | No | "today" | End date |
| `--output` | `-o` | No | auto | Output directory |

**Examples:**

```bash
# Basic analysis for Jakarta
flood-llm analyze -l "Jakarta, Indonesia"

# Specify date range
flood-llm analyze -l "Bangkok, Thailand" -s "2024-01-01" -e "2024-01-07"

# Use relative dates
flood-llm analyze -l "Manila, Philippines" -s "last 14 days"

# Custom output directory
flood-llm analyze -l "Jakarta" -o "./my-analysis"
```

**Sample Output:**
```
🌍 FloodLLM Analysis
Location: Jakarta, Indonesia
Period: last 7 days to today
----------------------------------------
📍 Geocoding location...
   Bounding box: (106.5, -6.5, 107.0, -6.0)
🛰️ Downloading satellite data...
   ✓ Sentinel-1: 3 images
   ✓ Sentinel-2: 2 images
🌧️ Downloading rainfall data...
   Total rainfall: 185.0 mm
🔍 Processing flood detection...
   Flood area: 45.50 km²
   Flooded pixels: 125000
📄 Generating report...
   Report: /output/reports/cli_20240115_103000.pdf

✅ Analysis complete!
Output directory: /Users/akmal/Desktop/2_smt2/01_komgeo/project-floodLLM/output
```

---

### `flood-llm parse` - Parse Natural Language Prompt

Parse a natural language prompt to extract location, dates, and task type.

**Usage:**
```bash
flood-llm parse "<PROMPT>"
```

**Examples:**

```bash
# Parse a simple prompt
flood-llm parse "Show floods in Bangkok for the last week"

# Parse a complex prompt
flood-llm parse "Generate damage assessment report for recent floods in Manila"

# Parse risk prediction request
flood-llm parse "Assess flood risk in Jakarta this month"
```

**Sample Output:**
```
📋 Parsed Prompt
----------------------------------------
Location: Bangkok
Date Range: last 7 days to today
Task Type: flood_detection
```

**Extracted Fields:**

| Field | Description |
|-------|-------------|
| `Location` | Extracted place name |
| `Date Range` | Start and end dates |
| `Task Type` | One of: `flood_detection`, `risk_prediction`, `damage_assessment`, `all` |
| `Context` | Additional context (if provided) |

---

### `flood-llm status` - Check System Status

Display system configuration and check if all dependencies are available.

**Usage:**
```bash
flood-llm status
```

**Sample Output:**
```
🔧 FloodLLM System Status
========================================
✓ App modules: Available
  Data directory: /path/to/project-floodLLM/data
  Output directory: /path/to/project-floodLLM/output
✓ Google API: Configured
✓ Copernicus API: Configured
✓ NASA Earthdata: Configured

📦 Optional Dependencies:
✓ rasterio: Installed
✓ folium: Installed
✓ google-generativeai: Installed
```

**When Dependencies Are Missing:**
```
🔧 FloodLLM System Status
========================================
✗ App modules: Not available

Install dependencies:
  conda env create -f environment.yml
  conda activate flood-llm
```

---

### `flood-llm test` - Run Test Suite

Run the end-to-end test suite to verify the installation and configuration.

**Usage:**
```bash
flood-llm test
```

**Sample Output:**
```
🧪 Running FloodLLM Test Suite
========================================

=== Testing Data Structures ===
Base directory: /path/to/project-floodLLM
Data directory: /path/to/project-floodLLM/data
Output directory: /path/to/project-floodLLM/output
✓ All directories configured correctly

=== Testing LLM Prompt Parsing ===

Prompt: 'Show flood extent in Jakarta for the last 7 days'
  Location: Jakarta
  Date Range: last 7 days to today
  Task Type: flood_detection

=== Testing Geocoding ===
✓ Jakarta geocoded: (106.5, -6.5, 107.0, -6.0)
✓ Bangkok geocoded: (100.5, 13.5, 101.0, 14.0)

=== Testing Report Generation ===
Report generated: /path/to/output/reports/test123.html
✓ Report file created successfully

========================================
TEST SUMMARY
========================================
  ✓ PASS: Data Structures
  ✓ PASS: LLM Parsing
  ✓ PASS: Geocoding
  ✓ PASS: Report Generation

Total: 4/4 tests passed
✅ All tests passed!
```

---

## Workflow Examples

### Example 1: Quick Flood Check

```bash
# Check system status first
flood-llm status

# Run analysis for recent floods
flood-llm analyze -l "Jakarta, Indonesia" -s "last 7 days"
```

### Example 2: Historical Analysis

```bash
# Analyze a past flood event
flood-llm analyze -l "Bangkok, Thailand" -s "2023-09-01" -e "2023-09-30"
```

### Example 3: Parse and Verify

```bash
# First, parse the prompt to verify extraction
flood-llm parse "Show flood extent in Manila during the monsoon season"

# Then run the analysis
flood-llm analyze -l "Manila, Philippines" -s "last 30 days"
```

---

## Troubleshooting

### "App modules not available"

```bash
# Reinstall dependencies
conda activate flood-llm
pip install -r requirements.txt
```

### "Geocoding failed"

The geocoding service may be unavailable. The system falls back to default coordinates for major cities.

### "No Sentinel-1 data available"

- Check your Copernicus credentials in `.env`
- Verify the date range has available satellite passes
- The area may not have recent satellite coverage

### "LLM parsing error"

- Check your Google API key is valid
- The LLM service is optional; a simple regex fallback is used when unavailable

---

## Output Files

All generated files are stored in the `output/` directory:

```
output/
├── maps/
│   └── flood_map_<job_id>.html    # Interactive flood map
├── reports/
│   └── flood_report_<job_id>.html # Assessment report
└── flood_masks/
    ├── flood_mask_<job_id>.tiff   # Raw flood mask (GeoTIFF)
    └── flood_mask_<job_id>.json   # Mask metadata
```

---

## Next Steps

- See [API Reference](API_REFERENCE.md) for programmatic access
- See [Architecture](ARCHITECTURE.md) for how the system works
- See [Deployment](DEPLOYMENT.md) for production setup
