#!/bin/bash
# FloodLLM Setup Script

set -e

echo "================================"
echo "FloodLLM Setup"
echo "================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda first."
    exit 1
fi

# Create conda environment
echo ""
echo "Step 1: Creating conda environment..."
conda env create -f environment.yml 2>/dev/null || {
    echo "Environment may already exist. Updating..."
    conda env update -f environment.yml
}

# Activate environment
echo ""
echo "Step 2: Activating environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flood-llm

# Copy environment file
echo ""
echo "Step 3: Setting up environment variables..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file. Please edit with your API keys."
else
    echo ".env file already exists."
fi

# Run tests
echo ""
echo "Step 4: Running tests..."
python -m tests.test_end_to_end

echo ""
echo "================================"
echo "Setup Complete!"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your API keys:"
echo "   - GOOGLE_API_KEY (from Google AI Studio)"
echo "   - COPERNICUS_USERNAME/PASSWORD (from dataspace.copernicus.eu)"
echo "   - NASA_EARTHDATA_USERNAME/PASSWORD (from urs.earthdata.nasa.gov)"
echo ""
echo "2. Start the API server:"
echo "   conda activate flood-llm"
echo "   python -m app.api.main"
echo ""
echo "3. Or use the CLI:"
echo "   python cli.py --help"
echo "   python cli.py analyze -l 'Jakarta, Indonesia'"
echo ""
