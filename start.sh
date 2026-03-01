#!/bin/bash

# BlindAssit Navigation Assistant - Startup Script

echo "========================================="
echo "  BlindAssit Navigation Assistant"
echo "========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed!"
    echo "Please install Python 3.8 or higher."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo "Installing dependencies..."
    ./venv/bin/pip install -r requirements.txt
fi

# Activate virtual environment
source venv/bin/activate

# Load environment variables from .env if exists
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "Environment variables loaded from .env"
fi

# Check if OPENROUTER_API_KEY is set
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo ""
    echo "Warning: OPENROUTER_API_KEY not set"
    echo "AI queries will be limited."
    echo "Get your API key at: https://openrouter.ai/keys"
    echo "Set it with: export OPENROUTER_API_KEY='your-key'"
    echo ""
fi

# Check if SeaFormer model files exist
if [ ! -d "configs" ] || [ ! -d "mmseg" ]; then
    echo ""
    echo "Warning: SeaFormer model files not found!"
    echo "Please ensure the following directories exist:"
    echo "  - configs/"
    echo "  - mmseg/"
    echo ""
fi

# Start the Flask server
echo "Starting BlindAssit server on http://localhost:5000"
echo "Press Ctrl+C to stop"
echo ""

cd server
python3 client_camera_server.py
