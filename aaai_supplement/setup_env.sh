#!/bin/bash

# FineScope AAAI Supplement Environment Setup
echo "Setting up FineScope environment..."

# Create virtual environment
python -m venv finescope_env

# Activate virtual environment
source finescope_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo "Environment setup complete!"
echo "To activate the environment, run: source finescope_env/bin/activate"
echo "To run the demo, execute: python main.py" 