"""
Configuration module for the car price prediction project.
This module loads environment variables from the .env file using python-dotenv.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_PATH = Path(os.getenv("DATA_PATH", "data/"))
RAW_DATA_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH = DATA_PATH / "processed"

# Models path
MODELS_PATH = Path(os.getenv("MODELS_PATH", "models/"))

# Ensure paths are absolute
if not DATA_PATH.is_absolute():
    DATA_PATH = PROJECT_ROOT / DATA_PATH
    RAW_DATA_PATH = PROJECT_ROOT / RAW_DATA_PATH
    PROCESSED_DATA_PATH = PROJECT_ROOT / PROCESSED_DATA_PATH

if not MODELS_PATH.is_absolute():
    MODELS_PATH = PROJECT_ROOT / MODELS_PATH

# Create directories if they don't exist
DATA_PATH.mkdir(parents=True, exist_ok=True)
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)
