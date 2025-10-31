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

# Data paths - external data directory
DATA_PATH = Path(os.getenv("DATA_PATH", "/Users/brunobrumbrum/Documents/data/le_boncoin_13_oct_2025"))

# Local processed data paths
LOCAL_DATA_PATH = PROJECT_ROOT / "data"
RAW_DATA_PATH = LOCAL_DATA_PATH / "raw" 
PROCESSED_DATA_PATH = LOCAL_DATA_PATH / "processed"

# Models path
MODELS_PATH = Path(os.getenv("MODELS_PATH", "models/"))

# Ensure local paths are absolute
if not MODELS_PATH.is_absolute():
    MODELS_PATH = PROJECT_ROOT / MODELS_PATH

# Create local directories if they don't exist (but not external data path)
LOCAL_DATA_PATH.mkdir(parents=True, exist_ok=True)
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)
