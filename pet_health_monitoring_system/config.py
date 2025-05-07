"""
Configuration settings for the Pet Health Monitoring System.

This module contains all configurable parameters and settings used throughout
the system, including sensor settings, AI model parameters, and system paths.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Base paths
ROOT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
CONFIG_DIR = os.path.join(ROOT_DIR, "config")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

# Default configuration file path
DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIR, "config.yaml")

# Sensor settings
SENSOR_SAMPLE_RATE = 10  # Hz
SENSOR_DATA_PRECISION = 4  # Decimal places

# AI model settings
MODEL_INPUT_WINDOW = 300  # 30 seconds at 10Hz
MODEL_PREDICTION_INTERVAL = 60  # Predict every 60 seconds

# Data management settings
BLOCKCHAIN_NETWORK = "testnet"
BLOCKCHAIN_API_KEY = os.environ.get("BLOCKCHAIN_API_KEY", "")

# Remote veterinary settings
REMOTE_CONSULTATION_MAX_TIME = 30 * 60  # 30 minutes in seconds

# System settings
LOG_LEVEL = "INFO"
ENABLE_REAL_TIME_ALERTS = True
MAX_STORED_DATA_DAYS = 90  # Store data for up to 90 days

# Default pet types and their normal vital ranges
PET_TYPES = {
    "dog": {
        "heart_rate": {"min": 60, "max": 140},  # BPM
        "respiratory_rate": {"min": 10, "max": 30},  # breaths per minute
        "temperature": {"min": 37.5, "max": 39.2},  # Celsius
        "activity_level": {"min": 30, "max": 300},  # minutes of activity per day
    },
    "cat": {
        "heart_rate": {"min": 120, "max": 220},  # BPM
        "respiratory_rate": {"min": 16, "max": 40},  # breaths per minute
        "temperature": {"min": 38.0, "max": 39.2},  # Celsius
        "activity_level": {"min": 20, "max": 180},  # minutes of activity per day
    },
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file. If None, uses the default path.
        
    Returns:
        Dictionary containing configuration settings.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    if not os.path.exists(config_path):
        # Create default config file if it doesn't exist
        default_config = {
            "sensors": {
                "sample_rate": SENSOR_SAMPLE_RATE,
                "data_precision": SENSOR_DATA_PRECISION,
            },
            "ai_model": {
                "input_window": MODEL_INPUT_WINDOW,
                "prediction_interval": MODEL_PREDICTION_INTERVAL,
            },
            "data_management": {
                "blockchain_network": BLOCKCHAIN_NETWORK,
            },
            "remote_veterinary": {
                "max_consultation_time": REMOTE_CONSULTATION_MAX_TIME,
            },
            "system": {
                "log_level": LOG_LEVEL,
                "enable_real_time_alerts": ENABLE_REAL_TIME_ALERTS,
                "max_stored_data_days": MAX_STORED_DATA_DAYS,
            },
            "pet_types": PET_TYPES,
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        return default_config
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Load configuration on module import
config = load_config()
