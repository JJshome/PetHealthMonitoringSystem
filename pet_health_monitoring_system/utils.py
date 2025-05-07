"""
Utilities for the Pet Health Monitoring System.

This module provides common utility functions used across different components
of the system, including data processing, visualization, and logging.
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union

from .config import config


# Set up logging
def setup_logging(log_level: str = None) -> logging.Logger:
    """
    Set up and configure the logger.
    
    Args:
        log_level: The log level to use. If None, uses the level from config.
        
    Returns:
        Configured logger instance.
    """
    if log_level is None:
        log_level = config["system"]["log_level"]
        
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),  # Log to console
            logging.FileHandler(os.path.join("logs", "pet_health_monitor.log"))  # Log to file
        ]
    )
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    return logging.getLogger("pet_health_monitor")


logger = setup_logging()


# Data processing utilities
def smooth_signal(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply a moving average smoothing to a signal.
    
    Args:
        data: The input signal data as a numpy array.
        window_size: The size of the moving average window.
        
    Returns:
        Smoothed signal array.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def normalize_signal(data: np.ndarray) -> np.ndarray:
    """
    Normalize a signal to have zero mean and unit variance.
    
    Args:
        data: The input signal data as a numpy array.
        
    Returns:
        Normalized signal array.
    """
    return (data - np.mean(data)) / np.std(data)


def resample_signal(data: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    """
    Resample a signal to a different sampling rate.
    
    Args:
        data: The input signal data as a numpy array.
        original_rate: The original sampling rate in Hz.
        target_rate: The target sampling rate in Hz.
        
    Returns:
        Resampled signal array.
    """
    # Calculate the number of samples in the resampled signal
    n_samples = int(len(data) * target_rate / original_rate)
    
    # Create a time vector for the original signal
    t_original = np.linspace(0, len(data) / original_rate, len(data))
    
    # Create a time vector for the resampled signal
    t_resampled = np.linspace(0, len(data) / original_rate, n_samples)
    
    # Resample the signal using linear interpolation
    return np.interp(t_resampled, t_original, data)


# Data visualization utilities
def plot_vital_signs(data: pd.DataFrame, vital_signs: List[str], 
                     title: str = "Vital Signs Monitoring", 
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot vital signs data.
    
    Args:
        data: DataFrame containing the vital signs data.
        vital_signs: List of column names to plot.
        title: Plot title.
        save_path: Path to save the plot. If None, the plot is not saved.
        
    Returns:
        Matplotlib figure object.
    """
    fig, axes = plt.subplots(len(vital_signs), 1, figsize=(12, 3 * len(vital_signs)))
    if len(vital_signs) == 1:
        axes = [axes]
    
    for i, sign in enumerate(vital_signs):
        axes[i].plot(data.index, data[sign])
        axes[i].set_title(sign)
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Value")
        axes[i].grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


# Time and date utilities
def timestamp_to_iso(timestamp: float) -> str:
    """
    Convert a UNIX timestamp to ISO format.
    
    Args:
        timestamp: UNIX timestamp.
        
    Returns:
        ISO formatted date string.
    """
    return datetime.fromtimestamp(timestamp).isoformat()


def iso_to_timestamp(iso_str: str) -> float:
    """
    Convert an ISO format string to UNIX timestamp.
    
    Args:
        iso_str: ISO formatted date string.
        
    Returns:
        UNIX timestamp.
    """
    return datetime.fromisoformat(iso_str).timestamp()


# File utilities
def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save.
        filepath: Path to save the file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.
    
    Args:
        filepath: Path to the JSON file.
        
    Returns:
        Loaded data as a dictionary.
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_csv(data: pd.DataFrame, filepath: str) -> None:
    """
    Save a DataFrame to a CSV file.
    
    Args:
        data: DataFrame to save.
        filepath: Path to save the file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data.to_csv(filepath, index=True)


def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load a DataFrame from a CSV file.
    
    Args:
        filepath: Path to the CSV file.
        
    Returns:
        Loaded DataFrame.
    """
    return pd.read_csv(filepath, index_col=0)


# Alert and notification utilities
def is_value_abnormal(value: float, min_threshold: float, max_threshold: float, 
                      margin: float = 0.1) -> bool:
    """
    Check if a value is outside normal thresholds.
    
    Args:
        value: The value to check.
        min_threshold: The minimum normal threshold.
        max_threshold: The maximum normal threshold.
        margin: Margin of error as a fraction of the normal range.
        
    Returns:
        True if the value is abnormal, False otherwise.
    """
    range_size = max_threshold - min_threshold
    lower_bound = min_threshold - (range_size * margin)
    upper_bound = max_threshold + (range_size * margin)
    
    return value < lower_bound or value > upper_bound


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds.
        
    Returns:
        Formatted duration string (e.g., "2h 30m 15s").
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    result = ""
    if hours > 0:
        result += f"{hours}h "
    if minutes > 0 or hours > 0:
        result += f"{minutes}m "
    result += f"{seconds}s"
    
    return result.strip()
