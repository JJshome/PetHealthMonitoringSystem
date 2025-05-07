"""
Data Simulator for the Pet Health Monitoring System.

This module provides functionality to generate synthetic data that simulates
readings from various pet health monitoring sensors. It can create both normal
and abnormal health patterns for testing and demonstration purposes.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta

from .config import config, PET_TYPES
from .utils import save_csv, save_json


class PetDataSimulator:
    """
    Simulator for generating synthetic pet health monitoring data.
    """
    
    def __init__(self, pet_type: str = "dog", pet_name: str = "Buddy", 
                 pet_age: float = 5.0, pet_weight: float = 15.0):
        """
        Initialize the pet data simulator.
        
        Args:
            pet_type: Type of pet (e.g., "dog", "cat").
            pet_name: Name of the pet.
            pet_age: Age of the pet in years.
            pet_weight: Weight of the pet in kg.
        """
        if pet_type not in PET_TYPES:
            raise ValueError(f"Pet type '{pet_type}' not supported. Supported types: {list(PET_TYPES.keys())}")
        
        self.pet_type = pet_type
        self.pet_name = pet_name
        self.pet_age = pet_age
        self.pet_weight = pet_weight
        self.vital_ranges = PET_TYPES[pet_type]
        
        # Adjust vital ranges based on age and weight
        self._adjust_vital_ranges()
        
    def _adjust_vital_ranges(self):
        """Adjust vital sign ranges based on pet age and weight."""
        # Heart rate decreases with age
        if self.pet_age > 7:  # Senior pet
            self.vital_ranges["heart_rate"]["min"] *= 0.9
            self.vital_ranges["heart_rate"]["max"] *= 0.9
        elif self.pet_age < 1:  # Puppy/kitten
            self.vital_ranges["heart_rate"]["min"] *= 1.2
            self.vital_ranges["heart_rate"]["max"] *= 1.2
        
        # Adjust for weight (larger animals have slower heart rates)
        if self.pet_type == "dog":
            if self.pet_weight > 25:  # Large dog
                self.vital_ranges["heart_rate"]["min"] *= 0.8
                self.vital_ranges["heart_rate"]["max"] *= 0.8
            elif self.pet_weight < 5:  # Small dog
                self.vital_ranges["heart_rate"]["min"] *= 1.1
                self.vital_ranges["heart_rate"]["max"] *= 1.1
    
    def generate_vital_sign(self, vital_type: str, duration_hours: float = 24.0, 
                            sample_rate: int = 10, include_anomalies: bool = False,
                            anomaly_probability: float = 0.05) -> np.ndarray:
        """
        Generate a time series for a specific vital sign.
        
        Args:
            vital_type: Type of vital sign (e.g., "heart_rate", "temperature").
            duration_hours: Duration of the time series in hours.
            sample_rate: Number of samples per second.
            include_anomalies: Whether to include anomalies in the data.
            anomaly_probability: Probability of an anomaly occurring.
            
        Returns:
            NumPy array containing the generated vital sign data.
        """
        if vital_type not in self.vital_ranges:
            raise ValueError(f"Vital type '{vital_type}' not supported for {self.pet_type}. "
                           f"Supported types: {list(self.vital_ranges.keys())}")
        
        # Calculate number of samples
        total_samples = int(duration_hours * 3600 * sample_rate)
        
        # Get the min and max values for this vital sign
        min_val = self.vital_ranges[vital_type]["min"]
        max_val = self.vital_ranges[vital_type]["max"]
        mean_val = (min_val + max_val) / 2
        
        # Generate a baseline signal with diurnal variations
        t = np.linspace(0, duration_hours, total_samples)
        baseline = mean_val + (max_val - min_val) * 0.2 * np.sin(2 * np.pi * t / 24)
        
        # Add random variations
        random_variations = np.random.normal(0, (max_val - min_val) * 0.05, total_samples)
        signal = baseline + random_variations
        
        # Add some autocorrelation (smoothing)
        window_size = int(sample_rate * 10)  # 10-second window
        kernel = np.ones(window_size) / window_size
        signal = np.convolve(signal, kernel, mode='same')
        
        # Add periodic variations (e.g., for activity, feeding)
        if vital_type in ["heart_rate", "respiratory_rate"]:
            # Add activity spikes a few times a day
            activity_times = np.random.randint(0, total_samples, size=int(duration_hours / 3))
            for time in activity_times:
                if time + sample_rate * 300 < total_samples:  # 5-minute activity
                    activity_duration = np.random.randint(sample_rate * 60, sample_rate * 300)
                    activity_intensity = np.random.uniform(0.1, 0.3) * (max_val - min_val)
                    signal[time:time+activity_duration] += activity_intensity * np.sin(
                        np.linspace(0, np.pi, activity_duration)
                    )
        
        # Add anomalies if requested
        if include_anomalies:
            anomaly_count = int(total_samples * anomaly_probability / 100)  # Convert to percentage of total samples
            if anomaly_count > 0:
                anomaly_starts = np.random.randint(0, total_samples - sample_rate * 60, size=anomaly_count)
                for start in anomaly_starts:
                    anomaly_duration = np.random.randint(sample_rate * 10, sample_rate * 60)  # 10-60 seconds
                    end = min(start + anomaly_duration, total_samples)
                    
                    # Decide whether to generate high or low anomaly
                    is_high_anomaly = np.random.random() > 0.5
                    
                    if is_high_anomaly:
                        # High anomaly (spike)
                        anomaly_value = max_val * np.random.uniform(1.1, 1.5)
                        anomaly_shape = np.sin(np.linspace(0, np.pi, end - start)) * (anomaly_value - signal[start])
                        signal[start:end] += anomaly_shape
                    else:
                        # Low anomaly (drop)
                        anomaly_value = min_val * np.random.uniform(0.5, 0.9)
                        anomaly_shape = np.sin(np.linspace(0, np.pi, end - start)) * (signal[start] - anomaly_value)
                        signal[start:end] -= anomaly_shape
        
        return signal
    
    def generate_dataset(self, duration_hours: float = 24.0, sample_rate: int = 10,
                         include_anomalies: bool = False, output_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Generate a complete dataset with all vital signs.
        
        Args:
            duration_hours: Duration of the time series in hours.
            sample_rate: Number of samples per second.
            include_anomalies: Whether to include anomalies in the data.
            output_dir: Directory to save the generated data. If None, data is not saved.
            
        Returns:
            DataFrame containing the generated dataset.
        """
        # Calculate timestamps
        total_samples = int(duration_hours * 3600 * sample_rate)
        start_time = datetime.now() - timedelta(hours=duration_hours)
        timestamps = [start_time + timedelta(seconds=i/sample_rate) for i in range(total_samples)]
        
        # Initialize DataFrame with timestamps
        df = pd.DataFrame(index=timestamps)
        
        # Generate data for each vital sign
        for vital_type in self.vital_ranges.keys():
            df[vital_type] = self.generate_vital_sign(
                vital_type, duration_hours, sample_rate, include_anomalies
            )
        
        # Add environmental data
        df["ambient_temperature"] = 22.0 + 2.0 * np.sin(np.linspace(0, 2 * np.pi * duration_hours / 24, total_samples))
        df["ambient_temperature"] += np.random.normal(0, 0.5, total_samples)
        
        df["ambient_humidity"] = 50.0 + 10.0 * np.sin(np.linspace(0, 2 * np.pi * duration_hours / 24, total_samples))
        df["ambient_humidity"] += np.random.normal(0, 2.0, total_samples)
        
        # Add derived metrics
        df["stress_level"] = self._calculate_stress_level(df)
        df["activity_state"] = self._calculate_activity_state(df)
        
        # Save data if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the complete dataset
            save_csv(df, os.path.join(output_dir, f"{self.pet_name}_data.csv"))
            
            # Save pet metadata
            metadata = {
                "pet_name": self.pet_name,
                "pet_type": self.pet_type,
                "pet_age": self.pet_age,
                "pet_weight": self.pet_weight,
                "data_start": timestamps[0].isoformat(),
                "data_end": timestamps[-1].isoformat(),
                "sample_rate": sample_rate,
                "includes_anomalies": include_anomalies,
            }
            save_json(metadata, os.path.join(output_dir, f"{self.pet_name}_metadata.json"))
        
        return df
    
    def _calculate_stress_level(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate a synthetic stress level based on vital signs."""
        # Normalize heart rate and respiratory rate to 0-1 range
        hr_min = self.vital_ranges["heart_rate"]["min"]
        hr_max = self.vital_ranges["heart_rate"]["max"]
        hr_norm = (df["heart_rate"] - hr_min) / (hr_max - hr_min)
        
        rr_min = self.vital_ranges["respiratory_rate"]["min"]
        rr_max = self.vital_ranges["respiratory_rate"]["max"]
        rr_norm = (df["respiratory_rate"] - rr_min) / (rr_max - rr_min)
        
        # Calculate stress level (0-100 scale)
        stress = 20.0 + 60.0 * (0.7 * hr_norm + 0.3 * rr_norm)
        
        # Add some random variations
        stress += np.random.normal(0, 5.0, len(df))
        
        # Ensure values are within 0-100 range
        stress = np.clip(stress, 0, 100)
        
        return stress
    
    def _calculate_activity_state(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate activity state based on vital signs."""
        # Define thresholds for different activity states
        hr_mean = (self.vital_ranges["heart_rate"]["min"] + self.vital_ranges["heart_rate"]["max"]) / 2
        hr_range = self.vital_ranges["heart_rate"]["max"] - self.vital_ranges["heart_rate"]["min"]
        
        sleep_threshold = hr_mean - hr_range * 0.3
        rest_threshold = hr_mean - hr_range * 0.1
        active_threshold = hr_mean + hr_range * 0.2
        
        # Initialize with 'resting' state
        activity_state = np.ones(len(df)) * 1  # 1 = resting
        
        # Set states based on thresholds
        activity_state[df["heart_rate"] <= sleep_threshold] = 0  # 0 = sleeping
        activity_state[df["heart_rate"] >= rest_threshold] = 1  # 1 = resting
        activity_state[df["heart_rate"] >= active_threshold] = 2  # 2 = active
        
        # Apply some smoothing to avoid too rapid state changes
        window_size = int(len(df) / (24 * 60))  # Approximately 1-minute window
        if window_size > 1:
            kernel = np.ones(window_size) / window_size
            smoothed = np.convolve(activity_state, kernel, mode='same')
            activity_state = np.round(smoothed).astype(int)
        
        return activity_state
    
    def generate_health_event(self, event_type: str, duration_minutes: float = 30.0,
                              sample_rate: int = 10, severity: float = 0.5) -> pd.DataFrame:
        """
        Generate a specific health event for testing anomaly detection.
        
        Args:
            event_type: Type of health event (e.g., "fever", "tachycardia", "bradycardia").
            duration_minutes: Duration of the event in minutes.
            sample_rate: Number of samples per second.
            severity: Severity of the event (0.0-1.0).
            
        Returns:
            DataFrame containing the generated health event data.
        """
        # Calculate number of samples
        total_samples = int(duration_minutes * 60 * sample_rate)
        
        # Generate timestamps
        start_time = datetime.now() - timedelta(minutes=duration_minutes)
        timestamps = [start_time + timedelta(seconds=i/sample_rate) for i in range(total_samples)]
        
        # Initialize DataFrame with timestamps
        df = pd.DataFrame(index=timestamps)
        
        # Generate baseline vital signs
        for vital_type in self.vital_ranges.keys():
            df[vital_type] = self.generate_vital_sign(
                vital_type, duration_minutes / 60, sample_rate, False
            )
        
        # Add environmental data
        df["ambient_temperature"] = 22.0 + np.random.normal(0, 0.5, total_samples)
        df["ambient_humidity"] = 50.0 + np.random.normal(0, 2.0, total_samples)
        
        # Modify vital signs based on the health event
        if event_type == "fever":
            # Increase temperature gradually
            temp_increase = 1.5 * severity
            temp_profile = np.sin(np.linspace(0, np.pi, total_samples)) * temp_increase
            df["temperature"] += temp_profile
            
            # Increase heart rate
            hr_increase = 20 * severity
            hr_profile = np.sin(np.linspace(0, np.pi, total_samples)) * hr_increase
            df["heart_rate"] += hr_profile
        
        elif event_type == "tachycardia":
            # Sudden increase in heart rate
            hr_max = self.vital_ranges["heart_rate"]["max"]
            hr_increase = (hr_max * 1.5 - hr_max) * severity
            
            # Shape of the tachycardia event (sharp rise, plateau, gradual decline)
            rise_samples = int(total_samples * 0.1)
            plateau_samples = int(total_samples * 0.6)
            decline_samples = total_samples - rise_samples - plateau_samples
            
            hr_profile = np.zeros(total_samples)
            hr_profile[:rise_samples] = np.linspace(0, hr_increase, rise_samples)
            hr_profile[rise_samples:rise_samples+plateau_samples] = hr_increase
            hr_profile[rise_samples+plateau_samples:] = np.linspace(hr_increase, 0, decline_samples)
            
            df["heart_rate"] += hr_profile
            
            # Also increase respiratory rate
            df["respiratory_rate"] += hr_profile * 0.5
        
        elif event_type == "bradycardia":
            # Sudden decrease in heart rate
            hr_min = self.vital_ranges["heart_rate"]["min"]
            hr_decrease = (hr_min - hr_min * 0.6) * severity
            
            # Shape of the bradycardia event
            hr_profile = np.zeros(total_samples)
            for i in range(total_samples):
                hr_profile[i] = -hr_decrease * np.exp(-((i - total_samples/2) / (total_samples/4))**2)
            
            df["heart_rate"] += hr_profile
        
        elif event_type == "stress":
            # Increase in heart rate and respiratory rate
            hr_increase = 30 * severity
            rr_increase = 10 * severity
            
            # Add oscillations to simulate anxiety
            t = np.linspace(0, duration_minutes, total_samples)
            oscillations = np.sin(2 * np.pi * t * 0.2) * 5 * severity  # 0.2 Hz oscillation
            
            df["heart_rate"] += hr_increase + oscillations
            df["respiratory_rate"] += rr_increase + oscillations * 0.3
        
        elif event_type == "dehydration":
            # Gradual increase in temperature and decrease in activity
            temp_increase = 0.7 * severity
            temp_profile = np.linspace(0, temp_increase, total_samples)
            df["temperature"] += temp_profile
            
            # Decrease in activity level
            df["activity_level"] = df["activity_level"] * (1 - 0.5 * severity)
        
        else:
            raise ValueError(f"Unsupported health event type: {event_type}")
        
        # Recalculate derived metrics
        df["stress_level"] = self._calculate_stress_level(df)
        df["activity_state"] = self._calculate_activity_state(df)
        
        return df


# Helper functions to generate complex datasets

def generate_multi_day_dataset(pet_type: str = "dog", pet_name: str = "Buddy", 
                               days: int = 7, include_anomalies: bool = True,
                               output_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Generate a multi-day dataset with daily patterns and occasional anomalies.
    
    Args:
        pet_type: Type of pet (e.g., "dog", "cat").
        pet_name: Name of the pet.
        days: Number of days to simulate.
        include_anomalies: Whether to include anomalies in the data.
        output_dir: Directory to save the generated data. If None, data is not saved.
        
    Returns:
        DataFrame containing the generated dataset.
    """
    simulator = PetDataSimulator(pet_type=pet_type, pet_name=pet_name)
    
    # Generate data for each day and concatenate
    all_data = []
    for day in range(days):
        daily_data = simulator.generate_dataset(
            duration_hours=24.0, 
            include_anomalies=include_anomalies,
            output_dir=None  # Don't save individual days
        )
        all_data.append(daily_data)
    
    # Concatenate all days
    combined_df = pd.concat(all_data)
    
    # Save combined dataset if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the complete dataset
        save_csv(combined_df, os.path.join(output_dir, f"{pet_name}_{days}days_data.csv"))
        
        # Save pet metadata
        metadata = {
            "pet_name": pet_name,
            "pet_type": pet_type,
            "data_start": combined_df.index[0].isoformat(),
            "data_end": combined_df.index[-1].isoformat(),
            "days": days,
            "includes_anomalies": include_anomalies,
        }
        save_json(metadata, os.path.join(output_dir, f"{pet_name}_{days}days_metadata.json"))
    
    return combined_df


def generate_health_event_dataset(output_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Generate a dataset containing various health events for testing and demonstration.
    
    Args:
        output_dir: Directory to save the generated data. If None, data is not saved.
        
    Returns:
        Dictionary mapping event types to their respective DataFrames.
    """
    simulator = PetDataSimulator()
    
    # List of health events to generate
    events = ["fever", "tachycardia", "bradycardia", "stress", "dehydration"]
    
    # Generate data for each event
    event_datasets = {}
    for event in events:
        data = simulator.generate_health_event(
            event_type=event,
            duration_minutes=60.0,
            severity=0.8
        )
        event_datasets[event] = data
        
        # Save event data if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_csv(data, os.path.join(output_dir, f"{event}_event_data.csv"))
    
    return event_datasets
