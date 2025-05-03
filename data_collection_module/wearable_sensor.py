"""
Wearable Sensor Data Collection Module

This module implements the high-precision wearable sensor system described in the patent,
which includes capabilities for measuring heart rate, temperature, blood pressure,
respiration rate, oxygen saturation, and frequency-scanning impedance.

Technical Note: This implementation is based on patented technology filed by Ucaretron Inc.
"""

import time
import random
import numpy as np
from typing import Dict, List, Tuple, Optional

class WearableSensor:
    """
    Implementation of the high-precision wearable sensor system.
    
    This class simulates the ultra-small, high-performance wearable sensor
    based on ultra-narrow linewidth semiconductor sensor technology.
    
    Specifications:
    - Size: 2mm x 2mm x 0.5mm
    - Power consumption: <1mW
    - Data transmission: Integrated high-speed communication module
    """
    
    def __init__(self, pet_id: str, sensor_id: str):
        """
        Initialize the wearable sensor.
        
        Args:
            pet_id: Unique identifier for the pet
            sensor_id: Unique identifier for the sensor
        """
        self.pet_id = pet_id
        self.sensor_id = sensor_id
        self.is_active = False
        self.battery_level = 100.0  # Percentage
        self.sampling_rate = 100  # Hz
        self.last_reading_time = None
        
        # Simulated baseline values for the pet (to be calibrated)
        self.baseline = {
            'heart_rate': 80,  # bpm
            'temperature': 38.5,  # Celsius
            'blood_pressure_systolic': 120,  # mmHg
            'blood_pressure_diastolic': 80,  # mmHg
            'respiration_rate': 20,  # breaths per minute
            'oxygen_saturation': 98,  # percentage
            'impedance_baseline': 500  # ohms (baseline for impedance at reference frequency)
        }
        
        # Edge AI processor for real-time data pre-processing
        self.edge_ai_enabled = True
        
        print(f"Wearable sensor {sensor_id} initialized for pet {pet_id}")
    
    def activate(self) -> bool:
        """Activate the sensor"""
        self.is_active = True
        self.last_reading_time = time.time()
        print(f"Sensor {self.sensor_id} activated")
        return True
    
    def deactivate(self) -> bool:
        """Deactivate the sensor"""
        self.is_active = False
        print(f"Sensor {self.sensor_id} deactivated")
        return True
    
    def get_battery_level(self) -> float:
        """Get the current battery level"""
        # Simulate battery drain
        if self.is_active:
            time_elapsed = time.time() - self.last_reading_time
            self.battery_level -= time_elapsed * 0.001  # 0.1% drain per second when active
            self.battery_level = max(0, self.battery_level)
        return self.battery_level
    
    def read_vital_signs(self) -> Dict[str, float]:
        """
        Read the vital signs from the sensor.
        
        Returns:
            Dict containing vital sign measurements
        """
        if not self.is_active:
            raise ValueError("Sensor is not active. Please activate the sensor first.")
        
        # Update last reading time
        current_time = time.time()
        time_elapsed = current_time - self.last_reading_time
        self.last_reading_time = current_time
        
        # Simulate vital sign readings with small random variations around baseline
        heart_rate = self.baseline['heart_rate'] + random.uniform(-5, 5)
        temperature = self.baseline['temperature'] + random.uniform(-0.2, 0.2)
        blood_pressure_systolic = self.baseline['blood_pressure_systolic'] + random.uniform(-5, 5)
        blood_pressure_diastolic = self.baseline['blood_pressure_diastolic'] + random.uniform(-3, 3)
        respiration_rate = self.baseline['respiration_rate'] + random.uniform(-2, 2)
        oxygen_saturation = min(100, self.baseline['oxygen_saturation'] + random.uniform(-1, 1))
        
        # Create readings dictionary
        readings = {
            'timestamp': current_time,
            'pet_id': self.pet_id,
            'sensor_id': self.sensor_id,
            'heart_rate': heart_rate,
            'temperature': temperature,
            'blood_pressure_systolic': blood_pressure_systolic,
            'blood_pressure_diastolic': blood_pressure_diastolic,
            'respiration_rate': respiration_rate,
            'oxygen_saturation': oxygen_saturation,
        }
        
        # Perform edge AI processing if enabled
        if self.edge_ai_enabled:
            readings = self._process_with_edge_ai(readings)
        
        return readings
    
    def perform_impedance_scan(self, frequency_range: Tuple[float, float], steps: int = 10) -> Dict[str, List[float]]:
        """
        Perform a frequency-scanning impedance measurement.
        
        This is a key innovation in the patent, allowing for personalized monitoring
        that can distinguish individual differences between users.
        
        Args:
            frequency_range: Tuple of (min_frequency, max_frequency) in Hz
            steps: Number of frequency steps to measure
            
        Returns:
            Dictionary containing frequencies and corresponding impedance values
        """
        if not self.is_active:
            raise ValueError("Sensor is not active. Please activate the sensor first.")
        
        min_freq, max_freq = frequency_range
        frequencies = np.linspace(min_freq, max_freq, steps)
        
        # Simulate impedance measurements based on frequency
        # Using a simplified model where impedance varies with frequency
        impedances = []
        for freq in frequencies:
            # Baseline impedance that varies with frequency (higher at higher frequencies)
            baseline_impedance = self.baseline['impedance_baseline'] * (1 + 0.1 * freq / max_freq)
            
            # Add some random variation
            impedance = baseline_impedance + random.uniform(-20, 20)
            impedances.append(impedance)
        
        return {
            'timestamp': time.time(),
            'pet_id': self.pet_id,
            'sensor_id': self.sensor_id,
            'frequencies': frequencies.tolist(),
            'impedances': impedances
        }
    
    def _process_with_edge_ai(self, readings: Dict[str, float]) -> Dict[str, float]:
        """
        Process readings with the edge AI processor for anomaly detection.
        
        Args:
            readings: Raw sensor readings
            
        Returns:
            Processed readings with anomaly flags if detected
        """
        # Simulate edge AI processing
        # Check if any vital signs are significantly outside the expected range
        anomalies = []
        
        # Heart rate check
        if abs(readings['heart_rate'] - self.baseline['heart_rate']) > 15:
            anomalies.append('heart_rate')
            
        # Temperature check
        if abs(readings['temperature'] - self.baseline['temperature']) > 0.5:
            anomalies.append('temperature')
            
        # Blood pressure check
        if abs(readings['blood_pressure_systolic'] - self.baseline['blood_pressure_systolic']) > 15:
            anomalies.append('blood_pressure_systolic')
            
        if abs(readings['blood_pressure_diastolic'] - self.baseline['blood_pressure_diastolic']) > 10:
            anomalies.append('blood_pressure_diastolic')
            
        # Oxygen saturation check
        if readings['oxygen_saturation'] < 95:
            anomalies.append('oxygen_saturation')
        
        # Add anomaly flags to readings
        if anomalies:
            readings['anomalies_detected'] = True
            readings['anomaly_flags'] = anomalies
        else:
            readings['anomalies_detected'] = False
        
        return readings


# Example usage
if __name__ == "__main__":
    # Create a wearable sensor for a pet
    sensor = WearableSensor(pet_id="pet12345", sensor_id="sensor001")
    
    # Activate the sensor
    sensor.activate()
    
    # Read vital signs
    vital_signs = sensor.read_vital_signs()
    print("Vital signs:", vital_signs)
    
    # Perform impedance scan
    impedance_data = sensor.perform_impedance_scan((1000, 10000), steps=20)
    print("Impedance scan completed with", len(impedance_data['frequencies']), "frequency points")
    
    # Check battery level
    battery = sensor.get_battery_level()
    print(f"Battery level: {battery:.2f}%")
    
    # Deactivate the sensor
    sensor.deactivate()