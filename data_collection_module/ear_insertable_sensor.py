"""
Ear-Insertable Biosignal Sensor Module

This module implements the ear-insertable biosignal sensor described in the patent,
which includes capabilities for measuring brain waves (EEG), core temperature,
blood flow using laser Doppler technology, and frequency-scanning impedance.

Technical Note: This implementation is based on patented technology filed by Ucaretron Inc.
"""

import time
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class EarInsertableSensor:
    """
    Implementation of the ear-insertable biosignal sensor system.
    
    This class simulates the cylindrical ear-insertable sensor with medical-grade
    silicone exterior and high-performance sensor modules inside.
    
    Specifications:
    - Size: 5mm diameter, 10mm length cylindrical shape
    - Wireless transmission: Bluetooth 5.0 LE and high-speed communication
    - Transmission range: Up to 100m (without obstacles)
    - Battery: 50mAh lithium polymer, up to 72 hours continuous use
    - Charging: Wireless (dedicated charging case)
    """
    
    def __init__(self, pet_id: str, sensor_id: str):
        """
        Initialize the ear-insertable sensor.
        
        Args:
            pet_id: Unique identifier for the pet
            sensor_id: Unique identifier for the sensor
        """
        self.pet_id = pet_id
        self.sensor_id = sensor_id
        self.is_active = False
        self.battery_level = 100.0  # Percentage
        self.last_reading_time = None
        self.data_encryption = "AES-256"  # Encryption method
        
        # EEG channel configuration
        self.eeg_channels = 8
        self.eeg_sampling_rate = 250  # Hz
        
        # Temperature measurement precision
        self.temperature_precision = 0.1  # Celsius
        
        # Baseline values (to be calibrated for individual pets)
        self.baseline = {
            'core_temperature': 38.6,  # Celsius
            'blood_flow': 100,  # Arbitrary units
            'eeg_baseline': [10, 20, 15, 12, 18, 14, 16, 11],  # Baseline for each EEG channel
            'impedance_baseline': 600  # ohms
        }
        
        print(f"Ear-insertable sensor {sensor_id} initialized for pet {pet_id}")
    
    def activate(self) -> bool:
        """Activate the sensor"""
        self.is_active = True
        self.last_reading_time = time.time()
        print(f"Ear sensor {self.sensor_id} activated")
        return True
    
    def deactivate(self) -> bool:
        """Deactivate the sensor"""
        self.is_active = False
        print(f"Ear sensor {self.sensor_id} deactivated")
        return True
    
    def get_battery_level(self) -> float:
        """Get the current battery level"""
        # Simulate battery drain
        if self.is_active:
            time_elapsed = time.time() - self.last_reading_time
            # Battery drain rate: ~1.4% per hour (72-hour battery life)
            self.battery_level -= time_elapsed * 0.00039  # 0.039% drain per 100 seconds
            self.battery_level = max(0, self.battery_level)
        return self.battery_level
    
    def measure_core_temperature(self) -> Dict[str, Any]:
        """
        Measure the core temperature with 0.1°C precision.
        
        Returns:
            Dict containing temperature measurement data
        """
        if not self.is_active:
            raise ValueError("Sensor is not active. Please activate the sensor first.")
        
        # Update last reading time
        current_time = time.time()
        self.last_reading_time = current_time
        
        # Simulate temperature reading with small random variation
        # The precision is 0.1°C as specified in the patent
        base_temp = self.baseline['core_temperature']
        variation = random.uniform(-0.2, 0.2)
        # Round to nearest 0.1°C to simulate the specified precision
        temperature = round((base_temp + variation) * 10) / 10
        
        return {
            'timestamp': current_time,
            'pet_id': self.pet_id,
            'sensor_id': self.sensor_id,
            'measurement_type': 'core_temperature',
            'value': temperature,
            'unit': 'Celsius',
            'precision': self.temperature_precision
        }
    
    def measure_blood_flow(self) -> Dict[str, Any]:
        """
        Measure blood flow using laser Doppler technology.
        
        Returns:
            Dict containing blood flow measurement data
        """
        if not self.is_active:
            raise ValueError("Sensor is not active. Please activate the sensor first.")
        
        # Update last reading time
        current_time = time.time()
        self.last_reading_time = current_time
        
        # Simulate blood flow reading with random variation
        base_flow = self.baseline['blood_flow']
        variation = random.uniform(-10, 10)
        blood_flow = base_flow + variation
        
        # Add pulsatile component to simulate heartbeats
        heart_rate = 80 + random.uniform(-5, 5)  # bpm
        period = 60 / heart_rate  # seconds per beat
        phase = (current_time % period) / period  # 0 to 1
        pulsatile_component = 15 * np.sin(2 * np.pi * phase)  # pulsatile amplitude
        
        blood_flow += pulsatile_component
        
        return {
            'timestamp': current_time,
            'pet_id': self.pet_id,
            'sensor_id': self.sensor_id,
            'measurement_type': 'blood_flow',
            'value': blood_flow,
            'unit': 'arbitrary_units',
            'technology': 'laser_doppler'
        }
    
    def measure_eeg(self, duration_seconds: float = 1.0) -> Dict[str, Any]:
        """
        Measure EEG signals across 8 channels with high resolution.
        
        Args:
            duration_seconds: Duration of the EEG recording in seconds
            
        Returns:
            Dict containing EEG measurement data
        """
        if not self.is_active:
            raise ValueError("Sensor is not active. Please activate the sensor first.")
        
        # Update last reading time
        current_time = time.time()
        self.last_reading_time = current_time
        
        # Calculate number of samples based on sampling rate and duration
        num_samples = int(self.eeg_sampling_rate * duration_seconds)
        
        # Simulate EEG data for each channel
        eeg_data = []
        for channel in range(self.eeg_channels):
            # Base amplitude for this channel
            base_amplitude = self.baseline['eeg_baseline'][channel]
            
            # Generate time series for this channel
            # We'll simulate a mix of different frequency bands (delta, theta, alpha, beta, gamma)
            channel_data = np.zeros(num_samples)
            
            # Add delta waves (1-4 Hz)
            delta_freq = random.uniform(1, 4)
            delta_amplitude = base_amplitude * 0.4
            time_points = np.linspace(0, duration_seconds, num_samples)
            channel_data += delta_amplitude * np.sin(2 * np.pi * delta_freq * time_points)
            
            # Add theta waves (4-8 Hz)
            theta_freq = random.uniform(4, 8)
            theta_amplitude = base_amplitude * 0.3
            channel_data += theta_amplitude * np.sin(2 * np.pi * theta_freq * time_points)
            
            # Add alpha waves (8-13 Hz)
            alpha_freq = random.uniform(8, 13)
            alpha_amplitude = base_amplitude * 0.2
            channel_data += alpha_amplitude * np.sin(2 * np.pi * alpha_freq * time_points)
            
            # Add beta waves (13-30 Hz)
            beta_freq = random.uniform(13, 30)
            beta_amplitude = base_amplitude * 0.1
            channel_data += beta_amplitude * np.sin(2 * np.pi * beta_freq * time_points)
            
            # Add random noise
            noise = np.random.normal(0, base_amplitude * 0.05, num_samples)
            channel_data += noise
            
            eeg_data.append(channel_data.tolist())
        
        return {
            'timestamp': current_time,
            'pet_id': self.pet_id,
            'sensor_id': self.sensor_id,
            'measurement_type': 'eeg',
            'channels': self.eeg_channels,
            'sampling_rate': self.eeg_sampling_rate,
            'duration': duration_seconds,
            'data': eeg_data,
            'unit': 'microvolts'
        }
    
    def perform_impedance_scan(self, frequency_range: Tuple[float, float], steps: int = 10) -> Dict[str, Any]:
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
        # Using a model where impedance has a characteristic pattern for this pet
        impedances = []
        
        # Create a pet-specific pattern using a combination of sine waves
        # This simulates the unique biological characteristics of each pet
        phase_shift = hash(self.pet_id) % 360  # Unique phase shift based on pet ID
        phase_shift_rad = phase_shift * np.pi / 180
        
        for freq in frequencies:
            # Baseline impedance that varies with frequency
            freq_factor = freq / max_freq
            baseline_impedance = self.baseline['impedance_baseline']
            
            # Add a unique pattern based on the pet's characteristics
            pattern = (
                20 * np.sin(2 * np.pi * freq_factor + phase_shift_rad) +
                10 * np.sin(4 * np.pi * freq_factor + phase_shift_rad) +
                5 * np.sin(8 * np.pi * freq_factor + phase_shift_rad)
            )
            
            # Final impedance with the pattern and some random noise
            impedance = baseline_impedance + pattern + random.uniform(-5, 5)
            impedances.append(impedance)
        
        return {
            'timestamp': time.time(),
            'pet_id': self.pet_id,
            'sensor_id': self.sensor_id,
            'measurement_type': 'impedance_scan',
            'frequencies': frequencies.tolist(),
            'impedances': impedances,
            'unit': 'ohms'
        }
    
    def collect_all_data(self, impedance_scan: bool = True) -> Dict[str, Any]:
        """
        Collect all biosignal data from the ear sensor.
        
        Args:
            impedance_scan: Whether to include impedance scan
            
        Returns:
            Dictionary containing all measurements
        """
        if not self.is_active:
            raise ValueError("Sensor is not active. Please activate the sensor first.")
        
        # Collect all measurements
        data = {
            'timestamp': time.time(),
            'pet_id': self.pet_id,
            'sensor_id': self.sensor_id,
            'battery_level': self.get_battery_level(),
            'core_temperature': self.measure_core_temperature()['value'],
            'blood_flow': self.measure_blood_flow()['value'],
            'eeg': self.measure_eeg(duration_seconds=2.0)
        }
        
        # Add impedance scan if requested
        if impedance_scan:
            impedance_data = self.perform_impedance_scan((100, 10000), steps=20)
            data['impedance_scan'] = {
                'frequencies': impedance_data['frequencies'],
                'impedances': impedance_data['impedances']
            }
        
        return data


# Example usage
if __name__ == "__main__":
    # Create an ear-insertable sensor for a pet
    ear_sensor = EarInsertableSensor(pet_id="pet12345", sensor_id="ear_sensor001")
    
    # Activate the sensor
    ear_sensor.activate()
    
    # Measure core temperature
    temperature_data = ear_sensor.measure_core_temperature()
    print(f"Core temperature: {temperature_data['value']}°C")
    
    # Measure blood flow
    blood_flow_data = ear_sensor.measure_blood_flow()
    print(f"Blood flow: {blood_flow_data['value']} {blood_flow_data['unit']}")
    
    # Measure EEG
    eeg_data = ear_sensor.measure_eeg(duration_seconds=0.5)
    print(f"EEG data collected: {eeg_data['channels']} channels, {len(eeg_data['data'][0])} samples")
    
    # Perform impedance scan
    impedance_data = ear_sensor.perform_impedance_scan((100, 5000), steps=10)
    print("Impedance scan completed with", len(impedance_data['frequencies']), "frequency points")
    
    # Collect all data
    all_data = ear_sensor.collect_all_data()
    print("All biosignal data collected successfully")
    
    # Check battery level
    battery = ear_sensor.get_battery_level()
    print(f"Battery level: {battery:.2f}%")
    
    # Deactivate the sensor
    ear_sensor.deactivate()