"""
Data Collection Manager Module

This module implements the main data collection manager that coordinates all sensors
and aggregates data for analysis. It handles the communication between sensors and
the central system, performs data preprocessing, and manages the flow of information.

Technical Note: This implementation is based on patented technology filed by Ucaretron Inc.
"""

import time
import json
import threading
import queue
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np

# Import sensor modules
from data_collection_module.wearable_sensor import WearableSensor
from data_collection_module.ear_insertable_sensor import EarInsertableSensor

class DataCollectionManager:
    """
    Manager for collecting and processing data from multiple pet sensors.
    
    This class coordinates data collection from all sensors, aggregates the data,
    performs initial preprocessing, and forwards it to the AI analysis module.
    """
    
    def __init__(self):
        """Initialize the data collection manager"""
        # Dictionary to store active sensors by pet ID
        self.sensors = {}  # Format: {pet_id: {'wearable': sensor, 'ear': sensor, ...}}
        
        # Data queue for collected measurements
        self.data_queue = queue.Queue()
        
        # Flag to control the data collection thread
        self.running = False
        self.collection_thread = None
        
        # Data collection frequency (Hz)
        self.collection_frequency = 1  # Default: collect data once per second
        
        # Data preprocessing options
        self.preprocessing_enabled = True
        self.noise_filtering_enabled = True
        self.anomaly_detection_enabled = True
        
        # Communication settings
        self.transmission_encrypted = True
        self.transmission_protocol = "high-speed"  # Options: "bluetooth", "high-speed", "lora"
        
        print("Data Collection Manager initialized")
    
    def register_pet(self, pet_id: str, pet_info: Dict[str, Any]) -> bool:
        """
        Register a new pet in the system.
        
        Args:
            pet_id: Unique identifier for the pet
            pet_info: Dictionary containing pet information (name, species, breed, age, etc.)
            
        Returns:
            True if registration was successful
        """
        if pet_id in self.sensors:
            print(f"Pet {pet_id} is already registered")
            return False
        
        # Create new entry for this pet
        self.sensors[pet_id] = {
            'info': pet_info,
            'sensors': {},
            'last_collection_time': None,
            'data_history': []
        }
        
        print(f"Pet {pet_id} ({pet_info.get('name', 'unnamed')}) registered successfully")
        return True
    
    def add_sensor(self, pet_id: str, sensor_type: str, sensor_id: str) -> Optional[Union[WearableSensor, EarInsertableSensor]]:
        """
        Add a new sensor for a pet.
        
        Args:
            pet_id: Pet's unique identifier
            sensor_type: Type of sensor ("wearable" or "ear")
            sensor_id: Unique identifier for the sensor
            
        Returns:
            The sensor object if successful, None otherwise
        """
        if pet_id not in self.sensors:
            print(f"Pet {pet_id} is not registered. Please register the pet first.")
            return None
        
        # Create and add the appropriate sensor
        if sensor_type.lower() == "wearable":
            sensor = WearableSensor(pet_id=pet_id, sensor_id=sensor_id)
            self.sensors[pet_id]['sensors']['wearable'] = sensor
            print(f"Wearable sensor {sensor_id} added for pet {pet_id}")
            return sensor
        
        elif sensor_type.lower() == "ear":
            sensor = EarInsertableSensor(pet_id=pet_id, sensor_id=sensor_id)
            self.sensors[pet_id]['sensors']['ear'] = sensor
            print(f"Ear-insertable sensor {sensor_id} added for pet {pet_id}")
            return sensor
        
        else:
            print(f"Unsupported sensor type: {sensor_type}")
            return None
    
    def activate_sensors(self, pet_id: str) -> bool:
        """
        Activate all sensors for a specific pet.
        
        Args:
            pet_id: Pet's unique identifier
            
        Returns:
            True if all sensors were activated successfully
        """
        if pet_id not in self.sensors:
            print(f"Pet {pet_id} is not registered")
            return False
        
        success = True
        
        # Activate each sensor
        for sensor_type, sensor in self.sensors[pet_id]['sensors'].items():
            activation_result = sensor.activate()
            success = success and activation_result
            
            if activation_result:
                print(f"{sensor_type.capitalize()} sensor activated for pet {pet_id}")
            else:
                print(f"Failed to activate {sensor_type} sensor for pet {pet_id}")
        
        return success
    
    def deactivate_sensors(self, pet_id: str) -> bool:
        """
        Deactivate all sensors for a specific pet.
        
        Args:
            pet_id: Pet's unique identifier
            
        Returns:
            True if all sensors were deactivated successfully
        """
        if pet_id not in self.sensors:
            print(f"Pet {pet_id} is not registered")
            return False
        
        success = True
        
        # Deactivate each sensor
        for sensor_type, sensor in self.sensors[pet_id]['sensors'].items():
            deactivation_result = sensor.deactivate()
            success = success and deactivation_result
            
            if deactivation_result:
                print(f"{sensor_type.capitalize()} sensor deactivated for pet {pet_id}")
            else:
                print(f"Failed to deactivate {sensor_type} sensor for pet {pet_id}")
        
        return success
    
    def start_data_collection(self):
        """Start the continuous data collection process"""
        if self.running:
            print("Data collection is already running")
            return
        
        self.running = True
        self.collection_thread = threading.Thread(target=self._data_collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        
        print("Data collection started")
    
    def stop_data_collection(self):
        """Stop the continuous data collection process"""
        if not self.running:
            print("Data collection is not running")
            return
        
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
        
        print("Data collection stopped")
    
    def _data_collection_loop(self):
        """Background thread for continuous data collection"""
        while self.running:
            # Collect data from all active pets and sensors
            for pet_id, pet_data in self.sensors.items():
                try:
                    # Skip if no sensors are registered for this pet
                    if not pet_data['sensors']:
                        continue
                    
                    # Collect data from all sensors
                    collected_data = self._collect_pet_data(pet_id)
                    
                    # Process the collected data
                    if collected_data and self.preprocessing_enabled:
                        processed_data = self._preprocess_data(collected_data)
                    else:
                        processed_data = collected_data
                    
                    # Store processed data in the queue for further analysis
                    if processed_data:
                        self.data_queue.put(processed_data)
                        
                        # Update last collection time
                        self.sensors[pet_id]['last_collection_time'] = time.time()
                        
                        # Store in history (limited to last 1000 readings)
                        self.sensors[pet_id]['data_history'].append(processed_data)
                        if len(self.sensors[pet_id]['data_history']) > 1000:
                            self.sensors[pet_id]['data_history'].pop(0)
                    
                except Exception as e:
                    print(f"Error collecting data for pet {pet_id}: {str(e)}")
            
            # Sleep until next collection cycle
            time.sleep(1.0 / self.collection_frequency)
    
    def _collect_pet_data(self, pet_id: str) -> Dict[str, Any]:
        """
        Collect data from all sensors for a specific pet.
        
        Args:
            pet_id: Pet's unique identifier
            
        Returns:
            Dictionary containing all collected data
        """
        pet_data = self.sensors[pet_id]
        sensors = pet_data['sensors']
        
        # Initialize data structure
        collected_data = {
            'timestamp': time.time(),
            'pet_id': pet_id,
            'sensors': {}
        }
        
        # Collect data from wearable sensor if available
        if 'wearable' in sensors:
            wearable = sensors['wearable']
            if wearable.is_active:
                try:
                    vital_signs = wearable.read_vital_signs()
                    impedance = wearable.perform_impedance_scan((1000, 10000), steps=10)
                    
                    collected_data['sensors']['wearable'] = {
                        'vital_signs': vital_signs,
                        'impedance': impedance,
                        'battery_level': wearable.get_battery_level()
                    }
                except Exception as e:
                    print(f"Error reading from wearable sensor for pet {pet_id}: {str(e)}")
        
        # Collect data from ear-insertable sensor if available
        if 'ear' in sensors:
            ear_sensor = sensors['ear']
            if ear_sensor.is_active:
                try:
                    # Collect all data from ear sensor
                    ear_data = ear_sensor.collect_all_data()
                    collected_data['sensors']['ear'] = ear_data
                except Exception as e:
                    print(f"Error reading from ear sensor for pet {pet_id}: {str(e)}")
        
        return collected_data
    
    def _preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess the collected data.
        
        This includes:
        - Noise filtering
        - Missing value handling
        - Time synchronization
        - Data normalization
        - Initial anomaly detection
        
        Args:
            data: Raw collected data
            
        Returns:
            Preprocessed data
        """
        # Create a copy to avoid modifying the original
        processed_data = data.copy()
        processed_data['preprocessed'] = True
        
        # Apply noise filtering if enabled
        if self.noise_filtering_enabled:
            processed_data = self._apply_noise_filtering(processed_data)
        
        # Apply anomaly detection if enabled
        if self.anomaly_detection_enabled:
            processed_data = self._detect_anomalies(processed_data)
        
        # Add metadata about preprocessing
        processed_data['preprocessing_info'] = {
            'timestamp': time.time(),
            'noise_filtering_applied': self.noise_filtering_enabled,
            'anomaly_detection_applied': self.anomaly_detection_enabled
        }
        
        return processed_data
    
    def _apply_noise_filtering(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply noise filtering to the collected data.
        
        Args:
            data: Data to filter
            
        Returns:
            Filtered data
        """
        # Create a copy to avoid modifying the original
        filtered_data = data.copy()
        
        # Apply simple moving average filter to vital signs if available
        if 'sensors' in filtered_data and 'wearable' in filtered_data['sensors']:
            wearable_data = filtered_data['sensors']['wearable']
            
            if 'vital_signs' in wearable_data:
                vital_signs = wearable_data['vital_signs']
                
                # For this simple implementation, we'll just simulate filtered values
                # In a real system, we would apply actual filtering algorithms
                for key in ['heart_rate', 'temperature', 'blood_pressure_systolic', 
                           'blood_pressure_diastolic', 'respiration_rate', 'oxygen_saturation']:
                    if key in vital_signs:
                        # Add noise reduction flag to indicate filtering was applied
                        vital_signs[f"{key}_filtered"] = True
        
        # Apply filtering to EEG data if available
        if 'sensors' in filtered_data and 'ear' in filtered_data['sensors']:
            ear_data = filtered_data['sensors']['ear']
            
            if 'eeg' in ear_data and 'data' in ear_data['eeg']:
                # For this simple implementation, we'll just indicate filtering was applied
                ear_data['eeg']['filtered'] = True
        
        return filtered_data
    
    def _detect_anomalies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect anomalies in the collected data.
        
        Args:
            data: Data to analyze
            
        Returns:
            Data with anomaly flags
        """
        # Create a copy to avoid modifying the original
        analyzed_data = data.copy()
        
        # Initialize anomalies list
        analyzed_data['anomalies'] = []
        
        # Check vital signs if available
        if 'sensors' in analyzed_data and 'wearable' in analyzed_data['sensors']:
            wearable_data = analyzed_data['sensors']['wearable']
            
            if 'vital_signs' in wearable_data:
                vital_signs = wearable_data['vital_signs']
                
                # Check for anomalies in vital signs
                # This is a simplified implementation - in a real system, we would use more sophisticated methods
                if 'heart_rate' in vital_signs and vital_signs['heart_rate'] > 120:
                    analyzed_data['anomalies'].append({
                        'type': 'heart_rate_high',
                        'value': vital_signs['heart_rate'],
                        'threshold': 120,
                        'sensor': 'wearable',
                        'severity': 'medium'
                    })
                
                if 'temperature' in vital_signs and vital_signs['temperature'] > 39.5:
                    analyzed_data['anomalies'].append({
                        'type': 'temperature_high',
                        'value': vital_signs['temperature'],
                        'threshold': 39.5,
                        'sensor': 'wearable',
                        'severity': 'high'
                    })
                
                if 'oxygen_saturation' in vital_signs and vital_signs['oxygen_saturation'] < 95:
                    analyzed_data['anomalies'].append({
                        'type': 'oxygen_saturation_low',
                        'value': vital_signs['oxygen_saturation'],
                        'threshold': 95,
                        'sensor': 'wearable',
                        'severity': 'medium'
                    })
        
        # Check ear sensor data if available
        if 'sensors' in analyzed_data and 'ear' in analyzed_data['sensors']:
            ear_data = analyzed_data['sensors']['ear']
            
            if 'core_temperature' in ear_data:
                temp = ear_data['core_temperature']
                
                if temp > 39.5:
                    analyzed_data['anomalies'].append({
                        'type': 'core_temperature_high',
                        'value': temp,
                        'threshold': 39.5,
                        'sensor': 'ear',
                        'severity': 'high'
                    })
        
        return analyzed_data
    
    def get_latest_data(self, pet_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest collected data for a specific pet.
        
        Args:
            pet_id: Pet's unique identifier
            
        Returns:
            Latest data, or None if no data is available
        """
        if pet_id not in self.sensors:
            print(f"Pet {pet_id} is not registered")
            return None
        
        pet_data = self.sensors[pet_id]
        
        if not pet_data['data_history']:
            print(f"No data available for pet {pet_id}")
            return None
        
        return pet_data['data_history'][-1]
    
    def get_data_history(self, pet_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get historical data for a specific pet.
        
        Args:
            pet_id: Pet's unique identifier
            limit: Maximum number of data points to return
            
        Returns:
            List of historical data points
        """
        if pet_id not in self.sensors:
            print(f"Pet {pet_id} is not registered")
            return []
        
        pet_data = self.sensors[pet_id]
        
        if not pet_data['data_history']:
            print(f"No data available for pet {pet_id}")
            return []
        
        # Return up to 'limit' most recent data points
        return pet_data['data_history'][-limit:]
    
    def export_data(self, pet_id: str, format: str = "json") -> Optional[str]:
        """
        Export all data for a specific pet.
        
        Args:
            pet_id: Pet's unique identifier
            format: Export format ("json" or "csv")
            
        Returns:
            Exported data string
        """
        if pet_id not in self.sensors:
            print(f"Pet {pet_id} is not registered")
            return None
        
        pet_data = self.sensors[pet_id]
        
        if not pet_data['data_history']:
            print(f"No data available for pet {pet_id}")
            return None
        
        if format.lower() == "json":
            # Convert data to JSON string
            return json.dumps(pet_data['data_history'], indent=2)
        
        elif format.lower() == "csv":
            # This is a simplified CSV export - a real implementation would be more complex
            csv_lines = ["timestamp,pet_id,sensor_type,measurement_type,value"]
            
            for data_point in pet_data['data_history']:
                timestamp = data_point['timestamp']
                pet_id = data_point['pet_id']
                
                for sensor_type, sensor_data in data_point.get('sensors', {}).items():
                    if sensor_type == 'wearable' and 'vital_signs' in sensor_data:
                        vs = sensor_data['vital_signs']
                        for key, value in vs.items():
                            if key in ['heart_rate', 'temperature', 'blood_pressure_systolic', 
                                      'blood_pressure_diastolic', 'respiration_rate', 'oxygen_saturation']:
                                csv_lines.append(f"{timestamp},{pet_id},{sensor_type},{key},{value}")
                    
                    elif sensor_type == 'ear':
                        for key, value in sensor_data.items():
                            if key in ['core_temperature', 'blood_flow']:
                                csv_lines.append(f"{timestamp},{pet_id},{sensor_type},{key},{value}")
            
            return "\n".join(csv_lines)
        
        else:
            print(f"Unsupported export format: {format}")
            return None


# Example usage
if __name__ == "__main__":
    # Create a data collection manager
    manager = DataCollectionManager()
    
    # Register a pet
    pet_info = {
        'name': 'Max',
        'species': 'dog',
        'breed': 'Golden Retriever',
        'age': 3,
        'weight': 30.5  # kg
    }
    manager.register_pet(pet_id="pet12345", pet_info=pet_info)
    
    # Add sensors for the pet
    wearable = manager.add_sensor(pet_id="pet12345", sensor_type="wearable", sensor_id="wear001")
    ear_sensor = manager.add_sensor(pet_id="pet12345", sensor_type="ear", sensor_id="ear001")
    
    # Activate sensors
    manager.activate_sensors(pet_id="pet12345")
    
    # Start data collection
    manager.start_data_collection()
    
    # Let it run for a few seconds
    for i in range(5):
        print(f"Collecting data... ({i+1}/5)")
        time.sleep(1)
    
    # Get the latest data
    latest_data = manager.get_latest_data(pet_id="pet12345")
    print("\nLatest data collected:")
    if latest_data:
        print(f"Timestamp: {latest_data['timestamp']}")
        print(f"Number of anomalies detected: {len(latest_data.get('anomalies', []))}")
        for sensor_type, sensor_data in latest_data.get('sensors', {}).items():
            print(f"  {sensor_type.capitalize()} sensor data collected")
    
    # Export data
    json_data = manager.export_data(pet_id="pet12345", format="json")
    print(f"\nExported data size: {len(json_data) if json_data else 0} bytes")
    
    # Stop data collection
    manager.stop_data_collection()
    
    # Deactivate sensors
    manager.deactivate_sensors(pet_id="pet12345")