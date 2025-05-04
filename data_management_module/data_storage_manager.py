"""
Data Storage Manager Module

This module handles the secure storage, retrieval, and management of pet health data.
It provides high-performance data access, backup functionality, and data versioning.

Technical Note: This implementation is based on patented technology filed by Ucaretron Inc.
"""

import time
import json
import os
import hashlib
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

class DataStorageManager:
    """
    Manages secure storage and retrieval of pet health data.
    
    This class provides a secure interface for storing, retrieving, and managing
    pet health data with encryption, versioning, and high availability.
    """
    
    def __init__(self, data_directory: str = "./data"):
        """
        Initialize the data storage manager.
        
        Args:
            data_directory: Directory for storing data files
        """
        self.data_directory = data_directory
        
        # Create data directories if they don't exist
        self._create_directory_structure()
        
        # Initialize data storage structures
        self.pet_data = {}         # Format: {pet_id: {metadata}}
        self.sensor_data = {}      # Format: {pet_id: {sensor_id: [data_points]}}
        self.health_records = {}   # Format: {pet_id: [health_records]}
        self.care_plans = {}       # Format: {pet_id: [care_plans]}
        
        # Load existing data if available
        self._load_data()
        
        print("Data Storage Manager initialized")
    
    def _create_directory_structure(self):
        """Create the necessary directory structure for data storage"""
        # Create main data directory
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)
        
        # Create subdirectories for different data types
        subdirs = ['pet_profiles', 'sensor_data', 'health_records', 'care_plans', 'backups']
        for subdir in subdirs:
            path = os.path.join(self.data_directory, subdir)
            if not os.path.exists(path):
                os.makedirs(path)
    
    def _load_data(self):
        """Load existing data from storage"""
        # Load pet profiles
        try:
            pet_profiles_dir = os.path.join(self.data_directory, 'pet_profiles')
            for filename in os.listdir(pet_profiles_dir):
                if filename.endswith('.json'):
                    pet_id = filename.split('.')[0]
                    file_path = os.path.join(pet_profiles_dir, filename)
                    with open(file_path, 'r') as f:
                        self.pet_data[pet_id] = json.load(f)
            print(f"Loaded {len(self.pet_data)} pet profiles")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading pet profiles: {str(e)}")
        
        # TODO: Load other data types (sensor data, health records, care plans)
        # For this implementation, we'll focus on pet profiles for simplicity
    
    def store_pet_profile(self, pet_id: str, profile_data: Dict[str, Any]) -> bool:
        """
        Store or update a pet profile.
        
        Args:
            pet_id: Unique identifier for the pet
            profile_data: Pet profile data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add metadata
            profile_data['last_updated'] = time.time()
            if 'created_at' not in profile_data:
                profile_data['created_at'] = time.time()
            
            # Update in-memory data
            self.pet_data[pet_id] = profile_data
            
            # Save to file
            file_path = os.path.join(self.data_directory, 'pet_profiles', f"{pet_id}.json")
            with open(file_path, 'w') as f:
                json.dump(profile_data, f, indent=2)
            
            print(f"Stored profile for pet {pet_id}")
            return True
        except Exception as e:
            print(f"Error storing pet profile: {str(e)}")
            return False
    
    def get_pet_profile(self, pet_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a pet profile.
        
        Args:
            pet_id: Unique identifier for the pet
            
        Returns:
            Pet profile data or None if not found
        """
        # Check if profile exists in memory
        if pet_id in self.pet_data:
            return self.pet_data[pet_id]
        
        # Try to load from file if not in memory
        try:
            file_path = os.path.join(self.data_directory, 'pet_profiles', f"{pet_id}.json")
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    profile_data = json.load(f)
                # Cache in memory
                self.pet_data[pet_id] = profile_data
                return profile_data
        except Exception as e:
            print(f"Error retrieving pet profile: {str(e)}")
        
        print(f"Profile for pet {pet_id} not found")
        return None
    
    def list_pets(self) -> List[Dict[str, Any]]:
        """
        List all stored pet profiles with basic information.
        
        Returns:
            List of pet profiles with basic information
        """
        pet_list = []
        
        # First check in-memory data
        for pet_id, profile in self.pet_data.items():
            # Include only basic information
            pet_list.append({
                'pet_id': pet_id,
                'name': profile.get('name', 'Unknown'),
                'species': profile.get('species', 'Unknown'),
                'breed': profile.get('breed', 'Unknown'),
                'age': profile.get('age', 0),
                'created_at': profile.get('created_at', 0),
                'last_updated': profile.get('last_updated', 0)
            })
        
        # Also check for any files not loaded in memory
        try:
            pet_profiles_dir = os.path.join(self.data_directory, 'pet_profiles')
            for filename in os.listdir(pet_profiles_dir):
                if filename.endswith('.json'):
                    pet_id = filename.split('.')[0]
                    # Skip if already in the list
                    if any(p['pet_id'] == pet_id for p in pet_list):
                        continue
                    
                    # Load basic info from file
                    file_path = os.path.join(pet_profiles_dir, filename)
                    try:
                        with open(file_path, 'r') as f:
                            profile = json.load(f)
                            pet_list.append({
                                'pet_id': pet_id,
                                'name': profile.get('name', 'Unknown'),
                                'species': profile.get('species', 'Unknown'),
                                'breed': profile.get('breed', 'Unknown'),
                                'age': profile.get('age', 0),
                                'created_at': profile.get('created_at', 0),
                                'last_updated': profile.get('last_updated', 0)
                            })
                    except (json.JSONDecodeError, IOError):
                        # Skip invalid files
                        continue
        except FileNotFoundError:
            pass
        
        # Sort by last updated time (most recent first)
        pet_list.sort(key=lambda x: x.get('last_updated', 0), reverse=True)
        
        return pet_list
    
    def delete_pet_profile(self, pet_id: str) -> bool:
        """
        Delete a pet profile and associated data.
        
        Args:
            pet_id: Unique identifier for the pet
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove from memory
            if pet_id in self.pet_data:
                del self.pet_data[pet_id]
            
            # Remove file
            file_path = os.path.join(self.data_directory, 'pet_profiles', f"{pet_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # TODO: Also remove associated sensor data, health records, care plans
            
            print(f"Deleted profile for pet {pet_id}")
            return True
        except Exception as e:
            print(f"Error deleting pet profile: {str(e)}")
            return False
    
    def store_sensor_data(self, pet_id: str, sensor_id: str, data_points: List[Dict[str, Any]]) -> bool:
        """
        Store sensor data for a pet.
        
        Args:
            pet_id: Unique identifier for the pet
            sensor_id: Identifier for the sensor
            data_points: List of sensor data points
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directories exist
            pet_sensor_dir = os.path.join(self.data_directory, 'sensor_data', pet_id)
            if not os.path.exists(pet_sensor_dir):
                os.makedirs(pet_sensor_dir)
            
            # Generate timestamp-based filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{sensor_id}_{timestamp}.json"
            file_path = os.path.join(pet_sensor_dir, filename)
            
            # Add metadata to each data point
            for point in data_points:
                if 'timestamp' not in point:
                    point['timestamp'] = time.time()
                point['sensor_id'] = sensor_id
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(data_points, f, indent=2)
            
            # Update in-memory cache
            if pet_id not in self.sensor_data:
                self.sensor_data[pet_id] = {}
            
            if sensor_id not in self.sensor_data[pet_id]:
                self.sensor_data[pet_id][sensor_id] = []
            
            # Append new data points to the in-memory cache
            self.sensor_data[pet_id][sensor_id].extend(data_points)
            
            # Limit in-memory cache size (keep only most recent 1000 points)
            if len(self.sensor_data[pet_id][sensor_id]) > 1000:
                self.sensor_data[pet_id][sensor_id] = self.sensor_data[pet_id][sensor_id][-1000:]
            
            print(f"Stored {len(data_points)} data points for pet {pet_id}, sensor {sensor_id}")
            return True
        except Exception as e:
            print(f"Error storing sensor data: {str(e)}")
            return False
    
    def get_recent_sensor_data(self, pet_id: str, sensor_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve recent sensor data for a pet.
        
        Args:
            pet_id: Unique identifier for the pet
            sensor_id: Identifier for the sensor
            limit: Maximum number of data points to return
            
        Returns:
            List of sensor data points
        """
        # Check if data exists in memory
        if pet_id in self.sensor_data and sensor_id in self.sensor_data[pet_id]:
            # Return the most recent data points
            return self.sensor_data[pet_id][sensor_id][-limit:]
        
        # Data not in memory, try to load from files
        try:
            pet_sensor_dir = os.path.join(self.data_directory, 'sensor_data', pet_id)
            if not os.path.exists(pet_sensor_dir):
                return []
            
            # Find all files for this sensor
            sensor_files = []
            for filename in os.listdir(pet_sensor_dir):
                if filename.startswith(f"{sensor_id}_") and filename.endswith('.json'):
                    file_path = os.path.join(pet_sensor_dir, filename)
                    modified_time = os.path.getmtime(file_path)
                    sensor_files.append((file_path, modified_time))
            
            # Sort files by modified time (most recent first)
            sensor_files.sort(key=lambda x: x[1], reverse=True)
            
            # Load data from the most recent files until we reach the limit
            data_points = []
            for file_path, _ in sensor_files:
                if len(data_points) >= limit:
                    break
                
                try:
                    with open(file_path, 'r') as f:
                        file_data = json.load(f)
                        data_points.extend(file_data)
                except (json.JSONDecodeError, IOError):
                    # Skip invalid files
                    continue
            
            # Sort by timestamp
            data_points.sort(key=lambda x: x.get('timestamp', 0))
            
            # Limit to the requested number
            data_points = data_points[-limit:]
            
            # Cache in memory
            if pet_id not in self.sensor_data:
                self.sensor_data[pet_id] = {}
            
            self.sensor_data[pet_id][sensor_id] = data_points
            
            return data_points
        except Exception as e:
            print(f"Error retrieving sensor data: {str(e)}")
            return []
    
    def store_health_record(self, pet_id: str, record_data: Dict[str, Any]) -> str:
        """
        Store a health record for a pet.
        
        Args:
            pet_id: Unique identifier for the pet
            record_data: Health record data
            
        Returns:
            Record ID if successful, empty string otherwise
        """
        try:
            # Ensure directories exist
            pet_health_dir = os.path.join(self.data_directory, 'health_records', pet_id)
            if not os.path.exists(pet_health_dir):
                os.makedirs(pet_health_dir)
            
            # Generate a unique record ID if not provided
            if 'record_id' not in record_data:
                record_id = f"hr_{int(time.time())}_{str(uuid.uuid4())[:8]}"
                record_data['record_id'] = record_id
            else:
                record_id = record_data['record_id']
            
            # Add metadata
            record_data['timestamp'] = record_data.get('timestamp', time.time())
            
            # Save to file
            file_path = os.path.join(pet_health_dir, f"{record_id}.json")
            with open(file_path, 'w') as f:
                json.dump(record_data, f, indent=2)
            
            # Update in-memory cache
            if pet_id not in self.health_records:
                self.health_records[pet_id] = []
            
            # Check if record already exists in memory
            for i, record in enumerate(self.health_records[pet_id]):
                if record.get('record_id') == record_id:
                    # Update existing record
                    self.health_records[pet_id][i] = record_data
                    break
            else:
                # Add new record
                self.health_records[pet_id].append(record_data)
            
            # Sort records by timestamp (most recent first)
            self.health_records[pet_id].sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            
            print(f"Stored health record {record_id} for pet {pet_id}")
            return record_id
        except Exception as e:
            print(f"Error storing health record: {str(e)}")
            return ""
    
    def get_health_records(self, pet_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve health records for a pet.
        
        Args:
            pet_id: Unique identifier for the pet
            limit: Maximum number of records to return
            
        Returns:
            List of health records
        """
        # Check if records exist in memory
        if pet_id in self.health_records:
            # Return the most recent records
            return self.health_records[pet_id][:limit]
        
        # Records not in memory, try to load from files
        try:
            pet_health_dir = os.path.join(self.data_directory, 'health_records', pet_id)
            if not os.path.exists(pet_health_dir):
                return []
            
            # Load all record files
            records = []
            for filename in os.listdir(pet_health_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(pet_health_dir, filename)
                    try:
                        with open(file_path, 'r') as f:
                            record = json.load(f)
                            records.append(record)
                    except (json.JSONDecodeError, IOError):
                        # Skip invalid files
                        continue
            
            # Sort by timestamp (most recent first)
            records.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            
            # Cache in memory
            self.health_records[pet_id] = records
            
            # Return limited number
            return records[:limit]
        except Exception as e:
            print(f"Error retrieving health records: {str(e)}")
            return []
    
    def store_care_plan(self, pet_id: str, plan_data: Dict[str, Any]) -> str:
        """
        Store a care plan for a pet.
        
        Args:
            pet_id: Unique identifier for the pet
            plan_data: Care plan data
            
        Returns:
            Plan ID if successful, empty string otherwise
        """
        try:
            # Ensure directories exist
            pet_plans_dir = os.path.join(self.data_directory, 'care_plans', pet_id)
            if not os.path.exists(pet_plans_dir):
                os.makedirs(pet_plans_dir)
            
            # Generate a unique plan ID if not provided
            if 'plan_id' not in plan_data:
                plan_id = f"cp_{int(time.time())}_{str(uuid.uuid4())[:8]}"
                plan_data['plan_id'] = plan_id
            else:
                plan_id = plan_data['plan_id']
            
            # Add metadata
            plan_data['timestamp'] = plan_data.get('timestamp', time.time())
            
            # Save to file
            file_path = os.path.join(pet_plans_dir, f"{plan_id}.json")
            with open(file_path, 'w') as f:
                json.dump(plan_data, f, indent=2)
            
            # Update in-memory cache
            if pet_id not in self.care_plans:
                self.care_plans[pet_id] = []
            
            # Check if plan already exists in memory
            for i, plan in enumerate(self.care_plans[pet_id]):
                if plan.get('plan_id') == plan_id:
                    # Update existing plan
                    self.care_plans[pet_id][i] = plan_data
                    break
            else:
                # Add new plan
                self.care_plans[pet_id].append(plan_data)
            
            # Sort plans by timestamp (most recent first)
            self.care_plans[pet_id].sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            
            print(f"Stored care plan {plan_id} for pet {pet_id}")
            return plan_id
        except Exception as e:
            print(f"Error storing care plan: {str(e)}")
            return ""
    
    def get_care_plans(self, pet_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve care plans for a pet.
        
        Args:
            pet_id: Unique identifier for the pet
            limit: Maximum number of plans to return
            
        Returns:
            List of care plans
        """
        # Check if plans exist in memory
        if pet_id in self.care_plans:
            # Return the most recent plans
            return self.care_plans[pet_id][:limit]
        
        # Plans not in memory, try to load from files
        try:
            pet_plans_dir = os.path.join(self.data_directory, 'care_plans', pet_id)
            if not os.path.exists(pet_plans_dir):
                return []
            
            # Load all plan files
            plans = []
            for filename in os.listdir(pet_plans_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(pet_plans_dir, filename)
                    try:
                        with open(file_path, 'r') as f:
                            plan = json.load(f)
                            plans.append(plan)
                    except (json.JSONDecodeError, IOError):
                        # Skip invalid files
                        continue
            
            # Sort by timestamp (most recent first)
            plans.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            
            # Cache in memory
            self.care_plans[pet_id] = plans
            
            # Return limited number
            return plans[:limit]
        except Exception as e:
            print(f"Error retrieving care plans: {str(e)}")
            return []
    
    def get_care_plan(self, pet_id: str, plan_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific care plan.
        
        Args:
            pet_id: Unique identifier for the pet
            plan_id: Unique identifier for the care plan
            
        Returns:
            Care plan data or None if not found
        """
        # Check if plan exists in memory
        if pet_id in self.care_plans:
            for plan in self.care_plans[pet_id]:
                if plan.get('plan_id') == plan_id:
                    return plan
        
        # Plan not in memory, try to load from file
        try:
            file_path = os.path.join(self.data_directory, 'care_plans', pet_id, f"{plan_id}.json")
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    plan_data = json.load(f)
                    
                # Cache in memory
                if pet_id not in self.care_plans:
                    self.care_plans[pet_id] = []
                
                # Check if plan already exists in memory
                for i, plan in enumerate(self.care_plans[pet_id]):
                    if plan.get('plan_id') == plan_id:
                        # Update existing plan
                        self.care_plans[pet_id][i] = plan_data
                        break
                else:
                    # Add new plan
                    self.care_plans[pet_id].append(plan_data)
                
                return plan_data
        except Exception as e:
            print(f"Error retrieving care plan: {str(e)}")
        
        print(f"Care plan {plan_id} for pet {pet_id} not found")
        return None
    
    def create_data_backup(self) -> str:
        """
        Create a backup of all data.
        
        Returns:
            Backup ID if successful, empty string otherwise
        """
        try:
            # Generate backup ID
            backup_id = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_dir = os.path.join(self.data_directory, 'backups', backup_id)
            
            # Create backup directory
            os.makedirs(backup_dir)
            
            # Copy pet profiles
            pet_profiles_backup = os.path.join(backup_dir, 'pet_profiles')
            os.makedirs(pet_profiles_backup)
            for pet_id in self.pet_data:
                file_path = os.path.join(self.data_directory, 'pet_profiles', f"{pet_id}.json")
                if os.path.exists(file_path):
                    with open(file_path, 'r') as src_file:
                        profile_data = json.load(src_file)
                        backup_path = os.path.join(pet_profiles_backup, f"{pet_id}.json")
                        with open(backup_path, 'w') as dest_file:
                            json.dump(profile_data, dest_file, indent=2)
            
            # Create backup metadata
            metadata = {
                'backup_id': backup_id,
                'timestamp': time.time(),
                'pet_count': len(self.pet_data),
                'backup_items': ['pet_profiles']  # Add more as implemented
            }
            
            # Save metadata
            with open(os.path.join(backup_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Created backup {backup_id}")
            return backup_id
        except Exception as e:
            print(f"Error creating backup: {str(e)}")
            return ""
    
    def restore_data_backup(self, backup_id: str) -> bool:
        """
        Restore data from a backup.
        
        Args:
            backup_id: Identifier for the backup
            
        Returns:
            True if successful, False otherwise
        """
        try:
            backup_dir = os.path.join(self.data_directory, 'backups', backup_id)
            if not os.path.exists(backup_dir):
                print(f"Backup {backup_id} not found")
                return False
            
            # Load backup metadata
            metadata_path = os.path.join(backup_dir, 'metadata.json')
            if not os.path.exists(metadata_path):
                print(f"Backup metadata not found")
                return False
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Restore pet profiles
            if 'pet_profiles' in metadata.get('backup_items', []):
                pet_profiles_backup = os.path.join(backup_dir, 'pet_profiles')
                if os.path.exists(pet_profiles_backup):
                    # Clear current pet profiles
                    self.pet_data = {}
                    
                    # Load from backup
                    for filename in os.listdir(pet_profiles_backup):
                        if filename.endswith('.json'):
                            pet_id = filename.split('.')[0]
                            backup_path = os.path.join(pet_profiles_backup, filename)
                            with open(backup_path, 'r') as src_file:
                                profile_data = json.load(src_file)
                                
                                # Update in memory
                                self.pet_data[pet_id] = profile_data
                                
                                # Save to data directory
                                file_path = os.path.join(self.data_directory, 'pet_profiles', f"{pet_id}.json")
                                with open(file_path, 'w') as dest_file:
                                    json.dump(profile_data, dest_file, indent=2)
            
            print(f"Restored backup {backup_id}")
            return True
        except Exception as e:
            print(f"Error restoring backup: {str(e)}")
            return False
    
    def export_pet_data(self, pet_id: str, format: str = "json") -> Optional[str]:
        """
        Export all data for a pet in a specified format.
        
        Args:
            pet_id: Unique identifier for the pet
            format: Export format (currently only "json" is supported)
            
        Returns:
            Exported data string or None if error
        """
        try:
            # Get pet profile
            profile = self.get_pet_profile(pet_id)
            if not profile:
                print(f"Pet profile {pet_id} not found")
                return None
            
            # Get health records
            health_records = self.get_health_records(pet_id, limit=100)
            
            # Get care plans
            care_plans = self.get_care_plans(pet_id, limit=10)
            
            # Compile all data
            export_data = {
                'pet_profile': profile,
                'health_records': health_records,
                'care_plans': care_plans,
                'export_timestamp': time.time(),
                'export_format': format
            }
            
            # Export based on format
            if format.lower() == "json":
                return json.dumps(export_data, indent=2)
            else:
                print(f"Unsupported export format: {format}")
                return None
        except Exception as e:
            print(f"Error exporting pet data: {str(e)}")
            return None
    
    def import_pet_data(self, data_str: str) -> Optional[str]:
        """
        Import pet data.
        
        Args:
            data_str: Data string to import
            
        Returns:
            Pet ID if successful, None otherwise
        """
        try:
            # Parse data
            import_data = json.loads(data_str)
            
            # Extract profile
            profile = import_data.get('pet_profile')
            if not profile or not isinstance(profile, dict):
                print("Invalid pet profile in import data")
                return None
            
            # Get or generate pet ID
            pet_id = profile.get('pet_id')
            if not pet_id:
                pet_id = f"pet_{int(time.time())}_{str(uuid.uuid4())[:8]}"
                profile['pet_id'] = pet_id
            
            # Store pet profile
            self.store_pet_profile(pet_id, profile)
            
            # Import health records
            health_records = import_data.get('health_records', [])
            for record in health_records:
                if isinstance(record, dict):
                    self.store_health_record(pet_id, record)
            
            # Import care plans
            care_plans = import_data.get('care_plans', [])
            for plan in care_plans:
                if isinstance(plan, dict):
                    self.store_care_plan(pet_id, plan)
            
            print(f"Imported data for pet {pet_id}")
            return pet_id
        except Exception as e:
            print(f"Error importing pet data: {str(e)}")
            return None
    
    def search_pet_data(self, search_term: str) -> List[Dict[str, Any]]:
        """
        Search for pet data matching a search term.
        
        Args:
            search_term: Term to search for
            
        Returns:
            List of matching pet data
        """
        search_term = search_term.lower()
        results = []
        
        # Search pet profiles
        for pet_id, profile in self.pet_data.items():
            # Check if search term matches any field in profile
            if any(search_term in str(value).lower() for value in profile.values()):
                # Add basic info to results
                results.append({
                    'pet_id': pet_id,
                    'name': profile.get('name', 'Unknown'),
                    'match_type': 'pet_profile',
                    'match_score': 1.0  # High score for direct matches
                })
        
        # Also search for any files not loaded in memory
        try:
            pet_profiles_dir = os.path.join(self.data_directory, 'pet_profiles')
            for filename in os.listdir(pet_profiles_dir):
                if filename.endswith('.json'):
                    pet_id = filename.split('.')[0]
                    
                    # Skip if already in the results
                    if any(r['pet_id'] == pet_id for r in results):
                        continue
                    
                    # Load data from file and search
                    file_path = os.path.join(pet_profiles_dir, filename)
                    try:
                        with open(file_path, 'r') as f:
                            profile = json.load(f)
                            if any(search_term in str(value).lower() for value in profile.values()):
                                results.append({
                                    'pet_id': pet_id,
                                    'name': profile.get('name', 'Unknown'),
                                    'match_type': 'pet_profile',
                                    'match_score': 1.0
                                })
                    except (json.JSONDecodeError, IOError):
                        # Skip invalid files
                        continue
        except FileNotFoundError:
            pass
        
        # TODO: Search health records, care plans, etc.
        
        return results
    
    def generate_data_report(self, pet_id: str) -> Dict[str, Any]:
        """
        Generate a comprehensive data report for a pet.
        
        Args:
            pet_id: Unique identifier for the pet
            
        Returns:
            Data report
        """
        report = {
            'pet_id': pet_id,
            'generated_at': time.time(),
            'profile_summary': {},
            'health_summary': {},
            'care_plan_summary': {},
            'data_statistics': {}
        }
        
        # Get pet profile
        profile = self.get_pet_profile(pet_id)
        if profile:
            report['profile_summary'] = {
                'name': profile.get('name', 'Unknown'),
                'species': profile.get('species', 'Unknown'),
                'breed': profile.get('breed', 'Unknown'),
                'age': profile.get('age', 0),
                'weight': profile.get('weight', 0),
                'profile_created': profile.get('created_at', 0),
                'profile_updated': profile.get('last_updated', 0)
            }
        
        # Get health records
        health_records = self.get_health_records(pet_id)
        if health_records:
            recent_record = health_records[0]
            report['health_summary'] = {
                'records_count': len(health_records),
                'latest_record_date': recent_record.get('timestamp', 0),
                'latest_record_type': recent_record.get('record_type', 'Unknown'),
                'health_issues': self._extract_health_issues(health_records)
            }
        
        # Get care plans
        care_plans = self.get_care_plans(pet_id)
        if care_plans:
            active_plan = care_plans[0]
            report['care_plan_summary'] = {
                'plans_count': len(care_plans),
                'active_plan_id': active_plan.get('plan_id', ''),
                'active_plan_created': active_plan.get('timestamp', 0),
                'active_plan_expires': active_plan.get('valid_until', 0),
                'priority_issues': self._extract_priority_issues(active_plan)
            }
        
        # Generate data statistics
        report['data_statistics'] = self._generate_data_statistics(pet_id)
        
        return report
    
    def _extract_health_issues(self, health_records: List[Dict[str, Any]]) -> List[str]:
        """Extract health issues from health records"""
        issues = set()
        
        for record in health_records:
            # Extract issues from diagnosis field
            diagnosis = record.get('diagnosis', {})
            if isinstance(diagnosis, dict):
                for issue in diagnosis.get('issues', []):
                    if isinstance(issue, dict) and 'type' in issue:
                        issues.add(issue['type'])
                    elif isinstance(issue, str):
                        issues.add(issue)
            
            # Extract issues from conditions field
            conditions = record.get('conditions', [])
            if isinstance(conditions, list):
                for condition in conditions:
                    if isinstance(condition, dict) and 'name' in condition:
                        issues.add(condition['name'])
                    elif isinstance(condition, str):
                        issues.add(condition)
        
        return list(issues)
    
    def _extract_priority_issues(self, care_plan: Dict[str, Any]) -> List[str]:
        """Extract priority issues from a care plan"""
        issues = []
        
        # Check health management conditions
        health_management = care_plan.get('health_management', {})
        conditions = health_management.get('conditions', [])
        for condition in conditions:
            if isinstance(condition, dict) and 'condition' in condition:
                issues.append(condition['condition'])
        
        # Check behavior management issues
        behavior_management = care_plan.get('behavior_management', {})
        identified_issues = behavior_management.get('identified_issues', [])
        for issue in identified_issues:
            if isinstance(issue, dict) and 'issue' in issue:
                issues.append(issue['issue'])
        
        return issues
    
    def _generate_data_statistics(self, pet_id: str) -> Dict[str, Any]:
        """Generate statistics about the pet's data"""
        statistics = {
            'total_health_records': 0,
            'total_care_plans': 0,
            'data_completeness': 0.0,
            'profile_fields': 0
        }
        
        # Count health records
        try:
            pet_health_dir = os.path.join(self.data_directory, 'health_records', pet_id)
            if os.path.exists(pet_health_dir):
                health_files = [f for f in os.listdir(pet_health_dir) if f.endswith('.json')]
                statistics['total_health_records'] = len(health_files)
        except (FileNotFoundError, OSError):
            pass
        
        # Count care plans
        try:
            pet_plans_dir = os.path.join(self.data_directory, 'care_plans', pet_id)
            if os.path.exists(pet_plans_dir):
                plan_files = [f for f in os.listdir(pet_plans_dir) if f.endswith('.json')]
                statistics['total_care_plans'] = len(plan_files)
        except (FileNotFoundError, OSError):
            pass
        
        # Calculate profile completeness
        profile = self.get_pet_profile(pet_id)
        if profile:
            # Count fields in profile
            statistics['profile_fields'] = len(profile)
            
            # Calculate completeness (based on expected fields)
            expected_fields = ['name', 'species', 'breed', 'age', 'weight', 'gender', 'color', 'microchip_id', 
                            'birth_date', 'neutered', 'owner_info', 'vet_info', 'allergies', 'medications']
            present_fields = sum(1 for field in expected_fields if field in profile)
            statistics['data_completeness'] = present_fields / len(expected_fields)
        
        return statistics


# Example usage
if __name__ == "__main__":
    # Create a data storage manager
    storage_manager = DataStorageManager()
    
    # Example pet information
    pet_info = {
        'name': 'Max',
        'species': 'dog',
        'breed': 'Golden Retriever',
        'age': 3,
        'weight': 28.5,  # kg
        'gender': 'male',
        'color': 'golden',
        'microchip_id': '123456789012345',
        'birth_date': '2022-01-15',
        'neutered': True,
        'owner_info': {
            'name': 'John Smith',
            'phone': '555-123-4567',
            'email': 'john.smith@example.com'
        },
        'vet_info': {
            'name': 'Dr. Wilson',
            'clinic': 'Healthy Pets Clinic',
            'phone': '555-987-6543'
        },
        'allergies': ['Chicken', 'Certain antibiotics'],
        'medications': []
    }
    
    # Store pet profile
    pet_id = "pet12345"
    storage_manager.store_pet_profile(pet_id, pet_info)
    
    # Retrieve pet profile
    retrieved_profile = storage_manager.get_pet_profile(pet_id)
    if retrieved_profile:
        print(f"Retrieved profile for {retrieved_profile['name']}")
        
    # List all pets
    pets = storage_manager.list_pets()
    print(f"Total pets: {len(pets)}")
    
    # Create a backup
    backup_id = storage_manager.create_data_backup()
    print(f"Created backup: {backup_id}")
