"""
Care Plan Manager Module

This module manages the creation, storage, and updating of personalized care plans.
It serves as a coordinator between the AI diagnosis module, care plan generator,
and the data management system.

Technical Note: This implementation is based on patented technology filed by Ucaretron Inc.
"""

import time
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta

from personalized_care_module.care_plan_generator import CarePlanGenerator

class CarePlanManager:
    """
    Manages personalized care plans for pets.
    
    This class coordinates the creation, storage, retrieval, and updating of 
    care plans based on AI diagnosis results and health data.
    """
    
    def __init__(self, care_plan_generator=None):
        """
        Initialize the care plan manager.
        
        Args:
            care_plan_generator: Optional custom care plan generator instance.
                If None, a new CarePlanGenerator will be created.
        """
        self.care_plan_generator = care_plan_generator or CarePlanGenerator()
        
        # Store active care plans in memory
        # In a real implementation, this would be backed by a database
        self.active_care_plans = {}  # Format: {pet_id: [care_plan1, care_plan2, ...]}
        
        # Store care plan history
        self.care_plan_history = {}  # Format: {pet_id: [archived_plan1, archived_plan2, ...]}
        
        print("Care Plan Manager initialized")
    
    def create_care_plan(self, pet_id: str, diagnosis_results: Dict[str, Any], 
                         pet_info: Dict[str, Any], health_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new personalized care plan based on diagnosis results.
        
        Args:
            pet_id: Pet's unique identifier
            diagnosis_results: Results from the AI diagnosis module
            pet_info: Information about the pet (species, breed, age, etc.)
            health_history: Optional historical health data
            
        Returns:
            The generated care plan
        """
        # Generate care plan using the care plan generator
        care_plan = self.care_plan_generator.generate_care_plan(
            pet_id, diagnosis_results, pet_info, health_history)
        
        # Add plan to active care plans
        if pet_id not in self.active_care_plans:
            self.active_care_plans[pet_id] = []
        
        # Add the new care plan to the beginning of the list (most recent first)
        self.active_care_plans[pet_id].insert(0, care_plan)
        
        # Trim the list if there are more than 3 active plans
        if len(self.active_care_plans[pet_id]) > 3:
            # Move the oldest plan to history
            oldest_plan = self.active_care_plans[pet_id].pop()
            
            if pet_id not in self.care_plan_history:
                self.care_plan_history[pet_id] = []
            
            self.care_plan_history[pet_id].append(oldest_plan)
        
        print(f"Created new care plan for pet {pet_id} with ID {care_plan['plan_id']}")
        return care_plan
    
    def update_care_plan(self, pet_id: str, plan_id: str, 
                         updates: Dict[str, Any], update_reason: str) -> Optional[Dict[str, Any]]:
        """
        Update an existing care plan with new information.
        
        Args:
            pet_id: Pet's unique identifier
            plan_id: Unique identifier of the care plan to update
            updates: Dictionary of updates to apply to the care plan
            update_reason: Reason for the update
            
        Returns:
            The updated care plan or None if the plan was not found
        """
        # Find the care plan to update
        if pet_id not in self.active_care_plans:
            print(f"No active care plans found for pet {pet_id}")
            return None
        
        care_plan = None
        plan_index = None
        
        for i, plan in enumerate(self.active_care_plans[pet_id]):
            if plan['plan_id'] == plan_id:
                care_plan = plan
                plan_index = i
                break
        
        if care_plan is None:
            print(f"Care plan with ID {plan_id} not found for pet {pet_id}")
            return None
        
        # Create a copy of the existing care plan
        updated_plan = care_plan.copy()
        
        # Apply updates to each section
        for section_key, section_updates in updates.items():
            if section_key in updated_plan:
                # If the section is a dictionary, we need to merge updates
                if isinstance(updated_plan[section_key], dict) and isinstance(section_updates, dict):
                    self._recursive_update(updated_plan[section_key], section_updates)
                # Otherwise, replace the section entirely
                else:
                    updated_plan[section_key] = section_updates
        
        # Update metadata
        updated_plan['last_updated'] = time.time()
        updated_plan['plan_version'] = updated_plan.get('plan_version', 1) + 1
        updated_plan['update_history'] = updated_plan.get('update_history', [])
        updated_plan['update_history'].append({
            'timestamp': time.time(),
            'reason': update_reason,
            'updated_sections': list(updates.keys())
        })
        
        # Replace the old plan with the updated one
        self.active_care_plans[pet_id][plan_index] = updated_plan
        
        print(f"Updated care plan {plan_id} for pet {pet_id} (version {updated_plan['plan_version']})")
        return updated_plan
    
    def _recursive_update(self, target_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """
        Recursively update a nested dictionary.
        
        Args:
            target_dict: The dictionary to update
            update_dict: The dictionary containing updates
        """
        for key, value in update_dict.items():
            if key in target_dict and isinstance(target_dict[key], dict) and isinstance(value, dict):
                self._recursive_update(target_dict[key], value)
            elif key in target_dict and isinstance(target_dict[key], list) and isinstance(value, list):
                # For lists, append new items rather than replacing the entire list
                target_dict[key].extend([item for item in value if item not in target_dict[key]])
            else:
                target_dict[key] = value
    
    def get_active_care_plan(self, pet_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the most recent active care plan for a pet.
        
        Args:
            pet_id: Pet's unique identifier
            
        Returns:
            The most recent care plan or None if no active plans exist
        """
        if pet_id in self.active_care_plans and self.active_care_plans[pet_id]:
            return self.active_care_plans[pet_id][0]  # Return the most recent plan
        
        print(f"No active care plans found for pet {pet_id}")
        return None
    
    def get_all_active_care_plans(self, pet_id: str) -> List[Dict[str, Any]]:
        """
        Get all active care plans for a pet.
        
        Args:
            pet_id: Pet's unique identifier
            
        Returns:
            List of active care plans or empty list if none exist
        """
        return self.active_care_plans.get(pet_id, [])
    
    def get_care_plan_history(self, pet_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get historical care plans for a pet.
        
        Args:
            pet_id: Pet's unique identifier
            limit: Maximum number of historical plans to return
            
        Returns:
            List of historical care plans or empty list if none exist
        """
        history = self.care_plan_history.get(pet_id, [])
        return history[:limit]
    
    def archive_care_plan(self, pet_id: str, plan_id: str) -> bool:
        """
        Archive an active care plan.
        
        Args:
            pet_id: Pet's unique identifier
            plan_id: Unique identifier of the care plan to archive
            
        Returns:
            True if successful, False otherwise
        """
        if pet_id not in self.active_care_plans:
            print(f"No active care plans found for pet {pet_id}")
            return False
        
        for i, plan in enumerate(self.active_care_plans[pet_id]):
            if plan['plan_id'] == plan_id:
                # Remove the plan from active plans
                archived_plan = self.active_care_plans[pet_id].pop(i)
                
                # Add to history
                if pet_id not in self.care_plan_history:
                    self.care_plan_history[pet_id] = []
                
                self.care_plan_history[pet_id].insert(0, archived_plan)  # Add to the beginning of history
                
                print(f"Archived care plan {plan_id} for pet {pet_id}")
                return True
        
        print(f"Care plan with ID {plan_id} not found for pet {pet_id}")
        return False
    
    def export_care_plan(self, pet_id: str, plan_id: str, format: str = "json") -> Optional[str]:
        """
        Export a care plan to a specific format.
        
        Args:
            pet_id: Pet's unique identifier
            plan_id: Unique identifier of the care plan to export
            format: Export format (currently only "json" is supported)
            
        Returns:
            Exported care plan string or None if the plan was not found
        """
        # Find the care plan
        care_plan = None
        
        # Check active plans
        if pet_id in self.active_care_plans:
            for plan in self.active_care_plans[pet_id]:
                if plan['plan_id'] == plan_id:
                    care_plan = plan
                    break
        
        # Check historical plans if not found in active plans
        if care_plan is None and pet_id in self.care_plan_history:
            for plan in self.care_plan_history[pet_id]:
                if plan['plan_id'] == plan_id:
                    care_plan = plan
                    break
        
        if care_plan is None:
            print(f"Care plan with ID {plan_id} not found for pet {pet_id}")
            return None
        
        # Export based on format
        if format.lower() == "json":
            return json.dumps(care_plan, indent=2)
        else:
            print(f"Unsupported export format: {format}")
            return None
    
    def generate_reminder(self, pet_id: str, days_ahead: int = 7) -> List[Dict[str, Any]]:
        """
        Generate reminders for upcoming care activities.
        
        Args:
            pet_id: Pet's unique identifier
            days_ahead: Number of days to look ahead for reminders
            
        Returns:
            List of reminder objects
        """
        active_plan = self.get_active_care_plan(pet_id)
        if not active_plan:
            print(f"No active care plan found for pet {pet_id}")
            return []
        
        reminders = []
        current_time = time.time()
        future_time = current_time + (days_ahead * 24 * 60 * 60)  # Convert days to seconds
        
        # Check veterinary follow-ups
        for visit in active_plan.get('follow_up', {}).get('veterinary_visits', []):
            # Parse timeframe to a timestamp
            visit_timeframe = visit.get('timeframe', '')
            visit_timestamp = self._parse_timeframe_to_timestamp(visit_timeframe, current_time)
            
            if visit_timestamp and current_time <= visit_timestamp <= future_time:
                reminders.append({
                    'type': 'veterinary_visit',
                    'timestamp': visit_timestamp,
                    'details': {
                        'purpose': visit.get('purpose', 'Check-up'),
                        'preparations': visit.get('preparations', [])
                    },
                    'priority': 'high'
                })
        
        # Check medication refills
        # Assuming medications have a refill_date or days_supply field
        for medication in active_plan.get('health_management', {}).get('medication_plan', {}).get('list', []):
            if 'refill_date' in medication:
                refill_timestamp = medication['refill_date']
                if current_time <= refill_timestamp <= future_time:
                    reminders.append({
                        'type': 'medication_refill',
                        'timestamp': refill_timestamp,
                        'details': {
                            'medication': medication.get('name', 'Unknown medication'),
                            'dosage': medication.get('dosage', '')
                        },
                        'priority': 'medium'
                    })
        
        # Check for plan expiration
        if 'valid_until' in active_plan:
            expiration_timestamp = active_plan['valid_until']
            if current_time <= expiration_timestamp <= future_time:
                reminders.append({
                    'type': 'care_plan_update',
                    'timestamp': expiration_timestamp,
                    'details': {
                        'plan_id': active_plan['plan_id'],
                        'reason': 'Plan expiration'
                    },
                    'priority': 'medium'
                })
        
        return reminders
    
    def _parse_timeframe_to_timestamp(self, timeframe: str, reference_time: float) -> Optional[float]:
        """
        Convert a timeframe string to a timestamp.
        
        Args:
            timeframe: String description of a timeframe (e.g., "2 weeks", "3 months")
            reference_time: Reference timestamp to calculate from
            
        Returns:
            Timestamp or None if parsing failed
        """
        try:
            # Common timeframe formats
            if 'week' in timeframe:
                weeks = int(''.join(filter(str.isdigit, timeframe)))
                return reference_time + (weeks * 7 * 24 * 60 * 60)
            elif 'month' in timeframe:
                months = int(''.join(filter(str.isdigit, timeframe)))
                # Approximate months as 30 days
                return reference_time + (months * 30 * 24 * 60 * 60)
            elif 'day' in timeframe:
                days = int(''.join(filter(str.isdigit, timeframe)))
                return reference_time + (days * 24 * 60 * 60)
            elif 'year' in timeframe:
                years = int(''.join(filter(str.isdigit, timeframe)))
                # Approximate years as 365 days
                return reference_time + (years * 365 * 24 * 60 * 60)
        except (ValueError, TypeError):
            pass
        
        return None
    
    def add_progress_update(self, pet_id: str, plan_id: str, 
                           progress_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Add a progress update to a care plan.
        
        Args:
            pet_id: Pet's unique identifier
            plan_id: Unique identifier of the care plan
            progress_data: Progress update data
            
        Returns:
            The updated care plan or None if the plan was not found
        """
        # Find the care plan
        care_plan = None
        
        if pet_id in self.active_care_plans:
            for i, plan in enumerate(self.active_care_plans[pet_id]):
                if plan['plan_id'] == plan_id:
                    care_plan = plan
                    plan_index = i
                    break
        
        if care_plan is None:
            print(f"Care plan with ID {plan_id} not found for pet {pet_id}")
            return None
        
        # Create a copy of the care plan
        updated_plan = care_plan.copy()
        
        # Add progress update
        if 'progress_updates' not in updated_plan:
            updated_plan['progress_updates'] = []
        
        progress_data['timestamp'] = time.time()
        updated_plan['progress_updates'].append(progress_data)
        
        # Update the care plan
        self.active_care_plans[pet_id][plan_index] = updated_plan
        
        print(f"Added progress update to care plan {plan_id} for pet {pet_id}")
        return updated_plan


# Example usage
if __name__ == "__main__":
    # Create a care plan manager
    manager = CarePlanManager()
    
    # Example pet information
    pet_info = {
        'name': 'Max',
        'species': 'dog',
        'breed': 'Golden Retriever',
        'age': 5,
        'weight': 28.5  # kg
    }
    
    # Example diagnosis results
    diagnosis_results = {
        'health_issues': [
            {
                'type': 'skin allergy',
                'severity': 'moderate',
                'confidence': 0.85
            }
        ],
        'behavior_issues': [
            {
                'type': 'separation anxiety',
                'severity': 'mild',
                'confidence': 0.75
            }
        ],
        'risk_factors': [
            {
                'type': 'obesity risk',
                'level': 'moderate',
                'confidence': 0.7
            }
        ],
        'overall_health_status': 'good'
    }
    
    # Create a care plan
    care_plan = manager.create_care_plan('pet123', diagnosis_results, pet_info)
    
    # Print the plan ID
    print(f"Created care plan with ID: {care_plan['plan_id']}")
    
    # Get the active care plan
    active_plan = manager.get_active_care_plan('pet123')
    print(f"Active care plan: {active_plan['plan_id']}")
    
    # Update the care plan
    updates = {
        'health_management': {
            'diet_plan': {
                'special_requirements': ['grain-free', 'limited ingredient']
            }
        }
    }
    
    updated_plan = manager.update_care_plan('pet123', care_plan['plan_id'], updates, 'Vet recommendation')
    
    # Generate reminders
    reminders = manager.generate_reminder('pet123', days_ahead=30)
    print(f"Generated {len(reminders)} reminders")
