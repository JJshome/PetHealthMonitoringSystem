"""
Personalized Care module for the Pet Health Monitoring System.

This module provides customized health management and behavior improvement solutions
based on pet characteristics, health conditions, and environmental factors.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta

from .config import config, PET_TYPES
from .utils import save_json, load_json, logger


class CareRecommendationEngine:
    """
    Generates personalized care recommendations based on health analysis results.
    """
    
    def __init__(self, pet_type: str = "dog", pet_name: str = None, 
                 pet_age: float = None, pet_weight: float = None):
        """
        Initialize the care recommendation engine.
        
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
        
        # Load recommendation database
        self.care_recommendations = self._load_care_recommendations()
        self.nutrition_recommendations = self._load_nutrition_recommendations()
        self.exercise_recommendations = self._load_exercise_recommendations()
        self.behavior_recommendations = self._load_behavior_recommendations()
    
    def _load_care_recommendations(self) -> Dict[str, Any]:
        """Load general care recommendations."""
        # This would ideally load from a database or external file
        # Here defined inline for demonstration
        
        common_recs = {
            "regular_checkup": {
                "title": "Regular Veterinary Check-ups",
                "description": "Regular veterinary check-ups are essential for maintaining your pet's health.",
                "frequency": {
                    "puppy_kitten": "every 3-4 weeks until 16 weeks old",
                    "adult": "annually",
                    "senior": "every 6 months"
                },
                "importance": "high"
            },
            "dental_care": {
                "title": "Dental Care",
                "description": "Regular dental care prevents periodontal disease and improves overall health.",
                "frequency": "daily",
                "methods": ["brushing", "dental treats", "professional cleaning"],
                "importance": "high"
            },
            "parasite_prevention": {
                "title": "Parasite Prevention",
                "description": "Regular parasite prevention protects against fleas, ticks, heartworm, and intestinal parasites.",
                "frequency": "monthly",
                "importance": "high"
            },
            "grooming": {
                "title": "Regular Grooming",
                "description": "Regular grooming keeps your pet clean and allows early detection of skin issues.",
                "frequency": {
                    "short_hair": "weekly",
                    "medium_hair": "2-3 times per week",
                    "long_hair": "daily"
                },
                "importance": "medium"
            }
        }
        
        # Pet type specific recommendations
        if self.pet_type == "dog":
            common_recs.update({
                "socialization": {
                    "title": "Regular Socialization",
                    "description": "Regular interaction with other dogs and people is important for behavioral health.",
                    "frequency": "weekly",
                    "importance": "medium"
                },
                "training": {
                    "title": "Consistent Training",
                    "description": "Regular training sessions maintain good behavior and mental stimulation.",
                    "frequency": "daily",
                    "importance": "medium"
                }
            })
        elif self.pet_type == "cat":
            common_recs.update({
                "environmental_enrichment": {
                    "title": "Environmental Enrichment",
                    "description": "Providing environmental enrichment prevents boredom and associated behavioral issues.",
                    "elements": ["climbing spaces", "scratching posts", "interactive toys", "hiding spots"],
                    "frequency": "always available",
                    "importance": "high"
                },
                "litter_box": {
                    "title": "Litter Box Maintenance",
                    "description": "Clean litter boxes prevent behavioral and urinary issues.",
                    "frequency": "daily",
                    "importance": "high"
                }
            })
        
        return common_recs
    
    def _load_nutrition_recommendations(self) -> Dict[str, Any]:
        """Load nutrition recommendations."""
        # This would ideally load from a database or external file
        # Here defined inline for demonstration
        
        common_nutrition = {
            "water": {
                "title": "Fresh Water",
                "description": "Always provide clean, fresh water.",
                "amount": "ad libitum",
                "importance": "critical"
            },
            "feeding_schedule": {
                "title": "Regular Feeding Schedule",
                "description": "A consistent feeding schedule helps with digestion and behavior.",
                "frequency": {
                    "puppy_kitten": "3-4 times per day",
                    "adult": "2 times per day",
                    "senior": "2-3 times per day"
                },
                "importance": "high"
            }
        }
        
        # Pet type specific nutrition
        if self.pet_type == "dog":
            common_nutrition.update({
                "diet_type": {
                    "title": "Appropriate Diet Type",
                    "description": "Select a diet appropriate for your dog's age, size, and activity level.",
                    "options": ["puppy formula", "adult maintenance", "senior formula", "weight management", "performance"],
                    "importance": "high"
                },
                "portion_control": {
                    "title": "Proper Portion Control",
                    "description": "Follow feeding guidelines adjusted for your dog's specific needs to prevent obesity.",
                    "calculation": "Based on weight, age, and activity level",
                    "importance": "high"
                },
                "treats": {
                    "title": "Healthy Treats",
                    "description": "Use healthy treats in moderation for training and bonding.",
                    "limit": "Treats should not exceed 10% of daily caloric intake",
                    "importance": "medium"
                }
            })
        elif self.pet_type == "cat":
            common_nutrition.update({
                "diet_type": {
                    "title": "Appropriate Diet Type",
                    "description": "Select a diet appropriate for your cat's age and health status.",
                    "options": ["kitten formula", "adult maintenance", "senior formula", "weight management", "hairball control"],
                    "importance": "high"
                },
                "wet_food": {
                    "title": "Include Wet Food",
                    "description": "Include wet food in the diet to increase water intake and support urinary health.",
                    "frequency": "daily if possible",
                    "importance": "medium"
                },
                "feeding_method": {
                    "title": "Appropriate Feeding Method",
                    "description": "Choose an appropriate feeding method based on your cat's eating habits.",
                    "options": ["meal feeding", "free feeding", "puzzle feeders", "automated feeders"],
                    "importance": "medium"
                }
            })
        
        return common_nutrition
    
    def _load_exercise_recommendations(self) -> Dict[str, Any]:
        """Load exercise recommendations."""
        # This would ideally load from a database or external file
        # Here defined inline for demonstration
        
        # Pet type specific exercise
        if self.pet_type == "dog":
            return {
                "daily_exercise": {
                    "title": "Daily Exercise",
                    "description": "Regular exercise is essential for physical and mental health.",
                    "duration": {
                        "small_breed": "30-60 minutes",
                        "medium_breed": "60-90 minutes",
                        "large_breed": "1-2 hours",
                        "working_breed": "2+ hours"
                    },
                    "frequency": "daily",
                    "importance": "high"
                },
                "mental_stimulation": {
                    "title": "Mental Stimulation",
                    "description": "Mental exercise is as important as physical exercise.",
                    "methods": ["puzzle toys", "training sessions", "nose work", "new experiences"],
                    "frequency": "daily",
                    "importance": "high"
                },
                "activity_variety": {
                    "title": "Variety of Activities",
                    "description": "A variety of activities keeps exercise interesting and works different muscle groups.",
                    "examples": ["walking", "running", "swimming", "fetch", "agility", "tug-of-war"],
                    "frequency": "mix throughout the week",
                    "importance": "medium"
                }
            }
        elif self.pet_type == "cat":
            return {
                "play_sessions": {
                    "title": "Interactive Play Sessions",
                    "description": "Regular play sessions provide exercise and strengthen the human-cat bond.",
                    "duration": "10-15 minutes",
                    "frequency": "2-3 times daily",
                    "importance": "high"
                },
                "hunting_stimulation": {
                    "title": "Hunting Behavior Stimulation",
                    "description": "Activities that simulate hunting satisfy natural instincts and provide exercise.",
                    "methods": ["wand toys", "laser pointers (followed by a physical reward)", "treat puzzles", "electronic moving toys"],
                    "frequency": "daily",
                    "importance": "high"
                },
                "climbing_opportunities": {
                    "title": "Vertical Space",
                    "description": "Vertical spaces allow cats to exercise and satisfy their natural climbing instincts.",
                    "examples": ["cat trees", "shelves", "window perches"],
                    "availability": "always accessible",
                    "importance": "medium"
                }
            }
        else:
            return {}
    
    def _load_behavior_recommendations(self) -> Dict[str, Any]:
        """Load behavior recommendations."""
        # This would ideally load from a database or external file
        # Here defined inline for demonstration
        
        common_behavior = {
            "positive_reinforcement": {
                "title": "Positive Reinforcement",
                "description": "Reward desired behaviors to encourage their repetition.",
                "methods": ["treats", "praise", "play", "attention"],
                "importance": "high"
            },
            "consistency": {
                "title": "Consistency",
                "description": "Consistent rules and responses help pets understand expectations.",
                "application": "All family members should follow the same rules and commands",
                "importance": "high"
            }
        }
        
        # Pet type specific behavior
        if self.pet_type == "dog":
            common_behavior.update({
                "basic_training": {
                    "title": "Basic Obedience Training",
                    "description": "Basic commands provide structure and safety.",
                    "commands": ["sit", "stay", "come", "leave it", "drop it"],
                    "frequency": "practice daily",
                    "importance": "high"
                },
                "socialization": {
                    "title": "Ongoing Socialization",
                    "description": "Regular exposure to different people, animals, and environments prevents fear and aggression.",
                    "methods": ["dog parks", "playdates", "urban walks", "new environments"],
                    "frequency": "weekly",
                    "importance": "high"
                },
                "structure": {
                    "title": "Routine and Structure",
                    "description": "A predictable routine provides security and reduces anxiety.",
                    "elements": ["consistent meal times", "regular exercise schedule", "predictable sleep times"],
                    "importance": "medium"
                }
            })
        elif self.pet_type == "cat":
            common_behavior.update({
                "scratching_outlets": {
                    "title": "Appropriate Scratching Outlets",
                    "description": "Providing appropriate scratching surfaces protects furniture and satisfies natural scratching instincts.",
                    "types": ["vertical posts", "horizontal pads", "different materials"],
                    "placement": "In social areas where the cat spends time",
                    "importance": "high"
                },
                "litter_box_management": {
                    "title": "Proper Litter Box Management",
                    "description": "Clean, accessible litter boxes prevent elimination issues.",
                    "guidelines": ["one more box than number of cats", "clean daily", "placed in quiet, accessible locations"],
                    "importance": "critical"
                },
                "territory": {
                    "title": "Respect for Territory",
                    "description": "Cats are territorial animals that need their own space.",
                    "elements": ["safe spaces", "multiple resources", "escape routes"],
                    "importance": "high"
                }
            })
        
        return common_behavior
    
    def generate_care_plan(self, health_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a personalized care plan based on health analysis.
        
        Args:
            health_analysis: Dictionary containing health analysis results.
            
        Returns:
            Dictionary with personalized care plan.
        """
        # Initialize care plan
        care_plan = {
            "timestamp": datetime.now().isoformat(),
            "pet_info": {
                "name": self.pet_name,
                "type": self.pet_type,
                "age": self.pet_age,
                "weight": self.pet_weight
            },
            "health_summary": self._extract_health_summary(health_analysis),
            "immediate_actions": [],
            "daily_care": {
                "nutrition": [],
                "exercise": [],
                "medical": [],
                "behavioral": []
            },
            "weekly_care": [],
            "monthly_care": [],
            "resources": []
        }
        
        # Process immediate recommendations from health analysis
        if "overall_assessment" in health_analysis and "recommendations" in health_analysis["overall_assessment"]:
            for rec in health_analysis["overall_assessment"]["recommendations"]:
                if rec.get("priority") == "high":
                    care_plan["immediate_actions"].append({
                        "title": f"Address {rec.get('type', 'health issue').replace('_', ' ')}",
                        "description": rec.get("description", ""),
                        "actions": rec.get("specific_actions", []),
                        "urgency": "high"
                    })
        
        # Add general daily nutrition recommendations
        self._add_nutrition_recommendations(care_plan, health_analysis)
        
        # Add exercise recommendations
        self._add_exercise_recommendations(care_plan, health_analysis)
        
        # Add behavior recommendations
        self._add_behavior_recommendations(care_plan, health_analysis)
        
        # Add medical care recommendations
        self._add_medical_recommendations(care_plan, health_analysis)
        
        # Add weekly and monthly care items
        self._add_periodic_care_items(care_plan)
        
        # Add resources
        self._add_resources(care_plan, health_analysis)
        
        return care_plan
    
    def _extract_health_summary(self, health_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract a concise health summary from the analysis."""
        summary = {
            "overall_status": health_analysis.get("overall_assessment", {}).get("health_status", "unknown"),
            "health_score": health_analysis.get("overall_assessment", {}).get("health_score", 0),
            "key_concerns": []
        }
        
        # Extract key concerns from vital signs
        for vital, info in health_analysis.get("vital_signs", {}).items():
            if not info.get("is_in_normal_range", True) or info.get("anomaly_percentage", 0) > 10:
                concern = {
                    "type": vital,
                    "current_value": info.get("latest_value"),
                    "normal_range": info.get("normal_range"),
                    "severity": "high" if info.get("anomaly_percentage", 0) > 20 else "medium"
                }
                summary["key_concerns"].append(concern)
        
        # Add stress information if available
        if "health_indicators" in health_analysis and "stress" in health_analysis["health_indicators"]:
            stress_info = health_analysis["health_indicators"]["stress"]
            if stress_info.get("status") in ["high", "moderate"]:
                summary["key_concerns"].append({
                    "type": "stress_level",
                    "current_value": stress_info.get("average_level"),
                    "severity": "high" if stress_info.get("status") == "high" else "medium"
                })
        
        return summary
    
    def _add_nutrition_recommendations(self, care_plan: Dict[str, Any], health_analysis: Dict[str, Any]):
        """Add nutrition recommendations to the care plan."""
        # Add basic nutrition recommendations
        care_plan["daily_care"]["nutrition"].append({
            "title": "Fresh Water",
            "description": "Always provide clean, fresh water throughout the day.",
            "importance": "critical"
        })
        
        # Add age-appropriate feeding schedule
        age_category = "adult"
        if self.pet_age is not None:
            if self.pet_age < 1:
                age_category = "puppy_kitten"
            elif self.pet_age > 7:
                age_category = "senior"
        
        feeding_schedule = self.nutrition_recommendations.get("feeding_schedule", {})
        feeding_frequency = feeding_schedule.get("frequency", {}).get(age_category, "2 times per day")
        
        care_plan["daily_care"]["nutrition"].append({
            "title": "Regular Meals",
            "description": f"Feed {feeding_frequency} at consistent times.",
            "importance": "high"
        })
        
        # Add weight management if needed
        if "vital_signs" in health_analysis and "weight" in health_analysis["vital_signs"]:
            weight_info = health_analysis["vital_signs"]["weight"]
            if not weight_info.get("is_in_normal_range", True):
                if weight_info.get("latest_value", 0) > weight_info.get("normal_range", {}).get("max", 0):
                    care_plan["daily_care"]["nutrition"].append({
                        "title": "Weight Management",
                        "description": "Implement portion control and avoid high-calorie treats to help reduce weight.",
                        "actions": [
                            "Measure food precisely",
                            "Use low-calorie treats for training",
                            "Consult with veterinarian about a weight management diet"
                        ],
                        "importance": "high"
                    })
                elif weight_info.get("latest_value", 0) < weight_info.get("normal_range", {}).get("min", 0):
                    care_plan["daily_care"]["nutrition"].append({
                        "title": "Weight Gain Support",
                        "description": "Focus on nutrient-dense foods to help gain weight appropriately.",
                        "actions": [
                            "Consider a higher calorie diet (consult veterinarian)",
                            "Feed smaller, more frequent meals",
                            "Monitor weight weekly"
                        ],
                        "importance": "high"
                    })
        
        # Add pet-specific nutrition recommendations
        if self.pet_type == "dog":
            care_plan["daily_care"]["nutrition"].append({
                "title": "Appropriate Dog Food",
                "description": "Use high-quality dog food appropriate for your dog's age, size, and activity level.",
                "importance": "high"
            })
        elif self.pet_type == "cat":
            care_plan["daily_care"]["nutrition"].append({
                "title": "Balanced Cat Diet",
                "description": "Include both wet and dry food in your cat's diet to ensure proper hydration and nutrition.",
                "importance": "high"
            })
    
    def _add_exercise_recommendations(self, care_plan: Dict[str, Any], health_analysis: Dict[str, Any]):
        """Add exercise recommendations to the care plan."""
        # Check if there are activity level concerns
        activity_concern = False
        if "health_indicators" in health_analysis and "activity_distribution" in health_analysis["health_indicators"]:
            activity = health_analysis["health_indicators"]["activity_distribution"]
            if activity.get("active", 0) < 10:  # Less than 10% active time
                activity_concern = True
        
        # Get basic exercise recommendations for pet type
        if self.pet_type == "dog":
            # Determine breed size category
            size_category = "medium_breed"
            if self.pet_weight is not None:
                if self.pet_weight < 10:
                    size_category = "small_breed"
                elif self.pet_weight > 25:
                    size_category = "large_breed"
            
            # Get recommended duration
            daily_exercise = self.exercise_recommendations.get("daily_exercise", {})
            exercise_duration = daily_exercise.get("duration", {}).get(size_category, "60-90 minutes")
            
            # Add standard recommendation
            recommendation = {
                "title": "Daily Exercise",
                "description": f"Provide {exercise_duration} of physical activity daily.",
                "examples": ["Walking", "Playing fetch", "Running in a secure area"],
                "importance": "high"
            }
            
            # Modify based on health status
            if activity_concern:
                recommendation["description"] += " Since activity levels have been low, gradually increase exercise duration."
                recommendation["examples"].append("Start with shorter sessions and gradually increase")
            
            care_plan["daily_care"]["exercise"].append(recommendation)
            
            # Add mental stimulation
            care_plan["daily_care"]["exercise"].append({
                "title": "Mental Stimulation",
                "description": "Provide mental exercise through games, training, and enrichment activities.",
                "examples": ["Puzzle toys", "New routes for walks", "Training sessions", "Sniff walks"],
                "importance": "high"
            })
            
        elif self.pet_type == "cat":
            # Add standard recommendation
            recommendation = {
                "title": "Interactive Play",
                "description": "Engage in 2-3 play sessions daily, 10-15 minutes each.",
                "examples": ["Wand toys", "Fetch with small toys", "Puzzle feeders"],
                "importance": "high"
            }
            
            # Modify based on health status
            if activity_concern:
                recommendation["description"] += " Activity levels have been low, so try to make play sessions more enticing."
                recommendation["examples"].append("Try different toys to find what most engages your cat")
            
            care_plan["daily_care"]["exercise"].append(recommendation)
            
            # Add environmental enrichment
            care_plan["daily_care"]["exercise"].append({
                "title": "Environmental Enrichment",
                "description": "Provide climbing opportunities, scratching surfaces, and observation spots.",
                "examples": ["Cat trees", "Window perches", "Shelves for climbing"],
                "importance": "high"
            })
    
    def _add_behavior_recommendations(self, care_plan: Dict[str, Any], health_analysis: Dict[str, Any]):
        """Add behavior recommendations to the care plan."""
        # Check if there are stress concerns
        stress_concern = False
        if "health_indicators" in health_analysis and "stress" in health_analysis["health_indicators"]:
            stress_info = health_analysis["health_indicators"]["stress"]
            if stress_info.get("status") in ["high", "moderate"]:
                stress_concern = True
        
        # Add general behavior recommendations
        care_plan["daily_care"]["behavioral"].append({
            "title": "Positive Reinforcement",
            "description": "Use rewards to encourage good behavior instead of punishment for unwanted behavior.",
            "examples": ["Treats", "Praise", "Play", "Attention"],
            "importance": "high"
        })
        
        # Add pet-specific behavior recommendations
        if self.pet_type == "dog":
            # Add training recommendation
            care_plan["daily_care"]["behavioral"].append({
                "title": "Basic Training",
                "description": "Practice basic commands daily to maintain good behavior and mental stimulation.",
                "examples": ["Sit", "Stay", "Come", "Leave it"],
                "importance": "medium"
            })
            
            # Add stress management if needed
            if stress_concern:
                care_plan["daily_care"]["behavioral"].append({
                    "title": "Stress Reduction",
                    "description": "Implement a predictable routine and provide a safe space to reduce stress.",
                    "actions": [
                        "Maintain consistent daily schedule",
                        "Provide a quiet retreat area",
                        "Use calming pheromone products if needed",
                        "Consider anxiety-reducing supplements (consult veterinarian)"
                    ],
                    "importance": "high"
                })
            
        elif self.pet_type == "cat":
            # Add litter box recommendation
            care_plan["daily_care"]["behavioral"].append({
                "title": "Litter Box Maintenance",
                "description": "Scoop litter boxes daily and provide one more box than the number of cats.",
                "importance": "critical"
            })
            
            # Add stress management if needed
            if stress_concern:
                care_plan["daily_care"]["behavioral"].append({
                    "title": "Stress Reduction",
                    "description": "Create a cat-friendly environment with multiple resources and hiding places.",
                    "actions": [
                        "Provide multiple elevated resting areas",
                        "Create hiding spots (boxes, cat caves)",
                        "Use Feliway diffusers in main living areas",
                        "Minimize loud noises and sudden changes"
                    ],
                    "importance": "high"
                })
    
    def _add_medical_recommendations(self, care_plan: Dict[str, Any], health_analysis: Dict[str, Any]):
        """Add medical care recommendations to the care plan."""
        # Add standard preventive care
        care_plan["daily_care"]["medical"].append({
            "title": "Medication Administration",
            "description": "Administer any prescribed medications according to veterinary instructions.",
            "importance": "critical"
        })
        
        # Add dental care
        care_plan["daily_care"]["medical"].append({
            "title": "Dental Care",
            "description": "Maintain dental health through regular teeth brushing or dental treats.",
            "examples": ["Brushing teeth with pet-safe toothpaste", "Dental treats", "Dental toys"],
            "importance": "high"
        })
        
        # Add monitoring recommendation
        care_plan["daily_care"]["medical"].append({
            "title": "Health Monitoring",
            "description": "Monitor for any changes in behavior, appetite, water intake, or elimination.",
            "importance": "high"
        })
        
        # Add specific recommendations based on health issues
        for concern in health_analysis.get("health_summary", {}).get("key_concerns", []):
            concern_type = concern.get("type", "")
            
            if concern_type == "heart_rate":
                care_plan["daily_care"]["medical"].append({
                    "title": "Heart Rate Monitoring",
                    "description": "Monitor heart rate daily and note any abnormalities.",
                    "actions": [
                        "Check heart rate when pet is calm and resting",
                        "Record observations",
                        "Contact veterinarian if abnormalities persist"
                    ],
                    "importance": "high"
                })
            
            elif concern_type == "temperature":
                care_plan["daily_care"]["medical"].append({
                    "title": "Temperature Management",
                    "description": "Monitor temperature and maintain appropriate environmental conditions.",
                    "actions": [
                        "Provide cool areas if temperature is elevated",
                        "Ensure warm resting areas if temperature is low",
                        "Limit physical activity during temperature abnormalities"
                    ],
                    "importance": "high"
                })
            
            elif concern_type == "respiratory_rate":
                care_plan["daily_care"]["medical"].append({
                    "title": "Respiratory Monitoring",
                    "description": "Monitor breathing patterns and limit strenuous activity if respiratory rate is elevated.",
                    "importance": "high"
                })
            
            elif concern_type == "stress_level":
                # Already addressed in behavior recommendations
                pass
    
    def _add_periodic_care_items(self, care_plan: Dict[str, Any]):
        """Add weekly and monthly care recommendations."""
        # Add weekly recommendations
        care_plan["weekly_care"].append({
            "title": "Thorough Grooming",
            "description": "Perform a more thorough grooming session to check for skin issues, lumps, or parasites.",
            "importance": "medium"
        })
        
        if self.pet_type == "dog":
            care_plan["weekly_care"].append({
                "title": "Socialization Activity",
                "description": "Ensure your dog has at least one opportunity for social interaction with other dogs or people.",
                "examples": ["Dog park visit", "Playdate", "Training class", "Pet store visit"],
                "importance": "medium"
            })
        
        elif self.pet_type == "cat":
            care_plan["weekly_care"].append({
                "title": "Litter Box Deep Clean",
                "description": "Perform a thorough cleaning of the litter box(es) beyond daily maintenance.",
                "importance": "high"
            })
        
        # Add monthly recommendations
        care_plan["monthly_care"].append({
            "title": "Parasite Prevention",
            "description": "Administer monthly parasite preventatives as prescribed by your veterinarian.",
            "importance": "high"
        })
        
        care_plan["monthly_care"].append({
            "title": "Weight Check",
            "description": "Weigh your pet to monitor for any significant weight changes.",
            "importance": "medium"
        })
        
        care_plan["monthly_care"].append({
            "title": "Care Plan Review",
            "description": "Review your pet's care plan and make adjustments based on changing needs or seasons.",
            "importance": "medium"
        })
    
    def _add_resources(self, care_plan: Dict[str, Any], health_analysis: Dict[str, Any]):
        """Add helpful resources to the care plan."""
        # Add general resources
        care_plan["resources"].append({
            "title": "Emergency Veterinary Information",
            "description": "Keep contact information for your regular and emergency veterinarians easily accessible.",
            "importance": "critical"
        })
        
        # Add resources based on pet type
        if self.pet_type == "dog":
            care_plan["resources"].append({
                "title": "Training Resources",
                "description": "Resources for training and behavior management.",
                "links": [
                    {"title": "American Kennel Club Training Resources", "url": "https://www.akc.org/expert-advice/training/"},
                    {"title": "Dog Training Books and Videos", "url": ""}
                ],
                "importance": "medium"
            })
        
        elif self.pet_type == "cat":
            care_plan["resources"].append({
                "title": "Cat Behavior Resources",
                "description": "Resources for understanding and managing cat behavior.",
                "links": [
                    {"title": "International Cat Care", "url": "https://icatcare.org/advice/"},
                    {"title": "Cat Behavior Books and Videos", "url": ""}
                ],
                "importance": "medium"
            })
        
        # Add resources based on health concerns
        for concern in health_analysis.get("health_summary", {}).get("key_concerns", []):
            concern_type = concern.get("type", "")
            
            if concern_type == "stress_level":
                care_plan["resources"].append({
                    "title": "Stress Management Resources",
                    "description": "Resources for managing pet stress and anxiety.",
                    "links": [
                        {"title": "Understanding Pet Stress", "url": ""},
                        {"title": "Calming Products and Techniques", "url": ""}
                    ],
                    "importance": "high"
                })


class BehaviorImprovementPlan:
    """
    Creates personalized behavior improvement plans based on behavioral analysis.
    """
    
    def __init__(self, pet_type: str = "dog", pet_name: str = None, 
                 pet_age: float = None, pet_weight: float = None):
        """
        Initialize the behavior improvement plan generator.
        
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
        
        # Load behavior solutions database
        self.behavior_solutions = self._load_behavior_solutions()
    
    def _load_behavior_solutions(self) -> Dict[str, Any]:
        """Load behavior solutions for common problems."""
        # This would ideally load from a database or external file
        # Here defined inline for demonstration
        
        common_solutions = {
            "excessive_barking": {
                "title": "Excessive Barking",
                "description": "Excessive barking can be triggered by various factors including boredom, anxiety, territorial behavior, or seeking attention.",
                "solutions": [
                    {
                        "title": "Identify Triggers",
                        "description": "Observe and note what triggers the barking to address the root cause.",
                        "steps": ["Keep a log of barking episodes", "Note time, duration, and potential triggers"]
                    },
                    {
                        "title": "Increase Exercise",
                        "description": "A tired dog is less likely to bark excessively.",
                        "steps": ["Ensure adequate physical exercise daily", "Include mental stimulation activities"]
                    },
                    {
                        "title": "Desensitization Training",
                        "description": "Gradually expose your dog to barking triggers at a low intensity and reward calm behavior.",
                        "steps": ["Start with trigger at a distance", "Gradually decrease distance as tolerance improves"]
                    },
                    {
                        "title": "Teach 'Quiet' Command",
                        "description": "Train your dog to stop barking on command.",
                        "steps": ["When dog barks, say 'quiet' in a calm, firm voice", "Reward with treat when dog stops barking"]
                    }
                ],
                "applicable_to": ["dog"]
            },
            "inappropriate_elimination": {
                "title": "Inappropriate Elimination",
                "description": "Eliminating outside the litter box or in inappropriate areas can be caused by medical issues, stress, litter box aversion, or territorial marking.",
                "solutions": [
                    {
                        "title": "Rule Out Medical Issues",
                        "description": "First ensure there are no underlying medical problems causing the behavior.",
                        "steps": ["Consult with a veterinarian", "Describe the pattern of inappropriate elimination"]
                    },
                    {
                        "title": "Optimize Litter Box Setup",
                        "description": "Ensure litter box conditions are appealing to your pet.",
                        "steps": [
                            "Provide one more box than the number of cats",
                            "Clean boxes daily",
                            "Use unscented litter",
                            "Try different litter types",
                            "Place boxes in quiet, accessible locations"
                        ]
                    },
                    {
                        "title": "Clean Soiled Areas Thoroughly",
                        "description": "Remove odors from soiled areas to prevent re-marking.",
                        "steps": ["Use enzymatic cleaners designed for pet urine", "Avoid ammonia-based cleaners"]
                    },
                    {
                        "title": "Reduce Stress",
                        "description": "Address potential stressors that may be contributing to the behavior.",
                        "steps": [
                            "Provide hiding places and vertical space",
                            "Maintain a consistent routine",
                            "Consider pheromone diffusers"
                        ]
                    }
                ],
                "applicable_to": ["cat", "dog"]
            },
            "separation_anxiety": {
                "title": "Separation Anxiety",
                "description": "Distress when separated from owners, often manifesting as destructive behavior, excessive vocalization, or inappropriate elimination.",
                "solutions": [
                    {
                        "title": "Gradual Desensitization",
                        "description": "Help your pet become comfortable with your departures and absences.",
                        "steps": [
                            "Practice departure cues without leaving",
                            "Leave for very short periods, gradually extending time",
                            "Remain calm during departures and arrivals"
                        ]
                    },
                    {
                        "title": "Create Positive Associations",
                        "description": "Make alone time a positive experience.",
                        "steps": [
                            "Provide special toys only available when alone",
                            "Leave food puzzles or long-lasting treats",
                            "Play calming music or leave TV on"
                        ]
                    },
                    {
                        "title": "Establish a Safe Space",
                        "description": "Create a comfortable area where your pet feels secure when alone.",
                        "steps": [
                            "Use a crate (for dogs) or dedicated room with familiar items",
                            "Include bedding with your scent",
                            "Provide access to water and toys"
                        ]
                    },
                    {
                        "title": "Consider Professional Help",
                        "description": "Severe separation anxiety may require professional intervention.",
                        "steps": [
                            "Consult with a veterinary behaviorist",
                            "Discuss potential for anti-anxiety medications",
                            "Consider working with a certified trainer"
                        ]
                    }
                ],
                "applicable_to": ["dog", "cat"]
            },
            "aggression": {
                "title": "Aggression",
                "description": "Aggressive behavior can be directed toward humans, other animals, or specific triggers and may be motivated by fear, resource guarding, territorial behavior, or pain.",
                "solutions": [
                    {
                        "title": "Consult a Professional",
                        "description": "Aggression is a serious behavior problem that typically requires professional help.",
                        "steps": [
                            "Consult with a veterinarian to rule out medical causes",
                            "Work with a certified animal behaviorist",
                            "Follow management protocols to ensure safety"
                        ],
                        "importance": "critical"
                    },
                    {
                        "title": "Identify Triggers",
                        "description": "Determine what situations or stimuli trigger the aggressive behavior.",
                        "steps": [
                            "Keep a detailed log of incidents",
                            "Note body language preceding aggression",
                            "Identify patterns in environmental factors"
                        ]
                    },
                    {
                        "title": "Management Strategies",
                        "description": "Implement management techniques to prevent aggressive incidents while working on behavior modification.",
                        "steps": [
                            "Use barriers, leashes, or muzzles when necessary",
                            "Avoid known trigger situations when possible",
                            "Provide adequate space and resources"
                        ]
                    }
                ],
                "applicable_to": ["dog", "cat"]
            },
            "destructive_behavior": {
                "title": "Destructive Behavior",
                "description": "Destructive behaviors like chewing, scratching, or digging can result from boredom, excess energy, anxiety, or natural instincts without appropriate outlets.",
                "solutions": [
                    {
                        "title": "Provide Appropriate Outlets",
                        "description": "Redirect natural behaviors to appropriate items.",
                        "steps": [
                            "For dogs: Provide appropriate chew toys",
                            "For cats: Install scratching posts in strategic locations",
                            "Rotate toys to maintain interest"
                        ]
                    },
                    {
                        "title": "Increase Mental and Physical Stimulation",
                        "description": "A tired, mentally stimulated pet is less likely to engage in destructive behavior.",
                        "steps": [
                            "Increase exercise appropriate to species and age",
                            "Provide food puzzles and interactive toys",
                            "Practice training exercises or tricks daily"
                        ]
                    },
                    {
                        "title": "Manage the Environment",
                        "description": "Prevent access to items that should not be destroyed.",
                        "steps": [
                            "Pet-proof areas where pet is left alone",
                            "Use deterrents on furniture (for cats)",
                            "Consider crate training (for dogs) when unsupervised"
                        ]
                    }
                ],
                "applicable_to": ["dog", "cat"]
            }
        }
        
        # Add dog-specific behavior solutions
        if self.pet_type == "dog":
            common_solutions.update({
                "leash_pulling": {
                    "title": "Leash Pulling",
                    "description": "Pulling on the leash during walks can be dangerous, uncomfortable, and reinforces undesirable behavior.",
                    "solutions": [
                        {
                            "title": "Stop and Stand Still",
                            "description": "Teach your dog that pulling causes forward movement to stop.",
                            "steps": [
                                "When dog pulls, stop walking immediately",
                                "Wait for slack in the leash before proceeding",
                                "Be consistent - never allow pulling to be successful"
                            ]
                        },
                        {
                            "title": "Direction Changes",
                            "description": "Change direction when your dog pulls to teach leash awareness.",
                            "steps": [
                                "When dog pulls, use a verbal cue like 'this way'",
                                "Change direction and walk the opposite way",
                                "Reward when dog catches up and walks nicely"
                            ]
                        },
                        {
                            "title": "Reward Position Training",
                            "description": "Reinforce the position you want your dog to maintain.",
                            "steps": [
                                "Reward dog frequently for walking in the desired position",
                                "Use high-value treats initially",
                                "Gradually increase duration between rewards"
                            ]
                        },
                        {
                            "title": "Consider Training Tools",
                            "description": "Certain tools can help manage pulling while training.",
                            "steps": [
                                "Consider front-clip harnesses or head halters",
                                "Avoid punitive tools like choke or prong collars",
                                "Get professional guidance on proper use of tools"
                            ]
                        }
                    ]
                },
                "jumping_on_people": {
                    "title": "Jumping on People",
                    "description": "Jumping up is often attention-seeking behavior that can be irritating or even dangerous with larger dogs.",
                    "solutions": [
                        {
                            "title": "Ignore Jumping",
                            "description": "Remove the reward of attention for jumping behavior.",
                            "steps": [
                                "Turn away when dog jumps",
                                "Avoid eye contact, touching, or speaking to the dog",
                                "Wait for four paws on the floor before giving attention"
                            ]
                        },
                        {
                            "title": "Teach an Incompatible Behavior",
                            "description": "Train an alternative greeting behavior that cannot be performed simultaneously with jumping.",
                            "steps": [
                                "Teach 'sit' or 'down' for greetings",
                                "Have visitors ask for and reward this behavior",
                                "Practice with different people in different contexts"
                            ]
                        },
                        {
                            "title": "Consistent Management",
                            "description": "Prevent rehearsal of the jumping behavior during training.",
                            "steps": [
                                "Keep dog on leash during greetings initially",
                                "Use baby gates or tethers when visitors arrive",
                                "Ensure all family members and visitors follow the same rules"
                            ]
                        }
                    ]
                }
            })
        
        # Add cat-specific behavior solutions
        elif self.pet_type == "cat":
            common_solutions.update({
                "scratching_furniture": {
                    "title": "Scratching Furniture",
                    "description": "Scratching is a natural behavior for cats, but can be destructive when directed at furniture.",
                    "solutions": [
                        {
                            "title": "Provide Appropriate Scratching Surfaces",
                            "description": "Ensure adequate scratching options that meet your cat's preferences.",
                            "steps": [
                                "Offer both horizontal and vertical scratchers",
                                "Try different materials (sisal, cardboard, carpet)",
                                "Place scratchers near furniture currently being scratched",
                                "Have at least one scratcher in each main living area"
                            ]
                        },
                        {
                            "title": "Make Furniture Less Appealing",
                            "description": "Temporarily deter scratching on furniture while training.",
                            "steps": [
                                "Cover with double-sided tape or aluminum foil",
                                "Use citrus-scented deterrents (most cats dislike citrus)",
                                "Consider plastic claw covers as a temporary solution"
                            ]
                        },
                        {
                            "title": "Positive Reinforcement",
                            "description": "Reward appropriate scratching behavior.",
                            "steps": [
                                "When cat uses scratching post, offer praise and treats",
                                "Play with toys near or on scratching posts",
                                "Use catnip or silver vine to make posts more attractive"
                            ]
                        },
                        {
                            "title": "Regular Nail Maintenance",
                            "description": "Keep nails trimmed to minimize damage.",
                            "steps": [
                                "Trim nails every 1-2 weeks",
                                "Use proper cat nail trimmers",
                                "Make trimming a positive experience with treats"
                            ]
                        }
                    ]
                },
                "play_aggression": {
                    "title": "Play Aggression",
                    "description": "Overly rough play involving biting or scratching of people, often caused by early play experiences or insufficient appropriate play outlets.",
                    "solutions": [
                        {
                            "title": "Appropriate Play Sessions",
                            "description": "Provide regular, appropriate play outlets to satisfy hunting instincts.",
                            "steps": [
                                "Schedule 2-3 interactive play sessions daily",
                                "Use fishing-rod type toys to keep hands away",
                                "Mimic natural prey movements (scurrying, darting)",
                                "End sessions with 'prey' capture and a treat"
                            ]
                        },
                        {
                            "title": "Redirect to Toys",
                            "description": "Teach your cat to direct play behaviors toward toys, not people.",
                            "steps": [
                                "Keep appropriate toys readily available",
                                "Redirect attention to toy when play with hands begins",
                                "Never use hands or feet as play objects"
                            ]
                        },
                        {
                            "title": "End Play When Aggression Occurs",
                            "description": "Teach that aggressive play ends the fun.",
                            "steps": [
                                "If cat becomes too rough, freeze and redirect",
                                "If aggression continues, calmly end play session",
                                "Leave the room if necessary"
                            ]
                        },
                        {
                            "title": "Provide Environmental Enrichment",
                            "description": "A stimulating environment helps prevent play aggression.",
                            "steps": [
                                "Provide climbing opportunities",
                                "Rotate toys to maintain novelty",
                                "Consider a second cat for play companionship if appropriate"
                            ]
                        }
                    ]
                }
            })
        
        return common_solutions
    
    def generate_behavior_plan(self, behavior_problem: str, severity: str = "moderate") -> Dict[str, Any]:
        """
        Generate a behavior improvement plan for a specific problem.
        
        Args:
            behavior_problem: Type of behavior problem (e.g., "excessive_barking").
            severity: Severity of the problem ("mild", "moderate", "severe").
            
        Returns:
            Dictionary with personalized behavior improvement plan.
        """
        # Check if behavior problem exists in solutions database
        if behavior_problem not in self.behavior_solutions:
            return {
                "status": "error",
                "message": f"Behavior problem '{behavior_problem}' not recognized. Available problems: {list(self.behavior_solutions.keys())}"
            }
        
        # Check if solution is applicable to this pet type
        solution = self.behavior_solutions[behavior_problem]
        if self.pet_type not in solution.get("applicable_to", []):
            return {
                "status": "error",
                "message": f"Behavior problem '{behavior_problem}' is not applicable to {self.pet_type}s."
            }
        
        # Initialize plan with basic information
        plan = {
            "timestamp": datetime.now().isoformat(),
            "pet_info": {
                "name": self.pet_name,
                "type": self.pet_type,
                "age": self.pet_age,
                "weight": self.pet_weight
            },
            "behavior_problem": {
                "title": solution.get("title", behavior_problem),
                "description": solution.get("description", ""),
                "severity": severity
            },
            "approach": "multi-faceted",
            "strategies": [],
            "implementation_plan": {
                "immediate_actions": [],
                "short_term_goals": [],
                "long_term_goals": []
            },
            "progress_tracking": {
                "metrics": [],
                "log_template": {}
            },
            "resources": []
        }
        
        # Add solutions as strategies
        for s in solution.get("solutions", []):
            strategy = {
                "title": s.get("title", ""),
                "description": s.get("description", ""),
                "steps": s.get("steps", []),
                "importance": s.get("importance", "medium")
            }
            
            plan["strategies"].append(strategy)
            
            # Add to implementation plan based on importance
            if s.get("importance") == "critical" or severity == "severe":
                plan["implementation_plan"]["immediate_actions"].append({
                    "action": s.get("title", ""),
                    "timeframe": "Start immediately",
                    "details": s.get("steps", [])[0] if s.get("steps") else ""
                })
            else:
                plan["implementation_plan"]["short_term_goals"].append({
                    "goal": s.get("title", ""),
                    "timeframe": "Within 1-2 weeks",
                    "success_criteria": "Consistent implementation of all steps"
                })
        
        # Add progress tracking metrics
        if behavior_problem == "excessive_barking":
            plan["progress_tracking"]["metrics"] = [
                "Frequency of barking episodes per day",
                "Duration of barking episodes",
                "Response to 'quiet' command (%)",
                "Situations where barking was avoided"
            ]
            plan["progress_tracking"]["log_template"] = {
                "date": "",
                "barking_episodes": 0,
                "total_duration": 0,
                "triggers_noted": [],
                "successful_interventions": 0,
                "notes": ""
            }
        
        elif behavior_problem == "inappropriate_elimination":
            plan["progress_tracking"]["metrics"] = [
                "Incidents per day",
                "Locations of incidents",
                "Successful litter box/outdoor eliminations",
                "Environmental or routine changes"
            ]
            plan["progress_tracking"]["log_template"] = {
                "date": "",
                "incidents": 0,
                "locations": [],
                "successful_eliminations": 0,
                "environmental_changes": "",
                "notes": ""
            }
        
        # Add appropriate resources
        if behavior_problem == "separation_anxiety":
            plan["resources"] = [
                {
                    "title": "Separation Anxiety Resources",
                    "description": "Books, videos, and products that can help with separation anxiety",
                    "links": [
                        {"title": "Separation Anxiety in Dogs", "author": "Patricia McConnell", "type": "book"},
                        {"title": "Calming Music for Pets", "type": "product"}
                    ]
                },
                {
                    "title": "Professional Help",
                    "description": "When to seek professional assistance",
                    "content": "If you see no improvement after 2-3 weeks of consistent implementation, or if the behavior worsens, consult with a certified animal behaviorist or veterinary behaviorist."
                }
            ]
        
        elif behavior_problem == "aggression":
            plan["resources"] = [
                {
                    "title": "Safety Resources",
                    "description": "Tools and techniques for safely managing aggressive behavior",
                    "links": [
                        {"title": "Proper Muzzle Training", "type": "guide"},
                        {"title": "Management Equipment", "type": "products"}
                    ]
                },
                {
                    "title": "Professional Help",
                    "description": "Professional assistance is strongly recommended for aggression issues",
                    "content": "Aggression requires professional guidance. Contact a certified animal behaviorist or veterinary behaviorist as soon as possible.",
                    "importance": "critical"
                }
            ]
        
        # Add long-term goals
        plan["implementation_plan"]["long_term_goals"] = [
            {
                "goal": f"Significant reduction in {solution.get('title', behavior_problem).lower()}",
                "timeframe": "Within 2-3 months",
                "success_criteria": "Problem behavior reduced by at least 80%"
            },
            {
                "goal": "Maintenance of improved behavior",
                "timeframe": "Ongoing",
                "success_criteria": "Consistently appropriate behavior with occasional reinforcement"
            }
        ]
        
        return plan
