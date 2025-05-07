"""
AI Diagnosis module for the Pet Health Monitoring System.

This module provides advanced AI-based health analysis and diagnosis capabilities,
including anomaly detection, health status assessment, and disease prediction using
multimodal data analysis.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .config import config, PET_TYPES
from .utils import normalize_signal, smooth_signal, save_json, load_json, logger


class HealthAnalyzer:
    """
    Performs health analysis and anomaly detection using AI algorithms.
    """
    
    def __init__(self, pet_type: str = "dog", pet_name: str = None, 
                 pet_age: float = None, pet_weight: float = None,
                 model_dir: str = None):
        """
        Initialize the health analyzer.
        
        Args:
            pet_type: Type of pet (e.g., "dog", "cat").
            pet_name: Name of the pet.
            pet_age: Age of the pet in years.
            pet_weight: Weight of the pet in kg.
            model_dir: Directory to load/save models.
        """
        if pet_type not in PET_TYPES:
            raise ValueError(f"Pet type '{pet_type}' not supported. Supported types: {list(PET_TYPES.keys())}")
        
        self.pet_type = pet_type
        self.pet_name = pet_name
        self.pet_age = pet_age
        self.pet_weight = pet_weight
        
        # Get vital sign ranges for this pet type
        self.vital_ranges = PET_TYPES[pet_type].copy()
        
        # Adjust vital ranges based on age and weight if provided
        if pet_age is not None and pet_weight is not None:
            self._adjust_vital_ranges()
        
        # Set the model directory
        if model_dir is None:
            model_dir = os.path.join(config["MODELS_DIR"], "ai_diagnosis")
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        
        # Initialize models
        self.anomaly_detectors = {}
        self.health_classifier = None
        self.scalers = {}
        
        # Load pre-trained models if available
        self._load_models()
    
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
    
    def _load_models(self):
        """Load pre-trained models if they exist."""
        # Try to load anomaly detection models for each vital sign
        for vital in self.vital_ranges.keys():
            model_path = os.path.join(self.model_dir, f"{self.pet_type}_{vital}_anomaly_detector.joblib")
            scaler_path = os.path.join(self.model_dir, f"{self.pet_type}_{vital}_scaler.joblib")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    self.anomaly_detectors[vital] = joblib.load(model_path)
                    self.scalers[vital] = joblib.load(scaler_path)
                    logger.info(f"Loaded anomaly detection model for {vital}")
                except Exception as e:
                    logger.warning(f"Failed to load anomaly detection model for {vital}: {e}")
        
        # Try to load health classification model
        classifier_path = os.path.join(self.model_dir, f"{self.pet_type}_health_classifier.joblib")
        if os.path.exists(classifier_path):
            try:
                self.health_classifier = joblib.load(classifier_path)
                logger.info(f"Loaded health classification model")
            except Exception as e:
                logger.warning(f"Failed to load health classification model: {e}")
    
    def train_anomaly_detectors(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train anomaly detection models for each vital sign.
        
        Args:
            training_data: DataFrame containing historical health data.
            
        Returns:
            Dictionary with training results for each vital sign.
        """
        results = {}
        
        for vital in self.vital_ranges.keys():
            if vital in training_data.columns:
                # Extract and preprocess the data
                X = training_data[vital].values.reshape(-1, 1)
                
                # Create and fit the scaler
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Create and train the anomaly detector
                detector = IsolationForest(
                    contamination=0.05,  # Assume 5% of data points are anomalies
                    random_state=42
                )
                detector.fit(X_scaled)
                
                # Save the model and scaler
                model_path = os.path.join(self.model_dir, f"{self.pet_type}_{vital}_anomaly_detector.joblib")
                scaler_path = os.path.join(self.model_dir, f"{self.pet_type}_{vital}_scaler.joblib")
                
                joblib.dump(detector, model_path)
                joblib.dump(scaler, scaler_path)
                
                # Update the class instance
                self.anomaly_detectors[vital] = detector
                self.scalers[vital] = scaler
                
                # Record result
                results[vital] = {"status": "success", "model_path": model_path}
            else:
                results[vital] = {"status": "error", "message": f"Vital sign '{vital}' not found in training data"}
        
        return results
    
    def train_health_classifier(self, training_data: pd.DataFrame, labels: List[int]) -> Dict[str, Any]:
        """
        Train a health classification model.
        
        Args:
            training_data: DataFrame containing historical health data.
            labels: List of health status labels (0: healthy, 1: abnormal).
            
        Returns:
            Dictionary with training results.
        """
        # Extract features from training data
        features = self._extract_features(training_data)
        
        # Create and train the classifier
        classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        classifier.fit(features, labels)
        
        # Save the model
        model_path = os.path.join(self.model_dir, f"{self.pet_type}_health_classifier.joblib")
        joblib.dump(classifier, model_path)
        
        # Update the class instance
        self.health_classifier = classifier
        
        return {
            "status": "success",
            "model_path": model_path,
            "feature_importance": dict(zip(training_data.columns, classifier.feature_importances_))
        }
    
    def detect_anomalies(self, data: pd.DataFrame, window_size: int = 300) -> Dict[str, pd.DataFrame]:
        """
        Detect anomalies in health data.
        
        Args:
            data: DataFrame containing health data.
            window_size: Size of the sliding window for anomaly detection.
            
        Returns:
            Dictionary mapping vital signs to DataFrames with anomaly scores.
        """
        results = {}
        
        for vital in self.vital_ranges.keys():
            if vital in data.columns:
                # Create a DataFrame to store results
                result_df = pd.DataFrame(index=data.index)
                result_df[vital] = data[vital]
                
                # Use ML-based anomaly detection if available
                if vital in self.anomaly_detectors and vital in self.scalers:
                    # Scale the data
                    X = data[vital].values.reshape(-1, 1)
                    X_scaled = self.scalers[vital].transform(X)
                    
                    # Predict anomaly scores (-1 for anomalies, 1 for normal)
                    scores = self.anomaly_detectors[vital].decision_function(X_scaled)
                    predictions = self.anomaly_detectors[vital].predict(X_scaled)
                    
                    # Convert to anomaly scores (higher means more anomalous)
                    anomaly_scores = -scores
                    anomaly_flags = predictions == -1
                    
                    result_df["anomaly_score"] = anomaly_scores
                    result_df["is_anomaly"] = anomaly_flags
                
                # Also use rule-based detection
                min_val = self.vital_ranges[vital]["min"]
                max_val = self.vital_ranges[vital]["max"]
                
                # Mark values outside normal range
                result_df["out_of_range"] = (data[vital] < min_val) | (data[vital] > max_val)
                
                # Calculate moving average and standard deviation
                result_df["moving_avg"] = data[vital].rolling(window=window_size).mean()
                result_df["moving_std"] = data[vital].rolling(window=window_size).std()
                
                # Mark values more than 3 standard deviations from moving average
                result_df["deviation_anomaly"] = abs(data[vital] - result_df["moving_avg"]) > 3 * result_df["moving_std"]
                
                # Combine anomaly signals (if ML detection is available)
                if "is_anomaly" in result_df.columns:
                    result_df["combined_anomaly"] = result_df["is_anomaly"] | result_df["out_of_range"] | result_df["deviation_anomaly"]
                else:
                    result_df["combined_anomaly"] = result_df["out_of_range"] | result_df["deviation_anomaly"]
                
                results[vital] = result_df
            else:
                logger.warning(f"Vital sign '{vital}' not found in data")
        
        return results
    
    def analyze_health_status(self, data: pd.DataFrame, window_size: int = 300) -> Dict[str, Any]:
        """
        Analyze overall health status based on multiple vital signs.
        
        Args:
            data: DataFrame containing health data.
            window_size: Size of the window for feature extraction.
            
        Returns:
            Dictionary with health status analysis results.
        """
        # Check if required vital signs are present
        required_vitals = ["heart_rate", "respiratory_rate", "temperature"]
        for vital in required_vitals:
            if vital not in data.columns:
                return {"status": "error", "message": f"Required vital sign '{vital}' not found in data"}
        
        # Extract features for health classification
        features = self._extract_features(data, window_size)
        
        # Prepare results dictionary
        results = {
            "timestamp": datetime.now().isoformat(),
            "pet_type": self.pet_type,
            "analysis_period": {
                "start": data.index[0].isoformat(),
                "end": data.index[-1].isoformat(),
            },
            "vital_signs": {},
            "health_indicators": {},
            "anomalies": {},
            "overall_assessment": {}
        }
        
        # Analyze each vital sign
        anomaly_results = self.detect_anomalies(data, window_size)
        for vital, result_df in anomaly_results.items():
            # Calculate summary statistics
            latest_value = data[vital].iloc[-1]
            avg_value = data[vital].mean()
            min_value = data[vital].min()
            max_value = data[vital].max()
            
            # Check if current value is in normal range
            normal_min = self.vital_ranges[vital]["min"]
            normal_max = self.vital_ranges[vital]["max"]
            is_normal = normal_min <= latest_value <= normal_max
            
            # Calculate percentage of anomalies
            anomaly_percentage = result_df["combined_anomaly"].mean() * 100
            
            # Store vital sign analysis
            results["vital_signs"][vital] = {
                "latest_value": latest_value,
                "average_value": avg_value,
                "minimum_value": min_value,
                "maximum_value": max_value,
                "normal_range": {"min": normal_min, "max": normal_max},
                "is_in_normal_range": is_normal,
                "anomaly_percentage": anomaly_percentage
            }
            
            # Store anomaly information
            if anomaly_percentage > 0:
                anomaly_periods = self._identify_anomaly_periods(result_df)
                results["anomalies"][vital] = {
                    "percentage": anomaly_percentage,
                    "periods": anomaly_periods
                }
        
        # Calculate derived health indicators
        if "activity_state" in data.columns:
            # Calculate activity distribution
            activity_counts = data["activity_state"].value_counts(normalize=True) * 100
            results["health_indicators"]["activity_distribution"] = {
                "sleeping": activity_counts.get(0, 0),
                "resting": activity_counts.get(1, 0),
                "active": activity_counts.get(2, 0)
            }
        
        if "stress_level" in data.columns:
            # Calculate stress statistics
            stress_avg = data["stress_level"].mean()
            stress_max = data["stress_level"].max()
            results["health_indicators"]["stress"] = {
                "average_level": stress_avg,
                "maximum_level": stress_max,
                "status": self._assess_stress_level(stress_avg)
            }
        
        # Calculate overall health score and status
        health_score, health_status = self._calculate_overall_health(results)
        results["overall_assessment"] = {
            "health_score": health_score,
            "health_status": health_status,
            "recommendations": self._generate_recommendations(results)
        }
        
        return results
    
    def predict_health_issues(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict potential health issues based on current data.
        
        Args:
            data: DataFrame containing health data.
            
        Returns:
            Dictionary with health issue predictions.
        """
        # Extract features for prediction
        features = self._extract_features(data)
        
        # Initialize results
        results = {
            "timestamp": datetime.now().isoformat(),
            "predictions": {}
        }
        
        # Apply rule-based predictions
        heart_rate = data["heart_rate"].iloc[-1]
        resp_rate = data["respiratory_rate"].iloc[-1]
        temperature = data["temperature"].iloc[-1]
        
        # Check for fever
        if temperature > self.vital_ranges["temperature"]["max"] * 1.05:
            fever_severity = (temperature - self.vital_ranges["temperature"]["max"]) / (self.vital_ranges["temperature"]["max"] * 0.05)
            results["predictions"]["fever"] = {
                "probability": min(0.9, 0.5 + fever_severity * 0.4),
                "indicators": ["elevated_temperature"]
            }
            
            # Check for potential infections based on other signs
            if heart_rate > self.vital_ranges["heart_rate"]["max"] * 1.1:
                results["predictions"]["infection"] = {
                    "probability": 0.7,
                    "indicators": ["elevated_temperature", "elevated_heart_rate"]
                }
        
        # Check for stress or anxiety
        if heart_rate > self.vital_ranges["heart_rate"]["max"] * 1.1 and resp_rate > self.vital_ranges["respiratory_rate"]["max"] * 1.1:
            results["predictions"]["stress_anxiety"] = {
                "probability": 0.8,
                "indicators": ["elevated_heart_rate", "elevated_respiratory_rate"]
            }
        
        # Check for potential heart issues
        if heart_rate < self.vital_ranges["heart_rate"]["min"] * 0.9 or heart_rate > self.vital_ranges["heart_rate"]["max"] * 1.2:
            results["predictions"]["heart_issue"] = {
                "probability": 0.6,
                "indicators": ["abnormal_heart_rate"]
            }
        
        # Check for dehydration
        if "activity_level" in data.columns and data["activity_level"].iloc[-1] < self.vital_ranges["activity_level"]["min"] * 0.7:
            if temperature > self.vital_ranges["temperature"]["max"] * 1.02:
                results["predictions"]["dehydration"] = {
                    "probability": 0.65,
                    "indicators": ["reduced_activity", "elevated_temperature"]
                }
        
        # Apply ML-based prediction if classifier is available
        if self.health_classifier is not None:
            try:
                # Predict probability of health issues
                proba = self.health_classifier.predict_proba(features)
                
                # Only include if probability is significant
                if proba[0][1] > 0.3:  # If abnormal probability > 0.3
                    results["predictions"]["general_health_issue"] = {
                        "probability": float(proba[0][1]),
                        "based_on": "machine_learning_model"
                    }
            except Exception as e:
                logger.warning(f"Failed to apply health classifier: {e}")
        
        return results
    
    def _extract_features(self, data: pd.DataFrame, window_size: int = 300) -> np.ndarray:
        """
        Extract features from health data for model training and prediction.
        
        Args:
            data: DataFrame containing health data.
            window_size: Size of the window for feature extraction.
            
        Returns:
            NumPy array containing extracted features.
        """
        features = []
        
        # Process each vital sign
        for vital in self.vital_ranges.keys():
            if vital in data.columns:
                # Basic statistics
                features.append(data[vital].mean())
                features.append(data[vital].std())
                features.append(data[vital].min())
                features.append(data[vital].max())
                
                # Percentiles
                features.append(data[vital].quantile(0.25))
                features.append(data[vital].quantile(0.75))
                
                # Rate of change
                if len(data) > 1:
                    features.append(data[vital].diff().mean())
                    features.append(data[vital].diff().std())
                else:
                    features.append(0)
                    features.append(0)
                
                # Out of range percentage
                min_val = self.vital_ranges[vital]["min"]
                max_val = self.vital_ranges[vital]["max"]
                out_of_range = ((data[vital] < min_val) | (data[vital] > max_val)).mean()
                features.append(out_of_range)
        
        # Add derived features if available
        if "stress_level" in data.columns:
            features.append(data["stress_level"].mean())
            features.append(data["stress_level"].max())
        
        if "activity_state" in data.columns:
            # Activity distribution
            activity_counts = data["activity_state"].value_counts(normalize=True)
            features.append(activity_counts.get(0, 0))  # sleeping
            features.append(activity_counts.get(1, 0))  # resting
            features.append(activity_counts.get(2, 0))  # active
        
        return np.array(features).reshape(1, -1)
    
    def _identify_anomaly_periods(self, anomaly_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identify periods of anomalies in the data.
        
        Args:
            anomaly_df: DataFrame with anomaly detection results.
            
        Returns:
            List of dictionaries describing anomaly periods.
        """
        if "combined_anomaly" not in anomaly_df.columns:
            return []
        
        # Find contiguous anomaly periods
        anomaly_series = anomaly_df["combined_anomaly"]
        anomaly_changes = anomaly_series.astype(int).diff().fillna(0)
        
        # Find start and end indices of anomaly periods
        starts = anomaly_df.index[anomaly_changes == 1].tolist()
        ends = anomaly_df.index[anomaly_changes == -1].tolist()
        
        # Handle edge cases
        if anomaly_series.iloc[0]:
            starts.insert(0, anomaly_df.index[0])
        if anomaly_series.iloc[-1]:
            ends.append(anomaly_df.index[-1])
        
        # Create periods list
        periods = []
        for start, end in zip(starts, ends):
            periods.append({
                "start": start.isoformat(),
                "end": end.isoformat(),
                "duration_seconds": (end - start).total_seconds(),
                "average_severity": float(anomaly_df.loc[start:end, "anomaly_score"].mean()) if "anomaly_score" in anomaly_df.columns else None
            })
        
        return periods
    
    def _assess_stress_level(self, stress_avg: float) -> str:
        """
        Assess the stress level status.
        
        Args:
            stress_avg: Average stress level.
            
        Returns:
            Stress level assessment as a string.
        """
        if stress_avg < 30:
            return "low"
        elif stress_avg < 60:
            return "moderate"
        else:
            return "high"
    
    def _calculate_overall_health(self, results: Dict[str, Any]) -> Tuple[float, str]:
        """
        Calculate overall health score and status.
        
        Args:
            results: Health analysis results.
            
        Returns:
            Tuple of (health_score, health_status).
        """
        # Initialize score at 100 (perfect health)
        score = 100.0
        
        # Deduct points for vital sign anomalies
        for vital, info in results.get("vital_signs", {}).items():
            anomaly_percentage = info.get("anomaly_percentage", 0)
            
            # Deduct based on anomaly percentage (max 30 points deduction per vital sign)
            score -= min(30, anomaly_percentage / 2)
            
            # Additional deduction if current value is outside normal range
            if not info.get("is_in_normal_range", True):
                score -= 5
        
        # Deduct for stress if available
        if "health_indicators" in results and "stress" in results["health_indicators"]:
            stress_status = results["health_indicators"]["stress"]["status"]
            if stress_status == "high":
                score -= 15
            elif stress_status == "moderate":
                score -= 5
        
        # Ensure score is within 0-100 range
        score = max(0, min(100, score))
        
        # Determine health status based on score
        if score >= 90:
            status = "excellent"
        elif score >= 75:
            status = "good"
        elif score >= 60:
            status = "fair"
        elif score >= 40:
            status = "concerning"
        else:
            status = "poor"
        
        return score, status
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate health recommendations based on analysis results.
        
        Args:
            results: Health analysis results.
            
        Returns:
            List of recommendation dictionaries.
        """
        recommendations = []
        
        # Check vital signs for abnormalities
        for vital, info in results.get("vital_signs", {}).items():
            if not info.get("is_in_normal_range", True) or info.get("anomaly_percentage", 0) > 10:
                if vital == "heart_rate":
                    if info["latest_value"] > info["normal_range"]["max"]:
                        recommendations.append({
                            "type": "heart_rate_high",
                            "priority": "high",
                            "description": "Heart rate is above normal range. Reduce stressful situations and consult with a veterinarian if it persists.",
                            "specific_actions": ["Ensure a calm environment", "Monitor for additional symptoms", "Consult veterinarian if no improvement in 24 hours"]
                        })
                    else:
                        recommendations.append({
                            "type": "heart_rate_low",
                            "priority": "high",
                            "description": "Heart rate is below normal range. Keep pet warm and consult with a veterinarian as soon as possible.",
                            "specific_actions": ["Keep pet warm", "Monitor closely", "Consult veterinarian immediately"]
                        })
                
                elif vital == "temperature":
                    if info["latest_value"] > info["normal_range"]["max"]:
                        recommendations.append({
                            "type": "temperature_high",
                            "priority": "high",
                            "description": "Body temperature is elevated. Cool environment recommended. Consult veterinarian if fever persists or is severe.",
                            "specific_actions": ["Provide cool water", "Keep in cool environment", "Consult veterinarian if over 39.7°C (103.5°F) or if persistent"]
                        })
                    else:
                        recommendations.append({
                            "type": "temperature_low",
                            "priority": "high",
                            "description": "Body temperature is below normal range. Warm environment recommended. Consult veterinarian immediately.",
                            "specific_actions": ["Warm the pet gradually", "Do not use direct heat sources", "Consult veterinarian immediately"]
                        })
        
        # Check stress levels
        if "health_indicators" in results and "stress" in results["health_indicators"]:
            stress_status = results["health_indicators"]["stress"]["status"]
            if stress_status == "high":
                recommendations.append({
                    "type": "stress_high",
                    "priority": "medium",
                    "description": "High stress levels detected. Reduce stressors and provide a calm environment.",
                    "specific_actions": ["Identify and remove stressors", "Provide a quiet space", "Consider calming supplements after veterinary consultation"]
                })
        
        # Check activity levels
        if "health_indicators" in results and "activity_distribution" in results["health_indicators"]:
            activity = results["health_indicators"]["activity_distribution"]
            if activity.get("active", 0) < 10:  # Less than 10% active time
                recommendations.append({
                    "type": "activity_low",
                    "priority": "medium",
                    "description": "Low activity levels detected. Consider increasing exercise gradually.",
                    "specific_actions": ["Introduce short play sessions", "Gradually increase activity", "Check for signs of pain during movement"]
                })
            elif activity.get("sleeping", 0) < 30:  # Less than 30% sleep time
                recommendations.append({
                    "type": "sleep_insufficient",
                    "priority": "medium",
                    "description": "Insufficient sleep detected. Ensure pet has a comfortable and quiet sleeping area.",
                    "specific_actions": ["Provide comfortable bedding", "Establish a regular routine", "Reduce disturbances during rest periods"]
                })
        
        # Generic recommendation based on overall health
        overall_status = results.get("overall_assessment", {}).get("health_status")
        if overall_status in ["concerning", "poor"]:
            recommendations.append({
                "type": "overall_health_check",
                "priority": "high",
                "description": "Overall health indicators show concerns. A veterinary check-up is recommended.",
                "specific_actions": ["Schedule veterinary appointment", "Track symptoms", "Prepare health history for veterinarian"]
            })
        
        return recommendations


class DiseasePredictor:
    """
    Predicts potential diseases based on patterns in health data.
    """
    
    def __init__(self, pet_type: str = "dog"):
        """
        Initialize the disease predictor.
        
        Args:
            pet_type: Type of pet (e.g., "dog", "cat").
        """
        self.pet_type = pet_type
        
        # Load disease pattern database
        self.disease_patterns = self._load_disease_patterns()
    
    def _load_disease_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Load disease pattern database for the specific pet type.
        
        Returns:
            Dictionary of disease patterns.
        """
        # In a real system, this would load from a database
        # Here we define some common patterns for demonstration
        
        if self.pet_type == "dog":
            return {
                "canine_influenza": {
                    "symptoms": {
                        "temperature": {"direction": "high", "threshold_factor": 1.05},
                        "respiratory_rate": {"direction": "high", "threshold_factor": 1.2},
                        "heart_rate": {"direction": "high", "threshold_factor": 1.1},
                        "activity_level": {"direction": "low", "threshold_factor": 0.7}
                    },
                    "duration_days": 5-14,
                    "contagious": True,
                    "requires_treatment": True,
                    "typical_onset": "rapid"
                },
                "arthritis": {
                    "symptoms": {
                        "activity_level": {"direction": "low", "threshold_factor": 0.8},
                        "stress_level": {"direction": "high", "threshold_factor": 1.2}
                    },
                    "duration_days": "chronic",
                    "contagious": False,
                    "requires_treatment": True,
                    "typical_onset": "gradual",
                    "age_factor": "higher_risk_with_age"
                },
                "anxiety": {
                    "symptoms": {
                        "heart_rate": {"direction": "high", "threshold_factor": 1.15},
                        "respiratory_rate": {"direction": "high", "threshold_factor": 1.2},
                        "stress_level": {"direction": "high", "threshold_factor": 1.4}
                    },
                    "duration_days": "variable",
                    "contagious": False,
                    "requires_treatment": "depends",
                    "typical_onset": "situational",
                    "triggering_factors": ["loud_noises", "separation", "new_environments"]
                }
            }
        elif self.pet_type == "cat":
            return {
                "feline_upper_respiratory_infection": {
                    "symptoms": {
                        "temperature": {"direction": "high", "threshold_factor": 1.05},
                        "respiratory_rate": {"direction": "high", "threshold_factor": 1.3},
                        "activity_level": {"direction": "low", "threshold_factor": 0.6}
                    },
                    "duration_days": 7-21,
                    "contagious": True,
                    "requires_treatment": True,
                    "typical_onset": "gradual_to_rapid"
                },
                "stress_cystitis": {
                    "symptoms": {
                        "stress_level": {"direction": "high", "threshold_factor": 1.3},
                        "activity_level": {"direction": "variable", "pattern": "restlessness"}
                    },
                    "duration_days": 3-7,
                    "contagious": False,
                    "requires_treatment": True,
                    "typical_onset": "rapid",
                    "triggering_factors": ["stress", "diet_change", "litter_box_issues"]
                }
            }
        else:
            # Default to an empty database for unsupported pet types
            return {}
    
    def predict_diseases(self, data: pd.DataFrame, pet_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Predict potential diseases based on health data patterns.
        
        Args:
            data: DataFrame containing health data.
            pet_info: Additional pet information (age, weight, etc.).
            
        Returns:
            Dictionary with disease predictions.
        """
        predictions = {}
        
        # Get vital ranges for the pet type
        vital_ranges = PET_TYPES[self.pet_type]
        
        # Check each disease pattern
        for disease_name, pattern in self.disease_patterns.items():
            match_score = 0
            total_symptoms = len(pattern["symptoms"])
            matched_symptoms = []
            
            # Check each symptom in the pattern
            for symptom, criteria in pattern["symptoms"].items():
                if symptom in data.columns:
                    # Get the current value (average of last hour if available, otherwise last value)
                    if len(data) > 360:  # Assuming 10 samples per second, 1 hour = 36000 samples
                        current_value = data[symptom].iloc[-3600:].mean()
                    else:
                        current_value = data[symptom].iloc[-1]
                    
                    # Get normal range
                    if symptom in vital_ranges:
                        min_val = vital_ranges[symptom]["min"]
                        max_val = vital_ranges[symptom]["max"]
                        threshold = 0
                        
                        # Calculate threshold based on direction and factor
                        if criteria["direction"] == "high":
                            threshold = max_val * criteria["threshold_factor"]
                            if current_value >= threshold:
                                match_score += 1
                                matched_symptoms.append(f"{symptom}_elevated")
                        elif criteria["direction"] == "low":
                            threshold = min_val * criteria["threshold_factor"]
                            if current_value <= threshold:
                                match_score += 1
                                matched_symptoms.append(f"{symptom}_reduced")
                        elif criteria["direction"] == "variable" and "pattern" in criteria:
                            # For variable patterns, check for specific patterns like "restlessness"
                            if criteria["pattern"] == "restlessness" and "activity_state" in data.columns:
                                # Check for frequent changes in activity state
                                activity_changes = data["activity_state"].diff().abs().sum() / len(data)
                                if activity_changes > 0.2:  # 20% of samples have activity state changes
                                    match_score += 1
                                    matched_symptoms.append("restlessness")
                    else:
                        # For symptoms without defined ranges (e.g., stress_level)
                        if criteria["direction"] == "high" and current_value > 60:  # Arbitrary threshold
                            match_score += 1
                            matched_symptoms.append(f"{symptom}_elevated")
                        elif criteria["direction"] == "low" and current_value < 30:  # Arbitrary threshold
                            match_score += 1
                            matched_symptoms.append(f"{symptom}_reduced")
            
            # Calculate match percentage
            match_percentage = (match_score / total_symptoms) * 100
            
            # Consider age factor if applicable
            if pet_info and "age" in pet_info and "age_factor" in pattern:
                if pattern["age_factor"] == "higher_risk_with_age" and pet_info["age"] > 7:
                    match_percentage += 20  # Increase match for older pets
                elif pattern["age_factor"] == "higher_risk_when_young" and pet_info["age"] < 1:
                    match_percentage += 20  # Increase match for younger pets
            
            # Add prediction if match percentage is significant
            if match_percentage > 70:
                predictions[disease_name] = {
                    "probability": match_percentage / 100,
                    "matched_symptoms": matched_symptoms,
                    "requires_treatment": pattern.get("requires_treatment", False),
                    "contagious": pattern.get("contagious", False)
                }
        
        # Format the result
        result = {
            "timestamp": datetime.now().isoformat(),
            "pet_type": self.pet_type,
            "predictions": predictions
        }
        
        return result
