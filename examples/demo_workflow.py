#!/usr/bin/env python3
"""
Demo Workflow Example for the Pet Health Monitoring System.

This script demonstrates a complete workflow using the Pet Health Monitoring System,
from data generation to health analysis and personalized care recommendations.
"""

import os
import sys
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pet_health_monitoring_system.data_simulator import PetDataSimulator, generate_multi_day_dataset
from pet_health_monitoring_system.ai_diagnosis import HealthAnalyzer, DiseasePredictor
from pet_health_monitoring_system.personalized_care import CareRecommendationEngine
from pet_health_monitoring_system import config, utils


def main():
    """Run the demo workflow."""
    print("=== Pet Health Monitoring System Demo ===")
    print("This demo will showcase a complete workflow of the system.")
    
    # Step 1: Create output directories
    output_dir = os.path.join("demo_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 2: Configure pet
    pet_type = "dog"
    pet_name = "Buddy"
    pet_age = 5.0
    pet_weight = 15.0
    
    print(f"\nConfigured pet: {pet_name} ({pet_type}, {pet_age} years, {pet_weight} kg)")
    
    # Step 3: Generate sample data
    print("\nGenerating sample data...")
    
    # First, generate normal data
    simulator = PetDataSimulator(
        pet_type=pet_type,
        pet_name=pet_name,
        pet_age=pet_age,
        pet_weight=pet_weight
    )
    
    print("  Generating 2 days of normal data...")
    normal_data = simulator.generate_dataset(
        duration_hours=48,
        include_anomalies=False,
        output_dir=output_dir
    )
    
    # Then, generate a health event
    print("  Generating a fever event...")
    fever_data = simulator.generate_health_event(
        event_type="fever",
        duration_minutes=120,
        severity=0.8
    )
    
    # Combine the datasets
    print("  Combining datasets...")
    combined_data = normal_data.copy()
    combined_data = combined_data.append(fever_data)
    combined_data = combined_data.sort_index()
    
    # Save the combined dataset
    data_file = os.path.join(output_dir, f"{pet_name}_combined_data.csv")
    utils.save_csv(combined_data, data_file)
    
    print(f"  Data generated and saved to {data_file}")
    
    # Step 4: Visualize the data
    print("\nVisualizing the data...")
    
    # Visualize vital signs
    vital_signs = ["heart_rate", "respiratory_rate", "temperature"]
    fig = utils.plot_vital_signs(
        combined_data, 
        vital_signs, 
        title=f"{pet_name}'s Vital Signs",
        save_path=os.path.join(output_dir, f"{pet_name}_vital_signs.png")
    )
    
    print(f"  Vital signs visualization saved to {output_dir}/{pet_name}_vital_signs.png")
    
    # Step 5: Analyze the data
    print("\nAnalyzing the pet's health...")
    
    analyzer = HealthAnalyzer(
        pet_type=pet_type,
        pet_name=pet_name,
        pet_age=pet_age,
        pet_weight=pet_weight
    )
    
    health_analysis = analyzer.analyze_health_status(combined_data)
    
    # Save the analysis results
    analysis_file = os.path.join(output_dir, f"{pet_name}_health_analysis.json")
    with open(analysis_file, 'w') as f:
        json.dump(health_analysis, f, indent=2)
    
    print(f"  Health analysis completed and saved to {analysis_file}")
    
    # Print analysis summary
    print("\nHealth Analysis Summary:")
    print(f"  Overall Status: {health_analysis.get('overall_assessment', {}).get('health_status', 'unknown')}")
    print(f"  Health Score: {health_analysis.get('overall_assessment', {}).get('health_score', 0)}/100")
    
    if health_analysis.get('anomalies'):
        print("\n  Detected Anomalies:")
        for vital, anomaly_info in health_analysis.get('anomalies', {}).items():
            print(f"    - {vital}: {anomaly_info.get('percentage', 0):.1f}% of readings")
    
    # Step 6: Predict diseases
    print("\nPredicting potential diseases...")
    
    predictor = DiseasePredictor(pet_type=pet_type)
    
    pet_info = {
        "name": pet_name,
        "age": pet_age,
        "weight": pet_weight
    }
    
    disease_predictions = predictor.predict_diseases(combined_data, pet_info)
    
    # Save predictions
    pred_file = os.path.join(output_dir, f"{pet_name}_disease_predictions.json")
    with open(pred_file, 'w') as f:
        json.dump(disease_predictions, f, indent=2)
    
    print(f"  Disease predictions saved to {pred_file}")
    
    # Print predictions
    if disease_predictions.get('predictions'):
        print("\n  Potential Health Concerns:")
        for disease, info in disease_predictions.get('predictions', {}).items():
            print(f"    - {disease}: {info.get('probability', 0):.1%} probability")
            if info.get('matched_symptoms'):
                print(f"      Based on: {', '.join(info.get('matched_symptoms', []))}")
    else:
        print("\n  No significant disease patterns detected.")
    
    # Step 7: Generate personalized care plan
    print("\nGenerating personalized care plan...")
    
    care_engine = CareRecommendationEngine(
        pet_type=pet_type,
        pet_name=pet_name,
        pet_age=pet_age,
        pet_weight=pet_weight
    )
    
    care_plan = care_engine.generate_care_plan(health_analysis)
    
    # Save the care plan
    plan_file = os.path.join(output_dir, f"{pet_name}_care_plan.json")
    with open(plan_file, 'w') as f:
        json.dump(care_plan, f, indent=2)
    
    print(f"  Personalized care plan generated and saved to {plan_file}")
    
    # Print care plan summary
    print("\nCare Plan Highlights:")
    if care_plan.get('immediate_actions'):
        print("\n  Immediate Actions Required:")
        for action in care_plan.get('immediate_actions', []):
            print(f"    - {action.get('title')}: {action.get('description')}")
    
    print("\n  Daily Care Summary:")
    for category, items in care_plan.get('daily_care', {}).items():
        if items:
            print(f"    {category.capitalize()}: {len(items)} recommendations")
    
    # Step 8: Summarize the workflow
    print("\n=== Demo Workflow Complete ===")
    print("All results saved in the demo_output directory:")
    print(f"- Health data: {data_file}")
    print(f"- Health analysis: {analysis_file}")
    print(f"- Disease predictions: {pred_file}")
    print(f"- Care plan: {plan_file}")


if __name__ == "__main__":
    main()
