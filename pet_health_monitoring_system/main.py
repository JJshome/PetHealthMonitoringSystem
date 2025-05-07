"""
Main workflow module for the Pet Health Monitoring System.

This module serves as the entry point for the system, coordinating the various components
and providing a complete workflow from data collection to personalized care recommendations.
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from . import config
from . import utils
from .data_simulator import PetDataSimulator, generate_multi_day_dataset, generate_health_event_dataset
from .ai_diagnosis import HealthAnalyzer, DiseasePredictor
from .personalized_care import CareRecommendationEngine, BehaviorImprovementPlan


def setup_environment():
    """Set up the environment for the application."""
    # Create necessary directories
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.DATA_DIR, "simulated"), exist_ok=True)
    os.makedirs(os.path.join(config.DATA_DIR, "processed"), exist_ok=True)
    os.makedirs(os.path.join(config.DATA_DIR, "results"), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join("logs", "pet_health_monitor.log"))
        ]
    )
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    return logging.getLogger("pet_health_monitor")


def simulate_data(args):
    """
    Simulate pet health data for testing.
    
    Args:
        args: Command line arguments.
    """
    logger = logging.getLogger("pet_health_monitor.simulate")
    logger.info(f"Simulating data for pet type: {args.pet_type}, name: {args.pet_name}")
    
    # Output directory for simulated data
    output_dir = os.path.join(config.DATA_DIR, "simulated")
    
    if args.event:
        # Simulate specific health event
        logger.info(f"Simulating health event: {args.event}")
        
        simulator = PetDataSimulator(
            pet_type=args.pet_type,
            pet_name=args.pet_name,
            pet_age=args.pet_age,
            pet_weight=args.pet_weight
        )
        
        event_data = simulator.generate_health_event(
            event_type=args.event,
            duration_minutes=args.duration or 60,
            severity=0.8
        )
        
        # Save event data
        event_file = os.path.join(output_dir, f"{args.pet_name}_{args.event}_event.csv")
        utils.save_csv(event_data, event_file)
        
        logger.info(f"Simulated event data saved to {event_file}")
        print(f"Simulated event data saved to {event_file}")
        
    else:
        # Simulate normal data with potential anomalies
        logger.info(f"Simulating {args.days} days of data with anomalies={args.anomalies}")
        
        data = generate_multi_day_dataset(
            pet_type=args.pet_type,
            pet_name=args.pet_name,
            days=args.days,
            include_anomalies=args.anomalies,
            output_dir=output_dir
        )
        
        logger.info(f"Simulated {len(data)} data points")
        print(f"Simulated {len(data)} data points across {args.days} days with anomalies={args.anomalies}")
        print(f"Data saved to {output_dir}/{args.pet_name}_{args.days}days_data.csv")


def analyze_data(args):
    """
    Analyze pet health data.
    
    Args:
        args: Command line arguments.
    """
    logger = logging.getLogger("pet_health_monitor.analyze")
    logger.info(f"Analyzing data from {args.input_file}")
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        print(f"Error: Input file not found: {args.input_file}")
        return
    
    # Load the data
    try:
        data = utils.load_csv(args.input_file)
        logger.info(f"Loaded {len(data)} data points from {args.input_file}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        print(f"Error: Failed to load data: {e}")
        return
    
    # Create health analyzer
    analyzer = HealthAnalyzer(
        pet_type=args.pet_type,
        pet_name=args.pet_name,
        pet_age=args.pet_age,
        pet_weight=args.pet_weight
    )
    
    # Analyze the data
    logger.info("Performing health analysis...")
    health_analysis = analyzer.analyze_health_status(data)
    
    # Save the analysis results
    results_dir = os.path.join(config.DATA_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(results_dir, f"{args.pet_name}_health_analysis_{timestamp}.json")
    
    with open(result_file, 'w') as f:
        json.dump(health_analysis, f, indent=2)
    
    logger.info(f"Analysis results saved to {result_file}")
    
    # Print summary
    print(f"\nHealth Analysis Summary for {args.pet_name}:")
    print(f"Overall Status: {health_analysis.get('overall_assessment', {}).get('health_status', 'unknown')}")
    print(f"Health Score: {health_analysis.get('overall_assessment', {}).get('health_score', 0)}/100")
    
    if health_analysis.get('anomalies'):
        print("\nDetected Anomalies:")
        for vital, anomaly_info in health_analysis.get('anomalies', {}).items():
            print(f"  - {vital}: {anomaly_info.get('percentage', 0):.1f}% of readings")
    
    print(f"\nFull analysis saved to {result_file}")
    
    # Optionally predict diseases
    if args.predict_diseases:
        logger.info("Predicting potential diseases...")
        
        # Create disease predictor
        predictor = DiseasePredictor(pet_type=args.pet_type)
        
        # Get pet info
        pet_info = {
            "name": args.pet_name,
            "age": args.pet_age,
            "weight": args.pet_weight
        }
        
        # Predict diseases
        disease_predictions = predictor.predict_diseases(data, pet_info)
        
        # Save predictions
        pred_file = os.path.join(results_dir, f"{args.pet_name}_disease_predictions_{timestamp}.json")
        with open(pred_file, 'w') as f:
            json.dump(disease_predictions, f, indent=2)
        
        logger.info(f"Disease predictions saved to {pred_file}")
        
        # Print predictions
        if disease_predictions.get('predictions'):
            print("\nPotential Health Concerns:")
            for disease, info in disease_predictions.get('predictions', {}).items():
                print(f"  - {disease}: {info.get('probability', 0):.1%} probability")
                if info.get('matched_symptoms'):
                    print(f"    Based on: {', '.join(info.get('matched_symptoms', []))}")
        else:
            print("\nNo significant disease patterns detected.")
        
        print(f"\nFull predictions saved to {pred_file}")


def generate_care_plan(args):
    """
    Generate personalized care plan based on health analysis.
    
    Args:
        args: Command line arguments.
    """
    logger = logging.getLogger("pet_health_monitor.care_plan")
    logger.info(f"Generating care plan for {args.pet_name}")
    
    # Check if analysis file exists
    if not os.path.exists(args.analysis_file):
        logger.error(f"Analysis file not found: {args.analysis_file}")
        print(f"Error: Analysis file not found: {args.analysis_file}")
        return
    
    # Load the health analysis
    try:
        with open(args.analysis_file, 'r') as f:
            health_analysis = json.load(f)
        logger.info(f"Loaded health analysis from {args.analysis_file}")
    except Exception as e:
        logger.error(f"Error loading health analysis: {e}")
        print(f"Error: Failed to load health analysis: {e}")
        return
    
    # Create care recommendation engine
    care_engine = CareRecommendationEngine(
        pet_type=args.pet_type,
        pet_name=args.pet_name,
        pet_age=args.pet_age,
        pet_weight=args.pet_weight
    )
    
    # Generate care plan
    logger.info("Generating personalized care plan...")
    care_plan = care_engine.generate_care_plan(health_analysis)
    
    # Save the care plan
    results_dir = os.path.join(config.DATA_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan_file = os.path.join(results_dir, f"{args.pet_name}_care_plan_{timestamp}.json")
    
    with open(plan_file, 'w') as f:
        json.dump(care_plan, f, indent=2)
    
    logger.info(f"Care plan saved to {plan_file}")
    
    # Print summary
    print(f"\nPersonalized Care Plan for {args.pet_name}:")
    
    # Print immediate actions if any
    if care_plan.get('immediate_actions'):
        print("\nImmediate Actions Required:")
        for action in care_plan.get('immediate_actions', []):
            print(f"  - {action.get('title')}: {action.get('description')}")
    
    # Print daily care items
    print("\nDaily Care Recommendations:")
    for category, items in care_plan.get('daily_care', {}).items():
        if items:
            print(f"\n  {category.capitalize()}:")
            for item in items:
                print(f"    - {item.get('title')}: {item.get('description')}")
    
    print(f"\nFull care plan saved to {plan_file}")


def generate_behavior_plan(args):
    """
    Generate behavior improvement plan.
    
    Args:
        args: Command line arguments.
    """
    logger = logging.getLogger("pet_health_monitor.behavior")
    logger.info(f"Generating behavior plan for {args.pet_name}, problem: {args.problem}")
    
    # Create behavior improvement plan generator
    behavior_plan = BehaviorImprovementPlan(
        pet_type=args.pet_type,
        pet_name=args.pet_name,
        pet_age=args.pet_age,
        pet_weight=args.pet_weight
    )
    
    # Generate behavior plan
    logger.info(f"Generating behavior improvement plan for {args.problem}...")
    plan = behavior_plan.generate_behavior_plan(args.problem, args.severity)
    
    # Check if there was an error
    if plan.get('status') == 'error':
        logger.error(plan.get('message'))
        print(f"Error: {plan.get('message')}")
        return
    
    # Save the behavior plan
    results_dir = os.path.join(config.DATA_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan_file = os.path.join(results_dir, f"{args.pet_name}_{args.problem}_behavior_plan_{timestamp}.json")
    
    with open(plan_file, 'w') as f:
        json.dump(plan, f, indent=2)
    
    logger.info(f"Behavior plan saved to {plan_file}")
    
    # Print summary
    print(f"\nBehavior Improvement Plan for {args.pet_name}:")
    print(f"Problem: {plan.get('behavior_problem', {}).get('title')}")
    print(f"Severity: {plan.get('behavior_problem', {}).get('severity')}")
    
    # Print strategies
    print("\nRecommended Strategies:")
    for strategy in plan.get('strategies', []):
        print(f"\n  {strategy.get('title')}:")
        print(f"    {strategy.get('description')}")
        for step in strategy.get('steps', []):
            print(f"      - {step}")
    
    # Print implementation plan
    print("\nImplementation Plan:")
    print("\n  Immediate Actions:")
    for action in plan.get('implementation_plan', {}).get('immediate_actions', []):
        print(f"    - {action.get('action')}: {action.get('timeframe')}")
    
    print("\n  Short-Term Goals:")
    for goal in plan.get('implementation_plan', {}).get('short_term_goals', []):
        print(f"    - {goal.get('goal')}: {goal.get('timeframe')}")
    
    print(f"\nFull behavior plan saved to {plan_file}")


def run_complete_workflow(args):
    """
    Run a complete workflow from data simulation to care plan generation.
    
    Args:
        args: Command line arguments.
    """
    logger = logging.getLogger("pet_health_monitor.workflow")
    logger.info(f"Running complete workflow for {args.pet_type} {args.pet_name}")
    
    # Step 1: Simulate data
    print("\n=== Step 1: Simulating Pet Health Data ===")
    output_dir = os.path.join(config.DATA_DIR, "simulated")
    
    if args.event:
        logger.info(f"Simulating health event: {args.event}")
        
        simulator = PetDataSimulator(
            pet_type=args.pet_type,
            pet_name=args.pet_name,
            pet_age=args.pet_age,
            pet_weight=args.pet_weight
        )
        
        data = simulator.generate_health_event(
            event_type=args.event,
            duration_minutes=args.duration or 60,
            severity=0.8
        )
        
        data_file = os.path.join(output_dir, f"{args.pet_name}_{args.event}_event.csv")
    else:
        logger.info(f"Simulating {args.days} days of data with anomalies={args.anomalies}")
        
        data = generate_multi_day_dataset(
            pet_type=args.pet_type,
            pet_name=args.pet_name,
            days=args.days,
            include_anomalies=args.anomalies,
            output_dir=output_dir
        )
        
        data_file = os.path.join(output_dir, f"{args.pet_name}_{args.days}days_data.csv")
    
    utils.save_csv(data, data_file)
    print(f"Data simulated and saved to {data_file}")
    
    # Step 2: Analyze health data
    print("\n=== Step 2: Analyzing Health Data ===")
    analyzer = HealthAnalyzer(
        pet_type=args.pet_type,
        pet_name=args.pet_name,
        pet_age=args.pet_age,
        pet_weight=args.pet_weight
    )
    
    health_analysis = analyzer.analyze_health_status(data)
    
    results_dir = os.path.join(config.DATA_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_file = os.path.join(results_dir, f"{args.pet_name}_health_analysis_{timestamp}.json")
    
    with open(analysis_file, 'w') as f:
        json.dump(health_analysis, f, indent=2)
    
    print(f"Health analysis completed and saved to {analysis_file}")
    
    # Print analysis summary
    print(f"\nHealth Analysis Summary for {args.pet_name}:")
    print(f"Overall Status: {health_analysis.get('overall_assessment', {}).get('health_status', 'unknown')}")
    print(f"Health Score: {health_analysis.get('overall_assessment', {}).get('health_score', 0)}/100")
    
    if health_analysis.get('anomalies'):
        print("\nDetected Anomalies:")
        for vital, anomaly_info in health_analysis.get('anomalies', {}).items():
            print(f"  - {vital}: {anomaly_info.get('percentage', 0):.1f}% of readings")
    
    # Step 3: Generate personalized care plan
    print("\n=== Step 3: Generating Personalized Care Plan ===")
    care_engine = CareRecommendationEngine(
        pet_type=args.pet_type,
        pet_name=args.pet_name,
        pet_age=args.pet_age,
        pet_weight=args.pet_weight
    )
    
    care_plan = care_engine.generate_care_plan(health_analysis)
    
    plan_file = os.path.join(results_dir, f"{args.pet_name}_care_plan_{timestamp}.json")
    
    with open(plan_file, 'w') as f:
        json.dump(care_plan, f, indent=2)
    
    print(f"Personalized care plan generated and saved to {plan_file}")
    
    # Print care plan summary
    print("\nCare Plan Highlights:")
    if care_plan.get('immediate_actions'):
        print("\nImmediate Actions Required:")
        for action in care_plan.get('immediate_actions', []):
            print(f"  - {action.get('title')}: {action.get('description')}")
    
    print("\nDaily Care Summary:")
    for category, items in care_plan.get('daily_care', {}).items():
        if items:
            print(f"  {category.capitalize()}: {len(items)} recommendations")
    
    # Step 4: Final summary
    print("\n=== Workflow Complete ===")
    print(f"All results saved in {results_dir}")
    print(f"- Health data: {data_file}")
    print(f"- Health analysis: {analysis_file}")
    print(f"- Care plan: {plan_file}")


def train_models(args):
    """
    Train AI models using historical data.
    
    Args:
        args: Command line arguments.
    """
    logger = logging.getLogger("pet_health_monitor.train")
    logger.info(f"Training models for pet type: {args.pet_type}")
    
    # Check if training data exists
    if not os.path.exists(args.training_data):
        logger.error(f"Training data not found: {args.training_data}")
        print(f"Error: Training data not found: {args.training_data}")
        return
    
    # Load the training data
    try:
        data = utils.load_csv(args.training_data)
        logger.info(f"Loaded {len(data)} data points from {args.training_data}")
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        print(f"Error: Failed to load training data: {e}")
        return
    
    # Create health analyzer
    analyzer = HealthAnalyzer(pet_type=args.pet_type)
    
    # Train anomaly detection models
    logger.info("Training anomaly detection models...")
    training_results = analyzer.train_anomaly_detectors(data)
    
    # Print results
    print("\nTraining Results:")
    for vital, result in training_results.items():
        status = result.get('status')
        if status == 'success':
            print(f"  - {vital}: Model trained successfully")
        else:
            print(f"  - {vital}: {result.get('message', 'Training failed')}")
    
    print("\nModels saved to:", os.path.join(config.MODELS_DIR, "ai_diagnosis"))


def main():
    """Main entry point for the application."""
    # Set up the environment
    logger = setup_environment()
    logger.info("Starting Pet Health Monitoring System")
    
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Pet Health Monitoring System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Simulate data command
    simulate_parser = subparsers.add_parser("simulate", help="Simulate pet health data")
    simulate_parser.add_argument("--pet-type", type=str, default="dog", choices=["dog", "cat"], help="Type of pet")
    simulate_parser.add_argument("--pet-name", type=str, default="Buddy", help="Name of the pet")
    simulate_parser.add_argument("--pet-age", type=float, default=5.0, help="Age of the pet in years")
    simulate_parser.add_argument("--pet-weight", type=float, default=15.0, help="Weight of the pet in kg")
    simulate_parser.add_argument("--days", type=int, default=7, help="Number of days to simulate")
    simulate_parser.add_argument("--anomalies", action="store_true", help="Include anomalies in the data")
    simulate_parser.add_argument("--event", type=str, choices=["fever", "tachycardia", "bradycardia", "stress", "dehydration"], help="Simulate specific health event")
    simulate_parser.add_argument("--duration", type=int, help="Duration of the event in minutes (if --event is specified)")
    
    # Analyze data command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze pet health data")
    analyze_parser.add_argument("--input-file", type=str, required=True, help="Input data file path")
    analyze_parser.add_argument("--pet-type", type=str, default="dog", choices=["dog", "cat"], help="Type of pet")
    analyze_parser.add_argument("--pet-name", type=str, default="Buddy", help="Name of the pet")
    analyze_parser.add_argument("--pet-age", type=float, default=5.0, help="Age of the pet in years")
    analyze_parser.add_argument("--pet-weight", type=float, default=15.0, help="Weight of the pet in kg")
    analyze_parser.add_argument("--predict-diseases", action="store_true", help="Also predict potential diseases")
    
    # Generate care plan command
    care_parser = subparsers.add_parser("care-plan", help="Generate personalized care plan")
    care_parser.add_argument("--analysis-file", type=str, required=True, help="Health analysis JSON file path")
    care_parser.add_argument("--pet-type", type=str, default="dog", choices=["dog", "cat"], help="Type of pet")
    care_parser.add_argument("--pet-name", type=str, default="Buddy", help="Name of the pet")
    care_parser.add_argument("--pet-age", type=float, default=5.0, help="Age of the pet in years")
    care_parser.add_argument("--pet-weight", type=float, default=15.0, help="Weight of the pet in kg")
    
    # Generate behavior plan command
    behavior_parser = subparsers.add_parser("behavior-plan", help="Generate behavior improvement plan")
    behavior_parser.add_argument("--problem", type=str, required=True, 
                            choices=["excessive_barking", "inappropriate_elimination", "separation_anxiety", 
                                    "aggression", "destructive_behavior", "jumping_on_people", 
                                    "leash_pulling", "scratching_furniture", "play_aggression"], 
                            help="Type of behavior problem")
    behavior_parser.add_argument("--severity", type=str, default="moderate", 
                             choices=["mild", "moderate", "severe"], 
                             help="Severity of the behavior problem")
    behavior_parser.add_argument("--pet-type", type=str, default="dog", choices=["dog", "cat"], help="Type of pet")
    behavior_parser.add_argument("--pet-name", type=str, default="Buddy", help="Name of the pet")
    behavior_parser.add_argument("--pet-age", type=float, default=5.0, help="Age of the pet in years")
    behavior_parser.add_argument("--pet-weight", type=float, default=15.0, help="Weight of the pet in kg")
    
    # Run complete workflow command
    workflow_parser = subparsers.add_parser("workflow", help="Run complete workflow")
    workflow_parser.add_argument("--pet-type", type=str, default="dog", choices=["dog", "cat"], help="Type of pet")
    workflow_parser.add_argument("--pet-name", type=str, default="Buddy", help="Name of the pet")
    workflow_parser.add_argument("--pet-age", type=float, default=5.0, help="Age of the pet in years")
    workflow_parser.add_argument("--pet-weight", type=float, default=15.0, help="Weight of the pet in kg")
    workflow_parser.add_argument("--days", type=int, default=7, help="Number of days to simulate")
    workflow_parser.add_argument("--anomalies", action="store_true", help="Include anomalies in the data")
    workflow_parser.add_argument("--event", type=str, choices=["fever", "tachycardia", "bradycardia", "stress", "dehydration"], help="Simulate specific health event")
    workflow_parser.add_argument("--duration", type=int, help="Duration of the event in minutes (if --event is specified)")
    
    # Train models command
    train_parser = subparsers.add_parser("train", help="Train AI models")
    train_parser.add_argument("--training-data", type=str, required=True, help="Training data file path")
    train_parser.add_argument("--pet-type", type=str, default="dog", choices=["dog", "cat"], help="Type of pet")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the appropriate command
    if args.command == "simulate":
        simulate_data(args)
    elif args.command == "analyze":
        analyze_data(args)
    elif args.command == "care-plan":
        generate_care_plan(args)
    elif args.command == "behavior-plan":
        generate_behavior_plan(args)
    elif args.command == "workflow":
        run_complete_workflow(args)
    elif args.command == "train":
        train_models(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
