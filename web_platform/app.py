"""
Web interface for the Pet Health Monitoring System.

This module provides a web-based dashboard for visualizing pet health data,
viewing analysis results, and accessing personalized care recommendations.
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
import base64
from io import BytesIO

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pet_health_monitoring_system.data_simulator import PetDataSimulator, generate_multi_day_dataset
from pet_health_monitoring_system.ai_diagnosis import HealthAnalyzer, DiseasePredictor
from pet_health_monitoring_system.personalized_care import CareRecommendationEngine
from pet_health_monitoring_system import config, utils

# Create Flask app
app = Flask(__name__)

# Global variables to store data across routes
pet_data = {
    "pet_type": "dog",
    "pet_name": "Buddy",
    "pet_age": 5.0,
    "pet_weight": 15.0
}

sensor_data = None
health_analysis = None
care_plan = None

# Ensure directories exist
os.makedirs(os.path.join(config.DATA_DIR, "simulated"), exist_ok=True)
os.makedirs(os.path.join(config.DATA_DIR, "processed"), exist_ok=True)
os.makedirs(os.path.join(config.DATA_DIR, "results"), exist_ok=True)
os.makedirs(os.path.join("static", "images"), exist_ok=True)


@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html', pet_data=pet_data)


@app.route('/pet-profile', methods=['GET', 'POST'])
def pet_profile():
    """Render and handle pet profile page."""
    global pet_data
    
    if request.method == 'POST':
        # Update pet data from form
        pet_data["pet_type"] = request.form.get("pet_type", "dog")
        pet_data["pet_name"] = request.form.get("pet_name", "Buddy")
        pet_data["pet_age"] = float(request.form.get("pet_age", 5.0))
        pet_data["pet_weight"] = float(request.form.get("pet_weight", 15.0))
        
        return redirect(url_for('pet_profile'))
    
    return render_template('pet_profile.html', pet_data=pet_data)


@app.route('/generate-data', methods=['GET', 'POST'])
def generate_data():
    """Generate and display sample data."""
    global sensor_data
    
    if request.method == 'POST':
        # Get form parameters
        days = int(request.form.get("days", 7))
        include_anomalies = request.form.get("include_anomalies") == "yes"
        event_type = request.form.get("event_type")
        
        # Generate data
        if event_type and event_type != "none":
            # Generate health event
            simulator = PetDataSimulator(
                pet_type=pet_data["pet_type"],
                pet_name=pet_data["pet_name"],
                pet_age=pet_data["pet_age"],
                pet_weight=pet_data["pet_weight"]
            )
            
            sensor_data = simulator.generate_health_event(
                event_type=event_type,
                duration_minutes=120,
                severity=0.8
            )
        else:
            # Generate normal data
            sensor_data = generate_multi_day_dataset(
                pet_type=pet_data["pet_type"],
                pet_name=pet_data["pet_name"],
                days=days,
                include_anomalies=include_anomalies
            )
        
        # Save data
        output_dir = os.path.join(config.DATA_DIR, "simulated")
        data_file = os.path.join(output_dir, f"{pet_data['pet_name']}_data.csv")
        utils.save_csv(sensor_data, data_file)
        
        return redirect(url_for('data_visualization'))
    
    return render_template('generate_data.html', pet_data=pet_data)


@app.route('/data-visualization')
def data_visualization():
    """Visualize pet health data."""
    global sensor_data
    
    if sensor_data is None:
        return redirect(url_for('generate_data'))
    
    # Generate plots
    plot_urls = {}
    
    # Plot vital signs
    vital_signs = ["heart_rate", "respiratory_rate", "temperature"]
    plot_urls["vital_signs"] = _generate_plot(
        sensor_data, 
        vital_signs, 
        title=f"{pet_data['pet_name']}'s Vital Signs"
    )
    
    # Plot stress level
    if "stress_level" in sensor_data.columns:
        plot_urls["stress"] = _generate_plot(
            sensor_data,
            ["stress_level"],
            title=f"{pet_data['pet_name']}'s Stress Level"
        )
    
    # Plot activity state
    if "activity_state" in sensor_data.columns:
        plot_urls["activity"] = _generate_plot(
            sensor_data,
            ["activity_state"],
            title=f"{pet_data['pet_name']}'s Activity State"
        )
    
    return render_template(
        'data_visualization.html', 
        pet_data=pet_data, 
        plot_urls=plot_urls, 
        data_points=len(sensor_data),
        start_time=sensor_data.index[0].strftime('%Y-%m-%d %H:%M:%S'),
        end_time=sensor_data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
    )


@app.route('/analyze-health')
def analyze_health():
    """Analyze pet health data."""
    global sensor_data, health_analysis
    
    if sensor_data is None:
        return redirect(url_for('generate_data'))
    
    # Create health analyzer
    analyzer = HealthAnalyzer(
        pet_type=pet_data["pet_type"],
        pet_name=pet_data["pet_name"],
        pet_age=pet_data["pet_age"],
        pet_weight=pet_data["pet_weight"]
    )
    
    # Analyze the data
    health_analysis = analyzer.analyze_health_status(sensor_data)
    
    # Save the analysis results
    results_dir = os.path.join(config.DATA_DIR, "results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(results_dir, f"{pet_data['pet_name']}_health_analysis_{timestamp}.json")
    
    with open(result_file, 'w') as f:
        json.dump(health_analysis, f, indent=2)
    
    # Create disease predictor
    predictor = DiseasePredictor(pet_type=pet_data["pet_type"])
    
    # Get pet info
    pet_info = {
        "name": pet_data["pet_name"],
        "age": pet_data["pet_age"],
        "weight": pet_data["pet_weight"]
    }
    
    # Predict diseases
    disease_predictions = predictor.predict_diseases(sensor_data, pet_info)
    
    return render_template(
        'health_analysis.html',
        pet_data=pet_data,
        health_analysis=health_analysis,
        disease_predictions=disease_predictions
    )


@app.route('/care-plan')
def care_plan_view():
    """Generate and display personalized care plan."""
    global health_analysis, care_plan
    
    if health_analysis is None:
        return redirect(url_for('analyze_health'))
    
    # Create care recommendation engine
    care_engine = CareRecommendationEngine(
        pet_type=pet_data["pet_type"],
        pet_name=pet_data["pet_name"],
        pet_age=pet_data["pet_age"],
        pet_weight=pet_data["pet_weight"]
    )
    
    # Generate care plan
    care_plan = care_engine.generate_care_plan(health_analysis)
    
    # Save the care plan
    results_dir = os.path.join(config.DATA_DIR, "results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan_file = os.path.join(results_dir, f"{pet_data['pet_name']}_care_plan_{timestamp}.json")
    
    with open(plan_file, 'w') as f:
        json.dump(care_plan, f, indent=2)
    
    return render_template(
        'care_plan.html',
        pet_data=pet_data,
        care_plan=care_plan
    )


@app.route('/behavior-plan', methods=['GET', 'POST'])
def behavior_plan_view():
    """Generate and display behavior improvement plan."""
    if request.method == 'POST':
        # Get form parameters
        problem = request.form.get("problem")
        severity = request.form.get("severity", "moderate")
        
        # Create behavior improvement plan generator
        from pet_health_monitoring_system.personalized_care import BehaviorImprovementPlan
        behavior_plan_gen = BehaviorImprovementPlan(
            pet_type=pet_data["pet_type"],
            pet_name=pet_data["pet_name"],
            pet_age=pet_data["pet_age"],
            pet_weight=pet_data["pet_weight"]
        )
        
        # Generate behavior plan
        behavior_plan = behavior_plan_gen.generate_behavior_plan(problem, severity)
        
        if behavior_plan.get('status') == 'error':
            error_message = behavior_plan.get('message')
            return render_template(
                'behavior_plan_form.html',
                pet_data=pet_data,
                error_message=error_message
            )
        
        # Save the behavior plan
        results_dir = os.path.join(config.DATA_DIR, "results")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plan_file = os.path.join(results_dir, f"{pet_data['pet_name']}_{problem}_behavior_plan_{timestamp}.json")
        
        with open(plan_file, 'w') as f:
            json.dump(behavior_plan, f, indent=2)
        
        return render_template(
            'behavior_plan.html',
            pet_data=pet_data,
            behavior_plan=behavior_plan
        )
    
    return render_template(
        'behavior_plan_form.html',
        pet_data=pet_data
    )


def _generate_plot(data, columns, title):
    """
    Generate a plot image and return the URL.
    
    Args:
        data: DataFrame containing the data.
        columns: List of column names to plot.
        title: Plot title.
        
    Returns:
        URL to the generated plot image.
    """
    plt.figure(figsize=(10, 6))
    
    for column in columns:
        plt.plot(data.index, data[column], label=column)
    
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save to buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    
    # Encode to base64 for HTML embedding
    img_data = base64.b64encode(buffer.read()).decode()
    
    return f"data:image/png;base64,{img_data}"


if __name__ == '__main__':
    # Create basic template directory if it doesn't exist
    os.makedirs(os.path.join("web_platform", "templates"), exist_ok=True)
    
    # Create basic index.html template if it doesn't exist
    index_template = os.path.join("web_platform", "templates", "index.html")
    if not os.path.exists(index_template):
        with open(index_template, 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Pet Health Monitoring System</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
</head>
<body>
    <div class="container mt-4">
        <h1>Pet Health Monitoring System</h1>
        <p class="lead">AI and IoT-based Real-time Pet Health Monitoring and Personalized Care System</p>
        <p>Based on patented technology from Ucaretron Inc.</p>
        
        <div class="row mt-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">Current Pet Profile</h5>
                        <p><strong>Name:</strong> {{ pet_data.pet_name }}</p>
                        <p><strong>Type:</strong> {{ pet_data.pet_type }}</p>
                        <p><strong>Age:</strong> {{ pet_data.pet_age }} years</p>
                        <p><strong>Weight:</strong> {{ pet_data.pet_weight }} kg</p>
                        <a href="/pet-profile" class="btn btn-primary">Update Profile</a>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">System Workflow</h5>
                        <ol>
                            <li><a href="/pet-profile">Configure Pet Profile</a></li>
                            <li><a href="/generate-data">Generate Health Data</a></li>
                            <li><a href="/data-visualization">Visualize Health Data</a></li>
                            <li><a href="/analyze-health">Analyze Health Status</a></li>
                            <li><a href="/care-plan">Generate Care Plan</a></li>
                            <li><a href="/behavior-plan">Generate Behavior Plan</a></li>
                        </ol>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.9.3/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</body>
</html>
""")
    
    # Run the app
    app.run(debug=True)
