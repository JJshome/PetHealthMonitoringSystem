# Getting Started with the Pet Health Monitoring System

This guide will help you set up and start using the Pet Health Monitoring System for monitoring and managing your pet's health.

## Installation

### Prerequisites

Before installing the system, ensure you have the following prerequisites:

- Python 3.9 or higher
- Git
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/JJshome/PetHealthMonitoringSystem.git
cd PetHealthMonitoringSystem
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install the Package in Development Mode

```bash
pip install -e .
```

## Running the Demo

The system includes a demo script that showcases the complete workflow:

```bash
python -m examples.demo_workflow
```

This demo will:
1. Generate simulated pet health data
2. Visualize the data
3. Perform health analysis
4. Predict potential health issues
5. Generate a personalized care plan
6. Save all results in the `demo_output` directory

## Command Line Interface

The system provides a command-line interface for various tasks:

### Simulating Data

```bash
python -m pet_health_monitoring_system.main simulate --pet-type dog --pet-name Buddy --days 7 --anomalies
```

Options:
- `--pet-type`: Type of pet (dog or cat)
- `--pet-name`: Name of the pet
- `--pet-age`: Age of the pet in years (default: 5.0)
- `--pet-weight`: Weight of the pet in kg (default: 15.0)
- `--days`: Number of days to simulate (default: 7)
- `--anomalies`: Include anomalies in the data
- `--event`: Simulate a specific health event (fever, tachycardia, bradycardia, stress, dehydration)
- `--duration`: Duration of the event in minutes (if --event is specified)

### Analyzing Data

```bash
python -m pet_health_monitoring_system.main analyze --input-file data/simulated/Buddy_7days_data.csv --pet-type dog --pet-name Buddy --predict-diseases
```

Options:
- `--input-file`: Path to the input data file
- `--pet-type`: Type of pet (dog or cat)
- `--pet-name`: Name of the pet
- `--pet-age`: Age of the pet in years (default: 5.0)
- `--pet-weight`: Weight of the pet in kg (default: 15.0)
- `--predict-diseases`: Also predict potential diseases

### Generating Care Plans

```bash
python -m pet_health_monitoring_system.main care-plan --analysis-file data/results/Buddy_health_analysis_20250507_123456.json --pet-type dog --pet-name Buddy
```

Options:
- `--analysis-file`: Path to the health analysis JSON file
- `--pet-type`: Type of pet (dog or cat)
- `--pet-name`: Name of the pet
- `--pet-age`: Age of the pet in years (default: 5.0)
- `--pet-weight`: Weight of the pet in kg (default: 15.0)

### Generating Behavior Improvement Plans

```bash
python -m pet_health_monitoring_system.main behavior-plan --problem separation_anxiety --severity moderate --pet-type dog --pet-name Buddy
```

Options:
- `--problem`: Type of behavior problem (e.g., excessive_barking, separation_anxiety)
- `--severity`: Severity of the problem (mild, moderate, severe)
- `--pet-type`: Type of pet (dog or cat)
- `--pet-name`: Name of the pet
- `--pet-age`: Age of the pet in years (default: 5.0)
- `--pet-weight`: Weight of the pet in kg (default: 15.0)

### Running a Complete Workflow

```bash
python -m pet_health_monitoring_system.main workflow --pet-type dog --pet-name Buddy --days 7 --anomalies
```

Options: Same as for the `simulate` command

## Web Dashboard

The system provides a web-based dashboard for visualizing pet health data and accessing care recommendations:

```bash
python -m web_platform.app
```

Then open your browser and navigate to http://localhost:5000

The dashboard provides the following features:
- Configure pet profile
- Generate and visualize health data
- Analyze health status
- Generate personalized care plans
- Create behavior improvement plans

## Working with Sensor Data

### Supported Sensors

The system supports various types of sensors:
- Wearable sensors (collar, harness)
- Ear-insertable sensors
- Environmental sensors
- Smart IoT devices (feeding bowls, water dispensers, litter boxes)

### Data Collection

For real hardware sensors, implement the data collection by extending the appropriate classes in the `data_collection_module` directory.

### Simulator for Testing

For development and testing, use the data simulator:

```python
from pet_health_monitoring_system.data_simulator import PetDataSimulator

# Create a simulator
simulator = PetDataSimulator(
    pet_type="dog",
    pet_name="Buddy",
    pet_age=5.0,
    pet_weight=15.0
)

# Generate a dataset
data = simulator.generate_dataset(
    duration_hours=24,  # 24 hours of data
    sample_rate=10,     # 10 samples per second
    include_anomalies=True  # Include some health anomalies
)

# Generate a specific health event
fever_data = simulator.generate_health_event(
    event_type="fever",
    duration_minutes=60,
    severity=0.8
)

# Save the data
from pet_health_monitoring_system import utils
utils.save_csv(data, "my_pet_data.csv")
```

## Next Steps

- See the full [API Documentation](./API.md) for detailed information about all modules and classes
- Learn about [Hardware Integration](./HARDWARE.md) for connecting real sensor devices
- Explore the [Advanced Features](./ADVANCED.md) for customization options

## Troubleshooting

### Common Issues

1. **ImportError**: Make sure you've installed the package in development mode with `pip install -e .`
2. **Missing data directories**: The system will create required directories automatically, but you may need to create the `data` directory manually if you encounter errors
3. **Visualization issues**: Make sure you have a proper backend for matplotlib (`pip install python-tk` for Tkinter backend)

### Getting Help

If you encounter any issues, please:
1. Check the [Troubleshooting Guide](./TROUBLESHOOTING.md)
2. Review the [FAQ](./FAQ.md)
3. Contact support at support@ucaretron.com
