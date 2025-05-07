from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="pet_health_monitoring_system",
    version="0.1.0",
    author="Ucaretron Inc.",
    author_email="contact@ucaretron.com",
    description="AI and IoT-based Real-time Pet Health Monitoring and Personalized Care System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JJshome/PetHealthMonitoringSystem",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "pet-health-monitor=pet_health_monitoring_system.main:main",
        ],
    },
    include_package_data=True,
)
