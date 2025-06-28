"""
Setup script for Customer Churn Prediction project.

This setup file allows for easy installation of the project dependencies
and package structure for development and deployment.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="customer-churn-prediction",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced Customer Segmentation and Churn Prediction using Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/customer-churn-prediction",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/customer-churn-prediction/issues",
        "Documentation": "https://github.com/yourusername/customer-churn-prediction#readme",
        "Source Code": "https://github.com/yourusername/customer-churn-prediction",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "viz": [
            "plotly>=5.0.0",
            "streamlit>=1.0.0",
        ],
        "ml": [
            "shap>=0.40.0",
            "optuna>=2.10.0",
            "lightgbm>=3.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "churn-predict=src.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    zip_safe=False,
) 