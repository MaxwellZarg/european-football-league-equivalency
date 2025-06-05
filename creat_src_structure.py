#!/usr/bin/env python3
"""
Create Source Directory Structure
Sets up the proper directory structure for the transfer prediction system
"""

import os
from pathlib import Path

def create_directory_structure():
    """Create the complete source directory structure."""
    
    print("Creating source directory structure...")
    
    # Define the directory structure
    directories = [
        "src",
        "src/data_processing", 
        "src/equivalency_model",
        "src/transfer_prediction",
        "src/utils",
        "src/visualization",
        "logs",
        "models/trained",
        "results/analysis",
        "results/reports", 
        "results/dashboard",
        "data/processed"
    ]
    
    # Create directories
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}")
    
    # Create __init__.py files for Python packages
    init_files = [
        "src/__init__.py",
        "src/data_processing/__init__.py",
        "src/equivalency_model/__init__.py", 
        "src/transfer_prediction/__init__.py",
        "src/utils/__init__.py",
        "src/visualization/__init__.py"
    ]
    
    for init_file in init_files:
        init_path = Path(init_file)
        if not init_path.exists():
            init_path.write_text('"""Package initialization."""\n')
            print(f"Created: {init_file}")
    
    print("Directory structure creation complete!")

def copy_code_files():
    """Copy the code from artifacts to actual files."""
    
    print("\nYou'll need to copy the code from the artifacts to these files:")
    
    files_to_create = [
        ("src/data_processing/european_data_loader.py", "European Data Loader"),
        ("src/data_processing/transfer_labeler.py", "Transfer Labeling System"),
        ("src/data_processing/feature_engineer.py", "Feature Engineering Pipeline"),
        ("src/transfer_prediction/transfer_models.py", "Transfer Prediction Models"),
        ("main_pipeline.py", "Main Transfer Prediction Pipeline"),
        ("run_transfer_prediction.py", "Transfer Prediction Execution Script")
    ]
    
    for file_path, description in files_to_create:
        print(f"  {file_path} <- {description}")

if __name__ == "__main__":
    create_directory_structure()
    copy_code_files()
