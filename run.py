#!/usr/bin/env python3
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_commands():
    """Run the project components in the correct order"""
    try:
        # 1. Download images
        logger.info("Running download_images.py...")
        subprocess.run(["python", "src/data/download_images.py"], check=True)
        
        # 2. Run data analysis
        logger.info("Running data_analysis.py...")
        subprocess.run(["python", "src/data/data_analysis.py"], check=True)
        
        # 3. Train models
        logger.info("Running train_models.py...")
        subprocess.run(["python", "src/training/train_models.py"], check=True)
        
        logger.info("All setup steps completed successfully!")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running setup commands: {e}")
        raise

if __name__ == "__main__":
    run_commands()