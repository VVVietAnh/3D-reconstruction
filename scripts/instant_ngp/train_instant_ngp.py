#!/usr/bin/env python3
import os
import argparse
import subprocess
from pathlib import Path

def train_instant_ngp(data_dir, output_dir):
    """Train Instant-NGP model."""
    print("Training Instant-NGP model...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run Instant-NGP training
    cmd = [
        "instant-ngp",  # Path to instant-ngp executable
        "--scene", data_dir,
        "--output", output_dir,
        "--n_steps", "10000",  # Number of training steps
        "--save_mesh",  # Save mesh after training
        "--save_snapshot"  # Save model snapshot
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Instant-NGP: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Train Instant-NGP model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing input data in Instant-NGP format')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for saving trained model and outputs')
    args = parser.parse_args()

    # Train Instant-NGP model
    train_instant_ngp(args.data_dir, args.output_dir)

if __name__ == '__main__':
    main() 