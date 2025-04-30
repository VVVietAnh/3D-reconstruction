#!/usr/bin/env python3
import os
import argparse
import subprocess
import shutil
from pathlib import Path
import sys
import zipfile
import hashlib
import requests
from tqdm import tqdm
import random

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def create_scene_directory_structure(base_dir):
    """Create the necessary directory structure for each scene."""
    dirs = [
        os.path.join(base_dir, "images"),
        os.path.join(base_dir, "sparse"),
        os.path.join(base_dir, "input"),
        os.path.join(base_dir, "input", "images"),
        os.path.join(base_dir, "input", "sparse"),
        os.path.join(base_dir, "input", "dense"),
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def select_images(image_files, k=70, h="7:1"):
    """Select k images evenly from the original set and split into train/test sets."""
    # Sort image files to ensure consistent order
    image_files.sort()
    
    # Select k images evenly spaced
    if len(image_files) > k:
        step = len(image_files) / k
        selected_indices = [int(i * step) for i in range(k)]
        selected_files = [image_files[i] for i in selected_indices]
    else:
        selected_files = image_files
    
    # Parse train/test ratio
    train_ratio, test_ratio = map(int, h.split(':'))
    total_ratio = train_ratio + test_ratio
    
    # Calculate number of train and test images
    n_test = len(selected_files) * test_ratio // total_ratio
    n_train = len(selected_files) - n_test
    
    # Select test images evenly spaced
    test_step = len(selected_files) / n_test
    test_indices = [int(i * test_step) for i in range(n_test)]
    test_files = [selected_files[i] for i in test_indices]
    
    # Train files are the remaining ones
    train_files = [f for f in selected_files if f not in test_files]
    
    return train_files, test_files

def prepare_tanks_and_temples_data(input_dir, output_dir, k=70, h="7:1"):
    """Prepare Tanks and Temples dataset for COLMAP and NeRF pipelines."""
    print("Preparing Tanks and Temples dataset...")
    print(f"Selecting {k} images with train/test ratio {h}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each scene
    images_dir = os.path.join(input_dir, "images")
    if os.path.exists(images_dir):
        for scene_dir in os.listdir(images_dir):
            scene_path = os.path.join(images_dir, scene_dir)
            if os.path.isdir(scene_path):
                print(f"\nProcessing scene: {scene_dir}")
                
                # Create scene directory structure
                scene_output_dir = os.path.join(output_dir, scene_dir)
                dirs = create_scene_directory_structure(scene_output_dir)
                
                # Get all image files
                image_files = [f for f in os.listdir(scene_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if not image_files:
                    print(f"No images found in {scene_path}")
                    continue
                
                # Select and split images
                train_files, test_files = select_images(image_files, k, h)
                
                # Copy train images to input/images
                print(f"Copying {len(train_files)} train images...")
                for img_file in train_files:
                    src = os.path.join(scene_path, img_file)
                    dst = os.path.join(dirs[3], img_file)  # input/images
                    shutil.copy2(src, dst)
                
                # Save test file names to test.txt
                test_txt_path = os.path.join(scene_output_dir, "test.txt")
                with open(test_txt_path, 'w') as f:
                    for img_file in test_files:
                        f.write(f"{img_file}\n")
                print(f"Saved {len(test_files)} test image names to {test_txt_path}")
                
                # Copy all images to images directory
                print(f"Copying all {len(image_files)} images...")
                for img_file in image_files:
                    src = os.path.join(scene_path, img_file)
                    dst = os.path.join(dirs[0], img_file)  # images
                    shutil.copy2(src, dst)
    
    print("Tanks and Temples data preparation completed.")

def prepare_data(args):
    """Prepare data for COLMAP and NeRF pipelines."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.dataset == "tanks_and_temples":
        t2_input_dir = os.path.join(args.input_dir, "tanks_and_temples")
        t2_output_dir = os.path.join(args.output_dir, "tanks_and_temples")
        prepare_tanks_and_temples_data(t2_input_dir, t2_output_dir, args.k, args.h)
    
    print("Data preparation completed.")

def main():
    parser = argparse.ArgumentParser(description='Prepare data for COLMAP and NeRF pipelines')
    parser.add_argument('--input_dir', type=str, default='data',
                        help='Input directory containing raw datasets')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='Output directory for processed data')
    parser.add_argument('--dataset', type=str, default='tanks_and_temples',
                        choices=['tanks_and_temples'],
                        help='Dataset to prepare')
    parser.add_argument('--k', type=int, default=70,
                        help='Number of images to select from the original set')
    parser.add_argument('--h', type=str, default='7:1',
                        help='Train/test ratio (e.g., 7:1 means 1 test image for every 7 train images)')
    args = parser.parse_args()
    
    prepare_data(args)

if __name__ == '__main__':
    main()
