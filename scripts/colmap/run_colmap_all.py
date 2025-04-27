#!/usr/bin/env python3
import subprocess
import os
import argparse
from pathlib import Path

# Default parameters for COLMAP sparse reconstruction
DEFAULT_PARAMS = {
    "SiftExtraction.max_num_features": 6144,    # default: 6144
    "SiftExtraction.max_image_size": 2000,      # default: 2000
    "SiftExtraction.estimate_affine_shape": 1,  # default: 1 (0 or 1)
    "SiftExtraction.domain_size_pooling": 1,    # default: 1
    "SiftMatching.max_ratio": 0.85,             # default: 0.85
    "Mapper.min_num_matches": 15,               # default: 15 
    "Mapper.ba_global_max_num_iterations": 50,  # default: 50
}

def run_colmap_sparse(dataset_path, params):
    """Run COLMAP sparse reconstruction pipeline."""
    print(f"\nProcessing dataset: {dataset_path}")
    
    # Create necessary directories
    os.makedirs(os.path.join(dataset_path, "sparse"), exist_ok=True)
    
    # Clean up old database if exists
    database_path = os.path.join(dataset_path, "database.db")
    if os.path.exists(database_path):
        print(f"Deleting old database: {database_path}")
        os.remove(database_path)
    
    # Clean up old sparse reconstruction if exists
    sparse_path = os.path.join(dataset_path, "sparse")
    if os.path.exists(sparse_path):
        print(f"Deleting old sparse reconstruction: {sparse_path}")
        for file in os.listdir(sparse_path):
            os.remove(os.path.join(sparse_path, file))
    
    commands = [
        # Feature extraction
        f"colmap feature_extractor --database_path {database_path} --image_path {dataset_path}/images "
        f"--SiftExtraction.max_num_features {params['SiftExtraction.max_num_features']} "
        f"--SiftExtraction.max_image_size {params['SiftExtraction.max_image_size']} "
        f"--SiftExtraction.estimate_affine_shape {params['SiftExtraction.estimate_affine_shape']} "
        f"--SiftExtraction.domain_size_pooling {params['SiftExtraction.domain_size_pooling']}",
        
        # Feature matching
        f"colmap exhaustive_matcher --database_path {database_path} "
        f"--SiftMatching.max_ratio {params['SiftMatching.max_ratio']}",
        
        # Sparse reconstruction
        f"colmap mapper --database_path {database_path} --image_path {dataset_path}/images --output_path {sparse_path} "
        f"--Mapper.min_num_matches {params['Mapper.min_num_matches']} "
        f"--Mapper.ba_global_max_num_iterations {params['Mapper.ba_global_max_num_iterations']}"
    ]

    for command in commands:
        print(f"Running command: {command}")
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {command}")
            print(f"Error message: {e}")
            return False
    
    return True

def process_dataset(input_dir, dataset_name):
    """Process a single dataset."""
    dataset_path = os.path.join(input_dir, dataset_name)
    if not os.path.exists(dataset_path):
        print(f"Dataset directory not found: {dataset_path}")
        return
    
    dataset_path_list = os.listdir(dataset_path)
    # do the last 3 scenes
    dataset_path_list = ['Courthouse', 'Barn', 'Ignatius']
    # Process each scene in the dataset
    for scene in dataset_path_list:
        scene_path = os.path.join(dataset_path, scene)
        if os.path.isdir(scene_path):
            print(f"\nProcessing scene: {scene}")
            success = run_colmap_sparse(scene_path, DEFAULT_PARAMS)
            if success:
                print(f"Successfully processed scene: {scene}")
            else:
                print(f"Failed to process scene: {scene}")

def main():
    parser = argparse.ArgumentParser(description='Run COLMAP sparse reconstruction on all images')
    parser.add_argument('--input_dir', type=str, default='data/processed',
                        help='Input directory containing processed datasets')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['dtu', 'tanks_and_temples', 'all'],
                        help='Dataset to process')
    args = parser.parse_args()
    
    if args.dataset == "dtu" or args.dataset == "all":
        process_dataset(args.input_dir, "dtu")
    
    if args.dataset == "tanks_and_temples" or args.dataset == "all":
        process_dataset(args.input_dir, "tanks_and_temples")

if __name__ == '__main__':
    main() 