#!/usr/bin/env python3
import os
import argparse
import subprocess
from pathlib import Path

def run_pipeline(args):
    """Run the complete evaluation pipeline."""
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'colmap'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'instant_ngp'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'evaluation'), exist_ok=True)
    
    # Step 1: Prepare data
    print("\n=== Step 1: Preparing data ===")
    prepare_data_cmd = [
        "python", "scripts/data/prepare_data.py",
        "--output_dir", args.data_dir,
        "--datasets", args.dataset
    ]
    subprocess.run(prepare_data_cmd, check=True)
    
    # Step 2: Run COLMAP
    print("\n=== Step 2: Running COLMAP ===")
    colmap_cmd = [
        "python", "scripts/colmap/run_colmap.py",
        "--image_dir", os.path.join(args.data_dir, args.dataset),
        "--workspace_dir", os.path.join(args.output_dir, 'colmap')
    ]
    subprocess.run(colmap_cmd, check=True)
    
    # # Step 3: Train Instant-NGP
    # print("\n=== Step 3: Training Instant-NGP ===")
    # train_ngp_cmd = [
    #     "python", "scripts/instant_ngp/train_instant_ngp.py",
    #     "--data_dir", os.path.join(args.data_dir, f"{args.dataset}_instant_ngp"),
    #     "--output_dir", os.path.join(args.output_dir, 'instant_ngp')
    # ]
    # subprocess.run(train_ngp_cmd, check=True)
    
    # # Step 4: Extract geometry from Instant-NGP
    # print("\n=== Step 4: Extracting geometry from Instant-NGP ===")
    # extract_cmd = [
    #     "python", "scripts/instant_ngp/extract_geometry.py",
    #     "--model_path", os.path.join(args.output_dir, 'instant_ngp', 'model.msgpack'),
    #     "--output_dir", os.path.join(args.output_dir, 'instant_ngp')
    # ]
    # subprocess.run(extract_cmd, check=True)
    
    # # Step 5: Evaluate results
    # print("\n=== Step 5: Evaluating results ===")
    # evaluate_cmd = [
    #     "python", "scripts/evaluation/evaluate.py",
    #     "--pred_path", os.path.join(args.output_dir, 'instant_ngp', 'mesh.obj'),
    #     "--gt_path", os.path.join(args.data_dir, args.dataset, 'ground_truth.obj'),
    #     "--output_dir", os.path.join(args.output_dir, 'evaluation')
    # ]
    # subprocess.run(evaluate_cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description='Run complete 3D reconstruction evaluation pipeline')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing input data')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory for saving results')
    parser.add_argument('--dataset', type=str, choices=['dtu', 'tanks_and_temples'],
                        default='dtu',
                        help='Dataset to use')
    args = parser.parse_args()
    
    run_pipeline(args)

if __name__ == '__main__':
    main() 