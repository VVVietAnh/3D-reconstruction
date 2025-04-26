#!/usr/bin/env python3
import os
import argparse
import subprocess
from pathlib import Path

def run_colmap_feature_extraction(image_dir, workspace_dir):
    """Run COLMAP feature extraction."""
    print("Running COLMAP feature extraction...")
    cmd = [
        "colmap", "feature_extractor",
        "--database_path", os.path.join(workspace_dir, "database.db"),
        "--image_path", image_dir,
        "--ImageReader.single_camera", "1"
    ]
    subprocess.run(cmd, check=True)

def run_colmap_feature_matching(workspace_dir):
    """Run COLMAP feature matching."""
    print("Running COLMAP feature matching...")
    cmd = [
        "colmap", "exhaustive_matcher",
        "--database_path", os.path.join(workspace_dir, "database.db")
    ]
    subprocess.run(cmd, check=True)

def run_colmap_mapper(workspace_dir):
    """Run COLMAP sparse reconstruction."""
    print("Running COLMAP sparse reconstruction...")
    sparse_dir = os.path.join(workspace_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)
    
    cmd = [
        "colmap", "mapper",
        "--database_path", os.path.join(workspace_dir, "database.db"),
        "--image_path", os.path.dirname(workspace_dir),
        "--output_path", sparse_dir
    ]
    subprocess.run(cmd, check=True)

def run_colmap_dense_reconstruction(workspace_dir):
    """Run COLMAP dense reconstruction."""
    print("Running COLMAP dense reconstruction...")
    dense_dir = os.path.join(workspace_dir, "dense")
    os.makedirs(dense_dir, exist_ok=True)
    
    # Convert sparse reconstruction to dense
    cmd = [
        "colmap", "image_undistorter",
        "--image_path", os.path.dirname(workspace_dir),
        "--input_path", os.path.join(workspace_dir, "sparse/0"),
        "--output_path", dense_dir,
        "--output_type", "COLMAP"
    ]
    subprocess.run(cmd, check=True)
    
    # Run dense reconstruction
    cmd = [
        "colmap", "patch_match_stereo",
        "--workspace_path", dense_dir
    ]
    subprocess.run(cmd, check=True)
    
    # Export dense point cloud
    cmd = [
        "colmap", "stereo_fusion",
        "--workspace_path", dense_dir,
        "--output_path", os.path.join(dense_dir, "fused.ply")
    ]
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description='Run COLMAP reconstruction pipeline')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing input images')
    parser.add_argument('--workspace_dir', type=str, required=True,
                        help='Directory for COLMAP workspace')
    args = parser.parse_args()

    # Create workspace directory
    os.makedirs(args.workspace_dir, exist_ok=True)

    # Run COLMAP pipeline
    run_colmap_feature_extraction(args.image_dir, args.workspace_dir)
    run_colmap_feature_matching(args.workspace_dir)
    run_colmap_mapper(args.workspace_dir)
    run_colmap_dense_reconstruction(args.workspace_dir)

if __name__ == '__main__':
    main() 