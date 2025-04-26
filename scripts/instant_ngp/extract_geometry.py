#!/usr/bin/env python3
import os
import argparse
import subprocess
import trimesh
import numpy as np
from pathlib import Path

def extract_mesh_from_instant_ngp(model_path, output_path):
    """Extract mesh from Instant-NGP model."""
    print("Extracting mesh from Instant-NGP model...")
    
    # Run Instant-NGP mesh extraction
    cmd = [
        "instant-ngp",
        "--load", model_path,
        "--extract_mesh",
        "--output", output_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error extracting mesh: {e}")
        raise

def post_process_mesh(input_path, output_path):
    """Post-process extracted mesh."""
    print("Post-processing mesh...")
    
    # Load mesh
    mesh = trimesh.load(input_path)
    
    # Clean mesh
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    
    # Save processed mesh
    mesh.export(output_path)

def main():
    parser = argparse.ArgumentParser(description='Extract geometry from Instant-NGP model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained Instant-NGP model')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for saving extracted geometry')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract mesh
    raw_mesh_path = os.path.join(args.output_dir, "raw_mesh.obj")
    final_mesh_path = os.path.join(args.output_dir, "mesh.obj")
    
    extract_mesh_from_instant_ngp(args.model_path, raw_mesh_path)
    post_process_mesh(raw_mesh_path, final_mesh_path)

if __name__ == '__main__':
    main() 