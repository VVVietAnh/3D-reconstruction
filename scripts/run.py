#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run NeRF processing pipeline on a folder of images')
    parser.add_argument('images_dir', help='Directory containing input images')
    args = parser.parse_args()
    
    # Get current directory
    current_dir = os.getcwd()
    
    # Change to the instant directory
    os.chdir('/home/doan/instant-ngp')
    
    # Đường dẫn đến python trong môi trường ảo
    venv_python = os.path.join(os.getcwd(), '.venv', 'bin', 'python3')
    
    # Convert COLMAP to NeRF format
    image_path = os.path.join(current_dir, "data", args.images_dir)
    colmap_cmd = [
        venv_python, "scripts/colmap2nerf.py",
        "--colmap_matcher", "exhaustive",
        "--run_colmap",
        "--aabb_scale", "4",
        "--images", image_path,
        "--out", "./output.json",
        "--overwrite"
    ]
    
    print(f"Executing: {' '.join(colmap_cmd)}")
    subprocess.run(colmap_cmd, check=True)
    
    # Run the NeRF training/rendering
    mesh_cmd = [
        venv_python, "scripts/run.py",
        "--save_mesh", os.path.join(current_dir, "output.ply"),
        "output.json"
    ]
    
    print(f"Executing: {' '.join(mesh_cmd)}")
    subprocess.run(mesh_cmd, check=True)
    
    print(f"Success! Mesh saved to {current_dir}/output.ply")

if __name__ == "__main__":
    main()
