#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import shutil
import logging
import json
from colmap2nerf import convert_to_nerf_format, run_colmap_commands
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run NeRF processing pipeline on a folder of images")
    parser.add_argument("--input", type=str, required=True, help="Path to input images folder")
    parser.add_argument("--output", type=str, required=True, help="Path to output folder")
    parser.add_argument("--aabb_scale", type=float, default=1.0, help="Scale factor for AABB")
    args = parser.parse_args()

    # Get absolute paths
    current_dir = os.path.abspath(os.path.dirname(__file__))
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Create temporary directory for COLMAP processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Run COLMAP commands
        logger.info("Running COLMAP processing...")
        run_colmap_commands(input_path, temp_dir)
        
        # Convert COLMAP output to NeRF format
        logger.info("Converting COLMAP output to NeRF format...")
        output_json = os.path.join(output_path, "transforms.json")
        nerf_data = convert_to_nerf_format(os.path.join(temp_dir, "sparse", "0"), args.aabb_scale, input_path)
        
        # Save NeRF data
        logger.info(f"Saving NeRF format to {output_json}")
        with open(output_json, "w") as f:
            json.dump(nerf_data, f, indent=2)
        
        # Run NeRF to create mesh
        venv_python = os.path.join(current_dir, "..", "instant-ngp", ".venv", "bin", "python")
        run_script = os.path.join(current_dir, "..", "instant-ngp", "scripts", "run.py")
        mesh_cmd = [
            venv_python, run_script,
            "--save_mesh", os.path.join(output_path, "output.ply"),
            "--n_steps", "10000",
            output_json
        ]
        
        logger.info(f"Executing: {' '.join(mesh_cmd)}")
        subprocess.run(mesh_cmd, check=True)
        
        logger.info(f"Success! Mesh saved to {os.path.join(output_path, 'output.ply')}")

if __name__ == "__main__":
    main()
