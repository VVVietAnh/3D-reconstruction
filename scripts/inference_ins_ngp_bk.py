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
import open3d as o3d
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('inference_log.txt', mode='a')
    ]
)
logger = logging.getLogger(__name__)

def convert_ply_to_obj(ply_path, obj_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12)
    o3d.io.write_triangle_mesh(obj_path, mesh)

def main():
    parser = argparse.ArgumentParser(description="Run NeRF inference with instance segmentation")
    parser.add_argument("--input", type=str, required=True, help="Path to input images")
    parser.add_argument("--output", type=str, default="output/ins_ngp", help="Path to output directory")
    parser.add_argument("--colmap_matcher", type=str, default="exhaustive", help="COLMAP matcher type")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    scene_name = os.path.basename(args.input)
    args.output = os.path.join(args.output, scene_name)
    
    # Create temporary directory for COLMAP output
    temp_dir = os.path.join(args.output, "temp_colmap")
    os.makedirs(temp_dir, exist_ok=True)

    # Create log file
    log_file = os.path.join(args.output, "log_3D_reconstruction.txt")
    with open(log_file, "w") as f:
        f.write("")
    
    # Run COLMAP
    logger.info("Running COLMAP processing...")
    logger.info(f"Input: {args.input}")
    colmap_path = run_colmap_commands(args.input, temp_dir, args.colmap_matcher)
    
    # Convert COLMAP output to NeRF format
    logger.info("Converting COLMAP output to NeRF format...")
    output_json = os.path.join(args.output, "transforms.json")
    nerf_data = convert_to_nerf_format(os.path.join(temp_dir, "sparse", "0"), 1.0, args.input)
    
    # Save NeRF data
    logger.info(f"Saving NeRF format to {output_json}")
    with open(output_json, "w") as f:
        json.dump(nerf_data, f, indent=2)
    
    # Run NeRF to create mesh
    venv_python = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "instant-ngp", ".venv", "bin", "python")
    run_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "instant-ngp", "scripts", "run.py")
    mesh_cmd = [
        venv_python, run_script,
        "--save_mesh", os.path.join(args.output, "output.ply"),
        "--n_steps", "5000",
        output_json
    ]
    
    logger.info(f"Executing: {' '.join(mesh_cmd)}")
    subprocess.run(mesh_cmd, check=True)
    
    # logger.info(f"Success! Cloud point saved to {os.path.join(args.output, 'output.ply')}")
    convert_ply_to_obj(os.path.join(args.output, 'output.ply'), os.path.join(args.output, 'output.obj'))
    logger.info(f"Success! Mesh saved to {os.path.join(args.output, 'output.obj')}")


if __name__ == "__main__":
    main()
