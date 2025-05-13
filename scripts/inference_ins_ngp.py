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

def setup_logging(log_file):
    """
    Set up logging configuration to write to both console and file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Create a new logger
    logger = logging.getLogger('inference_ins_ngp')
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def convert_ply_to_obj(ply_path, obj_path, logger):
    """
    Convert PLY point cloud to OBJ mesh using Poisson surface reconstruction
    """
    logger.info(f"Loading point cloud from {ply_path}...")
    pcd = o3d.io.read_point_cloud(ply_path)
    
    num_points = len(pcd.points)
    logger.info(f"Number of points in point cloud: {num_points}")
    
    if not pcd.has_normals():
        logger.info("Estimating normals for point cloud...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    logger.info("Performing Poisson surface reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12)
    
    if not mesh.is_empty():
        logger.info("Mesh created successfully!")
        o3d.io.write_triangle_mesh(obj_path, mesh)
        logger.info(f"Mesh saved to {obj_path}")
    else:
        logger.error("Failed to create mesh from point cloud")
        raise RuntimeError("Mesh creation failed")

def run_colmap_commands_with_logging(input_path, output_path, colmap_matcher, logger):
    """
    Run COLMAP commands with logging
    """
    logger.info(f"Running COLMAP commands with matcher: {colmap_matcher}")
    try:
        colmap_path = run_colmap_commands(input_path, output_path, colmap_matcher)
        logger.info("COLMAP processing completed successfully")
        return colmap_path
    except Exception as e:
        logger.error(f"COLMAP processing failed: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Run NeRF inference with instance segmentation")
    parser.add_argument("--input", type=str, required=True, help="Path to input images")
    parser.add_argument("--output", type=str, default="output/ins_ngp", help="Path to output directory")
    parser.add_argument("--colmap_matcher", type=str, default="exhaustive", help="COLMAP matcher type")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = os.path.join(args.output, os.path.basename(args.input))
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
        else:
            logger.info(f"Output directory already exists: {output_dir}")
    except Exception as e:
        logger.error(f"Failed to create output directory: {e}")
        raise

    # Create COLMAP output directory if it doesn't exist
    colmap_output_dir = os.path.join(output_dir, "colmap")
    try:
        if not os.path.exists(colmap_output_dir):
            os.makedirs(colmap_output_dir, exist_ok=True)
            logger.info(f"Created COLMAP output directory: {colmap_output_dir}")
        else:
            logger.info(f"COLMAP output directory already exists: {colmap_output_dir}")
    except Exception as e:
        logger.error(f"Failed to create COLMAP output directory: {e}")
        raise

    # Create NeRF output directory if it doesn't exist
    nerf_output_dir = os.path.join(output_dir, "nerf")
    try:
        if not os.path.exists(nerf_output_dir):
            os.makedirs(nerf_output_dir, exist_ok=True)
            logger.info(f"Created NeRF output directory: {nerf_output_dir}")
        else:
            logger.info(f"NeRF output directory already exists: {nerf_output_dir}")
    except Exception as e:
        logger.error(f"Failed to create NeRF output directory: {e}")
        raise

    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(output_dir, "logs")
    try:
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir, exist_ok=True)
            logger.info(f"Created logs directory: {logs_dir}")
        else:
            logger.info(f"Logs directory already exists: {logs_dir}")
    except Exception as e:
        logger.error(f"Failed to create logs directory: {e}")
        raise

    # Setup logging
    log_file = os.path.join(logs_dir, "log_3D_reconstruction.txt")
    logger = setup_logging(log_file)
    
    try:
        logger.info("="*50)
        logger.info(f"Starting NeRF inference for scene: {os.path.basename(args.input)}")
        logger.info(f"Input directory: {args.input}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("="*50)
        
        # Run COLMAP
        logger.info("Running COLMAP processing...")
        colmap_path = run_colmap_commands_with_logging(args.input, colmap_output_dir, args.colmap_matcher, logger)
        
        # Convert COLMAP output to NeRF format
        logger.info("Converting COLMAP output to NeRF format...")
        output_json = os.path.join(nerf_output_dir, "transforms.json")
        nerf_data = convert_to_nerf_format(os.path.join(colmap_output_dir, "sparse", "0"), 1.0, args.input)
        
        # Save NeRF data
        logger.info(f"Saving NeRF format to {output_json}")
        with open(output_json, "w") as f:
            json.dump(nerf_data, f, indent=2)
        
        # Run NeRF to create mesh
        venv_python = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "instant-ngp", ".venv", "bin", "python")
        run_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "instant-ngp", "scripts", "run.py")
        mesh_cmd = [
            venv_python, run_script,
            "--save_mesh", os.path.join(nerf_output_dir, "output.ply"),
            "--n_steps", "5000",
            output_json
        ]
        
        logger.info(f"Executing NeRF mesh generation: {' '.join(mesh_cmd)}")
        logger.info("Starting NeRF training...")
        
        # Run NeRF and capture output
        process = subprocess.Popen(
            mesh_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Log training progress in real-time
        for line in process.stdout:
            if 'PROGRESS' in line:
                continue
            logger.info(line.strip())
        
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, mesh_cmd)
        
        # Convert PLY to OBJ
        logger.info("Converting PLY to OBJ format...")
        convert_ply_to_obj(os.path.join(nerf_output_dir, 'output.ply'), os.path.join(nerf_output_dir, 'output.obj'), logger)
        
        logger.info("="*50)
        logger.info("Process completed successfully!")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
