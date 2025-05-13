#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import shutil
import logging
import json
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


def main():
    parser = argparse.ArgumentParser(description="Run NeRF inference with instance segmentation")
    parser.add_argument("--input", type=str, required=True, help="Path to input images")
    parser.add_argument("--output", type=str, default="output/ins_ngp_old", help="Path to output directory")
    parser.add_argument("--colmap_matcher", type=str, default="exhaustive", help="COLMAP matcher type")
    args = parser.parse_args()

    # Get absolute paths
    current_dir = os.getcwd()
    instant_ngp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "instant-ngp")
    
    # Create output directory structure with absolute paths
    scene_name = os.path.basename(args.input)
    output_dir = os.path.abspath(os.path.join(current_dir, args.output, scene_name))
    os.makedirs(output_dir, exist_ok=True)
    
    # Create temporary directory for COLMAP output
    temp_dir = os.path.join(output_dir, "temp_colmap")
    os.makedirs(temp_dir, exist_ok=True)

    # Setup logging
    log_file = os.path.join(output_dir, "log_3D_reconstruction.txt")
    logger = setup_logging(log_file)
    
    try:
        logger.info("="*50)
        logger.info(f"Starting NeRF inference for scene: {scene_name}")
        logger.info(f"Input directory: {args.input}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("="*50)
        
        # Change to instant-ngp directory
        os.chdir(instant_ngp_dir)
        
        # Setup paths
        venv_python = os.path.join(instant_ngp_dir, ".venv", "bin", "python3")
        output_json = os.path.join(output_dir, "transforms.json")
        
        # Run COLMAP to NeRF conversion
        colmap_cmd = [
            venv_python, "scripts/colmap2nerf.py",
            "--colmap_matcher", args.colmap_matcher,
            "--run_colmap",
            "--aabb_scale", "4",
            "--images", os.path.abspath(args.input),
            "--out", output_json,
            "--overwrite"
        ]
        
        logger.info(f"Executing COLMAP to NeRF conversion: {' '.join(colmap_cmd)}")
        process = subprocess.Popen(
            colmap_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Log COLMAP output
        for line in process.stdout:
            logger.info(line.strip())
        
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, colmap_cmd)
        
        # Run NeRF training
        mesh_cmd = [
            venv_python, "scripts/run.py",
            "--save_mesh", os.path.join(output_dir, "output.ply"),
            "--n_steps", "5000",
            output_json
        ]
        
        logger.info(f"Executing NeRF training: {' '.join(mesh_cmd)}")
        process = subprocess.Popen(
            mesh_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Log training progress
        for line in process.stdout:
            if "PROGRESS" in line:
                continue
            logger.info(line.strip())
        
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, mesh_cmd)
        
        # Convert PLY to OBJ
        logger.info("Converting PLY to OBJ format...")
        convert_ply_to_obj(os.path.join(output_dir, 'output.ply'), os.path.join(output_dir, 'output.obj'), logger)
        
        # Change back to original directory
        os.chdir(current_dir)
        
        logger.info("="*50)
        logger.info("Process completed successfully!")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
