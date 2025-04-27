#!/usr/bin/env python3
import subprocess
import os
import shutil
import open3d as o3d
import time
import argparse
from pathlib import Path

# Default parameters for COLMAP
DEFAULT_PARAMS = {
    "SiftExtraction.max_num_features": 6144,    # default: 6144
    "SiftExtraction.max_image_size": 2000,      # default: 2000
    "SiftExtraction.estimate_affine_shape": 1,  # default: 1 (0 hoặc 1)
    "SiftExtraction.domain_size_pooling": 1,    # 
    "SiftMatching.max_ratio": 0.85,             # default: 0.85
    "Mapper.min_num_matches": 15,               # default: 15 
    "Mapper.ba_global_max_num_iterations": 50,  # defualt: 50
    "PatchMatchStereo.geom_consistency": True,  # default: True
    "PatchMatchStereo.num_samples": 4,          # default: 4
    "PatchMatchStereo.window_radius": 2,        # default: 2
}

# Test parameters for different configurations
TEST_PARAMS = {
    "default": DEFAULT_PARAMS,
    "high_quality": {
        "SiftExtraction.max_num_features": 8192,
        "SiftExtraction.max_image_size": 3000,
        "SiftExtraction.estimate_affine_shape": 1,
        "SiftExtraction.domain_size_pooling": 1,
        "SiftMatching.max_ratio": 0.8,
        "Mapper.min_num_matches": 20,
        "Mapper.ba_global_max_num_iterations": 100,
        "PatchMatchStereo.geom_consistency": True,
        "PatchMatchStereo.num_samples": 8,
        "PatchMatchStereo.window_radius": 3,
    },
    "fast": {
        "SiftExtraction.max_num_features": 4096,
        "SiftExtraction.max_image_size": 1500,
        "SiftExtraction.estimate_affine_shape": 0,
        "SiftExtraction.domain_size_pooling": 0,
        "SiftMatching.max_ratio": 0.9,
        "Mapper.min_num_matches": 10,
        "Mapper.ba_global_max_num_iterations": 30,
        "PatchMatchStereo.geom_consistency": False,
        "PatchMatchStereo.num_samples": 2,
        "PatchMatchStereo.window_radius": 1,
    }
}

def run_colmap_commands(dataset_path, params):
    """Run COLMAP pipeline commands."""
    start_time = time.time()
    
    # Create necessary directories
    os.makedirs(os.path.join(dataset_path, "sparse"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "dense"), exist_ok=True)

    commands = [
        # Feature extraction
        f"colmap feature_extractor --database_path {dataset_path}/database.db --image_path {dataset_path}/images "
        f"--SiftExtraction.max_num_features {params['SiftExtraction.max_num_features']} "
        f"--SiftExtraction.max_image_size {params['SiftExtraction.max_image_size']} "
        f"--SiftExtraction.estimate_affine_shape {params['SiftExtraction.estimate_affine_shape']} "
        f"--SiftExtraction.domain_size_pooling {params['SiftExtraction.domain_size_pooling']}",
        
        # Feature matching
        f"colmap exhaustive_matcher --database_path {dataset_path}/database.db "
        f"--SiftMatching.max_ratio {params['SiftMatching.max_ratio']}",
        
        # Sparse reconstruction
        f"colmap mapper --database_path {dataset_path}/database.db --image_path {dataset_path}/images --output_path {dataset_path}/sparse "
        f"--Mapper.min_num_matches {params['Mapper.min_num_matches']} "
        f"--Mapper.ba_global_max_num_iterations {params['Mapper.ba_global_max_num_iterations']}",
        
        # Image undistortion
        f"colmap image_undistorter --image_path {dataset_path}/images --input_path {dataset_path}/sparse/0 --output_path {dataset_path}/dense --output_type COLMAP --max_image_size 2000",
        
        # Dense reconstruction
        f"colmap patch_match_stereo --workspace_path {dataset_path}/dense --workspace_format COLMAP "
        f"--PatchMatchStereo.geom_consistency {str(params['PatchMatchStereo.geom_consistency']).lower()} "
        f"--PatchMatchStereo.num_samples {params['PatchMatchStereo.num_samples']} "
        f"--PatchMatchStereo.window_radius {params['PatchMatchStereo.window_radius']}",
        
        # Stereo fusion
        f"colmap stereo_fusion --workspace_path {dataset_path}/dense --workspace_format COLMAP --input_type geometric --output_path {dataset_path}/dense/fused.ply"
    ]

    for command in commands:
        print(f"Running command: {command}")        
        subprocess.run(command, shell=True, check=True)

    end_time = time.time()
    total_time = end_time - start_time
    
    return total_time

def convert_ply_to_obj(dataset_path):
    """Convert fused.ply to mesh using Poisson surface reconstruction."""
    fused_ply_path = os.path.join(dataset_path, "dense", "fused.ply")
    if os.path.exists(fused_ply_path):
        print(f"Loading point cloud from {fused_ply_path}...")
        pcd = o3d.io.read_point_cloud(fused_ply_path)
        
        num_points = len(pcd.points)
        print(f"Number of points in point cloud: {num_points}")

        # Compute normals if not present
        if not pcd.has_normals():
            print("Computing normals...")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # Poisson surface reconstruction
        print("Performing Poisson surface reconstruction...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12)

        # Save the mesh
        if not mesh.is_empty():
            output_path = os.path.join(dataset_path, "dense", "mesh.obj")
            o3d.io.write_triangle_mesh(output_path, mesh)
            print(f"Mesh saved to {output_path}")
            return True
        else:
            print("Failed to create mesh")
            return False
    else:
        print(f"Point cloud file not found: {fused_ply_path}")
        return False

def process_scene(scene_path, params, test_mode=False):
    """Process a single scene with COLMAP."""
    print(f"\nProcessing scene: {scene_path}")
    if test_mode:
        print("Running in test mode with parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    
    # Run COLMAP pipeline
    total_time = run_colmap_commands(scene_path, params)
    print(f"COLMAP pipeline completed in {total_time:.2f} seconds")
    
    # Convert to mesh
    success = convert_ply_to_obj(scene_path)
    if success:
        print("Mesh conversion completed successfully")
    else:
        print("Mesh conversion failed")

def main():
    parser = argparse.ArgumentParser(description='Run COLMAP reconstruction pipeline')
    parser.add_argument('--input_dir', type=str, default='data/processed',
                        help='Input directory containing processed datasets')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['dtu', 'tanks_and_temples', 'all'],
                        help='Dataset to process')
    parser.add_argument('--scene', type=str, default=None,
                        help='Specific scene to process (optional)')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode with different parameter configurations')
    parser.add_argument('--test_config', type=str, default='default',
                        choices=['default', 'high_quality', 'fast'],
                        help='Test configuration to use')
    args = parser.parse_args()
    
    # Get parameters based on mode
    params = TEST_PARAMS[args.test_config] if args.test else DEFAULT_PARAMS
    
    # Process datasets
    if args.dataset == "dtu" or args.dataset == "all":
        dtu_dir = os.path.join(args.input_dir, "dtu")
        if os.path.exists(dtu_dir):
            for scene in os.listdir(dtu_dir):
                if args.scene is None or scene == args.scene:
                    scene_path = os.path.join(dtu_dir, scene)
                    process_scene(scene_path, params, args.test)
    
    if args.dataset == "tanks_and_temples" or args.dataset == "all":
        t2_dir = os.path.join(args.input_dir, "tanks_and_temples")
        if os.path.exists(t2_dir):
            for scene in os.listdir(t2_dir):
                if args.scene is None or scene == args.scene:
                    scene_path = os.path.join(t2_dir, scene)
                    process_scene(scene_path, params, args.test)

if __name__ == '__main__':
    main() 