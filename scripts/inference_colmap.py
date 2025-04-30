import subprocess
import os
import shutil
import open3d as o3d
import time

# Define tunable parameters
params = {
    "SiftExtraction.max_num_features": 8192,
    "SiftExtraction.max_image_size": 2000,
    "SiftExtraction.estimate_affine_shape": 1,
    "SiftExtraction.domain_size_pooling": 1,
    "SiftMatching.max_ratio": 0.85,
    "Mapper.min_num_matches": 15,
    "Mapper.ba_global_max_num_iterations": 50,
    "PatchMatchStereo.geom_consistency": True,
    "PatchMatchStereo.num_samples": 8,
    "PatchMatchStereo.window_radius": 2,
}

def clean_up(output_path):
    """
    Removes existing database, sparse, and dense reconstruction folders 
    in the output path to prepare for a fresh COLMAP run.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    database_path = os.path.join(output_path, "database.db")
    if os.path.exists(database_path):
        print(f"Deleting old database: {database_path}")
        os.remove(database_path)

    sparse_path = os.path.join(output_path, "sparse")
    if os.path.exists(sparse_path):
        print(f"Deleting old sparse folder: {sparse_path}")
        shutil.rmtree(sparse_path)

    dense_path = os.path.join(output_path, "dense")
    if os.path.exists(dense_path):
        print(f"Deleting old dense folder: {dense_path}")
        shutil.rmtree(dense_path)

def run_colmap_commands(input_path, output_path):
    """
    Executes a sequence of COLMAP commands to perform:
    - Feature extraction
    - Feature matching
    - Sparse reconstruction
    - Image undistortion
    - Dense reconstruction
    - Stereo fusion
    """
    start_time = time.time()
    
    # Create folders
    os.makedirs(os.path.join(output_path, "sparse"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "dense"), exist_ok=True)

    commands = [
        f"colmap feature_extractor --database_path {output_path}/database.db --image_path {input_path} "
        f"--SiftExtraction.max_num_features {params['SiftExtraction.max_num_features']} "
        f"--SiftExtraction.max_image_size {params['SiftExtraction.max_image_size']} "
        f"--SiftExtraction.estimate_affine_shape {params['SiftExtraction.estimate_affine_shape']} "
        f"--SiftExtraction.domain_size_pooling {params['SiftExtraction.domain_size_pooling']}",
        
        f"colmap exhaustive_matcher --database_path {output_path}/database.db "
        f"--SiftMatching.max_ratio {params['SiftMatching.max_ratio']}",
        
        f"colmap mapper --database_path {output_path}/database.db --image_path {input_path} --output_path {output_path}/sparse "
        f"--Mapper.min_num_matches {params['Mapper.min_num_matches']} "
        f"--Mapper.ba_global_max_num_iterations {params['Mapper.ba_global_max_num_iterations']}",
        
        f"colmap image_undistorter --image_path {input_path} --input_path {output_path}/sparse/0 --output_path {output_path}/dense --output_type COLMAP --max_image_size 2000",
        
        f"colmap patch_match_stereo --workspace_path {output_path}/dense --workspace_format COLMAP "
        f"--PatchMatchStereo.geom_consistency {str(params['PatchMatchStereo.geom_consistency']).lower()} "
        f"--PatchMatchStereo.num_samples {params['PatchMatchStereo.num_samples']} "
        f"--PatchMatchStereo.window_radius {params['PatchMatchStereo.window_radius']}",
        
        f"colmap stereo_fusion --workspace_path {output_path}/dense --workspace_format COLMAP --input_type geometric --output_path {output_path}/dense/fused.ply"
    ]

    for command in commands:
        print(f"Running command: {command}")        
        subprocess.run(command, shell=True, check=True)

    end_time = time.time()
    total_time = end_time - start_time
    
    return total_time

def convert_ply_to_obj(output_path):
    """
    Converts a dense point cloud (.ply file) to a surface mesh (.obj file) using 
    Poisson surface reconstruction.
    """
    fused_ply_path = os.path.join(output_path, "dense", "fused.ply")
    if os.path.exists(fused_ply_path):
        print(f"Loading point cloud from {fused_ply_path}...")
        pcd = o3d.io.read_point_cloud(fused_ply_path)
        
        num_points = len(pcd.points)
        print(f"Number of points in the point cloud: {num_points}")

        # Check if the point cloud has normals
        if not pcd.has_normals():
            print("Estimating normals...")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # Poisson surface reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12)

        # Check if the mesh is successfully created
        if not mesh.is_empty():
            print("Mesh loaded successfully!")
            output_obj_path = os.path.join(output_path, "Mesh.obj")
            o3d.io.write_triangle_mesh(output_obj_path, mesh)
            print(f"Mesh saved as .obj file at {output_obj_path}.")
        else:
            print("Failed to create mesh.")
    else:
        print(f"File {fused_ply_path} does not exist.")
        return None
    return num_points

if __name__ == "__main__":
    # Define input and output paths
    INPUT_PATH = "data/processed_images/xe_F1_khan"
    ROOT_OUTPUT_PATH = "output/colmap"

    output_dir_name = os.path.basename(INPUT_PATH).split(".")[0]
    OUTPUT_PATH = os.path.join(ROOT_OUTPUT_PATH, output_dir_name)

    clean_up(OUTPUT_PATH)
    total_time = run_colmap_commands(INPUT_PATH, OUTPUT_PATH)
    num_points = convert_ply_to_obj(OUTPUT_PATH)
    
    print(f"Execution time: {total_time} seconds")
    print(f"Number of points: {num_points}")

    # Save results into CSV file
    # import pandas as pd
    
    # data = {
    #     "Execution Time (seconds)": [total_time],
    #     "Number of Points": [num_points]
    # }
    
    # df = pd.DataFrame(data)
    # output_csv_path = os.path.join(OUTPUT_PATH, "result.csv")
    # df.to_csv(output_csv_path, index=False)
    # print(f"Results saved to {output_csv_path}")


