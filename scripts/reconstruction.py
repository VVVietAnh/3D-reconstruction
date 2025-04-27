import subprocess
import os
import shutil
import open3d as o3d
import time

DATASET_PATH = r"D:\doan\Run_Colmap\VA_test\xe_F1_khan_70_VA"

# Định nghĩa các tham số có thể tinh chỉnh
params = {
    "SiftExtraction.max_num_features": 6144,    # default: 6144
    "SiftExtraction.max_image_size": 2000,      # default: 2000
    "SiftExtraction.estimate_affine_shape": 1,  # default: 1 (0 hoặc 1)
    "SiftExtraction.domain_size_pooling": 1,    # 
    "SiftMatching.max_ratio": 0.85,              # default: 0.85
    "Mapper.min_num_matches": 15,               # default: 15 
    "Mapper.ba_global_max_num_iterations": 50,  # defualt: 50
    "PatchMatchStereo.geom_consistency": True,  # default: True
    "PatchMatchStereo.num_samples": 4,          # default: 4
    "PatchMatchStereo.window_radius": 2,        # default: 2
}

def clean_up():
    database_path = os.path.join(DATASET_PATH, "database.db")
    if os.path.exists(database_path):
        print(f"Deleting old database: {database_path}")
        os.remove(database_path)

    sparse_path = os.path.join(DATASET_PATH, "sparse")
    if os.path.exists(sparse_path):
        print(f"Deleting old sparse folder: {sparse_path}")
        shutil.rmtree(sparse_path)

    dense_path = os.path.join(DATASET_PATH, "dense")
    if os.path.exists(dense_path):
        print(f"Deleting old dense folder: {dense_path}")
        shutil.rmtree(dense_path)


def run_colmap_commands():
    start_time = time.time()
    
    # Tạo thư mục bằng Python thay vì dùng mkdir
    os.makedirs(os.path.join(DATASET_PATH, "sparse"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_PATH, "dense"), exist_ok=True)

    commands = [
        f"colmap feature_extractor --database_path {DATASET_PATH}/database.db --image_path {DATASET_PATH}/images "
        f"--SiftExtraction.max_num_features {params['SiftExtraction.max_num_features']} "
        f"--SiftExtraction.max_image_size {params['SiftExtraction.max_image_size']} "
        f"--SiftExtraction.estimate_affine_shape {params['SiftExtraction.estimate_affine_shape']} "
        f"--SiftExtraction.domain_size_pooling {params['SiftExtraction.domain_size_pooling']}",
        
        f"colmap exhaustive_matcher --database_path {DATASET_PATH}/database.db "
        f"--SiftMatching.max_ratio {params['SiftMatching.max_ratio']}",
        
        f"colmap mapper --database_path {DATASET_PATH}/database.db --image_path {DATASET_PATH}/images --output_path {DATASET_PATH}/sparse "
        f"--Mapper.min_num_matches {params['Mapper.min_num_matches']} "
        f"--Mapper.ba_global_max_num_iterations {params['Mapper.ba_global_max_num_iterations']}",
        
        f"colmap image_undistorter --image_path {DATASET_PATH}/images --input_path {DATASET_PATH}/sparse/0 --output_path {DATASET_PATH}/dense --output_type COLMAP --max_image_size 2000",
        
        f"colmap patch_match_stereo --workspace_path {DATASET_PATH}/dense --workspace_format COLMAP "
        f"--PatchMatchStereo.geom_consistency {str(params['PatchMatchStereo.geom_consistency']).lower()} "
        f"--PatchMatchStereo.num_samples {params['PatchMatchStereo.num_samples']} "
        f"--PatchMatchStereo.window_radius {params['PatchMatchStereo.window_radius']}",
        
        f"colmap stereo_fusion --workspace_path {DATASET_PATH}/dense --workspace_format COLMAP --input_type geometric --output_path {DATASET_PATH}/dense/fused.ply"
    ]

    for command in commands:
        print(f"Running command: {command}")        
        subprocess.run(command, shell=True, check=True)

    end_time = time.time()
    total_time = end_time - start_time
    
    return total_time


def convert_ply_to_obj():
    fused_ply_path = os.path.join(DATASET_PATH, "dense", "fused.ply")
    if os.path.exists(fused_ply_path):
        print(f"Loading point cloud from {fused_ply_path}...")
        pcd = o3d.io.read_point_cloud(fused_ply_path)
        
        num_points = len(pcd.points)
        print(f"Số lượng điểm trong point cloud: {num_points}")

        # Kiểm tra xem point cloud có normals hay không
        if not pcd.has_normals():
            print("Tính toán normals...")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))  # Điều chỉnh radius và max_nn nếu cần

        # Poisson surface reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12)

        # Optional: check if the mesh is successfully loaded and visualize
        if not mesh.is_empty():
            print("Mesh loaded successfully!")
            # Optional: visualize the mesh
            # o3d.visualization.draw_geometries([mesh])

            # Write the mesh to .obj format
            output_obj_path = os.path.join(DATASET_PATH, "Mesh.obj")
            o3d.io.write_triangle_mesh(output_obj_path, mesh)
            print(f"Mesh saved as .obj file at {output_obj_path}.")
        else:
            print("Failed to load the mesh.")
    else:
        print(f"File {fused_ply_path} does not exist.")
        return None, None
    return num_points

# if __name__ == "__main__":
#     clean_up()
#     total_time = run_colmap_commands()
#     num_points = convert_ply_to_obj()
    
#     # Ghi dữ liệu vào bảng
#     print(f"Thời gian chạy: {total_time} giây")
#     print(f"Số lượng điểm: {num_points}")
    
#     # Bạn có thể lưu dữ liệu này vào một file CSV hoặc bảng Excel
#     # Ví dụ:
#     import pandas as pd
    
#     data = {
#         "Thời gian chạy (giây)": [total_time],
#         "Số lượng điểm": [num_points]
#     }
    
#     df = pd.DataFrame(data)
#     df.to_csv("result.csv", index=False)