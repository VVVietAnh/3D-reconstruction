import subprocess
import os
import shutil
import open3d as o3d
import time

DATASET_PATH = r"D:\doan\Run_Colmap\VA_test\xe_F1_khan_70_train"

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

import torch, cv2
import lpips
import numpy as np
# import open3d as o3d
import pyrender
import trimesh
from tqdm import tqdm
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from read_write_model import read_cameras_binary, read_images_binary

# =======================
# Load Camera Intrinsics
# =======================
def load_intrinsics(camera):
    fx, fy, cx, cy = camera.params[:4]
    width, height = int(camera.width), int(camera.height)
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])
    return K, width, height

# =======================
# Load Camera Extrinsics
# =======================
def load_extrinsics(image):
    R = image.qvec2rotmat()
    t = image.tvec.reshape(3, 1)
    # Colmap: world to camera => cần nghịch đảo để camera to world
    Rt = np.eye(4)
    Rt[:3, :3] = R
    Rt[:3, 3] = t[:, 0]
    return np.linalg.inv(Rt)

# =======================
# Render Image
# =======================
def render_scene(mesh, K, width, height, pose):
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.3, 0.3, 0.3])
    scene.add(mesh)

    camera = pyrender.IntrinsicsCamera(
        fx=K[0, 0], fy=K[1, 1],
        cx=K[0, 2], cy=K[1, 2],
        znear=0.1, zfar=100.0
    )

    # Convert Colmap to OpenGL camera frame
    flip_transform = np.eye(4)
    flip_transform[1, 1] = -1
    flip_transform[2, 2] = -1
    pose = pose @ flip_transform

    scene.add(camera, pose=pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.FLAT | pyrender.RenderFlags.SKIP_CULL_FACES)

    renderer.delete()
    return color

# =======================
# Compute Metrics
# =======================
def compute_metrics(gt_image, pred_image, loss_fn_alex):
    # Resize pred_image if needed
    if gt_image.shape != pred_image.shape:
        pred_image = cv2.resize(pred_image, (gt_image.shape[1], gt_image.shape[0]))

    # Convert to float32
    gt_float = gt_image.astype(np.float32) / 255.0
    pred_float = pred_image.astype(np.float32) / 255.0

    # PSNR
    psnr = compare_psnr(gt_float, pred_float, data_range=1.0)

    # SSIM
    print(f"GT image shape: {gt_image.shape}")
    print(f"Predicted image shape: {pred_image.shape}")
    # ssim = compare_ssim(gt_float, pred_float, multichannel=True, data_range=1.0)
    ssim = compare_ssim(gt_float, pred_float, multichannel=True, data_range=1.0, win_size=3)
    
    # LPIPS
    gt_tensor = torch.tensor(gt_float).permute(2, 0, 1).unsqueeze(0).float()
    pred_tensor = torch.tensor(pred_float).permute(2, 0, 1).unsqueeze(0).float()
    lpips_value = loss_fn_alex(gt_tensor, pred_tensor).item()

    return psnr, ssim, lpips_value


def compute_camera_center(images):
    centers = []
    for image in images.values():
        pose = load_extrinsics(image)
        center = pose[:3, 3]
        centers.append(center)
    return np.mean(centers, axis=0)

# Sử dụng hàm
if __name__ == "__main__":
    # SiftExtraction_max_num_features_test_list = [2048, 3072, 4096, 5120, 7168, 8192, 9216, 10240]
    # SiftExtraction_max_image_size_test_list = [1000, 1500, 2500, 3000, 3500, 4000, 4500, 5000, 6000]
    # SiftMatching_max_ratio_test_list = [0.65, 0.7, 0.75, 0.8, 0.9, 0.92, 0.95, 0.98]
    SiftMatching_max_ratio_test_list = [0.87]
    # Mapper_min_num_matches_test_list = [5, 8, 10, 12, 18, 20, 25, 30, 40]
    # PatchMatchStereo_num_samples_test_list = [5, 6, 7, 8, 9, 10, 11, 12, 13]
    # PatchMatchStereo_window_radius_test_list = [1, 3, 4, 5, 6, 7, 8, 9, 10]
    # Mapper_ba_global_max_num_iterations_test_list = [20, 30, 40, 60, 75, 100, 125, 150, 200]   
    # SiftExtraction_estimate_affine_shape_test_list = [0]
    # SiftExtraction_domain_size_pooling_test_list = [0]
    # PatchMatchStereo_geom_consistency = [False]
    result_path = "result1.csv"
    with open(result_path, "a") as f:
        f.write("SiftMatching.max_ratio, Num_points, Runtime, PSNR, SSIM, LPIPS\n")
    for test_parameter in SiftMatching_max_ratio_test_list:
        params["SiftMatching.max_ratio"] = test_parameter

        clean_up()
        total_time = run_colmap_commands()
        num_points = convert_ply_to_obj()
        
        sparse_dir = r"D:\doan\Run_Colmap\VA_test\xe_F1_khan_70\test"
        images_dir = r"D:\doan\Run_Colmap\VA_test\xe_F1_khan_70\images_test"
        ply_file = f"{DATASET_PATH}\\Mesh.obj"
        assert os.path.isdir(sparse_dir)
        assert os.path.isdir(images_dir)
        assert os.path.isfile(ply_file)
        # # # output_dir= "/Volumes/Projects/3drn/data/new_project_xeF1_100/dense/0/rendered_images"
        # # # os.makedirs(output_dir, exist_ok=True)

        print("✅ Loading cameras and images...")
        cameras = read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
        # print('cameras : ', cameras)
        images = read_images_binary(os.path.join(sparse_dir, "images.bin"))
        # print('images: ', images)

        print("✅ Loading mesh...")
        # Load the .ply file
        mesh = trimesh.load(ply_file)

        # Check if the loaded object is a Scene
        if isinstance(mesh, trimesh.Scene):
            # Extract the first geometry from the Scene
            mesh = mesh.dump(concatenate=True)
        # mesh = pyrender.Mesh.from_trimesh(mesh)

        # dịch vật thể về trung tâm
        camera_center = compute_camera_center(images)
        # Move mesh to camera center
        mesh.apply_translation(-camera_center)
        # Optional: scale mesh to better fit view
        scale_factor = 1.0 / np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
        mesh.apply_scale(scale_factor)
        mesh = pyrender.Mesh.from_trimesh(mesh)

        # Assume single camera
        camera = list(cameras.values())[0]
        K, width, height = load_intrinsics(camera)

        loss_fn_alex = lpips.LPIPS(net='alex')

        psnr_list, ssim_list, lpips_list = [], [], []

        for image_name, image in tqdm(images.items(), desc="Rendering and comparing"):
            pose = load_extrinsics(image)
            # dịch camera về giống cloud points
            pose[:3, 3] -= camera_center
            # Optional: scale camera to better fit view
            pose[:3, 3] *= scale_factor

            # Render
            color = render_scene(mesh, K, width, height, pose)
            # output_path = os.path.join(output_dir, os.path.splitext(image.name)[0] + ".png")
            # cv2.imwrite(output_path, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

            # Load ground truth
            gt_path = os.path.join(images_dir, image.name)
            if not os.path.exists(gt_path):
                print(f"⚠️ Ground truth image {gt_path} not found, skipping.")
                continue
            gt_image = cv2.imread(gt_path)
            pred_image = color

            # Compute metrics
            psnr, ssim, lpips_value = compute_metrics(gt_image, pred_image, loss_fn_alex)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            lpips_list.append(lpips_value)

            print(f"{image.name}: PSNR={psnr:.2f}, SSIM={ssim:.3f}, LPIPS={lpips_value:.3f}")

        print("\n✅ Done! Average metrics:")
        # Ghi dữ liệu vào bảng
        print(f"Thời gian chạy: {total_time} giây")
        print(f"Số lượng điểm: {num_points}")
        print(f"PSNR: {np.mean(psnr_list):.2f}")
        print(f"SSIM: {np.mean(ssim_list):.3f}")
        print(f"LPIPS: {np.mean(lpips_list):.3f}")

        with open(result_path, "a") as f:
            f.write(f"{test_parameter}, {num_points}, {int(total_time)}, "
                    f"{np.mean(psnr_list):.2f}, {np.mean(ssim_list):.3f}, {np.mean(lpips_list):.3f}\n")
