import os
import cv2
import torch
import lpips
import numpy as np
import open3d as o3d
import pyrender
import trimesh
from tqdm import tqdm
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from read_write_model import read_cameras_binary, read_images_binary

# =======================
# Load Camera Intrinsics
# =======================
# def load_intrinsics(camera):
#     print('camera info: ', camera)
#     fx, fy, cx, cy = camera.params[:4]
#     width, height = int(camera.width), int(camera.height)
#     K = np.array([[fx, 0, cx],
#                   [0, fy, cy],
#                   [0,  0,  1]])
#     return K, width, height
def load_intrinsics(camera):
    model = camera.model

    if model in ['SIMPLE_PINHOLE', 'SIMPLE_RADIAL']:
        f, cx, cy = camera.params[:3]
        fx = fy = f
        if cy < 1e-2:
            print("⚠️ Detected invalid cy (≈ 0), overriding to image center.")
            cy = camera.height / 2
    elif model == 'PINHOLE':
        fx, fy, cx, cy = camera.params[:4]
    else:
        raise ValueError(f"⚠️ Unsupported camera model: {model}")

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])
    width, height = int(camera.width), int(camera.height)

    if abs(cx - width / 2) > width * 0.1 or abs(cy - height / 2) > height * 0.1:
        print("⚠️  Warning: cx, cy lệch xa trung tâm ảnh. Kiểm tra lại nội tại COLMAP hoặc ảnh đã resize.")
        print(f"fx: {fx}  - fy: {fy}  - cx: {cx}  - cy: {cy}")
        print("🎯 Intrinsic matrix K:\n", K)
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

    # light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    # scene.add(light, pose=pose)

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
    # print(lpips_value)
    if lpips_value=='nan':
        lpips_value = 1
    return psnr, ssim, lpips_value

# =======================
# Main
# =======================
def main():
    # sparse_dir = "/Volumes/Projects/3drn/data/new_project_xeF1_100/sparse/0"
    sparse_dir = "/home/doan/VA_0223/20250411_Final/Reconstruction_3D/output/colmap/xe_F1_khan/sparse/0"
    ply_file = "/home/doan/VA_0223/20250411_Final/Reconstruction_3D/output/colmap/xe_F1_khan/Mesh.obj"
    images_dir = "/home/doan/VA_0223/20250411_Final/Reconstruction_3D/data/processed_images/xe_F1_khan"
    output_dir= "/home/doan/VA_0223/20250411_Final/Reconstruction_3D/output/colmap/xe_F1_khan/rendered_images"

    # data_dir = "data"
    # images_dir = os.path.join(data_dir, "images")
    # sparse_dir = os.path.join(data_dir, "sparse/0")
    # ply_file = os.path.join(data_dir, "dense/fused.ply")
    # output_dir = os.path.join(data_dir, "rendered")
    os.makedirs(output_dir, exist_ok=True)

    print("✅ Loading cameras and images...")
    cameras = read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
    images = read_images_binary(os.path.join(sparse_dir, "images.bin"))


    print("✅ Loading mesh...")
    # Load the .ply file
    mesh = trimesh.load(ply_file)


    # Check if the loaded object is a Scene
    if isinstance(mesh, trimesh.Scene):
        # Extract the first geometry from the Scene
        mesh = mesh.dump(concatenate=True)


    print('point3D.bin : ', os.path.join(sparse_dir, "points3D.bin"))
    # mesh = align_mesh_to_sparse(mesh, os.path.join(sparse_dir, "points3D.bin"))
    # mesh = align_mesh_with_colmap(mesh, os.path.join(sparse_dir, "points3D.bin"))

    mesh = pyrender.Mesh.from_trimesh(mesh)
    

    # # Assume single camera
    # camera = list(cameras.values())[0]
    # # camera = list(cameras.values())[3]
    # K, width, height = load_intrinsics(camera)

    loss_fn_alex = lpips.LPIPS(net='alex')

    psnr_list, ssim_list, lpips_list = [], [], []

    for image_name, image in tqdm(images.items(), desc="Rendering and comparing"):
        pose = load_extrinsics(image)

        camera = cameras[image.camera_id]
        K, width, height = load_intrinsics(camera)

        # Render
        color = render_scene(mesh, K, width, height, pose)
        output_path = os.path.join(output_dir, os.path.splitext(image.name)[0] + ".png")
        cv2.imwrite(output_path, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

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
    print(f"PSNR: {np.mean(psnr_list):.2f}")
    print(f"SSIM: {np.mean(ssim_list):.3f}")
    print(f"LPIPS: {np.mean(lpips_list):.3f}")

if __name__ == "__main__":
    main()
