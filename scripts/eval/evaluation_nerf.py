import os
import torch
import cv2
import lpips
import numpy as np
import pyrender
import trimesh
from tqdm import tqdm
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from read_write_model import read_cameras_binary, read_images_binary
import argparse

# Base paths
BASE_PATH = "/home/doan/VA_0223/20250411_Final/Reconstruction_3D/eval/data/processed/tanks_and_temples"
GT_BASE_PATH = "/home/doan/VA_0223/20250411_Final/Reconstruction_3D/eval/data/tanks_and_temples/gt"  # Added ground truth base path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for computation')
    return parser.parse_args()

# =======================
# Load Camera Intrinsics
# =======================
def load_intrinsics(camera):
    """Load camera intrinsics from COLMAP camera model"""
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
    """Load camera extrinsics from COLMAP image model"""
    R = image.qvec2rotmat()
    t = image.tvec.reshape(3, 1)
    # Colmap: world to camera => need to invert for camera to world
    Rt = np.eye(4)
    Rt[:3, :3] = R
    Rt[:3, 3] = t[:, 0]
    return np.linalg.inv(Rt)

# =======================
# Render Image
# =======================
def render_scene(mesh, K, width, height, pose):
    """Render scene from given camera pose"""
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
def compute_metrics(gt_image, pred_image, loss_fn_alex, use_gpu=False):
    """Compute image quality metrics"""
    # Resize pred_image if needed
    if gt_image.shape != pred_image.shape:
        pred_image = cv2.resize(pred_image, (gt_image.shape[1], gt_image.shape[0]))

    # Convert to float32
    gt_float = gt_image.astype(np.float32) / 255.0
    pred_float = pred_image.astype(np.float32) / 255.0

    # PSNR
    psnr = compare_psnr(gt_float, pred_float, data_range=1.0)

    # SSIM
    ssim = compare_ssim(gt_float, pred_float, multichannel=True, data_range=1.0, win_size=3)
    
    # LPIPS
    try:
        if use_gpu and torch.cuda.is_available():
            gt_tensor = torch.tensor(gt_float).permute(2, 0, 1).unsqueeze(0).float().cuda()
            pred_tensor = torch.tensor(pred_float).permute(2, 0, 1).unsqueeze(0).float().cuda()
        else:
            gt_tensor = torch.tensor(gt_float).permute(2, 0, 1).unsqueeze(0).float()
            pred_tensor = torch.tensor(pred_float).permute(2, 0, 1).unsqueeze(0).float()
        lpips_value = loss_fn_alex(gt_tensor, pred_tensor).item()
    except Exception as e:
        print(f"Error computing LPIPS: {e}")
        lpips_value = 0.0

    return psnr, ssim, lpips_value

def process_scene(scene_name, use_gpu=False):
    """Process a single scene"""
    print(f"\nProcessing scene: {scene_name}")
    
    # Set up paths for this scene
    input_path = os.path.join(BASE_PATH, scene_name, "input")
    output_path = os.path.join(BASE_PATH, scene_name, "output")
    
    # Compute 3D metrics
    gt_mesh_path = os.path.join(GT_BASE_PATH, f"{scene_name}.obj")  # Changed from .ply to .obj
    pred_mesh_path = os.path.join(output_path, "nerf.obj")
    
    if os.path.exists(gt_mesh_path) and os.path.exists(pred_mesh_path):
        print("Computing 3D reconstruction metrics...")
        metrics_3d = compute_3d_metrics(gt_mesh_path, pred_mesh_path, use_gpu=use_gpu)
        print(f"Chamfer Distance: {metrics_3d['chamfer_distance']:.6f}")
        print(f"Hausdorff Distance: {metrics_3d['hausdorff_distance']:.6f}")
    else:
        print("Warning: Ground truth or predicted mesh not found. Skipping 3D metrics.")
        if not os.path.exists(gt_mesh_path):
            print(f"Ground truth mesh not found at: {gt_mesh_path}")
        if not os.path.exists(pred_mesh_path):
            print(f"Predicted mesh not found at: {pred_mesh_path}")
        metrics_3d = None
    
    # Initialize LPIPS
    try:
        if use_gpu and torch.cuda.is_available():
            print("Using GPU for LPIPS computation")
            loss_fn_alex = lpips.LPIPS(net='alex').cuda()
        else:
            print("Using CPU for LPIPS computation")
            loss_fn_alex = lpips.LPIPS(net='alex')
    except Exception as e:
        print(f"Error initializing LPIPS: {e}")
        return None
    
    # Load reconstructed mesh
    # mesh_path = os.path.join(input_path, "dense", "mesh.obj")
    mesh_path = os.path.join(output_path, "nerf.obj")
    if not os.path.exists(mesh_path):
        print(f"Error: Mesh file not found at {mesh_path}")
        return None
    
    mesh = trimesh.load(mesh_path)
    mesh = pyrender.Mesh.from_trimesh(mesh)
    
    # Load COLMAP cameras and images
    sparse_dir = os.path.join(BASE_PATH, scene_name, "sparse", "0")
    if not os.path.exists(sparse_dir):
        print(f"Error: Sparse reconstruction directory not found at {sparse_dir}")
        return None
    
    print("Loading cameras and images...")
    cameras = read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
    images = read_images_binary(os.path.join(sparse_dir, "images.bin"))
    
    # Create output directory for rendered images
    rendered_dir = os.path.join(output_path, "rendered_nerf")
    os.makedirs(rendered_dir, exist_ok=True)
    
    # Initialize metrics
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    num_images = 0
    
    # Get test images directory
    test_images_dir = os.path.join(input_path, "images_test")
    if not os.path.exists(test_images_dir):
        print(f"Error: Test images directory not found at {test_images_dir}")
        return None
    
    # Process each test image
    test_images = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.png'))]
    for image_name in tqdm(test_images, desc="Processing test images"):
        # Find corresponding image in COLMAP reconstruction
        image_id = None
        for img_id, img in images.items():
            if img.name == image_name:
                image_id = img_id
                break
        
        if image_id is None:
            print(f"Warning: No COLMAP data found for test image {image_name}")
            continue
            
        # Load ground truth image
        gt_image_path = os.path.join(test_images_dir, image_name)
        gt_image = cv2.imread(gt_image_path)
        if gt_image is None:
            print(f"Warning: Could not read image at {gt_image_path}")
            continue
            
        # Get camera parameters
        image = images[image_id]
        camera = cameras[image.camera_id]
        K, width, height = load_intrinsics(camera)
        pose = load_extrinsics(image)
        
        # Render image
        rendered_image = render_scene(mesh, K, width, height, pose)
        
        # Save rendered image
        rendered_path = os.path.join(rendered_dir, f"rendered_{image_name}")
        cv2.imwrite(rendered_path, rendered_image)
        
        # Compute metrics
        psnr, ssim, lpips_value = compute_metrics(gt_image, rendered_image, loss_fn_alex, use_gpu)
        
        # Update totals
        total_psnr += psnr
        total_ssim += ssim
        total_lpips += lpips_value
        num_images += 1
        
        # Print per-image metrics
        print(f"\nImage: {image_name}")
        print(f"PSNR: {psnr:.2f}")
        print(f"SSIM: {ssim:.4f}")
        print(f"LPIPS: {lpips_value:.4f}")
    
    # Compute and save metrics
    if num_images > 0:
        avg_psnr = total_psnr / num_images
        avg_ssim = total_ssim / num_images
        avg_lpips = total_lpips / num_images
        
        print("\nAverage Metrics:")
        print(f"PSNR: {avg_psnr:.2f}")
        print(f"SSIM: {avg_ssim:.4f}")
        print(f"LPIPS: {avg_lpips:.4f}")
        
        # Save metrics to file
        metrics_file = os.path.join(output_path, "metrics_nerf.txt")
        with open(metrics_file, "w") as f:
            f.write(f"Average PSNR: {avg_psnr:.2f}\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")
            f.write(f"Average LPIPS: {avg_lpips:.4f}\n")
            f.write(f"Number of test images processed: {num_images}\n")
        
        return {
            "scene": scene_name,
            "psnr": avg_psnr,
            "ssim": avg_ssim,
            "lpips": avg_lpips,
            "num_images": num_images
        }
    else:
        print("No test images were processed successfully")
        return None

def main():
    args = parse_args()
    use_gpu = args.use_gpu and torch.cuda.is_available()
    
    if use_gpu:
        print("Using GPU for computation")
    else:
        print("Using CPU for computation")
    
    # Get all scene directories
    scenes = [d for d in os.listdir(BASE_PATH) 
             if os.path.isdir(os.path.join(BASE_PATH, d)) 
             and d != "output"]
    
    # Process each scene
    all_metrics = []
    for scene in scenes:
        metrics = process_scene(scene, use_gpu)
        if metrics:
            all_metrics.append(metrics)
    
    # Save overall results
    if all_metrics:
        overall_file = os.path.join(BASE_PATH, "overall_metrics_test.txt")
        with open(overall_file, "w") as f:
            f.write("Scene\tPSNR\tSSIM\tLPIPS\tNum_Images\n")
            for metrics in all_metrics:
                f.write(f"{metrics['scene']}\t{metrics['psnr']:.2f}\t{metrics['ssim']:.4f}\t{metrics['lpips']:.4f}\t{metrics['num_images']}\n")
            
            # Compute and write averages
            avg_psnr = sum(m['psnr'] for m in all_metrics) / len(all_metrics)
            avg_ssim = sum(m['ssim'] for m in all_metrics) / len(all_metrics)
            avg_lpips = sum(m['lpips'] for m in all_metrics) / len(all_metrics)
            total_images = sum(m['num_images'] for m in all_metrics)
            
            f.write("\nOverall Averages:\n")
            f.write(f"Average PSNR: {avg_psnr:.2f}\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")
            f.write(f"Average LPIPS: {avg_lpips:.4f}\n")
            f.write(f"Total Test Images Processed: {total_images}\n")

if __name__ == "__main__":
    main()
