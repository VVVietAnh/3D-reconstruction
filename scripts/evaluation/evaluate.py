#!/usr/bin/env python3
import os
import argparse
import numpy as np
import open3d as o3d
import trimesh
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import torch
from PIL import Image

def load_point_cloud(path):
    """Load point cloud from file."""
    if path.endswith('.ply'):
        pcd = o3d.io.read_point_cloud(path)
        return np.asarray(pcd.points)
    elif path.endswith('.obj'):
        mesh = trimesh.load(path)
        return np.array(mesh.vertices)
    else:
        raise ValueError(f"Unsupported file format: {path}")

def compute_chamfer_distance(points1, points2):
    """Compute Chamfer distance between two point clouds."""
    # Compute distances from points1 to points2
    dist1 = np.min(np.sum((points1[:, np.newaxis] - points2) ** 2, axis=2), axis=1)
    # Compute distances from points2 to points1
    dist2 = np.min(np.sum((points2[:, np.newaxis] - points1) ** 2, axis=2), axis=1)
    # Return average Chamfer distance
    return (np.mean(dist1) + np.mean(dist2)) / 2

def compute_hausdorff_distance(points1, points2):
    """Compute Hausdorff distance between two point clouds."""
    # Compute distances from points1 to points2
    dist1 = np.min(np.sum((points1[:, np.newaxis] - points2) ** 2, axis=2), axis=1)
    # Compute distances from points2 to points1
    dist2 = np.min(np.sum((points2[:, np.newaxis] - points1) ** 2, axis=2), axis=1)
    # Return maximum Hausdorff distance
    return max(np.max(dist1), np.max(dist2))

def compute_f_score(points1, points2, threshold):
    """Compute F-score between two point clouds."""
    # Compute precision
    dist1 = np.min(np.sum((points1[:, np.newaxis] - points2) ** 2, axis=2), axis=1)
    precision = np.mean(dist1 <= threshold)
    
    # Compute recall
    dist2 = np.min(np.sum((points2[:, np.newaxis] - points1) ** 2, axis=2), axis=1)
    recall = np.mean(dist2 <= threshold)
    
    # Compute F-score
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)

def compute_image_metrics(img1_path, img2_path):
    """Compute image quality metrics."""
    # Load images
    img1 = np.array(Image.open(img1_path).convert('RGB'))
    img2 = np.array(Image.open(img2_path).convert('RGB'))
    
    # Compute PSNR
    psnr_value = psnr(img1, img2)
    
    # Compute SSIM
    ssim_value = ssim(img1, img2, channel_axis=2)
    
    # Compute LPIPS
    loss_fn = lpips.LPIPS(net='alex')
    img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    lpips_value = loss_fn(img1_tensor, img2_tensor).item()
    
    return {
        'psnr': psnr_value,
        'ssim': ssim_value,
        'lpips': lpips_value
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate 3D reconstruction quality')
    parser.add_argument('--pred_path', type=str, required=True,
                        help='Path to predicted reconstruction')
    parser.add_argument('--gt_path', type=str, required=True,
                        help='Path to ground truth reconstruction')
    parser.add_argument('--pred_img_path', type=str,
                        help='Path to predicted rendered image')
    parser.add_argument('--gt_img_path', type=str,
                        help='Path to ground truth image')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for saving evaluation results')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load point clouds
    pred_points = load_point_cloud(args.pred_path)
    gt_points = load_point_cloud(args.gt_path)
    
    # Compute 3D metrics
    chamfer_dist = compute_chamfer_distance(pred_points, gt_points)
    hausdorff_dist = compute_hausdorff_distance(pred_points, gt_points)
    f_score = compute_f_score(pred_points, gt_points, threshold=0.01)
    
    # Save 3D metrics
    metrics_3d = {
        'chamfer_distance': chamfer_dist,
        'hausdorff_distance': hausdorff_dist,
        'f_score': f_score
    }
    
    # Compute image metrics if images are provided
    metrics_img = {}
    if args.pred_img_path and args.gt_img_path:
        metrics_img = compute_image_metrics(args.pred_img_path, args.gt_img_path)
    
    # Combine all metrics
    all_metrics = {**metrics_3d, **metrics_img}
    
    # Save metrics to file
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        for metric, value in all_metrics.items():
            f.write(f'{metric}: {value}\n')
    
    print("Evaluation completed. Results saved to:", os.path.join(args.output_dir, 'metrics.txt'))

if __name__ == '__main__':
    main() 