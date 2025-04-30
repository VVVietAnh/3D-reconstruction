import numpy as np
import open3d as o3d
import argparse
import time
import trimesh

def load_mesh(file_path):
    """Load mesh file (PLY or OBJ) and convert to point cloud."""
    if file_path.endswith('.ply'):
        return o3d.io.read_point_cloud(file_path)
    elif file_path.endswith('.obj'):
        # Load mesh using trimesh
        mesh = trimesh.load(file_path)
        # Sample points from mesh surface
        points = mesh.sample(100000)  # Adjust number of points as needed
        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

class PointCloudMetrics:
    def __init__(self, pcd1, pcd2):
        """
        Initialize point clouds for metrics computation.
        
        Args:
            pcd1: Open3D point cloud (ground truth)
            pcd2: Open3D point cloud (prediction)
        """
        print("Initializing point clouds...")
        start_time = time.time()
        
        self.pcd1 = pcd1  # ground truth
        self.pcd2 = pcd2  # prediction
        
        # Build KD-trees using Open3D
        self.pcd1_tree = o3d.geometry.KDTreeFlann(pcd1)
        self.pcd2_tree = o3d.geometry.KDTreeFlann(pcd2)
        
        print(f"Initialization completed in {time.time() - start_time:.2f} seconds")

    def chamfer_distance(self):
        """Calculate Chamfer Distance using Open3D."""
        dist1 = np.asarray(self.pcd1.compute_point_cloud_distance(self.pcd2))
        dist2 = np.asarray(self.pcd2.compute_point_cloud_distance(self.pcd1))
        
        chamfer_dist = (np.mean(dist1) + np.mean(dist2)) / 2.0
        return chamfer_dist

    def hausdorff_distance(self):
        """Calculate Hausdorff Distance using Open3D."""
        dist1 = np.asarray(self.pcd1.compute_point_cloud_distance(self.pcd2))
        dist2 = np.asarray(self.pcd2.compute_point_cloud_distance(self.pcd1))
        
        return max(np.max(dist1), np.max(dist2))

    def compute_all_metrics(self):
        """Compute all metrics at once."""
        cd = self.chamfer_distance()
        hd = self.hausdorff_distance()
        
        return {
            'chamfer_distance': cd,
            'hausdorff_distance': hd
        }

def evaluate_point_clouds(gt_path, pred_path):
    """Evaluate predicted point cloud against ground truth."""
    print(f"\nEvaluating: {pred_path}")
    print(f"Loading point clouds...")
    start_time = time.time()
    
    gt_pcd = load_mesh(gt_path)
    pred_pcd = load_mesh(pred_path)
    print(f"Point clouds loaded in {time.time() - start_time:.2f} seconds")
    
    print(f"Ground truth points: {len(np.asarray(gt_pcd.points))}")
    print(f"Predicted points: {len(np.asarray(pred_pcd.points))}")
    
    metrics_calculator = PointCloudMetrics(gt_pcd, pred_pcd)
    
    print("\nCalculating metrics...")
    start_time = time.time()
    results = metrics_calculator.compute_all_metrics()
    print(f"Metrics calculated in {time.time() - start_time:.2f} seconds")
    
    print("\nResults:")
    print(f"Chamfer Distance: {results['chamfer_distance']:.6f}")
    print(f"Hausdorff Distance: {results['hausdorff_distance']:.6f}")
    
    return results

def main():
    default_gt = "/home/doan/VA_0223/20250411_Final/Reconstruction_3D/eval/data/2T_3D/gt/Courthouse.ply"
    colmap_pred = "/home/doan/VA_0223/20250411_Final/Reconstruction_3D/eval/data/2T_3D/Courthouse/fused.ply"
    nerf_pred = "/home/doan/VA_0223/20250411_Final/Reconstruction_3D/eval/data/2T_3D/Courthouse/nerf.obj"
    
    parser = argparse.ArgumentParser(description='Evaluate 3D point clouds using multiple metrics')
    parser.add_argument('--gt', default=default_gt, help='Path to ground truth PLY file')
    parser.add_argument('--output', help='Path to save results as numpy file')
    
    args = parser.parse_args()
    
    # Evaluate both predictions
    results = {
        'colmap': evaluate_point_clouds(args.gt, colmap_pred),
        'nerf': evaluate_point_clouds(args.gt, nerf_pred)
    }
    
    # Print comparison
    print("\n=== Comparison Summary ===")
    print("default gt: ", default_gt)
    metrics = ['chamfer_distance', 'hausdorff_distance']
    
    print("\nMetric\t\tColmap\t\tNeRF")
    print("-" * 50)
    for metric in metrics:
        print(f"{metric}:\t{results['colmap'][metric]:.6f}\t{results['nerf'][metric]:.6f}")
    
    # Save results if output path is provided
    if args.output:
        np.save(args.output, results)
        print(f"\nResults saved to {args.output}")

if __name__ == '__main__':
    main()