import os
import json
import numpy as np
from read_write_model import read_cameras_binary, read_images_binary
import sqlite3
from pathlib import Path

def get_transform_json(scene_path):
    """Generate transforms.json from COLMAP reconstruction for a specific scene"""
    # Paths
    sparse_path = os.path.join(scene_path, "input", "sparse", "0")
    db_path = os.path.join(scene_path, "input", "database.db")
    output_path = os.path.join(scene_path, "input", "transforms.json")
    
    # Check if required files exist
    if not os.path.exists(sparse_path):
        print(f"Error: Sparse reconstruction not found at {sparse_path}")
        return None
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return None
    
    try:
        # Read COLMAP cameras and images
        cameras = read_cameras_binary(os.path.join(sparse_path, "cameras.bin"))
        images = read_images_binary(os.path.join(sparse_path, "images.bin"))
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get image names from database
        cursor.execute("SELECT image_id, name FROM images")
        image_names = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Get first camera parameters for global settings
        first_camera = next(iter(cameras.values()))
        fx, fy, cx, cy = first_camera.params[:4]
        width, height = int(first_camera.width), int(first_camera.height)
        
        # Calculate camera angles
        camera_angle_x = 2 * np.arctan(width / (2 * fx))
        camera_angle_y = 2 * np.arctan(height / (2 * fy))
        
        # Prepare transforms data
        transforms = {
            "camera_angle_x": float(camera_angle_x),
            "camera_angle_y": float(camera_angle_y),
            "fl_x": float(fx),
            "fl_y": float(fy),
            "k1": 0.0,  # Radial distortion coefficients
            "k2": 0.0,
            "k3": 0.0,
            "k4": 0.0,
            "p1": 0.0,  # Tangential distortion coefficients
            "p2": 0.0,
            "is_fisheye": False,
            "cx": float(cx),
            "cy": float(cy),
            "w": float(width),
            "h": float(height),
            "aabb_scale": 4,
            "frames": []
        }
        
        # Process each image
        for img_id, img in images.items():
            # Get image name
            image_name = image_names.get(img_id, f"image_{img_id}.jpg")
            
            # Get camera pose
            R = img.qvec2rotmat()
            t = img.tvec.reshape(3, 1)
            
            # Convert to camera-to-world transformation
            c2w = np.eye(4)
            c2w[:3, :3] = R
            c2w[:3, 3] = t.reshape(3)
            
            # Convert to NeRF format (OpenGL convention)
            c2w[1:3] *= -1  # Flip y and z axes
            
            # Add frame data
            frame = {
                "file_path": f"images/{image_name}",
                "sharpness": 100.0,  # Default sharpness value
                "transform_matrix": c2w.tolist()
            }
            transforms["frames"].append(frame)
        
        # Sort frames by image name
        transforms["frames"].sort(key=lambda x: x["file_path"])
        
        # Save transforms.json
        with open(output_path, 'w') as f:
            json.dump(transforms, f, indent=2)
        
        print(f"Successfully generated transforms.json for scene {os.path.basename(scene_path)}")
        return transforms
        
    except Exception as e:
        print(f"Error generating transforms.json for scene {os.path.basename(scene_path)}: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()

def process_all_scenes():
    """Process all scenes in the tanks_and_temples directory"""
    base_path = "/home/doan/VA_0223/eval/data/processed/tanks_and_temples"
    
    # Get all scene directories
    scene_dirs = [d for d in os.listdir(base_path) 
                 if os.path.isdir(os.path.join(base_path, d))]
    
    print(f"Found {len(scene_dirs)} scenes to process")
    
    # Process each scene
    for scene_dir in scene_dirs:
        scene_path = os.path.join(base_path, scene_dir)
        print(f"\nProcessing scene: {scene_dir}")
        get_transform_json(scene_path)

if __name__ == "__main__":
    process_all_scenes() 