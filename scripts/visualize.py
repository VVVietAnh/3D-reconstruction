import os
import numpy as np
import open3d as o3d
from read_write_model import read_images_binary, read_cameras_binary

def load_camera_frustum(K, width, height, scale=0.2):
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

    # Camera frustum points in camera space
    points = [
        [0, 0, 0],  # camera center
        [(0 - cx) / fx, (0 - cy) / fy, 1.0],
        [(width - cx) / fx, (0 - cy) / fy, 1.0],
        [(width - cx) / fx, (height - cy) / fy, 1.0],
        [(0 - cx) / fx, (height - cy) / fy, 1.0],
    ]
    points = np.array(points) * scale

    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1],
    ]

    colors = [[1, 0, 0] for _ in lines]  # red frustum
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def create_camera_geometry(pose, K, width, height, scale=0.2):
    frustum = load_camera_frustum(K, width, height, scale)
    frustum.transform(pose)
    return frustum

def main_mesh():
    # === PATHS ===
    sparse_dir = "/home/doan/VA_0223/20250411_Final/Reconstruction_3D/output/colmap/xe_F1_khan/sparse"
    mesh_path = "/home/doan/VA_0223/20250411_Final/Reconstruction_3D/output/colmap/xe_F1_khan/Mesh.obj"

    # === LOAD COLMAP ===
    cameras = read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
    images = read_images_binary(os.path.join(sparse_dir, "images.bin"))
    camera = list(cameras.values())[0]
    fx, fy, cx, cy = camera.params[:4]
    width, height = int(camera.width), int(camera.height)
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])

    # === LOAD MESH ===
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    # === CREATE CAMERA GEOMETRY ===
    geometries = [mesh]
    for image in images.values():
        R = image.qvec2rotmat()
        t = image.tvec.reshape(3, 1)
        Rt = np.eye(4)
        Rt[:3, :3] = R
        Rt[:3, 3] = t[:, 0]

        cam_to_world = np.linalg.inv(Rt)

        # Flip to Open3D (z forward, y up)
        flip_transform = np.eye(4)
        flip_transform[1, 1] = -1
        flip_transform[2, 2] = -1
        cam_to_world = cam_to_world @ flip_transform

        frustum = create_camera_geometry(cam_to_world, K, width, height, scale=0.3)
        geometries.append(frustum)

    # === VISUALIZE ===
    o3d.visualization.draw_geometries(geometries)

def main_point_cloud():
    mesh_path = "/home/doan/VA_0223/20250411_Final/Reconstruction_3D/output_old/colmap/xe_F1_khan/dense/fused.ply"

    mesh = o3d.io.read_triangle_mesh(mesh_path)

    # Kiểm tra xem mesh có hợp lệ không
    if not mesh.is_empty():
        print("Mesh is loaded successfully.")
    else:
        print("Failed to load mesh.")

    # Hiển thị mesh
    o3d.visualization.draw_geometries([mesh])

if __name__ == "__main__":
    main_point_cloud()