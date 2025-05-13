import open3d as o3d
import os

# root_dir = "/home/doan/VA_0223/20250411_Final/Reconstruction_3D/output_old/colmap/xe_F1_khan"
# # Load fused point cloud
# pcd = o3d.io.read_point_cloud(os.path.join(root_dir, "dense/fused.ply"))

# # Poisson surface reconstruction
# mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

# # Save as OBJ
# o3d.io.write_triangle_mesh(os.path.join(root_dir, "dense/fused.obj"), mesh)


# # Load the .ply file
# pcd = o3d.io.read_point_cloud(os.path.join(root_dir, "output.ply"))

# # Poisson surface reconstruction
# mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

# # Save as OBJ
# o3d.io.write_triangle_mesh(os.path.join(root_dir, "output.obj"), mesh)

def convert_ply_to_obj(ply_path, obj_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12)
    o3d.io.write_triangle_mesh(obj_path, mesh)

root_dir = "/home/doan/VA_0223/20250411_Final/Reconstruction_3D/output_old/nerf"
convert_ply_to_obj(os.path.join(root_dir, "output.ply"), os.path.join(root_dir, "output.obj"))