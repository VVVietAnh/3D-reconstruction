import open3d as o3d

# Load fused point cloud
pcd = o3d.io.read_point_cloud("dense/fused.ply")

# Poisson surface reconstruction
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

# Save as OBJ
o3d.io.write_triangle_mesh("output.obj", mesh)