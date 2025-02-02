import pyassimp
import open3d as o3d

# Load the .abc file
scene = pyassimp.load('input.abc')

# Extract mesh data from the scene
mesh = scene.meshes[0]  # Assuming there's at least one mesh

# Convert mesh to Open3D format
vertices = mesh.vertices
faces = mesh.faces

# Create an Open3D TriangleMesh
o3d_mesh = o3d.geometry.TriangleMesh()
o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

# Save as .ply
o3d.io.write_triangle_mesh('output.ply', o3d_mesh)

# Cleanup
pyassimp.release(scene)