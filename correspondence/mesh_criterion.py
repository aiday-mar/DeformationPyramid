import open3d as o3d
import numpy as np
from collections import defaultdict

def get_mesh_criterion_edge_vertices(pcd_points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points)
    o3d.geometry.estimate_normals(pcd_points)
    radii = [0.01, 0.02, 0.04]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    mesh_vertices = np.asarray(mesh.vertices)
    mesh_triangles = np.asarray(mesh.triangles)

    dictionary = defaultdict(int)
    for triangle_indices in mesh_triangles:
        edge1 = [triangle_indices[0], triangle_indices[1]]
        edge2 = [triangle_indices[1], triangle_indices[2]]
        edge3 = [triangle_indices[2], triangle_indices[0]]

        dictionary[tuple(np.sort(edge1))] += 1
        dictionary[tuple(np.sort(edge2))] += 1
        dictionary[tuple(np.sort(edge3))] += 1

    dictionary = dict(filter(lambda elem: elem[1] == 1, dictionary.items()))
    edge_vertices = []
    for edge in dictionary:
        edge_vertices.append(edge[0])
        edge_vertices.append(edge[1])

    edge_vertices = mesh_vertices[edge_vertices]
    return edge_vertices