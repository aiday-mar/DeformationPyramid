import open3d as o3d
import numpy as np
from collections import defaultdict

surface_reconstruction='ball_pivoting'
# surface_reconstruction='poisson'
# surface_reconstruction='alpha'

def mesh_criterion_boundary_detection(filename, num):
    pcd = o3d.io.read_point_cloud(filename)
    if surface_reconstruction=='ball_pivoting':
        radii = [0.01, 0.02, 0.04]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    elif surface_reconstruction=='poisson':
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    elif surface_reconstruction=='alpha':
        alpha = 0.05
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    else:
        raise Exception('Specify a valid reconstruction method')
    o3d.io.write_triangle_mesh('Testing/exterior_boundary_detection_initial_experimentation/model' + num + '_mesh_' + surface_reconstruction + '.ply', mesh)
    print(mesh)
    mesh_vertices = np.asarray(mesh.vertices)
    mesh_triangles = np.asarray(mesh.triangles)

    # looking at the number of times the vertices are used
    '''
    dictionary = defaultdict(int)
    for triangle_indices in mesh_triangles:
        for index in triangle_indices:
            dictionary[index] += 1

    dictionary = dict(filter(lambda elem: elem[1] <= 4, dictionary.items()))
    edge_vertices = []
    for vertex_index in dictionary:
        edge_vertices.append(mesh_vertices[vertex_index])
    edge_vertices = np.asarray(edge_vertices)
    '''

    # looking at how many times the edges appear
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
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(edge_vertices)
    o3d.io.write_point_cloud('Testing/exterior_boundary_detection_initial_experimentation/mesh_criterion_' + num + '_' + surface_reconstruction + '.ply', final_pcd)

'''
filename='Testing/exterior_boundary_detection_initial_experimentation/model002.ply'
num='002'
mesh_criterion_boundary_detection(filename, num)

filename='Testing/exterior_boundary_detection_initial_experimentation/model042.ply'
num='042'
mesh_criterion_boundary_detection(filename, num)

filename='Testing/exterior_boundary_detection_initial_experimentation/model085.ply'
num='085'
mesh_criterion_boundary_detection(filename, num)

filename='Testing/exterior_boundary_detection_initial_experimentation/model126.ply'
num='126'
mesh_criterion_boundary_detection(filename, num)

filename='Testing/exterior_boundary_detection_initial_experimentation/model167.ply'
num='167'
mesh_criterion_boundary_detection(filename, num)
'''

filename='Testing/exterior_boundary_detection_initial_experimentation/model207.ply'
num='207'
mesh_criterion_boundary_detection(filename, num)