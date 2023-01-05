import numpy as np
from scipy.spatial import distance_matrix
import math 
import open3d as o3d

def apply_simple_boundary_detection(file_path, num):
    pcd = o3d.io.read_point_cloud(file_path)
    pcd_points = np.array(pcd.points)
    n_pcd_points = pcd_points.shape[0]
    dists = np.zeros((pcd_points, pcd_points))

    for i in range(n_pcd_points):
        src_point = pcd_points[i]
        difference = pcd_points - src_point
        square_difference = difference ** 2
        square_difference = np.array(square_difference.cpu())
        sum_by_row = np.sum(square_difference, axis=1)
        dists[i, :] = np.sqrt(sum_by_row)

    for i in range(n_pcd_points):
        dists[i][i] = float('inf')

    min_distances = dists.min(axis = 1)
    average_distance = np.average(min_distances)
    neighbors = dists < 1.5 * average_distance
    n_neighbors = np.sum(neighbors, axis=1)
    initial_edge_point_indices = np.argwhere(n_neighbors < 3)
    edge_points = pcd_points[initial_edge_point_indices]
    edge_points_pcd = o3d.geometry.PointCloud()
    edge_points_pcd.points = o3d.utility.Vector3dVector(np.array(edge_points))
    o3d.io.write_point_cloud('Testing/exterior_boundary_detection/simple_criterion_' + num + '.ply', edge_points_pcd)