import open3d as o3d
import numpy as np
from scipy.spatial import distance_matrix
import math

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def find_indices(points, n):
    k = 50
    final_edge_point_indices = []
    final_edge_point_angles = []
    dists = distance_matrix(points, points)
    
    for i in range(points.shape[0]):
        print(str(i) + '/' + str(points.shape[0]))
        dists_point = dists[i]
        res = sorted(range(len(dists_point)), key = lambda sub: dists_point[sub])[:k]
        if i == 0:
            print(res)
        neighborhood_points = points[res]
        neighborhood_points_ext = np.c_[ neighborhood_points, np.ones(neighborhood_points.shape[0]) ]
        matrix_multiplication = neighborhood_points_ext.T @ neighborhood_points_ext
        w, v = np.linalg.eigh(matrix_multiplication)
        plane_coeffs = v[:, np.argmax(w)]
        
        if i == 0:
            print(plane_coeffs)
        
        projected_neighborhood_points = np.zeros(neighborhood_points.shape)
        
        for j in range(neighborhood_points.shape[0]):
            neighborhood_point = neighborhood_points[j, :]
            coeff_k = (-plane_coeffs[3] - plane_coeffs[0]*neighborhood_point[0] - plane_coeffs[1]*neighborhood_point[1] - plane_coeffs[2]*neighborhood_point[2])/(plane_coeffs[0]**2 + plane_coeffs[1]**2 + plane_coeffs[2]**2)
            projected_neighborhood_points[j, :] = np.array([neighborhood_point[0] + coeff_k*plane_coeffs[0], neighborhood_point[1] + coeff_k*plane_coeffs[1], neighborhood_point[2] + coeff_k*plane_coeffs[2]])
        
        projected_neighborhood_points = np.array(projected_neighborhood_points)
        
        if i == 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(projected_neighborhood_points)
            colors = np.zeros(projected_neighborhood_points.shape)
            
            colors[0, :] = np.array([1, 0, 0])
            for l in range(1, projected_neighborhood_points.shape[0]):
                colors[l, :] = np.array([0, 1, 0])
                
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        vectors_from_center = np.zeros((projected_neighborhood_points.shape[0] - 1, 3))
        for j in range(vectors_from_center.shape[0]):
            vectors_from_center[j, :] = projected_neighborhood_points[j + 1, :] - projected_neighborhood_points[0, :]
            
        angles = np.zeros((vectors_from_center.shape[0],))
        
        for j in range(1, vectors_from_center.shape[0]):
            angles[j] = angle_between(vectors_from_center[0, :], vectors_from_center[j, :])

        indices_sort = np.argsort(angles)
        sorted_vectors_from_center = vectors_from_center.take(indices_sort, 0)
        
        final_angles = np.zeros((sorted_vectors_from_center.shape[0] - 1, ))
        for j in range(sorted_vectors_from_center.shape[0] - 1):
            final_angles[j] = angle_between(sorted_vectors_from_center[j], sorted_vectors_from_center[j + 1])
        
        max_angle = final_angles.max()
        final_edge_point_angles.append(max_angle)

    final_edge_point_angles = np.array(final_edge_point_angles)
    final_edge_point_indices = (-final_edge_point_angles).argsort()[:n]
    return final_edge_point_indices

def get_angle_criterion_mask(file_path, num):
    n = 2000
    pcd = o3d.io.read_point_cloud(file_path)
    pcd_points = np.array(pcd.points)
    edge_point_indices = find_indices(pcd_points, n)
    edge_points = pcd_points[edge_point_indices]
    print('number of pcd points : ', pcd_points.shape[0])
    print('number of edge points : ', edge_points.shape[0])

    n = 1000
    final_edge_point_indices = find_indices(edge_points, n)
    final_edge_points = edge_points[final_edge_point_indices]
    print('number of edge points : ', edge_points.shape[0])
    print('number of final edge points : ', final_edge_points.shape[0])

    n = 500
    final_final_edge_point_indices = find_indices(final_edge_points, n)
    final_final_edge_points = final_edge_points[final_final_edge_point_indices]
    print('number of final edge points : ', final_edge_points.shape[0])
    print('number of final final edge points : ', final_final_edge_points.shape[0])
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(np.array(final_final_edge_points))
    o3d.io.write_point_cloud('Testing/exterior_boundary_detection/angle_criterion_' + num + '.ply', final_pcd)

    angle_indices = edge_point_indices[final_edge_point_indices[final_final_edge_point_indices]]
    n_pcd_points = pcd_points.shape[0]
    mask = np.zeros((n_pcd_points,), dtype = bool)
    for index in angle_indices:
        mask[index] = True
    
    return mask