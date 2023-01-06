import open3d as o3d
import numpy as np
from scipy.spatial import distance_matrix
import math

'''
def gaussian_kernel(sigma, d):
    return math.exp(-d**2/sigma**2)

def find_probability_vector(pcd_points):
    number_points = pcd_points.shape[0]
    dists = distance_matrix(pcd_points, pcd_points)
    k = 10
    probabilities = []
    for i in range(number_points):
        print(i, '/', number_points - 1)
        point = pcd_points[i]
        dists_to_point = dists[i, :]
        indices_neighbors = np.argsort(dists_to_point)[:k]
        neighbor_points = pcd_points[indices_neighbors]
        n_neighbor_points = neighbor_points.shape[0]
        distances_to_neighbors = dists_to_point[indices_neighbors]
        rp = np.mean(distances_to_neighbors)
        sigma = 1/3*rp
        gaussian_values = np.array([gaussian_kernel(sigma, np.linalg.norm(neighbor - point)) for neighbor in neighbor_points])
        sum_gaussian_values = gaussian_values.sum()
        mup = np.array([0., 0., 0.])
        for j in range(n_neighbor_points):
            mup += gaussian_values[j]*neighbor_points[j]
        mup = mup/sum_gaussian_values

        # find tangent plane
        augmented_neighbor_points = np.c_[ neighbor_points, np.ones(n_neighbor_points) ]
        transformed_matrix = augmented_neighbor_points.T @ augmented_neighbor_points
        vals, vects = np.linalg.eig(transformed_matrix)
        plane_coeffs = vects[:, np.argmax(vals)]
        projection_coeff = -plane_coeffs[3] - plane_coeffs[0]*mup[0] - plane_coeffs[1]*mup[1] - plane_coeffs[2]*mup[2]
        projected_mup = np.array([
            mup[0] + projection_coeff * plane_coeffs[0],
            mup[1] + projection_coeff * plane_coeffs[1],
            mup[2] + projection_coeff * plane_coeffs[2]
        ])
        diff = np.linalg.norm(point - projected_mup)
        probability = min([diff/(4*rp/(3 * math.pi)), 1])
        probabilities.append(probability)

    return np.array(probabilities)

pcd = o3d.io.read_point_cloud('TestData/PartialDeformed/model002/020_0.ply')
pcd_points = np.array(pcd.points)
probabilities = find_probability_vector(pcd_points)
n = 200
indices = (-probabilities).argsort()[:n]
edge_points = pcd_points[indices]
edge_points_pcd = o3d.geometry.PointCloud()
edge_points_pcd.points = o3d.utility.Vector3dVector(edge_points)
o3d.io.write_point_cloud('half_disc_criterion.ply', edge_points_pcd)
'''

# Modified version of this code where we do not use the gaussian kernel
def find_indices(pcd_points,n):
    number_points = pcd_points.shape[0]
    dists = distance_matrix(pcd_points, pcd_points)
    k = 10
    differences = []
    for i in range(number_points):
        print(i, '/', number_points - 1)
        point = pcd_points[i]
        dists_to_point = dists[i, :]
        indices_neighbors = np.argsort(dists_to_point)[:k]
        neighbor_points = pcd_points[indices_neighbors]
        n_neighbor_points = neighbor_points.shape[0]

        denominator = 0
        mup = np.array([0., 0., 0.])
        for j in range(n_neighbor_points):
            mup += np.linalg.norm(point - neighbor_points[j])*neighbor_points[j]
            denominator += np.linalg.norm(point - neighbor_points[j])
        mup = mup/denominator

        diff = np.linalg.norm(point - mup)
        differences.append(diff)

    indices = np.array(differences).argsort()[:n]
    return indices

def get_half_disc_criterion_mask(file_path, num):
    n = 1000
    pcd = o3d.io.read_point_cloud(file_path)
    pcd_points = np.array(pcd.points)
    n_pcd_points = pcd_points.shape[0]
    edge_point_indices = find_indices(pcd_points, n)
    edge_points = pcd_points[edge_point_indices]
    print('number of pcd points : ', pcd_points.shape[0])
    print('number of edge points : ', edge_points.shape[0])

    n = 700
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
    o3d.io.write_point_cloud('Testing/exterior_boundary_detection_initial_experimentation/half_disc_criterion_' + num +'.ply', final_pcd)

    half_disc_indices = edge_point_indices[final_edge_point_indices[final_final_edge_point_indices]]
    mask = np.zeros((n_pcd_points,), dtype = bool)
    for index in half_disc_indices:
        mask[index] = True
    
    return mask
