import numpy as np
from scipy.spatial import distance_matrix
import math 
import open3d as o3d

characteristic_equations = {
    'boundary' : np.array([2/3, 1/3, 0]),
    'interior' : np.array([1/2, 1/2, 0]),
    'corner' : np.array([1/3, 1/3, 1/3]),
    'line' : np.array([1, 0, 0])
}

def gaussian_kernel(sigma, d):
    return math.exp(-d**2/sigma**2)

def find_indices(pcd_points, n):
    probabilities = []
    norms = []
    pi_tildas = {}
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

        covariance_matrix = np.zeros((3, 3))
        for neighbor_point in neighbor_points:
            covariance_matrix += (mup - neighbor_point)@(mup - neighbor_point).T
        
        eigenvalues, _ = np.linalg.eig(covariance_matrix)
        eigenvalues = sorted(eigenvalues, reverse=True)
        
        sum_eigenvalues = np.sum(eigenvalues)
        centroid_t_lambda = 1/3*(np.array([1/2, 1/2, 0]) + np.array([1/3, 1/3, 1/3]) + np.array([1, 0, 0]))
        lambdap = eigenvalues/sum_eigenvalues
        
        sum_pi_tildas = 0
        for type in characteristic_equations:
            characteristic_equation = characteristic_equations[type]
            sigma_phi = 1/3 * np.linalg.norm(characteristic_equation - centroid_t_lambda)**2
            pi_tilda = gaussian_kernel(sigma_phi, np.linalg.norm(lambdap - characteristic_equation))
            pi_tildas[type] = pi_tilda
            sum_pi_tildas += pi_tilda
        
        probability = pi_tildas['boundary']/sum_pi_tildas
        probabilities.append(probability)
        norms.append(np.linalg.norm(lambdap - characteristic_equations['boundary']))

    indices_proba = (-np.array(probabilities)).argsort()[:n]
    indices_norm = np.array(probabilities).argsort()[:n]

    return indices_proba, indices_norm

pcd = o3d.io.read_point_cloud('TestData/PartialDeformed/model002/020_0.ply')
pcd_points = np.array(pcd.points)

# Using the indices by looking at the norm
n = 2000
pcd = o3d.io.read_point_cloud('TestData/PartialDeformed/model002/020_0.ply')
pcd_points = np.array(pcd.points)
indices_proba, indices_norm = find_indices(pcd_points, n)
edge_points = pcd_points[indices_norm]
print('number of pcd points : ', pcd_points.shape[0])
print('number of edge points : ', edge_points.shape[0])

n = 1000
indices_proba, indices_norm = find_indices(edge_points, n)
final_edge_points = edge_points[indices_norm]
print('number of edge points : ', edge_points.shape[0])
print('number of final edge points : ', final_edge_points.shape[0])

n = 500
indices_proba, indices_norm = find_indices(final_edge_points, n)
final_final_edge_points = final_edge_points[indices_norm]
print('number of final edge points : ', final_edge_points.shape[0])
print('number of final final edge points : ', final_final_edge_points.shape[0])

final_pcd = o3d.geometry.PointCloud()
final_pcd.points = o3d.utility.Vector3dVector(np.array(final_final_edge_points))
o3d.io.write_point_cloud('shape_criterion_norms.ply', final_pcd)