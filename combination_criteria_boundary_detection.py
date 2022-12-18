from angle_criterion_boundary_detection import get_angle_criterion_mask 
from shape_criterion_boundary_detection import get_shape_criterion_mask
from half_disc_criterion_boundary_detection import get_half_disc_criterion_mask 
import open3d as o3d
import numpy as np

file_path = 'TestData/PartialDeformed/model002/020_0.ply'
angle_criterion_mask = get_angle_criterion_mask(file_path)
shape_criterion_mask = get_shape_criterion_mask(file_path)
half_disc_criterion_mask = get_half_disc_criterion_mask(file_path)

mask = angle_criterion_mask & shape_criterion_mask & half_disc_criterion_mask
pcd = o3d.io.read_point_cloud(file_path)
pcd_points = np.array(pcd.points)
final_pcd = o3d.geometry.PointCloud()
final_points = pcd_points[mask]
final_pcd.points = o3d.utility.Vector3dVector(final_points)
o3d.io.write_point_cloud('combination_criteria.ply', final_pcd)
