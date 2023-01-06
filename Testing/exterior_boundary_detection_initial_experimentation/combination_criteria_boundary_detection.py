from angle_criterion_boundary_detection import get_angle_criterion_mask 
from shape_criterion_boundary_detection import get_shape_criterion_mask
from half_disc_criterion_boundary_detection import get_half_disc_criterion_mask
from simple_boundary_detection import apply_simple_boundary_detection
import open3d as o3d
import numpy as np

def apply_different_criteria(file_path, num):
    angle_criterion_mask = get_angle_criterion_mask(file_path, num)
    shape_criterion_mask = get_shape_criterion_mask(file_path, num)
    half_disc_criterion_mask = get_half_disc_criterion_mask(file_path, num)

    mask = angle_criterion_mask & shape_criterion_mask & half_disc_criterion_mask
    pcd = o3d.io.read_point_cloud(file_path)
    pcd_points = np.array(pcd.points)
    final_pcd = o3d.geometry.PointCloud()
    final_points = pcd_points[mask]
    final_pcd.points = o3d.utility.Vector3dVector(final_points)
    o3d.io.write_point_cloud('Testing/exterior_boundary_detection_initial_experimentation/combination_criteria_' + num + '.ply', final_pcd)

   
file_path = 'TestData/PartialDeformed/model002/020_0.ply'
num = '002'
apply_different_criteria(file_path, num)
apply_simple_boundary_detection(file_path, num)

file_path = 'TestData/PartialDeformed/model042/020_0.ply'
num = '042'
apply_different_criteria(file_path, num)
apply_simple_boundary_detection(file_path, num)

file_path = 'TestData/PartialDeformed/model085/020_0.ply'
num = '085'
apply_different_criteria(file_path, num)
apply_simple_boundary_detection(file_path, num)

file_path = 'TestData/PartialDeformed/model126/020_0.ply'
num = '126'
apply_different_criteria(file_path, num)
apply_simple_boundary_detection(file_path, num)

file_path = 'TestData/PartialDeformed/model167/020_0.ply'
num = '167'
apply_different_criteria(file_path, num)
apply_simple_boundary_detection(file_path, num)

file_path = 'TestData/PartialDeformed/model207/020_0.ply'
num = '207'
apply_different_criteria(file_path, num)
apply_simple_boundary_detection(file_path, num)