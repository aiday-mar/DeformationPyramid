from angle_criterion_boundary_detection import get_angle_criterion_mask
import open3d as o3d
import numpy as np

use_proba = True
# use_proba = False

file_path = 'TestData/PartialDeformed/model002/020_0.ply'
num = '002'
get_angle_criterion_mask(file_path, num, use_proba)

file_path = 'TestData/PartialDeformed/model042/020_0.ply'
num = '042'
get_angle_criterion_mask(file_path, num, use_proba)

file_path = 'TestData/PartialDeformed/model085/020_0.ply'
num = '085'
get_angle_criterion_mask(file_path, num, use_proba)

file_path = 'TestData/PartialDeformed/model126/020_0.ply'
num = '126'
get_angle_criterion_mask(file_path, num, use_proba)

file_path = 'TestData/PartialDeformed/model167/020_0.ply'
num = '167'
get_angle_criterion_mask(file_path, num, use_proba)

file_path = 'TestData/PartialDeformed/model207/020_0.ply'
num = '207'
get_angle_criterion_mask(file_path, num, use_proba)
