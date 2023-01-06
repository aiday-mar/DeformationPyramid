from shape_criterion_boundary_detection import get_shape_criterion_mask
import open3d as o3d
import numpy as np

use_proba = True
# use_proba = False

file_path = 'TestData/PartialDeformed/model002/020_0.ply'
num = '002'
get_shape_criterion_mask(file_path, num, use_proba)

file_path = 'TestData/PartialDeformed/model042/020_0.ply'
num = '042'
get_shape_criterion_mask(file_path, num, use_proba)

file_path = 'TestData/PartialDeformed/model085/020_0.ply'
num = '085'
get_shape_criterion_mask(file_path, num, use_proba)

file_path = 'TestData/PartialDeformed/model126/020_0.ply'
num = '126'
get_shape_criterion_mask(file_path, num, use_proba)

file_path = 'TestData/PartialDeformed/model167/020_0.ply'
num = '167'
get_shape_criterion_mask(file_path, num, use_proba)

file_path = 'TestData/PartialDeformed/model207/020_0.ply'
num = '207'
get_shape_criterion_mask(file_path, num, use_proba)
