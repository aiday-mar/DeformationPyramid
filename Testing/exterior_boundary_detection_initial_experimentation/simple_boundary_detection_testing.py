from angle_criterion_boundary_detection import get_angle_criterion_mask 
from shape_criterion_boundary_detection import get_shape_criterion_mask
from half_disc_criterion_boundary_detection import get_half_disc_criterion_mask
from simple_boundary_detection import apply_simple_boundary_detection
import open3d as o3d
import numpy as np

file_path = 'TestData/PartialDeformed/model002/020_0.ply'
num = '020'
apply_simple_boundary_detection(file_path, num)

file_path = 'TestData/PartialDeformed/model042/020_0.ply'
num = '042'
apply_simple_boundary_detection(file_path, num)

file_path = 'TestData/PartialDeformed/model085/020_0.ply'
num = '085'
apply_simple_boundary_detection(file_path, num)

file_path = 'TestData/PartialDeformed/model126/020_0.ply'
num = '126'
apply_simple_boundary_detection(file_path, num)

file_path = 'TestData/PartialDeformed/model167/020_0.ply'
num = '167'
apply_simple_boundary_detection(file_path, num)

file_path = 'TestData/PartialDeformed/model207/020_0.ply'
num = '207'
apply_simple_boundary_detection(file_path, num)