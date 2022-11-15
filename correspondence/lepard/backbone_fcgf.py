import torch.nn.functional as F
import numpy as np
import os
import torch.nn as nn
import open3d as o3d

class FCGF(nn.Module):

    def __init__(self, config):
        super(FCGF, self).__init__()

    def forward(self, batch, phase = 'encode'):
        print('Inside of FCGF forward')
        source_pcd = batch['src_pcd_list'][0]
        target_pcd = batch['tgt_pcd_list'][0]
        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = source_pcd
        o3d.io.write_point_cloud('src_pcd.ply', src_pcd)
        tgt_pcd = o3d.geometry.PointCloud()
        tgt_pcd.points = target_pcd
        o3d.io.write_point_cloud('tgt_pcd.ply', tgt_pcd)
        
        command = 'python3 ../../../sfm/python/vision/features/feature_fcgf_cli.py --input="src_pcd.ply" --input="src_pcd.npz"'
        os.system(command)
        
        command = 'python3 ../../../sfm/python/vision/features/feature_fcgf_cli.py --input="tgt_pcd.ply" --input="tgt_pcd.npz"'
        os.system(command)