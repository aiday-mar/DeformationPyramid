import torch.nn.functional as F
import numpy as np
import os
import torch.nn as nn
import open3d as o3d
import subprocess
import torch

class FCGF(nn.Module):

    def __init__(self, config):
        super(FCGF, self).__init__()

    def forward(self, batch, phase = 'encode'):
        print('Inside of FCGF forward')
        source_pcd = batch['src_pcd_list'][0]
        target_pcd = batch['tgt_pcd_list'][0]
        print(source_pcd)
        print(target_pcd)
        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(np.array(source_pcd.cpu()))
        o3d.io.write_point_cloud('src_pcd.ply', src_pcd)
        tgt_pcd = o3d.geometry.PointCloud()
        tgt_pcd.points = o3d.utility.Vector3dVector(np.array(target_pcd.cpu()))
        o3d.io.write_point_cloud('tgt_pcd.ply', tgt_pcd)
        # TODO: not possible to easily change conda environments
        '''
        command = 'conda init bash'
        os.system(command)
        
        command = 'conda activate py3-fcgf-3'
        os.system(command)
        '''
        
        # subprocess.run('conda init bash & conda activate py3-fcgf-3 && python3 ../sfm/python/vision/features/feature_fcgf_cli.py --input="src_pcd.ply" --input="src_pcd.npz" && conda deactivate', shell=True)

        # command = 'python3 ../sfm/python/vision/features/feature_fcgf_cli.py --input="src_pcd.ply" --input="src_pcd.npz"'
        # os.system(command)
        
        # command = 'python3 ../sfm/python/vision/features/feature_fcgf_cli.py --input="tgt_pcd.ply" --input="tgt_pcd.npz"'
        # os.system(command)
        
        '''
        command = 'conda deactivate'
        os.system(command)
        '''
        
        # TODO: Use for testing purposes pregenerated features
        features_src = np.load('020.npz')
        features_tgt = np.load('104.npz')
        
        print('list(features_src.keys()) : ', list(features_src.keys()))
        features_src = features_src['arr_0']
        features_tgt = features_tgt['arr_0']
        
        features_src = features_src[:133]
        features_tgt = features_tgt[:128]
        coarse_features = np.concatenate((features_src, features_tgt), axis=0)
        return torch.tensor(coarse_features)