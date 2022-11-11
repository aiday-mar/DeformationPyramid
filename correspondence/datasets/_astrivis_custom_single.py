import os, sys, glob, torch
sys.path.append("../")
[sys.path.append(i) for i in ['.', '..']]
import numpy as np
import torch
import random
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
import h5py
import open3d as o3d

HMN_intrin = np.array( [443, 256, 443, 250 ])
cam_intrin = np.array( [443, 256, 443, 250 ])


class _AstrivisCustomSingle(Dataset):

    def __init__(self, config, source_file, target_file, matches, source_trans, target_trans):
        # Only one phase the testing phase
        super(_AstrivisCustomSingle, self).__init__()
        self.matches = {}
        self.number_matches = 0
        self.n_files_per_folder = 0
        self.config = config        
        self.path = '/home/aiday.kyzy/dataset/Synthetic/'
        self.source_file = source_file
        self.target_file = target_file
        self.matches = matches
        self.source_trans = source_trans
        self.target_trans = target_trans
                    
    def __len__(self):
        return 1

    def __getitem__(self, index):
        
        src_pcd = o3d.io.read_point_cloud(self.path + self.source_file)
        src_pcd = np.array(src_pcd.points).astype(np.float32)
        tgt_pcd = o3d.io.read_point_cloud(self.path + self.target_file)
        tgt_pcd = np.array(tgt_pcd.points).astype(np.float32)
        
        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)
        
        matches = np.load(self.path + self.matches)
        correspondences = np.array(matches['matches'])
        indices_src = correspondences[:, 0]
        indices_tgt = correspondences[:, 1]
        
        src_pcd_centered = src_pcd - np.mean(src_pcd, axis=0)
        tgt_pcd_centered = tgt_pcd - np.mean(tgt_pcd, axis=0)
        
        src_flow = np.array([src_pcd_centered[i] for i in indices_src])
        tgt_flow = np.array([tgt_pcd_centered[i] for i in indices_tgt])
        
        s2t_flow = tgt_flow - src_flow
        
        src_trans_file=h5py.File(self.path + self.source_trans, "r")
        src_pcd_transform = np.array(src_trans_file['transformation'])

        tgt_trans_file=h5py.File(self.path + self.target_trans, "r")
        tgt_pcd_transform_inverse = np.linalg.inv(np.array(tgt_trans_file['transformation']))
        
        rot = np.matmul(tgt_pcd_transform_inverse[:3, :3], src_pcd_transform[:3, :3])
        trans = tgt_pcd_transform_inverse[:3, :3]@src_pcd_transform[:3, 3] + tgt_pcd_transform_inverse[:3, 3]
        
        trans = np.expand_dims(trans, axis=0)
        trans = trans.transpose()
        
        metric_index = None
        depth_paths = None 
        cam_intrin = None
        
        return src_pcd, tgt_pcd, src_feats, tgt_feats, correspondences, rot.astype(np.float32), trans.astype(np.float32), s2t_flow.astype(np.float32), metric_index, depth_paths, cam_intrin