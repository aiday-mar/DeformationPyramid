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


class _AstrivisCustom(Dataset):

    def __init__(self, config, split):
        super(_AstrivisCustom, self).__init__()

        assert split in ['train','val','test']

        self.matches = {}
        self.number_matches = 0
        self.n_files_per_folder = 0
        self.config = config
        
        n_files_per_folder_found = False
        
        for folder in os.listdir('/home/aiday.kyzy/dataset/TrainingDataDeformedFinal'):
            self.matches[folder] = []
            for filename in os.listdir('/home/aiday.kyzy/dataset/TrainingDataDeformedFinal/' + folder + '/matches'):
                self.matches[folder].append(filename)
                self.number_matches += 1
                if not n_files_per_folder_found:
                    self.n_files_per_folder += 1
            
            n_files_per_folder_found = True
    
        print('self.n_files_per_folder : ', self.n_files_per_folder)
        print('number folders : ', len(self.matches))
        
    def __len__(self):
        # Removing one in order to avoid indexing problems on the last index
        return self.number_matches


    def __getitem__(self, index):

        folder_number = index // self.n_files_per_folder
        idx_inside_folder = index % self.n_files_per_folder
        
        print('folder_number : ', folder_number)
        print('idx_inside_folder : ', idx_inside_folder)
        
        folder_string = 'model' + str(folder_number).zfill(3)
        files_array = self.matches[folder_string]
        filename = files_array[idx_inside_folder]
                
        file_pointers = filename[:-4]
        file_pointers = file_pointers.split('_')
        print('file_pointers : ', file_pointers)
        
        src_pcd_file = file_pointers[0] + '_' + file_pointers[2] + '.ply'
        tgt_pcd_file = file_pointers[1] + '_' + file_pointers[3] + '.ply'
        
        src_pcd = o3d.io.read_point_cloud('/home/aiday.kyzy/dataset/TrainingDataDeformedFinal/' + folder_string + '/transformed/' + src_pcd_file)
        src_pcd = np.array(src_pcd.points)
        tgt_pcd = o3d.io.read_point_cloud('/home/aiday.kyzy/dataset/TrainingDataDeformedFinal/' + folder_string + '/transformed/' + tgt_pcd_file)
        tgt_pcd = np.array(tgt_pcd.points)
        
        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)
        
        matches = np.load('/home/aiday.kyzy/dataset/TrainingDataDeformedFinal/' + folder_string + '/matches/' + filename)
        correspondences = np.array(matches['matches'])
        indices_src = correspondences[:, 0]
        indices_tgt = correspondences[:, 1]
        src_flow = np.array([src_pcd[i] for i in indices_src])
        tgt_flow = np.array([tgt_pcd[i] for i in indices_tgt])
        
        s2t_flow = tgt_flow - src_flow
        
        src_pcd_trans = file_pointers[0] + '_' + file_pointers[2] + '_se4.h5'
        tgt_pcd_trans = file_pointers[1] + '_' + file_pointers[3] + '_se4.h5'
        
        src_trans_file=h5py.File('/home/aiday.kyzy/dataset/TrainingDataDeformedFinal/' + folder_string + '/transformed/' + src_pcd_trans, "r")
        src_pcd_transform = np.array(src_trans_file['transformation'])
        
        tgt_trans_file=h5py.File('/home/aiday.kyzy/dataset/TrainingDataDeformedFinal/' + folder_string + '/transformed/' + tgt_pcd_trans, "r")
        tgt_pcd_transform = np.array(tgt_trans_file['transformation'])
        print('src_pcd_transform : ', tgt_pcd_transform)
        
        final_transform = np.dot(src_pcd_transform, np.linalg.inv(tgt_pcd_transform))
        rot = final_transform[:3, :3]
        trans = final_transform[:3, 3]
        
        metric_index = None
        depth_paths = None 
        cam_intrin = None
        
        print('src_pcd : ', src_pcd)
        print('src_pcd length : ', len(src_pcd))
        print('tgt_pcd : ', tgt_pcd)
        print('tgt_pcd length : ', len(tgt_pcd))
        print('src_feats : ', src_feats)
        print('tgt_feats : ', tgt_feats)
        print('correspondences : ', correspondences)
        print('correspondences length : ', len(correspondences))
        print('rot : ', rot)
        print('trans : ', trans)
        print('s2t_flow : ', s2t_flow)
        print('s2t_flow length : ', len(s2t_flow))
        print('metric_index : ', metric_index)
        print('depth_paths : ', depth_paths)
        print('cam_intrin : ', cam_intrin)
        return src_pcd, tgt_pcd, src_feats, tgt_feats, correspondences, rot, trans, s2t_flow, metric_index, depth_paths, cam_intrin



