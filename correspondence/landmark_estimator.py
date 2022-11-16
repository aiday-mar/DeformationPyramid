import torch
import yaml
from easydict import EasyDict as edict

import sys
import open3d as o3d
import numpy as np
import os

sys.path.append("")
from correspondence.lepard.pipeline import Pipeline as Matcher
from correspondence.outlier_rejection.pipeline import   Outlier_Rejection
from correspondence.outlier_rejection.loss import   NeCoLoss

path = '/home/aiday.kyzy/dataset/Synthetic/'

class Landmark_Model():

    def __init__(self, config_file, device ):

        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
            config = edict(config)

        with open(config['matcher_config'], 'r') as f_:
            matcher_config = yaml.load(f_, Loader=yaml.Loader)
            matcher_config = edict(matcher_config)

        with open(config['outlier_rejection_config'], 'r') as f_:
            outlier_rejection_config = yaml.load(f_, Loader=yaml.Loader)
            outlier_rejection_config = edict(outlier_rejection_config)
            
        config['kpfcn_config'] = matcher_config['kpfcn_config']

        # matcher initialization
        self.matcher = Matcher(matcher_config).to(device)  # pretrained point cloud matcher model
        state = torch.load(config.matcher_weights)
        self.matcher.load_state_dict(state['state_dict'])

        # outlier model initialization
        self.outlier_model = Outlier_Rejection(outlier_rejection_config.model).to(device)
        state = torch.load(config.outlier_rejection_weights)
        self.outlier_model.load_state_dict(state['state_dict'])

        self.device = device

        self.kpfcn_config = config['kpfcn_config']


    def inference(self, inputs, intermediate_output_folder = None, base = None, reject_outliers=True, inlier_thr=0.5, timer=None):

        if base:
            self.path = base
        else:
            self.path = path 
            
        self.matcher.eval()
        self.outlier_model.eval()
        
        with torch.no_grad():

            if timer: timer.tic("matcher")
            data = self.matcher(inputs, timers=None)
            if timer: timer.toc("matcher")
            
            if intermediate_output_folder:
                if not os.path.exists(self.path + intermediate_output_folder + 'lepard_ldmk'):
                    os.mkdir(self.path + intermediate_output_folder + 'lepard_ldmk')
                
                b_size=len(data['s_pcd'])
                ind = data['coarse_match_pred']
                bi, si, ti = ind[:, 0], ind[:, 1], ind[:, 2]
            
                for i in range(b_size):
                    bmask = bi == i
                    rot = data['batched_rot'][0]

                    print('data.keys() : ', data.keys())
                    s_pos = data['s_pcd'][i][si[bmask]]
                    t_pos = data['t_pcd'][i][ti[bmask]]
                    
                    src_pcd_points = data['src_pcd_list'][0]
                    src_pcd = o3d.geometry.PointCloud()
                    src_pcd.points = o3d.utility.Vector3dVector(np.array(src_pcd_points.cpu()))
                    src_pcd.rotate(np.array(rot.cpu()), center=(0, 0, 0))
                    o3d.io.write_point_cloud(self.path + intermediate_output_folder + 'lepard_ldmk/' + 'src_pcd.ply', src_pcd)

                    tgt_pcd_points = data['tgt_pcd_list'][0]
                    tgt_pcd = o3d.geometry.PointCloud()
                    tgt_pcd.points = o3d.utility.Vector3dVector(np.array(tgt_pcd_points.cpu()))
                    tgt_pcd.rotate(np.array(rot.cpu()), center=(0, 0, 0))
                    o3d.io.write_point_cloud(self.path + intermediate_output_folder + 'lepard_ldmk/' + 'tgt_pcd.ply', tgt_pcd)

                    s_pos_pcd = o3d.geometry.PointCloud()
                    s_pos_pcd.points = o3d.utility.Vector3dVector(np.array(s_pos.cpu()))
                    s_pos_pcd.rotate(np.array(rot.cpu()), center=(0, 0, 0))
                    rotated_s_pos = np.array(s_pos_pcd.points)
                    o3d.io.write_point_cloud(self.path + intermediate_output_folder + 'lepard_ldmk/' + 's_lepard_pcd.ply', s_pos_pcd)
                    
                    t_pos_pcd = o3d.geometry.PointCloud()
                    t_pos_pcd.points = o3d.utility.Vector3dVector(np.array(t_pos.cpu()))
                    o3d.io.write_point_cloud(self.path + intermediate_output_folder +  'lepard_ldmk/' + 't_lepard_pcd.ply', t_pos_pcd)
                    
                    total_points = np.concatenate((rotated_s_pos, np.array(t_pos.cpu())), axis = 0)
                    number_points_src = s_pos.shape[0]
                    correspondences = [[i, i + number_points_src] for i in range(0, number_points_src)]
                    line_set = o3d.geometry.LineSet(
                        points=o3d.utility.Vector3dVector(total_points),
                        lines=o3d.utility.Vector2iVector(correspondences),
                    )
                    o3d.io.write_line_set(self.path + intermediate_output_folder +  'lepard_ldmk/' + 'lepard_line_set.ply', line_set)
                    
            if timer: timer.tic("outlier rejection")
            confidence = self.outlier_model(data)
            if timer: timer.toc("outlier rejection")

            inlier_conf = confidence[0]

            coarse_flow = data['coarse_flow'][0]
            inlier_mask, inlier_rate = NeCoLoss.compute_inlier_mask(data, inlier_thr, s2t_flow=coarse_flow)
            match_filtered = inlier_mask[0] [  inlier_conf > inlier_thr ]
            inlier_rate_2 = match_filtered.sum()/(match_filtered.shape[0])

            vec_6d = data['vec_6d'][0]

            if reject_outliers:
                vec_6d = vec_6d [inlier_conf > inlier_thr]

            ldmk_s, ldmk_t = vec_6d[:, :3], vec_6d[:, 3:]

            if intermediate_output_folder:
                if not os.path.exists(self.path + intermediate_output_folder + 'outlier_ldmk'):
                    os.mkdir(self.path + intermediate_output_folder + 'outlier_ldmk')

                rot = data['batched_rot'][0]

                src_pcd_points = data['src_pcd_list'][0]
                src_pcd = o3d.geometry.PointCloud()
                src_pcd.points = o3d.utility.Vector3dVector(np.array(src_pcd_points.cpu()))
                src_pcd.rotate(np.array(rot.cpu()), center=(0, 0, 0))
                o3d.io.write_point_cloud(self.path + intermediate_output_folder + 'outlier_ldmk/' + 'src_pcd.ply', src_pcd)

                tgt_pcd_points = data['tgt_pcd_list'][0]
                tgt_pcd = o3d.geometry.PointCloud()
                tgt_pcd.points = o3d.utility.Vector3dVector(np.array(tgt_pcd_points.cpu()))
                o3d.io.write_point_cloud(self.path + intermediate_output_folder + 'outlier_ldmk/' + 'tgt_pcd.ply', tgt_pcd)
            
                ldmk_s_pcd = o3d.geometry.PointCloud()
                ldmk_s_pcd.points = o3d.utility.Vector3dVector(np.array(ldmk_s.cpu()))
                ldmk_s_pcd.rotate(np.array(rot.cpu()), center=(0, 0, 0))
                rotated_ldmk_s = np.array(ldmk_s_pcd.points)
                o3d.io.write_point_cloud(self.path + intermediate_output_folder + 'outlier_ldmk/' + 's_outlier_rejected_pcd.ply', ldmk_s_pcd)
                
                ldmk_t_pcd = o3d.geometry.PointCloud()
                ldmk_t_pcd.points = o3d.utility.Vector3dVector(np.array(ldmk_t.cpu()))
                o3d.io.write_point_cloud(self.path + intermediate_output_folder + 'outlier_ldmk/' + 't_outlier_rejected_pcd.ply', ldmk_t_pcd)
                
                total_points = np.concatenate((rotated_ldmk_s, np.array(ldmk_t.cpu())), axis = 0)
                number_points_src = ldmk_s.shape[0]
                correspondences = [[i, i + number_points_src] for i in range(0, number_points_src)]
                line_set = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(total_points),
                    lines=o3d.utility.Vector2iVector(correspondences),
                )
                o3d.io.write_line_set(self.path + intermediate_output_folder +  'outlier_ldmk/' + 'outlier_line_set.ply', line_set)
                
            return ldmk_s, ldmk_t, inlier_rate, inlier_rate_2