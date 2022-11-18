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

from collections import defaultdict
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
            
            # After the outlier rejection network, add code in order to do some sort of custom filtering
            custom_filtering = True
            print_size = True
            if custom_filtering:
                ldmk_s_np = np.array(ldmk_s.cpu())
                print('ldmk_s_np.shape : ', ldmk_s_np.shape)
                ldmk_t_np = np.array(ldmk_t.cpu())
                # Suppose we choose to generate 100 transformations
                neighborhood_center_indices_list = np.linspace(0, ldmk_s_np.shape[0] - 1, num=20).astype(int)

                for neighborhood_center_index in neighborhood_center_indices_list:
                    if print_size:
                        print('neighborhood_center_index : ', neighborhood_center_index)
                    neighborhood_center_source = ldmk_s_np[neighborhood_center_index]
                    if print_size:
                        print('neighborhood_center_source : ', neighborhood_center_source)
                    neighborhood_center_target = ldmk_t_np[neighborhood_center_index]

                    # Find all the points closest and second closest to the centers (note that they are potentially stacked on top of each other)
                    # Set single=True in the matching algorithm
                    distance_to_neighborhood_center = np.linalg.norm(ldmk_s_np - neighborhood_center_source, axis = 1)
                    if print_size:
                        print('distance_to_neighborhood_center.shape : ', distance_to_neighborhood_center.shape)

                    indices_minimum_distance = np.where(distance_to_neighborhood_center == distance_to_neighborhood_center.min())[0]
                    if print_size:
                        print('indices_minimum_distance : ', indices_minimum_distance)
                        print('indices_minimum_distance.shape : ', indices_minimum_distance.shape)
                    
                    distance_to_neighborhood_center[indices_minimum_distance] = float('inf')
                    indices_second_minimum_distance = np.where(distance_to_neighborhood_center == distance_to_neighborhood_center.min())[0]
                    if print_size:
                        print('indices_second_minimum_distance : ', indices_second_minimum_distance)
                        print('indices_second_minimum_distance.shape : ', indices_second_minimum_distance.shape)

                    distance_to_neighborhood_center[indices_second_minimum_distance] = float('inf')
                    indices_third_minimum_distance = np.where(distance_to_neighborhood_center == distance_to_neighborhood_center.min())[0]
                    if print_size:
                        print('indices_third_minimum_distance : ', indices_third_minimum_distance)
                        print('indices_third_minimum_distance.shape : ', indices_third_minimum_distance.shape)                       

                    max_number_transformations = min(indices_minimum_distance.shape[0], indices_second_minimum_distance.shape[0], indices_third_minimum_distance.shape[0])
                    number_transformations = min(3, max_number_transformations)

                    indices_1 = np.random.choice(indices_minimum_distance.shape[0], number_transformations, replace=False)
                    indices_2 = np.random.choice(indices_second_minimum_distance.shape[0], number_transformations, replace=False)
                    indices_3 = np.random.choice(indices_third_minimum_distance.shape[0], number_transformations, replace=False)

                    if print_size:
                        print('indices_1 : ', indices_1)
                        print('indices_2 : ', indices_2)
                        print('indices_3 : ', indices_3)

                    outliers = defaultdict(int)
                    tau = 0.01
                    point_indices_close_to_center = np.where(distance_to_neighborhood_center < tau)[0]
                    source_points_close_to_center = ldmk_s_np[point_indices_close_to_center]
                    target_points_close_to_center = ldmk_t_np[point_indices_close_to_center]
                    print('source_points_close_to_center.shape[0] : ', source_points_close_to_center.shape[0])

                    for n_transform in range(number_transformations):
                        source_point_1 = ldmk_s_np[indices_minimum_distance[indices_1[n_transform]]]
                        target_point_1 = ldmk_t_np[indices_minimum_distance[indices_1[n_transform]]]
                        source_point_2 = ldmk_s_np[indices_second_minimum_distance[indices_2[n_transform]]]
                        target_point_2 = ldmk_t_np[indices_second_minimum_distance[indices_2[n_transform]]]
                        source_point_3 = ldmk_s_np[indices_third_minimum_distance[indices_3[n_transform]]]
                        target_point_3 = ldmk_t_np[indices_third_minimum_distance[indices_3[n_transform]]]

                        X = np.empty((0,3), int)
                        X = np.append(X, np.array(np.expand_dims(source_point_1, axis=0)), axis=0)
                        X = np.append(X, np.array(np.expand_dims(source_point_2, axis=0)), axis=0)
                        X = np.append(X, np.array(np.expand_dims(source_point_3, axis=0)), axis=0)
                        Y = np.empty((0,3), int)
                        Y = np.append(Y, np.array(np.expand_dims(target_point_1, axis=0)), axis=0)
                        Y = np.append(Y, np.array(np.expand_dims(target_point_2, axis=0)), axis=0)
                        Y = np.append(Y, np.array(np.expand_dims(target_point_3, axis=0)), axis=0)

                        if print_size:
                            print('X : ', X)
                            print('Y : ', Y)
                            
                        mean_X = np.mean(X, axis = 0)
                        mean_Y = np.mean(Y, axis = 0)

                        if print_size:
                            print('mean_X : ', mean_X)
                            print('mean_Y : ', mean_Y)
                            print('(Y - mean_Y) : ', (Y - mean_Y))
                            print('(X - mean_X) : ', (X - mean_X))
                    
                        Sxy = np.matmul( (Y - mean_Y).T, (X - mean_X) )
                        U, _, V = np.linalg.svd(Sxy, full_matrices=True)
                        S = np.eye(3)
                        UV_det = np.linalg.det(U) * np.linalg.det(V)
                        S[2, 2] = UV_det
                        sv = np.matmul( S, V )
                        R = np.matmul( U, sv)
                        t = mean_Y.T - np.matmul( R, mean_X.T )
                        
                        if print_size:
                            print('R : ', R)
                            print('t : ', t)
                        
                        # find points which should be inliers against these given transformations                        
                        points_after_transformation = R @ source_points_close_to_center + t
                        norm_error = np.linalg.norm(points_after_transformation - target_points_close_to_center)
                        outlier_indices = np.where(norm_error > tau)[0]
                        for outlier_idx in outlier_indices:
                            outliers[outlier_idx] = outliers[outlier_idx] + 1

                    if print_size:
                        print('outliers : ', outliers)

                    print_size = False
                
            return ldmk_s, ldmk_t, inlier_rate, inlier_rate_2