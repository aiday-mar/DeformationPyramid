import torch
import yaml
from easydict import EasyDict as edict

import sys
import open3d as o3d
import numpy as np
import os
import copy 
import random
import h5py

sys.path.append("")
from correspondence.lepard.pipeline import Pipeline as Matcher
from correspondence.lepard.pipeline_fcgf import PipelineFCGF as MatcherFCGF
from correspondence.outlier_rejection.pipeline import   Outlier_Rejection
from correspondence.outlier_rejection.loss import   NeCoLoss

from collections import defaultdict
path = '/home/aiday.kyzy/dataset/Synthetic/'

class Landmark_Model():

    def __init__(self, config_file, device, indent=None, feature_extractor='kpfcn'):

        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
            config = edict(config)

        with open(indent+config['matcher_config'] if indent else config['matcher_config'], 'r') as f_:
            matcher_config = yaml.load(f_, Loader=yaml.Loader)
            matcher_config = edict(matcher_config)

        with open(indent+config['outlier_rejection_config'] if indent else config['outlier_rejection_config'], 'r') as f_:
            outlier_rejection_config = yaml.load(f_, Loader=yaml.Loader)
            outlier_rejection_config = edict(outlier_rejection_config)
            
        config['kpfcn_config'] = matcher_config['kpfcn_config']

        # matcher initialization
        self.feature_extractor = feature_extractor
        print('feature_extractor : ', feature_extractor)
        if self.feature_extractor == 'kpfcn':
            self.matcher = Matcher(matcher_config).to(device)
        if self.feature_extractor == 'fcgf':
            self.matcher = MatcherFCGF(matcher_config).to(device)

        state = torch.load(indent + config.matcher_weights if indent else config.matcher_weights)
        self.matcher.load_state_dict(state['state_dict'])
        self.indent = indent
        
        # outlier model initialization
        self.outlier_model = Outlier_Rejection(outlier_rejection_config.model).to(device)
        state = torch.load(indent + config.outlier_rejection_weights if indent else config.outlier_rejection_weights)
        self.outlier_model.load_state_dict(state['state_dict'])

        self.device = device
        self.kpfcn_config = config['kpfcn_config']

    def inference(self, inputs, sampling = 'linspace', mesh_path = None, source_trans = None, inlier_outlier_thr = 0.05, matches_path = None, custom_filtering = None, number_iterations_custom_filtering = 1, average_distance_multiplier = 2.0, intermediate_output_folder = None, number_centers = 1000, base = None, preprocessing = 'mutual', confidence_threshold = None, coarse_level = None, reject_outliers=True, inlier_thr=0.5, index_at_which_to_return_coarse_feats = 1, timer=None, gt_thr = 0.01):
        if base:
            self.path = base
        else:
            self.path = path 
        
        print('confidence threshold in inference : ', confidence_threshold)
        self.matcher.eval()
        self.outlier_model.eval()
        lepard_true_correspondences_mask = None
        custom_filtering_true_correspondences_mask = None
        
        if self.feature_extractor == 'kpfcn':
            folder_name = 'lepard'
        elif self.feature_extractor == 'fcgf':
            folder_name = 'fcgf'
        else:
            raise Exception('Specify a valid feature extractor')
            
        with torch.no_grad():
            if timer: timer.tic("matcher")
            if self.feature_extractor == 'fcgf':
                data = self.matcher(inputs, confidence_threshold = confidence_threshold, preprocessing = preprocessing)
            elif self.feature_extractor == 'kpfcn':
                data = self.matcher(inputs, coarse_level = coarse_level, confidence_threshold = confidence_threshold, preprocessing = preprocessing, index_at_which_to_return_coarse_feats = index_at_which_to_return_coarse_feats, timers=None)
            else:
                raise Exception('Choose a valid feature extractor')
            if timer: timer.toc("matcher")
            
            if intermediate_output_folder:
                if not os.path.exists(self.path + intermediate_output_folder + folder_name + '_ldmk'):
                    os.mkdir(self.path + intermediate_output_folder + folder_name + '_ldmk')
                
                b_size=len(data['s_pcd'])
                ind = data['coarse_match_pred']
                bi, si, ti = ind[:, 0], ind[:, 1], ind[:, 2]
            
                for i in range(b_size):
                    bmask = bi == i
                    rot = data['batched_rot'][0]

                    s_pos = data['s_pcd'][i][si[bmask]]
                    t_pos = data['t_pcd'][i][ti[bmask]]
                    
                    src_pcd_points = data['src_pcd_list'][0]
                    src_pcd_points = np.array(src_pcd_points.cpu())
                    src_pcd = o3d.geometry.PointCloud()
                    src_pcd.points = o3d.utility.Vector3dVector(src_pcd_points)
                    src_pcd.rotate(np.array(rot.cpu()), center=(0, 0, 0))
                    o3d.io.write_point_cloud(self.path + intermediate_output_folder + folder_name + '_ldmk/' + 'src_pcd.ply', src_pcd)

                    tgt_pcd_points = data['tgt_pcd_list'][0]
                    tgt_pcd_points = np.array(tgt_pcd_points.cpu())
                    tgt_pcd = o3d.geometry.PointCloud()
                    tgt_pcd.points = o3d.utility.Vector3dVector(tgt_pcd_points)
                    o3d.io.write_point_cloud(self.path + intermediate_output_folder + folder_name + '_ldmk/' + 'tgt_pcd.ply', tgt_pcd)

                    s_pos_pcd = o3d.geometry.PointCloud()
                    s_pos_pcd_points = np.array(s_pos.cpu())
                    s_pos_pcd.points = o3d.utility.Vector3dVector(s_pos_pcd_points)
                    o3d.io.write_point_cloud(self.path + intermediate_output_folder + folder_name + '_ldmk/' + 's_lepard_pcd.ply', s_pos_pcd)
                    
                    s_pos_pcd_rotated = s_pos_pcd.rotate(np.array(rot.cpu()), center=(0, 0, 0))
                    s_pos_pcd_points_rotated = np.array(s_pos_pcd_rotated.points)
                    o3d.io.write_point_cloud(self.path + intermediate_output_folder + folder_name + '_ldmk/' + 's_lepard_pcd_rotated.ply', s_pos_pcd_rotated)
                    
                    t_pos_pcd = o3d.geometry.PointCloud()
                    t_pos_pcd_points = np.array(t_pos.cpu())
                    t_pos_pcd.points = o3d.utility.Vector3dVector(t_pos_pcd_points)
                    o3d.io.write_point_cloud(self.path + intermediate_output_folder + folder_name + '_ldmk/' + 't_lepard_pcd.ply', t_pos_pcd)
                    
                    total_points = np.concatenate((s_pos_pcd_points_rotated, t_pos_pcd_points), axis = 0)
                    number_points_src = s_pos.shape[0]
                    correspondences = [[i, i + number_points_src] for i in range(0, number_points_src)]
                    line_set = o3d.geometry.LineSet(
                        points=o3d.utility.Vector3dVector(total_points),
                        lines=o3d.utility.Vector2iVector(correspondences),
                    )
                    o3d.io.write_line_set(self.path + intermediate_output_folder + folder_name + '_ldmk/' + 'lepard_line_set.ply', line_set)
                    
            if matches_path:
                b_size=len(data['s_pcd'])
                ind = data['coarse_match_pred']
                bi, si, ti = ind[:, 0], ind[:, 1], ind[:, 2]
            
                for i in range(b_size):
                    bmask = bi == i
                    rot = data['batched_rot'][0]

                    s_pos = data['s_pcd'][i][si[bmask]]
                    t_pos = data['t_pcd'][i][ti[bmask]]
                    
                    mask = np.array([])
                    matches = np.load(self.path + matches_path)
                    correspondences = np.array(matches['matches'])
                    ind_src = correspondences[:, 0]
                    ind_tgt = correspondences[:, 1]
                    src_pcd_points = data['src_pcd_list'][0]
                    src_pcd_points = np.array(src_pcd_points.cpu())
                    matches_source = src_pcd_points[ind_src]
                    tgt_pcd_points = data['tgt_pcd_list'][0]
                    tgt_pcd_points = np.array(tgt_pcd_points.cpu())
                    matches_target = tgt_pcd_points[ind_tgt]
                    s_pos_pcd_points = np.array(s_pos.cpu())
                    t_pos_pcd_points = np.array(t_pos.cpu())
                    thr = 0.01
                    
                    for i in range(s_pos_pcd_points.shape[0]):
                        s_ldmk = s_pos_pcd_points[i]
                        t_ldmk = t_pos_pcd_points[i]
                        distance_to_s_ldmk = np.linalg.norm(matches_source - s_ldmk, axis=1)
                        distance_to_t_ldmk = np.linalg.norm(matches_target - t_ldmk, axis=1)
                        indices_neigh_s_ldmk = set(np.where(distance_to_s_ldmk < thr)[0])
                        indices_neigh_t_ldmk = set(np.where(distance_to_t_ldmk < thr)[0])
                        if indices_neigh_s_ldmk & indices_neigh_t_ldmk:
                            mask = np.append(mask, True)
                        else:
                            mask = np.append(mask, False)
                    
                    lepard_true_correspondences_mask = mask.astype(bool)
                    n_true_lepard_correspondences = int(lepard_true_correspondences_mask.sum())
                    n_total_lepard_correspondences = lepard_true_correspondences_mask.shape[0]
                    if n_total_lepard_correspondences == 0:
                        print('Early exit')
                        sys.exit(1)

                    if self.feature_extractor == 'kpfcn':
                        print('number of true landmarks correspondences returned from Lepard : ', n_true_lepard_correspondences , ' out of ', n_total_lepard_correspondences)
                        print('fraction of true landmark correspondences returned from Lepard : ', n_true_lepard_correspondences/n_total_lepard_correspondences if n_total_lepard_correspondences != 0 else 0)
                    elif self.feature_extractor == 'fcgf':
                        print('number of true landmarks correspondences returned from FCGF based Lepard : ', n_true_lepard_correspondences , ' out of ', n_total_lepard_correspondences)
                        print('fraction of true landmark correspondences returned from FCGF based Lepard : ', n_true_lepard_correspondences/n_total_lepard_correspondences if n_total_lepard_correspondences != 0 else 0)
                    
                    if intermediate_output_folder:
                        s_pos_pcd = o3d.geometry.PointCloud()
                        s_pos_pcd.points = o3d.utility.Vector3dVector(s_pos_pcd_points)
                        s_pos_pcd_rotated = s_pos_pcd.rotate(np.array(rot.cpu()), center=(0, 0, 0))
                        s_pos_pcd_points_rotated = np.array(s_pos_pcd_rotated.points)
                        gt_inlier_matches_s = s_pos_pcd_points_rotated[lepard_true_correspondences_mask]
                        gt_inlier_matches_t = t_pos_pcd_points[lepard_true_correspondences_mask]
                        total_inlier_points = np.concatenate((gt_inlier_matches_s, gt_inlier_matches_t), axis = 0)
                        number_gt_inliers = gt_inlier_matches_s.shape[0]
                        inlier_correspondences = np.array([[i, i + number_gt_inliers] for i in range(0, number_gt_inliers)])
                        colors = np.repeat([[0, 1, 0]], [inlier_correspondences.shape[0]], axis=0).astype(np.float64)
                        inliers_lepard_line_set = o3d.geometry.LineSet()
                        inliers_lepard_line_set.points=o3d.utility.Vector3dVector(total_inlier_points)
                        inliers_lepard_line_set.lines =o3d.utility.Vector2iVector(inlier_correspondences)
                        inliers_lepard_line_set.colors = o3d.utility.Vector3dVector(colors)
                        o3d.io.write_line_set(self.path + intermediate_output_folder + folder_name + '_ldmk/inliers.ply', inliers_lepard_line_set)
                        
                        inverse_mask = ~lepard_true_correspondences_mask
                        gt_outlier_matches_s = s_pos_pcd_points_rotated[inverse_mask]
                        gt_outlier_matches_t = t_pos_pcd_points[inverse_mask]
                        total_outlier_points = np.concatenate((gt_outlier_matches_s, gt_outlier_matches_t), axis = 0)
                        number_gt_outliers = gt_outlier_matches_s.shape[0]
                        outlier_correspondences = np.array([[i, i + number_gt_outliers] for i in range(0, number_gt_outliers)])
                        colors = np.repeat([[0, 1, 0]], [outlier_correspondences.shape[0]], axis=0).astype(np.float64)
                        outliers_lepard_line_set = o3d.geometry.LineSet()
                        outliers_lepard_line_set.points=o3d.utility.Vector3dVector(total_outlier_points)
                        outliers_lepard_line_set.lines =o3d.utility.Vector2iVector(outlier_correspondences)
                        outliers_lepard_line_set.colors = o3d.utility.Vector3dVector(colors)
                        o3d.io.write_line_set(self.path + intermediate_output_folder + folder_name + '_ldmk/outliers.ply', outliers_lepard_line_set)
                        
            if timer: timer.tic("outlier rejection")
            confidence = self.outlier_model(data, self.feature_extractor)
            if timer: timer.toc("outlier rejection")

            inlier_conf = confidence[0]

            coarse_flow = data['coarse_flow'][0]
            inlier_mask, inlier_rate = NeCoLoss.compute_inlier_mask(data, inlier_thr, s2t_flow=coarse_flow)
            match_filtered = inlier_mask[0] [  inlier_conf > inlier_thr ]
            inlier_rate_2 = match_filtered.sum()/(match_filtered.shape[0])
            vec_6d = data['vec_6d'][0]

            if not custom_filtering and reject_outliers:
                vec_6d = vec_6d[inlier_conf > inlier_thr] 
            
            var = inlier_conf > inlier_thr
            ldmk_s, ldmk_t = vec_6d[:, :3], vec_6d[:, 3:]
            
            if intermediate_output_folder and not custom_filtering:
                if not os.path.exists(self.path + intermediate_output_folder + folder_name + '_outlier_ldmk'):
                    os.mkdir(self.path + intermediate_output_folder + folder_name + '_outlier_ldmk')

                rot = data['batched_rot'][0]
                src_pcd_points = data['src_pcd_list'][0]
                src_pcd = o3d.geometry.PointCloud()
                src_pcd.points = o3d.utility.Vector3dVector(np.array(src_pcd_points.cpu()))
                src_pcd.rotate(np.array(rot.cpu()), center=(0, 0, 0))
                o3d.io.write_point_cloud(self.path + intermediate_output_folder + folder_name + '_outlier_ldmk/' + 'src_pcd.ply', src_pcd)

                tgt_pcd_points = data['tgt_pcd_list'][0]
                tgt_pcd = o3d.geometry.PointCloud()
                tgt_pcd.points = o3d.utility.Vector3dVector(np.array(tgt_pcd_points.cpu()))
                o3d.io.write_point_cloud(self.path + intermediate_output_folder + folder_name + '_outlier_ldmk/' + 'tgt_pcd.ply', tgt_pcd)
            
                ldmk_s_pcd = o3d.geometry.PointCloud()
                ldmk_s_pcd.points = o3d.utility.Vector3dVector(np.array(ldmk_s.cpu()))
                ldmk_s_pcd.rotate(np.array(rot.cpu()), center=(0, 0, 0))
                rotated_ldmk_s = np.array(ldmk_s_pcd.points)
                o3d.io.write_point_cloud(self.path + intermediate_output_folder + folder_name + '_outlier_ldmk/' + 's_outlier_rejected_pcd.ply', ldmk_s_pcd)
                
                ldmk_t_pcd = o3d.geometry.PointCloud()
                ldmk_t_pcd.points = o3d.utility.Vector3dVector(np.array(ldmk_t.cpu()))
                o3d.io.write_point_cloud(self.path + intermediate_output_folder + folder_name + '_outlier_ldmk/' + 't_outlier_rejected_pcd.ply', ldmk_t_pcd)
                
                total_points = np.concatenate((rotated_ldmk_s, np.array(ldmk_t.cpu())), axis = 0)
                number_points_src = ldmk_s.shape[0]
                correspondences = [[i, i + number_points_src] for i in range(0, number_points_src)]
                line_set = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(total_points),
                    lines=o3d.utility.Vector2iVector(correspondences),
                )
                o3d.io.write_line_set(self.path + intermediate_output_folder + folder_name + '_outlier_ldmk/' + 'outlier_line_set.ply', line_set)
                
            if matches_path:
                outlier_rejected_vec_6d = data['vec_6d'][0][inlier_conf > inlier_thr]
                ldmk_s_outlier_rejected, ldmk_t_outlier_rejected = outlier_rejected_vec_6d[:, :3], outlier_rejected_vec_6d[:, 3:]
                mask = np.array([])
                matches = np.load(self.path + matches_path)
                correspondences = np.array(matches['matches'])
                ind_src = correspondences[:, 0]
                ind_tgt = correspondences[:, 1]
                
                src_pcd_points = data['src_pcd_list'][0]
                src_pcd_points = np.array(src_pcd_points.cpu())
                tgt_pcd_points = data['tgt_pcd_list'][0]
                tgt_pcd_points = np.array(tgt_pcd_points.cpu())
                
                matches_source = src_pcd_points[ind_src]
                matches_target = tgt_pcd_points[ind_tgt]
                thr = 0.01
                
                for i in range(ldmk_s_outlier_rejected.shape[0]):
                    s_ldmk = np.array(ldmk_s_outlier_rejected[i].cpu())
                    t_ldmk = np.array(ldmk_t_outlier_rejected[i].cpu())
                    distance_to_s_ldmk = np.linalg.norm(matches_source - s_ldmk, axis=1)
                    distance_to_t_ldmk = np.linalg.norm(matches_target - t_ldmk, axis=1)
                    indices_neigh_s_ldmk = set(np.where(distance_to_s_ldmk < thr)[0])
                    indices_neigh_t_ldmk = set(np.where(distance_to_t_ldmk < thr)[0])
                    if indices_neigh_s_ldmk & indices_neigh_t_ldmk:
                        mask = np.append(mask, True)
                    else:
                        mask = np.append(mask, False)
                
                outlier_rejected_true_correspondences_mask = mask.astype(bool)
                n_true_outlier_rejected_correspondences = int(outlier_rejected_true_correspondences_mask.sum())
                n_total_outlier_rejected_correspondences = outlier_rejected_true_correspondences_mask.shape[0]
                if self.feature_extractor == 'kpfcn':
                    print('number of true landmark correspondences returned from Outlier Rejection : ', n_true_outlier_rejected_correspondences , ' out of ', n_total_outlier_rejected_correspondences)
                    print('fraction of true landmark correspondences returned from Outlier Rejection : ', n_true_outlier_rejected_correspondences/n_total_outlier_rejected_correspondences if n_total_outlier_rejected_correspondences != 0 else 0)
                elif self.feature_extractor == 'fcgf':
                    print('number of true landmark correspondences returned from FCGF based Outlier Rejection : ', n_true_outlier_rejected_correspondences , ' out of ', n_total_outlier_rejected_correspondences)
                    print('fraction of true landmark correspondences returned from FCGF based Outlier Rejection : ', n_true_outlier_rejected_correspondences/n_total_outlier_rejected_correspondences if n_total_outlier_rejected_correspondences != 0 else 0)

                if intermediate_output_folder:
                    if not os.path.exists(self.path + intermediate_output_folder + folder_name + '_outlier_ldmk'):
                        os.mkdir(self.path + intermediate_output_folder + folder_name + '_outlier_ldmk')
                    rot = data['batched_rot'][0]
                    ldmk_s_outlier_rejected_pcd = o3d.geometry.PointCloud()
                    ldmk_s_outlier_rejected = np.array(ldmk_s_outlier_rejected.cpu())
                    ldmk_s_outlier_rejected_pcd.points = o3d.utility.Vector3dVector(ldmk_s_outlier_rejected)                    
                    ldmk_s_outlier_rejected_rotated_pcd = ldmk_s_outlier_rejected_pcd.rotate(np.array(rot.cpu()), center=(0, 0, 0))
                    ldmk_s_outlier_rejected_rotated = np.array(ldmk_s_outlier_rejected_rotated_pcd.points)
                    
                    gt_inlier_matches_s = ldmk_s_outlier_rejected_rotated[outlier_rejected_true_correspondences_mask]
                    gt_inlier_matches_t = ldmk_t_outlier_rejected[outlier_rejected_true_correspondences_mask].cpu()
                    total_inlier_points = np.concatenate((gt_inlier_matches_s, gt_inlier_matches_t), axis = 0)
                    number_gt_inliers = gt_inlier_matches_s.shape[0]
                    inlier_correspondences = np.array([[i, i + number_gt_inliers] for i in range(0, number_gt_inliers)])
                    colors = np.repeat([[0, 1, 0]], [inlier_correspondences.shape[0]], axis=0).astype(np.float64)
                    inliers_outlier_rejected_line_set = o3d.geometry.LineSet()
                    inliers_outlier_rejected_line_set.points=o3d.utility.Vector3dVector(total_inlier_points)
                    inliers_outlier_rejected_line_set.lines =o3d.utility.Vector2iVector(inlier_correspondences)
                    inliers_outlier_rejected_line_set.colors = o3d.utility.Vector3dVector(colors)
                    o3d.io.write_line_set(self.path + intermediate_output_folder + folder_name + '_outlier_ldmk/inliers.ply', inliers_outlier_rejected_line_set)
                    
                    inverse_mask = ~outlier_rejected_true_correspondences_mask
                    gt_outlier_matches_s = ldmk_s_outlier_rejected_rotated[inverse_mask]
                    gt_outlier_matches_t = ldmk_t_outlier_rejected[inverse_mask].cpu()
                    total_outlier_points = np.concatenate((gt_outlier_matches_s, gt_outlier_matches_t), axis = 0)
                    number_gt_outliers = gt_outlier_matches_s.shape[0]
                    outlier_correspondences = np.array([[i, i + number_gt_outliers] for i in range(0, number_gt_outliers)])
                    colors = np.repeat([[0, 1, 0]], [outlier_correspondences.shape[0]], axis=0).astype(np.float64)
                    outliers_outlier_rejected_line_set = o3d.geometry.LineSet()
                    outliers_outlier_rejected_line_set.points=o3d.utility.Vector3dVector(total_outlier_points)
                    outliers_outlier_rejected_line_set.lines =o3d.utility.Vector2iVector(outlier_correspondences)
                    outliers_outlier_rejected_line_set.colors = o3d.utility.Vector3dVector(colors)
                    o3d.io.write_line_set(self.path + intermediate_output_folder + folder_name + '_outlier_ldmk/outliers.ply', outliers_outlier_rejected_line_set)
        
            # 1. Custom filtering done with the first method
            '''            
            custom_filtering = True
            print_size = True
            
            if custom_filtering and intermediate_output_folder:
                if not os.path.exists(self.path + intermediate_output_folder + 'custom_filtering_ldmk'):
                    os.mkdir(self.path + intermediate_output_folder + 'custom_filtering_ldmk')
                    
                print('When we do custom filtering')
                ldmk_s_np = np.array(ldmk_s.cpu())
                ldmk_t_np = np.array(ldmk_t.cpu())
                if print_size:
                    print('ldmk_s_np.shape : ', ldmk_s_np.shape)

                neighborhood_center_indices_list = np.linspace(0, ldmk_s_np.shape[0] - 1, num=1000).astype(int)
                print('neighborhood_center_indices_list.shape : ', neighborhood_center_indices_list.shape)
                outliers = defaultdict(int)

                for neighborhood_center_index in neighborhood_center_indices_list:
                    if print_size:
                        print('neighborhood_center_index : ', neighborhood_center_index)
                    neighborhood_center_source = ldmk_s_np[neighborhood_center_index]
                    if print_size:
                        print('neighborhood_center_source : ', neighborhood_center_source)
                    neighborhood_center_target = ldmk_t_np[neighborhood_center_index]

                    # Find all the points closest and second closest to the centers (note that they are potentially stacked on top of each other)
                    distance_to_neighborhood_center = np.linalg.norm(ldmk_s_np - neighborhood_center_source, axis = 1)
                    distances_to_center = copy.deepcopy(distance_to_neighborhood_center)

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
                    number_transformations = min(6, max_number_transformations)
                    print('number_transformations : ', number_transformations)

                    indices_1 = np.random.choice(indices_minimum_distance.shape[0], number_transformations, replace=False)
                    indices_2 = np.random.choice(indices_second_minimum_distance.shape[0], number_transformations, replace=False)
                    indices_3 = np.random.choice(indices_third_minimum_distance.shape[0], number_transformations, replace=False)

                    if print_size:
                        print('indices_1 : ', indices_1)
                        print('indices_2 : ', indices_2)
                        print('indices_3 : ', indices_3)

                    tau = 0.1
                    if print_size:
                        print('np.where(distances_to_center < tau)[0].shape : ', np.where(distances_to_center < tau)[0].shape)
                    point_indices_close_to_center = np.where(distances_to_center < tau)[0]
                    source_points_close_to_center = ldmk_s_np[point_indices_close_to_center]
                    target_points_close_to_center = ldmk_t_np[point_indices_close_to_center]
                    if print_size:
                        print('point_indices_close_to_center.shape : ', point_indices_close_to_center.shape)
                        print('source_points_close_to_center.shape : ', source_points_close_to_center.shape)

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
                        thr = 0.05                        
                        points_after_transformation = (R @ source_points_close_to_center.T + np.expand_dims(t, axis=1)).T
                        norm_error = np.linalg.norm(points_after_transformation - target_points_close_to_center, axis = 1)
                        if print_size:
                            print('norm_error.shape : ', norm_error.shape)
                            print('np.where(norm_error > thr)[0].shape : ', np.where(norm_error > thr)[0].shape)
                        outlier_indices = np.where(norm_error > thr)[0]
                        for outlier_idx in outlier_indices:
                            out_idx = point_indices_close_to_center[outlier_idx]
                            outliers[out_idx] = outliers[out_idx] + 1
                            
                    print_size = False

                outliers = dict((k, v) for k, v in outliers.items() if v >= 300)
                print('outliers : ', outliers)
                total_outliers = set(outliers.keys())                                  
                final_indices = np.array([i for i in range(0, len(ldmk_s_np)) if i not in total_outliers])
                print('final_indices : ', final_indices)
                ldmk_s = torch.tensor(ldmk_s_np[final_indices]).to('cuda:0')
                ldmk_t = torch.tensor(ldmk_t_np[final_indices]).to('cuda:0')
                print('ldmk_s.shape : ', ldmk_s.shape)
                
                rot = data['batched_rot'][0]
                ldmk_s_custom_filtering = o3d.geometry.PointCloud()
                ldmk_s_custom_filtering.points = o3d.utility.Vector3dVector(np.array(ldmk_s.cpu()))
                ldmk_s_custom_filtering.rotate(np.array(rot.cpu()), center=(0, 0, 0))
                rotated_ldmk_s = np.array(ldmk_s_custom_filtering.points)
                o3d.io.write_point_cloud(self.path + intermediate_output_folder + 'custom_filtering_ldmk/' + 's_custom_filtering.ply', ldmk_s_custom_filtering)
                
                ldmk_t_custom_filtering = o3d.geometry.PointCloud()
                ldmk_t_custom_filtering.points = o3d.utility.Vector3dVector(np.array(ldmk_t.cpu()))
                o3d.io.write_point_cloud(self.path + intermediate_output_folder + 'custom_filtering_ldmk/' + 't_custom_filtering_pcd.ply', ldmk_t_custom_filtering)
                
                total_points = np.concatenate((rotated_ldmk_s, np.array(ldmk_t.cpu())), axis = 0)
                number_points_src = ldmk_s.shape[0]
                correspondences = [[i, i + number_points_src] for i in range(0, number_points_src)]
                line_set = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(total_points),
                    lines=o3d.utility.Vector2iVector(correspondences),
                )
                o3d.io.write_line_set(self.path + intermediate_output_folder +  'custom_filtering_ldmk/' + 'custom_filtering_line_set.ply', line_set)
                
                data_mod = {}
                final_indices = list(final_indices)
                print('len(final_indices) : ', len(final_indices))

                print('data[vec_6d][0].shape : ', data['vec_6d'][0].shape)
                vec_6d = data['vec_6d'][0][final_indices]
                data_mod['vec_6d'] = vec_6d[None, :]
                
                print('data[vec_6d_mask][0].shape : ', data['vec_6d_mask'][0].shape)
                vec_6d_mask = data['vec_6d_mask'][0][final_indices]
                data_mod['vec_6d_mask'] = vec_6d_mask[None, :]
                
                print('data[vec_6d_ind][0].shape : ', data['vec_6d_ind'][0].shape)
                vec_6d_ind = data['vec_6d_ind'][0][final_indices]
                data_mod['vec_6d_ind'] = vec_6d_ind[None, :]
                
                data_mod['s_pcd'] = data['s_pcd']
                data_mod['t_pcd'] = data['t_pcd']
                data_mod['batched_rot'] = data['batched_rot']
                data_mod['batched_trn'] = data['batched_trn']
        
                print('coarse_flow.shape : ', coarse_flow.shape)
                inlier_mask, inlier_rate = NeCoLoss.compute_inlier_mask(data_mod, inlier_thr, s2t_flow=coarse_flow)
                print('inlier_conf.shape : ', inlier_conf.shape)
                inlier_conf = inlier_conf[final_indices]
                match_filtered = inlier_mask[0] [  inlier_conf > inlier_thr ]
                inlier_rate_2 = match_filtered.sum()/(match_filtered.shape[0])
            
            '''
            # 2. Custom filtering done with the second method
            # VERSION 1
            '''        
            if custom_filtering and intermediate_output_folder:
                print('custom filtering is used')
                if not os.path.exists(self.path + intermediate_output_folder + 'custom_filtering_ldmk'):
                    os.mkdir(self.path + intermediate_output_folder + 'custom_filtering_ldmk')
                    
                ldmk_s_np = np.array(ldmk_s.cpu())
                ldmk_t_np = np.array(ldmk_t.cpu())

                # Find map from initial points to all corresponding indices for correpondences that reference that point
                map_ldmk_s_correspondences = defaultdict(list)
                for idx, ldmk_s_point in enumerate(ldmk_s_np):
                    map_ldmk_s_correspondences[tuple(ldmk_s_point)].append(idx)
                
                neighborhood_center_indices_list = np.linspace(0, ldmk_s_np.shape[0] - 1, num=1000).astype(int)
                outliers = defaultdict(int)

                for neighborhood_center_index in neighborhood_center_indices_list:
                    neighborhood_center_source = ldmk_s_np[neighborhood_center_index]
                    neighborhood_center_target = ldmk_t_np[neighborhood_center_index]

                    # Find all the points closest and second closest to the centers (note that they are potentially stacked on top of each other)
                    distance_to_neighborhood_center = np.linalg.norm(ldmk_s_np - neighborhood_center_source, axis = 1)
                    distances_to_center = copy.deepcopy(distance_to_neighborhood_center)

                    indices_minimum_distance = np.where(distance_to_neighborhood_center == distance_to_neighborhood_center.min())[0]
                    distance_to_neighborhood_center[indices_minimum_distance] = float('inf')
                    indices_second_minimum_distance = np.where(distance_to_neighborhood_center == distance_to_neighborhood_center.min())[0]

                    distance_to_neighborhood_center[indices_second_minimum_distance] = float('inf')
                    indices_third_minimum_distance = np.where(distance_to_neighborhood_center == distance_to_neighborhood_center.min())[0]

                    max_number_transformations = min(indices_minimum_distance.shape[0], indices_second_minimum_distance.shape[0], indices_third_minimum_distance.shape[0])
                    number_transformations = min(6, max_number_transformations)

                    indices_1 = np.random.choice(indices_minimum_distance.shape[0], number_transformations, replace=False)
                    indices_2 = np.random.choice(indices_second_minimum_distance.shape[0], number_transformations, replace=False)
                    indices_3 = np.random.choice(indices_third_minimum_distance.shape[0], number_transformations, replace=False)
                    
                    tau = 0.1
                    point_indices_close_to_center = np.where(distances_to_center < tau)[0]
                    source_points_close_to_center = ldmk_s_np[point_indices_close_to_center]
                    target_points_close_to_center = ldmk_t_np[point_indices_close_to_center]

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

                        mean_X = np.mean(X, axis = 0)
                        mean_Y = np.mean(Y, axis = 0)
                    
                        Sxy = np.matmul( (Y - mean_Y).T, (X - mean_X) )
                        U, _, V = np.linalg.svd(Sxy, full_matrices=True)
                        S = np.eye(3)
                        UV_det = np.linalg.det(U) * np.linalg.det(V)
                        S[2, 2] = UV_det
                        sv = np.matmul( S, V )
                        R = np.matmul( U, sv)
                        t = mean_Y.T - np.matmul( R, mean_X.T )

                        thr = 0.05                        
                        points_after_transformation = (R @ source_points_close_to_center.T + np.expand_dims(t, axis=1)).T
                        norm_error = np.linalg.norm(points_after_transformation - target_points_close_to_center, axis = 1)
                        outlier_indices = np.where(norm_error > thr)[0]
                        for outlier_idx in outlier_indices:
                            out_idx = point_indices_close_to_center[outlier_idx]
                            outliers[out_idx] = outliers[out_idx] + 1
                            
                final_indices = np.array([])
                for ldmk_s_point in map_ldmk_s_correspondences:
                    correspondence_indices = map_ldmk_s_correspondences[ldmk_s_point]
                    correspondence_indices_to_outliers = {key: outliers[key] for key in correspondence_indices if key in outliers}
                    if correspondence_indices_to_outliers:
                        correspondence_min = min(correspondence_indices_to_outliers, key=correspondence_indices_to_outliers.get)
                        final_indices = np.append(final_indices, correspondence_min)
                
                final_indices = np.sort(final_indices).astype(int) 
                ldmk_s = torch.tensor(ldmk_s_np[final_indices]).to('cuda:0')
                ldmk_t = torch.tensor(ldmk_t_np[final_indices]).to('cuda:0')
                
                rot = data['batched_rot'][0]
                ldmk_s_custom_filtering = o3d.geometry.PointCloud()
                ldmk_s_custom_filtering.points = o3d.utility.Vector3dVector(np.array(ldmk_s.cpu()))
                ldmk_s_custom_filtering.rotate(np.array(rot.cpu()), center=(0, 0, 0))
                rotated_ldmk_s = np.array(ldmk_s_custom_filtering.points)
                o3d.io.write_point_cloud(self.path + intermediate_output_folder + 'custom_filtering_ldmk/' + 's_custom_filtering.ply', ldmk_s_custom_filtering)
                
                ldmk_t_custom_filtering = o3d.geometry.PointCloud()
                ldmk_t_custom_filtering.points = o3d.utility.Vector3dVector(np.array(ldmk_t.cpu()))
                o3d.io.write_point_cloud(self.path + intermediate_output_folder + 'custom_filtering_ldmk/' + 't_custom_filtering_pcd.ply', ldmk_t_custom_filtering)
                
                total_points = np.concatenate((rotated_ldmk_s, np.array(ldmk_t.cpu())), axis = 0)
                number_points_src = ldmk_s.shape[0]
                correspondences = [[i, i + number_points_src] for i in range(0, number_points_src)]
                line_set = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(total_points),
                    lines=o3d.utility.Vector2iVector(correspondences),
                )
                o3d.io.write_line_set(self.path + intermediate_output_folder +  'custom_filtering_ldmk/' + 'custom_filtering_line_set.ply', line_set)
                
                data_mod = {}
                final_indices = list(final_indices)

                vec_6d = data['vec_6d'][0][final_indices]
                data_mod['vec_6d'] = vec_6d[None, :]
                
                vec_6d_mask = data['vec_6d_mask'][0][final_indices]
                data_mod['vec_6d_mask'] = vec_6d_mask[None, :]
                
                vec_6d_ind = data['vec_6d_ind'][0][final_indices]
                data_mod['vec_6d_ind'] = vec_6d_ind[None, :]
                
                data_mod['s_pcd'] = data['s_pcd']
                data_mod['t_pcd'] = data['t_pcd']
                data_mod['batched_rot'] = data['batched_rot']
                data_mod['batched_trn'] = data['batched_trn']
        
                inlier_mask, inlier_rate = NeCoLoss.compute_inlier_mask(data_mod, inlier_thr, s2t_flow=coarse_flow)
                inlier_conf = inlier_conf[final_indices]
                match_filtered = inlier_mask[0] [  inlier_conf > inlier_thr ]
                inlier_rate_2 = match_filtered.sum()/(match_filtered.shape[0])
            '''
            # Custom filtering done with the 3rd method
            # VERSION 2
            '''
            if custom_filtering and intermediate_output_folder:
                print('Custom filtering is used')
                if not os.path.exists(self.path + intermediate_output_folder + 'custom_filtering_ldmk'):
                    os.mkdir(self.path + intermediate_output_folder + 'custom_filtering_ldmk')
                
                ldmk_s_np = np.array(ldmk_s.cpu())
                number_points = ldmk_s_np.shape[0]
                print('number points : ', number_points)
                ldmk_t_np = np.array(ldmk_t.cpu())

                distances = np.zeros((number_points,number_points))
                average_distance = 0
                for j in range(number_points):
                    for k in range(j + 1, number_points):
                        distances[j][k] = np.linalg.norm(ldmk_s_np[j] - ldmk_s_np[k])
                    row = distances[j]
                    non_zero_values = row[np.nonzero(row)]
                    if non_zero_values.size != 0:
                        average_distance += min(non_zero_values)/number_points
                
                print('average distance : ', average_distance)
                print('average distance multiplier : ', average_distance_multiplier)
                tau = average_distance_multiplier*average_distance
                number_transformations = 6
                    
                map_ldmk_s_correspondences = defaultdict(list)
                for idx, ldmk_s_point in enumerate(ldmk_s_np):
                    map_ldmk_s_correspondences[tuple(ldmk_s_point)].append(idx)
                
                print('number centers : ', number_centers)
                neighborhood_center_indices_list = np.linspace(0, ldmk_s_np.shape[0] - 1, num=number_centers).astype(int)
                outliers = defaultdict(int)

                print('number iterations custom filtering : ', number_iterations_custom_filtering)
                for _ in range(number_iterations_custom_filtering):
                    for neighborhood_center_index in neighborhood_center_indices_list:
                        neighborhood_center_source = ldmk_s_np[neighborhood_center_index]

                        distance_to_neighborhood_center = np.linalg.norm(ldmk_s_np - neighborhood_center_source, axis = 1)
                        distances_to_center = copy.deepcopy(distance_to_neighborhood_center)
                        
                        indices_neighborhood_points = np.where(distance_to_neighborhood_center < tau)[0]
                        
                        if indices_neighborhood_points.size < 3:
                            continue                 
                        
                        transformation_indices = []
                        for i in range(number_transformations):
                            random_indices = random.sample(list(indices_neighborhood_points), 3)
                            transformation_indices.append(random_indices)                       

                        point_indices_close_to_center = np.where(distances_to_center < tau)[0]
                        source_points_close_to_center = ldmk_s_np[point_indices_close_to_center]
                        target_points_close_to_center = ldmk_t_np[point_indices_close_to_center]

                        for n_transform in range(number_transformations):
                            source_point_1 = ldmk_s_np[transformation_indices[n_transform][0]]
                            target_point_1 = ldmk_t_np[transformation_indices[n_transform][0]]
                            source_point_2 = ldmk_s_np[transformation_indices[n_transform][1]]
                            target_point_2 = ldmk_t_np[transformation_indices[n_transform][1]]
                            source_point_3 = ldmk_s_np[transformation_indices[n_transform][2]]
                            target_point_3 = ldmk_t_np[transformation_indices[n_transform][2]]

                            X = np.empty((0,3), int)
                            X = np.append(X, np.array(np.expand_dims(source_point_1, axis=0)), axis=0)
                            X = np.append(X, np.array(np.expand_dims(source_point_2, axis=0)), axis=0)
                            X = np.append(X, np.array(np.expand_dims(source_point_3, axis=0)), axis=0)
                            Y = np.empty((0,3), int)
                            Y = np.append(Y, np.array(np.expand_dims(target_point_1, axis=0)), axis=0)
                            Y = np.append(Y, np.array(np.expand_dims(target_point_2, axis=0)), axis=0)
                            Y = np.append(Y, np.array(np.expand_dims(target_point_3, axis=0)), axis=0)

                            mean_X = np.mean(X, axis = 0)
                            mean_Y = np.mean(Y, axis = 0)
                        
                            Sxy = np.matmul( (Y - mean_Y).T, (X - mean_X) )
                            U, _, V = np.linalg.svd(Sxy, full_matrices=True)
                            S = np.eye(3)
                            UV_det = np.linalg.det(U) * np.linalg.det(V)
                            S[2, 2] = UV_det
                            sv = np.matmul( S, V )
                            R = np.matmul( U, sv)
                            t = mean_Y.T - np.matmul( R, mean_X.T )

                            thr = 0.05                        
                            points_after_transformation = (R @ source_points_close_to_center.T + np.expand_dims(t, axis=1)).T
                            norm_error = np.linalg.norm(points_after_transformation - target_points_close_to_center, axis = 1)
                            outlier_indices = np.where(norm_error > thr)[0]
                            for outlier_idx in outlier_indices:
                                weight = norm_error[outlier_idx]/tau
                                out_idx = point_indices_close_to_center[outlier_idx]
                                outliers[out_idx] = outliers[out_idx] + weight
                            
                final_indices = np.array([])
                for ldmk_s_point in map_ldmk_s_correspondences:
                    correspondence_indices = map_ldmk_s_correspondences[ldmk_s_point]
                    correspondence_indices_to_outliers = {key: outliers[key] for key in correspondence_indices if key in outliers}
                    if correspondence_indices_to_outliers:
                        correspondence_min = min(correspondence_indices_to_outliers, key=correspondence_indices_to_outliers.get)
                        final_indices = np.append(final_indices, correspondence_min)
                
                final_indices = np.sort(final_indices).astype(int) 
                ldmk_s = torch.tensor(ldmk_s_np[final_indices]).to('cuda:0')
                ldmk_t = torch.tensor(ldmk_t_np[final_indices]).to('cuda:0')
                
                rot = data['batched_rot'][0]
                ldmk_s_custom_filtering = o3d.geometry.PointCloud()
                ldmk_s_custom_filtering.points = o3d.utility.Vector3dVector(np.array(ldmk_s.cpu()))
                ldmk_s_custom_filtering.rotate(np.array(rot.cpu()), center=(0, 0, 0))
                rotated_ldmk_s = np.array(ldmk_s_custom_filtering.points)
                o3d.io.write_point_cloud(self.path + intermediate_output_folder + 'custom_filtering_ldmk/' + 's_custom_filtering.ply', ldmk_s_custom_filtering)
                
                ldmk_t_custom_filtering = o3d.geometry.PointCloud()
                ldmk_t_custom_filtering.points = o3d.utility.Vector3dVector(np.array(ldmk_t.cpu()))
                o3d.io.write_point_cloud(self.path + intermediate_output_folder + 'custom_filtering_ldmk/' + 't_custom_filtering_pcd.ply', ldmk_t_custom_filtering)
                
                total_points = np.concatenate((rotated_ldmk_s, np.array(ldmk_t.cpu())), axis = 0)
                number_points_src = ldmk_s.shape[0]
                correspondences = [[i, i + number_points_src] for i in range(0, number_points_src)]
                colors = np.tile(np.array([50, 50, 50]), (2*number_points_src, 1))
                line_set = o3d.geometry.LineSet()
                line_set.points=o3d.utility.Vector3dVector(total_points)
                line_set.lines =o3d.utility.Vector2iVector(correspondences)
                line_set.colors = o3d.utility.Vector3dVector(colors)
                    
                o3d.io.write_line_set(self.path + intermediate_output_folder +  'custom_filtering_ldmk/' + 'custom_filtering_line_set.ply', line_set)
                
                data_mod = {}
                final_indices = list(final_indices)

                vec_6d = data['vec_6d'][0][final_indices]
                data_mod['vec_6d'] = vec_6d[None, :]
                
                vec_6d_mask = data['vec_6d_mask'][0][final_indices]
                data_mod['vec_6d_mask'] = vec_6d_mask[None, :]
                
                vec_6d_ind = data['vec_6d_ind'][0][final_indices]
                data_mod['vec_6d_ind'] = vec_6d_ind[None, :]
                
                data_mod['s_pcd'] = data['s_pcd']
                data_mod['t_pcd'] = data['t_pcd']
                data_mod['batched_rot'] = data['batched_rot']
                data_mod['batched_trn'] = data['batched_trn']
        
                inlier_mask, inlier_rate = NeCoLoss.compute_inlier_mask(data_mod, inlier_thr, s2t_flow=coarse_flow)
                inlier_conf = inlier_conf[final_indices]
                match_filtered = inlier_mask[0] [  inlier_conf > inlier_thr ]
                inlier_rate_2 = match_filtered.sum()/(match_filtered.shape[0])
            '''
            
            # Custom filtering with the 4th method
            # VERSION 4
            if custom_filtering:
                print('Custom filtering is used')
                
                if intermediate_output_folder:
                    if not os.path.exists(self.path + intermediate_output_folder + folder_name + '_custom_filtering_ldmk'):
                        os.mkdir(self.path + intermediate_output_folder + folder_name + '_custom_filtering_ldmk')
                    
                    if matches_path:
                        matches = np.load(self.path + matches_path)
                        correspondences = np.array(matches['matches'])                    
                        src_pcd_points = data['src_pcd_list'][0]
                        src_pcd_points = np.array(src_pcd_points.cpu())
                        tgt_pcd_points = data['tgt_pcd_list'][0]
                        tgt_pcd_points = np.array(tgt_pcd_points.cpu())
                        
                        src_pcd = o3d.geometry.PointCloud()
                        src_pcd.points = o3d.utility.Vector3dVector(src_pcd_points)
                        rot = data['batched_rot'][0]
                        src_pcd.rotate(np.array(rot.cpu()), center=(0, 0, 0))
                        src_pcd_points_rotated = np.array(src_pcd.points)
                        n_points_source = src_pcd_points_rotated.shape[0]
                        correspondences[:, 1] = correspondences[:, 1] + n_points_source
                        
                        total_points = np.concatenate((src_pcd_points_rotated, tgt_pcd_points), axis = 0)
                        line_set = o3d.geometry.LineSet()
                        line_set.points=o3d.utility.Vector3dVector(total_points)
                        line_set.lines =o3d.utility.Vector2iVector(correspondences)
                        o3d.io.write_line_set(self.path + intermediate_output_folder + folder_name + '_custom_filtering_ldmk/' + 'gt_line_set.ply', line_set)
                        
                    ldmk_s_np = np.array(ldmk_s.cpu())
                    number_points = ldmk_s_np.shape[0]
                    print('number of landmarks : ', number_points)
                    ldmk_t_np = np.array(ldmk_t.cpu())

                    ldmk_s_pcd = o3d.geometry.PointCloud()
                    ldmk_s_pcd.points = o3d.utility.Vector3dVector(np.array(ldmk_s_np))
                    o3d.io.write_point_cloud(self.path + intermediate_output_folder + folder_name + '_custom_filtering_ldmk/ldmk_s_pcd.ply', ldmk_s_pcd)

                    rot = data['batched_rot'][0]
                    ldmk_s_pcd.rotate(np.array(rot.cpu()), center=(0, 0, 0))
                    o3d.io.write_point_cloud(self.path + intermediate_output_folder + folder_name + '_custom_filtering_ldmk/ldmk_s_rotated_pcd.ply', ldmk_s_pcd)
                    rotated_ldmk_s_np = np.array(ldmk_s_pcd.points)

                    ldmk_t_pcd = o3d.geometry.PointCloud()
                    ldmk_t_pcd.points = o3d.utility.Vector3dVector(np.array(ldmk_t_np))
                    o3d.io.write_point_cloud(self.path + intermediate_output_folder + folder_name + '_custom_filtering_ldmk/ldmk_t_pcd.ply', ldmk_t_pcd)
                else:
                    ldmk_s_np = np.array(ldmk_s.cpu())
                    ldmk_s_pcd = o3d.geometry.PointCloud()
                    ldmk_s_pcd.points = o3d.utility.Vector3dVector(np.array(ldmk_s_np))
                    rot = data['batched_rot'][0]
                    ldmk_s_pcd.rotate(np.array(rot.cpu()), center=(0, 0, 0))
                    rotated_ldmk_s_np = np.array(ldmk_s_pcd.points)
                    ldmk_t_np = np.array(ldmk_t.cpu())
                
                number_points = ldmk_s_np.shape[0]
                distances = np.zeros((number_points,number_points))
                average_distance = 0
                for j in range(number_points):
                    for k in range(j + 1, number_points):
                        distances[j][k] = np.linalg.norm(ldmk_s_np[j] - ldmk_s_np[k])
                    row = distances[j]
                    non_zero_values = row[np.nonzero(row)]
                    if non_zero_values.size != 0:
                        average_distance += min(non_zero_values)/number_points
                
                print('average distance : ', average_distance)
                print('average distance multiplier : ', average_distance_multiplier)
                tau = average_distance_multiplier*average_distance
                number_transformations = 6
                    
                map_ldmk_s_correspondences = defaultdict(list)
                for idx, ldmk_s_point in enumerate(ldmk_s_np):
                    map_ldmk_s_correspondences[tuple(ldmk_s_point)].append(idx)
                
                print('number of unique source landmarks : ', len(map_ldmk_s_correspondences))
                print('number of centers : ', number_centers)
                print('sampling : ', sampling)
                
                centers_pcd = o3d.geometry.PointCloud()
                if sampling == 'linspace':
                    neighborhood_center_indices_list = np.linspace(0, ldmk_s_np.shape[0] - 1, num=number_centers).astype(int)
                    centers_points = ldmk_s_np[neighborhood_center_indices_list]
                    centers_pcd.points = o3d.utility.Vector3dVector(centers_points)
                elif sampling == 'poisson' and mesh_path and source_trans:
                    source_mesh_pcd = o3d.io.read_point_cloud(self.path + mesh_path)
                    src_trans_file=h5py.File(self.path + source_trans, "r")
                    src_pcd_transform = np.array(src_trans_file['transformation'])
                    src_pcd_transform_inverse = np.linalg.inv(src_pcd_transform)
                    rot_inv = src_pcd_transform_inverse[:3, :3]
                    trn_inv = src_pcd_transform_inverse[:3, 3]
                    source_mesh_pcd.rotate(rot_inv, center=(0, 0, 0))
                    source_mesh_pcd.translate(trn_inv)
                    radii = [0.005]
                    source_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(source_mesh_pcd, o3d.utility.DoubleVector(radii))
                    if intermediate_output_folder:
                        o3d.io.write_triangle_mesh(self.path + intermediate_output_folder + folder_name + '_custom_filtering_ldmk/source_mesh.ply', source_mesh)
                    centers_pcd = source_mesh.sample_points_poisson_disk(number_of_points=number_centers)
                
                if intermediate_output_folder:
                    o3d.io.write_point_cloud(self.path + intermediate_output_folder + folder_name + '_custom_filtering_ldmk/centers_pcd.ply', centers_pcd)
                
                neighborhood_centers = np.array(centers_pcd.points)
                print('number of neighborhood centers : ', neighborhood_centers.shape[0])
                outliers = defaultdict(float)
                inliers = defaultdict(float)

                print('number of iterations of custom filtering : ', number_iterations_custom_filtering)
                print('inlier outlier threshold : ', inlier_outlier_thr)
                
                for _ in range(number_iterations_custom_filtering):
                    n_center = 0
                    for neighborhood_center in neighborhood_centers:

                        distance_to_neighborhood_center = np.linalg.norm(ldmk_s_np - neighborhood_center, axis = 1)
                        distances_to_center = copy.deepcopy(distance_to_neighborhood_center)
                        
                        indices_neighborhood_points = np.where(distance_to_neighborhood_center < tau)[0]
                        
                        if indices_neighborhood_points.size < 3:
                            continue                 
                        
                        transformation_indices = []
                        for i in range(number_transformations):
                            random_indices = random.sample(list(indices_neighborhood_points), 3)
                            transformation_indices.append(random_indices)                       

                        point_indices_close_to_center = np.where(distances_to_center < tau)[0]
                        source_points_close_to_center = ldmk_s_np[point_indices_close_to_center]
                        target_points_close_to_center = ldmk_t_np[point_indices_close_to_center]
                        
                        n_outlier_indices = float('inf')
                        final_outliers = np.array([])
                        final_inliers = np.array([])
                        final_norm_error = np.array([])
                        final_R = np.array([])
                        final_t = np.array([])
                        
                        for n_transform in range(number_transformations):
                            source_point_1 = ldmk_s_np[transformation_indices[n_transform][0]]
                            target_point_1 = ldmk_t_np[transformation_indices[n_transform][0]]
                            source_point_2 = ldmk_s_np[transformation_indices[n_transform][1]]
                            target_point_2 = ldmk_t_np[transformation_indices[n_transform][1]]
                            source_point_3 = ldmk_s_np[transformation_indices[n_transform][2]]
                            target_point_3 = ldmk_t_np[transformation_indices[n_transform][2]]

                            X = np.empty((0,3), int)
                            X = np.append(X, np.array(np.expand_dims(source_point_1, axis=0)), axis=0)
                            X = np.append(X, np.array(np.expand_dims(source_point_2, axis=0)), axis=0)
                            X = np.append(X, np.array(np.expand_dims(source_point_3, axis=0)), axis=0)
                            Y = np.empty((0,3), int)
                            Y = np.append(Y, np.array(np.expand_dims(target_point_1, axis=0)), axis=0)
                            Y = np.append(Y, np.array(np.expand_dims(target_point_2, axis=0)), axis=0)
                            Y = np.append(Y, np.array(np.expand_dims(target_point_3, axis=0)), axis=0)

                            mean_X = np.mean(X, axis = 0)
                            mean_Y = np.mean(Y, axis = 0)
                        
                            Sxy = np.matmul( (Y - mean_Y).T, (X - mean_X) )
                            U, _, V = np.linalg.svd(Sxy, full_matrices=True)
                            S = np.eye(3)
                            UV_det = np.linalg.det(U) * np.linalg.det(V)
                            S[2, 2] = UV_det
                            sv = np.matmul( S, V )
                            R = np.matmul( U, sv)
                            t = mean_Y.T - np.matmul( R, mean_X.T )
                        
                            points_after_transformation = (R @ source_points_close_to_center.T + np.expand_dims(t, axis=1)).T
                            norm_error = np.linalg.norm(points_after_transformation - target_points_close_to_center, axis = 1)
                            outlier_indices = np.where(norm_error > inlier_outlier_thr)[0]
                            inlier_indices = np.where(norm_error <= inlier_outlier_thr)[0]
                            
                            if outlier_indices.shape[0] < n_outlier_indices:
                                n_outlier_indices = outlier_indices.shape[0]
                                final_outliers = outlier_indices
                                final_inliers = inlier_indices
                                final_norm_error = norm_error
                                final_R = R
                                final_t = t
                        
                        if final_outliers.size == 0 or final_norm_error.size == 0 or final_inliers.size == 0 or final_R.size == 0 or final_t.size == 0:
                            n_center += 1
                            continue
                        
                        if intermediate_output_folder:
                            landmarks_transformed_s = (final_R @ ldmk_s_np.T + np.expand_dims(final_t, axis=1)).T
                            landmarks_transformed_s_pcd = o3d.geometry.PointCloud()
                            landmarks_transformed_s_pcd.points = o3d.utility.Vector3dVector(landmarks_transformed_s)
                            o3d.io.write_point_cloud(self.path + intermediate_output_folder + folder_name + '_custom_filtering_ldmk/landmarks_transformed_' + str(n_center) + '.ply', landmarks_transformed_s_pcd)
                        
                            neighborhood_center_pcd = o3d.geometry.PointCloud()
                            neighborhood_center_pcd.points = o3d.utility.Vector3dVector(neighborhood_center[None, :])
                            rot = data['batched_rot'][0]
                            neighborhood_center_pcd.rotate(np.array(rot.cpu()), center=(0, 0, 0))
                            
                            inliers_pcd_points = rotated_ldmk_s_np[point_indices_close_to_center[final_inliers]]
                            inliers_pcd_points = np.concatenate((inliers_pcd_points, np.array(neighborhood_center_pcd.points)))
                            inliers_colors = np.zeros((inliers_pcd_points.shape[0], inliers_pcd_points.shape[1]))
                            color = np.squeeze(np.random.rand(1,3), axis=0)
                            for inlier_idx in range(inliers_pcd_points.shape[0]):
                                if inlier_idx < inliers_pcd_points.shape[0] - 1:
                                    inliers_colors[inlier_idx] = color
                                else:
                                    inliers_colors[inlier_idx] = np.array([0, 1, 0])
                            inliers_pcd = o3d.geometry.PointCloud()
                            inliers_pcd.points = o3d.utility.Vector3dVector(np.array(inliers_pcd_points))
                            inliers_pcd.colors = o3d.utility.Vector3dVector(np.array(inliers_colors))
                            o3d.io.write_point_cloud(self.path + intermediate_output_folder + folder_name + '_custom_filtering_ldmk/inliers_' + str(n_center) + '.ply', inliers_pcd)
                        
                            inliers_pcd_points_s = rotated_ldmk_s_np[point_indices_close_to_center[final_inliers]]
                            inliers_pcd_points_t = ldmk_t_np[point_indices_close_to_center[final_inliers]]
                            total_inliers = np.concatenate((inliers_pcd_points_s, inliers_pcd_points_t), axis = 0)
                            number_inliers_src = inliers_pcd_points_s.shape[0]
                            inliers_correspondences = np.array([[i, i + number_inliers_src] for i in range(0, number_inliers_src)])
                            inliers_line_set = o3d.geometry.LineSet()
                            inliers_line_set.points=o3d.utility.Vector3dVector(total_inliers)
                            inliers_line_set.lines =o3d.utility.Vector2iVector(inliers_correspondences)
                            o3d.io.write_line_set(self.path + intermediate_output_folder + folder_name + '_custom_filtering_ldmk/inliers_line_set_' + str(n_center) + '.ply', inliers_line_set)
                            
                            outliers_pcd_points_s = rotated_ldmk_s_np[point_indices_close_to_center[final_outliers]]
                            outliers_pcd_points_t = ldmk_t_np[point_indices_close_to_center[final_outliers]]
                            total_outliers = np.concatenate((outliers_pcd_points_s, outliers_pcd_points_t), axis = 0)
                            number_outliers_src = outliers_pcd_points_s.shape[0]
                            outliers_correspondences = np.array([[i, i + number_outliers_src] for i in range(0, number_outliers_src)])
                            outliers_line_set = o3d.geometry.LineSet()
                            outliers_line_set.points=o3d.utility.Vector3dVector(total_outliers)
                            outliers_line_set.lines =o3d.utility.Vector2iVector(outliers_correspondences)
                            o3d.io.write_line_set(self.path + intermediate_output_folder + folder_name + '_custom_filtering_ldmk/outliers_line_set_' + str(n_center) + '.ply', outliers_line_set)

                        for outlier_idx in final_outliers:
                            weight = final_norm_error[outlier_idx]/tau
                            out_idx = point_indices_close_to_center[outlier_idx]
                            outliers[out_idx] = outliers[out_idx] + weight
                        
                        for inlier_idx in final_inliers:
                            weight = final_norm_error[inlier_idx]/tau
                            in_idx = point_indices_close_to_center[inlier_idx]
                            inliers[in_idx] = inliers[in_idx] + weight
                            
                        n_center += 1
                
                number_points = ldmk_s_np.shape[0]
                final_outlier_indices = np.arange(number_points)
                final_indices = np.array([])
                for ldmk_s_point in map_ldmk_s_correspondences:
                    correspondence_indices = map_ldmk_s_correspondences[ldmk_s_point]
                    correspondence_indices_to_inliers = {key: inliers[key] for key in correspondence_indices if key in inliers}
                    
                    if correspondence_indices_to_inliers:
                        correspondence_min = min(correspondence_indices_to_inliers, key=correspondence_indices_to_inliers.get)
                        final_indices = np.append(final_indices, correspondence_min)
                    # correspondence_indices_to_outliers = {key: outliers[key] for key in correspondence_indices if key in outliers}
                    # elif correspondence_indices_to_outliers:
                    # correspondence_min = min(correspondence_indices_to_outliers, key=correspondence_indices_to_outliers.get)
                    # final_indices = np.append(final_indices, correspondence_min)
                
                final_indices = set(final_indices)
                final_outlier_indices = [outlier_idx for outlier_idx in final_outlier_indices if outlier_idx not in final_indices]                 
                final_indices = np.sort(list(final_indices)).astype(int)
                print('number of landmarks after custom filtering : ', final_indices.shape[0])
                ldmk_s = torch.tensor(ldmk_s_np[final_indices]).to('cuda:0')
                ldmk_t = torch.tensor(ldmk_t_np[final_indices]).to('cuda:0')
                
                if matches_path:
                    mask = np.array([])
                    matches = np.load(self.path + matches_path)
                    correspondences = np.array(matches['matches'])
                    ind_src = correspondences[:, 0]
                    ind_tgt = correspondences[:, 1]
                    
                    src_pcd_points = data['src_pcd_list'][0]
                    src_pcd_points = np.array(src_pcd_points.cpu())
                    tgt_pcd_points = data['tgt_pcd_list'][0]
                    tgt_pcd_points = np.array(tgt_pcd_points.cpu())
                    
                    matches_source = src_pcd_points[ind_src]
                    matches_target = tgt_pcd_points[ind_tgt]
                    
                    for i in range(ldmk_s_np[final_indices].shape[0]):
                        s_ldmk = np.array(ldmk_s_np[final_indices][i])
                        t_ldmk = np.array(ldmk_t_np[final_indices][i])
                        distance_to_s_ldmk = np.linalg.norm(matches_source - s_ldmk, axis=1)
                        distance_to_t_ldmk = np.linalg.norm(matches_target - t_ldmk, axis=1)
                        indices_neigh_s_ldmk = set(np.where(distance_to_s_ldmk < gt_thr)[0])
                        indices_neigh_t_ldmk = set(np.where(distance_to_t_ldmk < gt_thr)[0])
                        if indices_neigh_s_ldmk & indices_neigh_t_ldmk:
                            mask = np.append(mask, True)
                        else:
                            mask = np.append(mask, False)
                    
                    custom_filtering_true_correspondences_mask = mask.astype(bool)
                    n_true_custom_filtering_correspondences = int(custom_filtering_true_correspondences_mask.sum())
                    n_total_custom_filtering_correspondences = custom_filtering_true_correspondences_mask.shape[0]
                    print('number of true landmark correspondences returned from custom filtering : ', n_true_custom_filtering_correspondences , ' out of ', n_total_custom_filtering_correspondences)
                    
                    if self.feature_extractor == 'kpfcn':
                        if n_total_custom_filtering_correspondences != 0:
                            print('fraction of true landmark correspondences returned from custom filtering : ', n_true_custom_filtering_correspondences/n_total_custom_filtering_correspondences )
                        else:
                            print('fraction of true landmark correspondences returned from custom filtering : ', 0 )
                    elif self.feature_extractor == 'fcgf':
                        if n_total_custom_filtering_correspondences != 0:
                            print('fraction of true landmark correspondences returned from FCGF based custom filtering : ', n_true_custom_filtering_correspondences/n_total_custom_filtering_correspondences )
                        else:
                            print('fraction of true landmark correspondences returned from FCGF based custom filtering : ', 0 )

                    final_custom_filtering_true_correspondences_mask = np.array([False for i in range(ldmk_s_np.shape[0])])
                    final_custom_filtering_true_correspondences_mask[final_indices] = custom_filtering_true_correspondences_mask
                    custom_and_lepard_true_correspondences_mask = final_custom_filtering_true_correspondences_mask & lepard_true_correspondences_mask

                    if self.feature_extractor == 'kpfcn':
                        if n_true_custom_filtering_correspondences != 0:
                            print('fraction of true landmark correspondences returned from custom filtering also returned from Lepard : ', int(custom_and_lepard_true_correspondences_mask.sum())/n_true_custom_filtering_correspondences)
                        else:
                            print('fraction of true landmark correspondences returned from custom filtering also returned from Lepard : ', 0)
                    elif self.feature_extractor == 'fcgf':
                        if n_true_custom_filtering_correspondences != 0:
                            print('fraction of true landmark correspondences returned from FCGF based custom filtering also returned from Lepard : ', int(custom_and_lepard_true_correspondences_mask.sum())/n_true_custom_filtering_correspondences)
                        else:
                            print('fraction of true landmark correspondences returned from FCGF based custom filtering also returned from Lepard : ', 0)
                    
                if final_indices.shape[0] != 0 and intermediate_output_folder:

                    # inliers
                    rot = data['batched_rot'][0]
                    ldmk_s_custom_filtering = o3d.geometry.PointCloud()
                    ldmk_s_custom_filtering.points = o3d.utility.Vector3dVector(np.array(ldmk_s.cpu()))
                    ldmk_s_custom_filtering.rotate(np.array(rot.cpu()), center=(0, 0, 0))
                    rotated_ldmk_s = np.array(ldmk_s_custom_filtering.points)
                    o3d.io.write_point_cloud(self.path + intermediate_output_folder + folder_name + '_custom_filtering_ldmk/' + 's_custom_filtering_pcd.ply', ldmk_s_custom_filtering)
                    
                    ldmk_t_custom_filtering = o3d.geometry.PointCloud()
                    ldmk_t_custom_filtering.points = o3d.utility.Vector3dVector(np.array(ldmk_t.cpu()))
                    o3d.io.write_point_cloud(self.path + intermediate_output_folder + folder_name + '_custom_filtering_ldmk/' + 't_custom_filtering_pcd.ply', ldmk_t_custom_filtering)
                    
                    total_points = np.concatenate((rotated_ldmk_s, np.array(ldmk_t.cpu())), axis = 0)
                    number_points_src = ldmk_s.shape[0]
                    correspondences = [[i, i + number_points_src] for i in range(0, number_points_src)]
                    colors = np.tile(np.array([0.5, 0.5, 0.5]), (2*number_points_src, 1))
                    line_set = o3d.geometry.LineSet()
                    line_set.points=o3d.utility.Vector3dVector(total_points)
                    line_set.lines =o3d.utility.Vector2iVector(correspondences)
                    line_set.colors = o3d.utility.Vector3dVector(colors)
                    o3d.io.write_line_set(self.path + intermediate_output_folder + folder_name + '_custom_filtering_ldmk/' + 'custom_filtering_line_set.ply', line_set)
                
                    # outliers 
                    ldmk_s_np_outliers = ldmk_s_np[final_outlier_indices]
                    ldmk_t_np_outliers = ldmk_t_np[final_outlier_indices]
                    ldmk_s_outliers_custom_filtering = o3d.geometry.PointCloud()
                    ldmk_s_outliers_custom_filtering.points = o3d.utility.Vector3dVector(ldmk_s_np_outliers)
                    ldmk_s_outliers_custom_filtering.rotate(np.array(rot.cpu()), center=(0, 0, 0))
                    rotated_ldmk_s_np_outliers = np.array(ldmk_s_outliers_custom_filtering.points)
                    o3d.io.write_point_cloud(self.path + intermediate_output_folder + folder_name + '_custom_filtering_ldmk/' + 's_custom_filtering_pcd_outliers.ply', ldmk_s_outliers_custom_filtering)
                    
                    ldmk_t_outliers_custom_filtering = o3d.geometry.PointCloud()
                    ldmk_t_outliers_custom_filtering.points = o3d.utility.Vector3dVector(ldmk_t_np_outliers)
                    o3d.io.write_point_cloud(self.path + intermediate_output_folder + folder_name + '_custom_filtering_ldmk/' + 't_custom_filtering_pcd_outliers.ply', ldmk_t_outliers_custom_filtering)
                    
                    total_points = np.concatenate((rotated_ldmk_s_np_outliers, ldmk_t_np_outliers), axis = 0)
                    number_points_src = rotated_ldmk_s_np_outliers.shape[0]
                    correspondences = [[i, i + number_points_src] for i in range(0, number_points_src)]
                    colors = np.tile(np.array([0.5, 0.5, 0.5]), (2*number_points_src, 1))
                    line_set_outliers = o3d.geometry.LineSet()
                    line_set_outliers.points=o3d.utility.Vector3dVector(total_points)
                    line_set_outliers.lines =o3d.utility.Vector2iVector(correspondences)
                    line_set_outliers.colors = o3d.utility.Vector3dVector(colors)
                    o3d.io.write_line_set(self.path + intermediate_output_folder + folder_name + '_custom_filtering_ldmk/' + 'custom_filtering_line_set_outliers.ply', line_set_outliers)
                
                data_mod = {}
                final_indices = list(final_indices)

                vec_6d = data['vec_6d'][0][final_indices]
                data_mod['vec_6d'] = vec_6d[None, :]
                vec_6d_mask = data['vec_6d_mask'][0][final_indices]
                data_mod['vec_6d_mask'] = vec_6d_mask[None, :]
                vec_6d_ind = data['vec_6d_ind'][0][final_indices]
                data_mod['vec_6d_ind'] = vec_6d_ind[None, :]
                
                data_mod['s_pcd'] = data['s_pcd']
                data_mod['t_pcd'] = data['t_pcd']
                data_mod['batched_rot'] = data['batched_rot']
                data_mod['batched_trn'] = data['batched_trn']
        
                inlier_mask, inlier_rate = NeCoLoss.compute_inlier_mask(data_mod, inlier_thr, s2t_flow=coarse_flow)
                inlier_conf = inlier_conf[final_indices]
                match_filtered = inlier_mask[0] [  inlier_conf > inlier_thr ]
                inlier_rate_2 = match_filtered.sum()/(match_filtered.shape[0])
                
            return ldmk_s, ldmk_t, inlier_rate, inlier_rate_2