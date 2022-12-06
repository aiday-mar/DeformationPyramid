import numpy as np
import open3d as o3d
from functools import partial
import torch
import os
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors
from correspondence.datasets._3dmatch import _3DMatch
from correspondence.datasets._4dmatch import _4DMatch
from correspondence.datasets._4dmatch_multiview import _4DMatch_Multiview
from correspondence.datasets.utils import blend_scene_flow, multual_nn_correspondence
from correspondence.lib.visualization import *
from torch.utils.data import DataLoader

path = '/home/aiday.kyzy/dataset/Synthetic/'

def batch_grid_subsampling_kpconv(points, batches_len, features=None, labels=None, sampleDl=0.1, max_p=0, verbose=0, random_grid_orient=True):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    """
    if (features is None) and (labels is None):
        s_points, s_len = cpp_subsampling.subsample_batch(points,
                                                          batches_len,
                                                          sampleDl=sampleDl,
                                                          max_p=max_p,
                                                          verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len)

    elif (labels is None):
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(points,
                                                                      batches_len,
                                                                      features=features,
                                                                      sampleDl=sampleDl,
                                                                      max_p=max_p,
                                                                      verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features)

    elif (features is None):
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(points,
                                                                    batches_len,
                                                                    classes=labels,
                                                                    sampleDl=sampleDl,
                                                                    max_p=max_p,
                                                                    verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_labels)

    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(points,
                                                                              batches_len,
                                                                              features=features,
                                                                              classes=labels,
                                                                              sampleDl=sampleDl,
                                                                              max_p=max_p,
                                                                              verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features), torch.from_numpy(s_labels)

def batch_neighbors_kpconv(queries, supports, q_batches, s_batches, radius, max_neighbors):
    """
    Computes neighbors for a batch of queries and supports, apply radius search
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """

    neighbors = cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)
    if max_neighbors > 0:
        return torch.from_numpy(neighbors[:, :max_neighbors])
    else:
        return torch.from_numpy(neighbors)


def collate_fn_4dmatch_multiview(multiview_data, config, neighborhood_limits ):

    assert len(multiview_data) == 1

    pcds, pcd_pairs, pairwise_flows, pairwise_overlap, _ , axis_node, poses = multiview_data [0]

    batched_points_list = []
    batched_features_list = []
    batched_lengths_list = []

    src_pcd_list = []
    tgt_pcd_list = []

    batched_rot = []
    batched_trn = []

    sflow_list = []
    metric_index_list = [] #for feature matching recall computation

    for ind in range ( len(pcd_pairs) ) :
        # for ind, ( src_pcd, tgt_pcd, src_feats, tgt_feats, correspondences, rot, trn, s2t_flow, metric_index) in enumerate(multiview_data):

        s_id , t_id = pcd_pairs[ind]
        s2t_flow = pairwise_flows[ind]

        s_pose, t_pose = poses[s_id], poses[t_id]
        Transform = np.matmul(np.linalg.inv(t_pose), s_pose) # relative transform from s2t
        rot = Transform[:3, :3]
        trn = Transform[:3, 3:]

        src_pcd, tgt_pcd = pcds[s_id], pcds[t_id]
        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)

        src_pcd_list.append(torch.from_numpy(src_pcd) )
        tgt_pcd_list.append(torch.from_numpy(tgt_pcd) )

        batched_points_list.append(src_pcd)
        batched_points_list.append(tgt_pcd)
        batched_features_list.append(src_feats)
        batched_features_list.append(tgt_feats)
        batched_lengths_list.append(len(src_pcd))
        batched_lengths_list.append(len(tgt_pcd))

        batched_rot.append( torch.from_numpy(rot).float())
        batched_trn.append( torch.from_numpy(trn).float())

        # gt_cov_list.append(gt_cov)
        sflow_list.append( torch.from_numpy(s2t_flow).float() )

        # if metric_index is None:
        #     metric_index_list = None
        # else :
        #     metric_index_list.append ( torch.from_numpy(metric_index))
        #

    # if timers: cnter['collate_load_batch'] = time.time() - st
    batched_features = torch.from_numpy(np.concatenate(batched_features_list, axis=0))
    batched_points = torch.from_numpy(np.concatenate(batched_points_list, axis=0))
    batched_lengths = torch.from_numpy(np.array(batched_lengths_list)).int()

    batched_rot = torch.stack(batched_rot,dim=0)
    batched_trn = torch.stack(batched_trn,dim=0)

    # Starting radius of convolutions
    r_normal = config.first_subsampling_dl * config.conv_radius

    # Starting layer
    layer_blocks = []
    layer = 0

    # Lists of inputs
    input_points = []
    input_neighbors = []
    input_pools = []
    input_upsamples = []
    input_batches_len = []

    # construt kpfcn inds
    for block_i, block in enumerate(config.architecture):

        # Stop when meeting a global pooling or upsampling
        if 'global' in block or 'upsample' in block:
            break

        # Get all blocks of the layer
        if not ('pool' in block or 'strided' in block):
            layer_blocks += [block]
            if block_i < len(config.architecture) - 1 and not ('upsample' in config.architecture[block_i + 1]):
                continue

        # Convolution neighbors indices
        # ****************************
        if layer_blocks:
            # Convolutions are done in this layer, compute the neighbors with the good radius
            if np.any(['deformable' in blck for blck in layer_blocks[:-1]]):
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal
            conv_i = batch_neighbors_kpconv(batched_points, batched_points, batched_lengths, batched_lengths, r,
                                            neighborhood_limits[layer])

        else:
            # This layer only perform pooling, no neighbors required
            conv_i = torch.zeros((0, 1), dtype=torch.int64)

        # Pooling neighbors indices
        # *************************
        # If end of layer is a pooling operation
        if 'pool' in block or 'strided' in block:

            # New subsampling length
            dl = 2 * r_normal / config.conv_radius

            # Subsampled points
            pool_p, pool_b = batch_grid_subsampling_kpconv(batched_points, batched_lengths, sampleDl=dl)

            # Radius of pooled neighbors
            if 'deformable' in block:
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal

            # Subsample indices
            pool_i = batch_neighbors_kpconv(pool_p, batched_points, pool_b, batched_lengths, r,
                                            neighborhood_limits[layer])

            # Upsample indices (with the radius of the next layer to keep wanted density)
            up_i = batch_neighbors_kpconv(batched_points, pool_p, batched_lengths, pool_b, 2 * r,
                                          neighborhood_limits[layer])

        else:
            # No pooling in the end of this layer, no pooling indices required
            pool_i = torch.zeros((0, 1), dtype=torch.int64)
            pool_p = torch.zeros((0, 3), dtype=torch.float32)
            pool_b = torch.zeros((0,), dtype=torch.int64)
            up_i = torch.zeros((0, 1), dtype=torch.int64)

        # Updating input lists
        input_points += [batched_points.float()]
        input_neighbors += [conv_i.long()]
        input_pools += [pool_i.long()]
        input_upsamples += [up_i.long()]
        input_batches_len += [batched_lengths]

        # New points for next layer
        batched_points = pool_p
        batched_lengths = pool_b

        # Update radius and reset blocks
        r_normal *= 2
        layer += 1
        layer_blocks = []

    # coarse infomation
    coarse_level = config.coarse_level
    pts_num_coarse = input_batches_len[coarse_level].view(-1, 2)
    b_size = pts_num_coarse.shape[0]
    src_pts_max, tgt_pts_max = pts_num_coarse.amax(dim=0)
    coarse_pcd = input_points[coarse_level] # .numpy()
    coarse_matches= []
    coarse_flow = []
    src_ind_coarse_split= [] # src_feats shape :[b_size * src_pts_max]
    src_ind_coarse = []
    tgt_ind_coarse_split= []
    tgt_ind_coarse = []
    accumu = 0
    src_mask = torch.zeros([b_size, src_pts_max], dtype=torch.bool)
    tgt_mask = torch.zeros([b_size, tgt_pts_max], dtype=torch.bool)

    for entry_id, cnt in enumerate( pts_num_coarse ): #input_batches_len[-1].numpy().reshape(-1,2)) :
        n_s_pts, n_t_pts = cnt

        '''split mask for bottlenect feats'''
        src_mask[entry_id][:n_s_pts] = 1
        tgt_mask[entry_id][:n_t_pts] = 1

        '''split indices of bottleneck feats'''
        src_ind_coarse_split.append( torch.arange( n_s_pts ) + entry_id * src_pts_max )
        tgt_ind_coarse_split.append( torch.arange( n_t_pts ) + entry_id * tgt_pts_max )
        src_ind_coarse.append( torch.arange( n_s_pts ) + accumu )
        tgt_ind_coarse.append( torch.arange( n_t_pts ) + accumu + n_s_pts )

        '''get match at coarse level'''
        c_src_pcd_np = coarse_pcd[accumu : accumu + n_s_pts].numpy()
        c_tgt_pcd_np = coarse_pcd[accumu + n_s_pts: accumu + n_s_pts + n_t_pts].numpy()
        #interpolate flow
        f_src_pcd = batched_points_list[entry_id * 2]
        c_flow = blend_scene_flow( c_src_pcd_np, f_src_pcd, sflow_list[entry_id].numpy(), knn=3)
        c_src_pcd_deformed = c_src_pcd_np + c_flow
        s_pc_wrapped = ( batched_rot[entry_id].numpy() @ c_src_pcd_deformed.T  + batched_trn [entry_id].numpy() ).T
        coarse_match_gt = torch.from_numpy( multual_nn_correspondence(s_pc_wrapped , c_tgt_pcd_np , search_radius=config['coarse_match_radius'])  )# 0.1m scaled
        coarse_matches.append(coarse_match_gt)
        coarse_flow.append(torch.from_numpy(c_flow))

        accumu = accumu + n_s_pts + n_t_pts
        vis=False # for debug
        if vis :
            viz_coarse_nn_correspondence_mayavi(c_src_pcd_np, c_tgt_pcd_np, coarse_match_gt, scale_factor=0.02)

    src_ind_coarse_split = torch.cat(src_ind_coarse_split)
    tgt_ind_coarse_split = torch.cat(tgt_ind_coarse_split)
    src_ind_coarse = torch.cat(src_ind_coarse)
    tgt_ind_coarse = torch.cat(tgt_ind_coarse)

    dict_inputs = {
        'src_pcd_list': src_pcd_list,
        'tgt_pcd_list': tgt_pcd_list,
        'points': input_points,
        'neighbors': input_neighbors,
        'pools': input_pools,
        'upsamples': input_upsamples,
        'features': batched_features.float(),
        'stack_lengths': input_batches_len,
        'coarse_matches': coarse_matches,
        'coarse_flow' : coarse_flow,
        'src_mask': src_mask,
        'tgt_mask': tgt_mask,
        'src_ind_coarse_split': src_ind_coarse_split,
        'tgt_ind_coarse_split': tgt_ind_coarse_split,
        'src_ind_coarse': src_ind_coarse,
        'tgt_ind_coarse': tgt_ind_coarse,
        'batched_rot': batched_rot,
        'batched_trn': batched_trn,
        'sflow_list': sflow_list,
        'pcd_pairs': torch.from_numpy( pcd_pairs ),
        'axis_node': axis_node
    }

    return dict_inputs

def collate_fn_4dmatch_multiview_sequence(multiview_data, config, neighborhood_limits):

    assert len(multiview_data) == 1

    pcds, pcd_pairs, pairwise_flows, pairwise_overlap, _ , axis_node, poses = multiview_data [0]
    pairwise_data_list = []

    for ind in range ( len(pcd_pairs) ) :

        s_id , t_id = pcd_pairs[ind]
        s2t_flow = pairwise_flows[ind]

        s_pose, t_pose = poses[s_id], poses[t_id]
        Transform = np.matmul(np.linalg.inv(t_pose), s_pose) # relative transform from s2t
        rot = Transform[:3, :3]
        trn = Transform[:3, 3:]

        src_pcd, tgt_pcd = pcds[s_id], pcds[t_id]
        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)

        pair_data = [(src_pcd, tgt_pcd, src_feats, tgt_feats, np.zeros([1]), rot, trn, s2t_flow, None, None, None)]
        pair_batched = collate_fn_4dmatch( pair_data, config, neighborhood_limits)
        pairwise_data_list.append(pair_batched)

    return pcd_pairs, pairwise_data_list

def collate_fn_4dmatch(pairwise_data, config, neighborhood_limits, output_folder = None, base = None, coarse_level = None, feature_extractor = None):

    print('feature extractor inside of collate_fn_4dmatch : ', feature_extractor)
    batched_points_list = []
    batched_features_list = []
    batched_lengths_list = []

    src_pcd_list = []
    tgt_pcd_list = []

    batched_rot = []
    batched_trn = []

    sflow_list = []
    correspondences_list = []
    depth_paths_list = {}
    src_pcd_colors_list = []

    for ind, ( src_pcd, tgt_pcd, src_feats, tgt_feats, correspondences, rot, trn, s2t_flow, metric_index, depth_paths, cam_intrin, src_pcd_colors, src_feats_indices, tgt_feats_indices) in enumerate(pairwise_data):
        src_pcd_list.append(torch.from_numpy(src_pcd))
        src_pcd_colors_list.append(torch.from_numpy(src_pcd_colors))
        tgt_pcd_list.append(torch.from_numpy(tgt_pcd))

        batched_points_list.append(src_pcd)
        batched_points_list.append(tgt_pcd)
        batched_features_list.append(src_feats)
        batched_features_list.append(tgt_feats)
        batched_lengths_list.append(len(src_pcd))
        batched_lengths_list.append(len(tgt_pcd))

        batched_rot.append( torch.from_numpy(rot).float())
        batched_trn.append( torch.from_numpy(trn).float())

        sflow_list.append( torch.from_numpy(s2t_flow).float() )

        correspondences_list.append( torch.from_numpy(correspondences) )

        depth_paths_list.update( { ind : depth_paths} )

    batched_features = torch.from_numpy(np.concatenate(batched_features_list, axis=0))
    batched_points = torch.from_numpy(np.concatenate(batched_points_list, axis=0))
    batched_lengths = torch.from_numpy(np.array(batched_lengths_list)).int()

    batched_rot = torch.stack(batched_rot, dim=0)
    batched_trn = torch.stack(batched_trn, dim=0)

    # Starting radius of convolutions
    r_normal = config.first_subsampling_dl * config.conv_radius

    # Starting layer
    layer_blocks = []
    layer = 0

    # Lists of inputs
    input_points = []
    input_neighbors = []
    input_pools = []
    input_upsamples = []
    input_batches_len = []
    coarse_matches= []
    coarse_flow = []
    src_ind_coarse_split= []
    src_ind_coarse = []
    tgt_ind_coarse_split= []
    tgt_ind_coarse = []
    accumu = 0

    # construt kpfcn inds
    if feature_extractor == 'kpfcn':
        for block_i, block in enumerate(config.architecture):

            # Stop when meeting a global pooling or upsampling
            if 'global' in block or 'upsample' in block:
                break

            # Get all blocks of the layer
            if not ('pool' in block or 'strided' in block):
                layer_blocks += [block]
                if block_i < len(config.architecture) - 1 and not ('upsample' in config.architecture[block_i + 1]):
                    continue

            # Convolution neighbors indices
            # *****************************
            if layer_blocks:
                # Convolutions are done in this layer, compute the neighbors with the good radius
                if np.any(['deformable' in blck for blck in layer_blocks[:-1]]):
                    r = r_normal * config.deform_radius / config.conv_radius
                else:
                    r = r_normal
                conv_i = batch_neighbors_kpconv(batched_points, batched_points, batched_lengths, batched_lengths, r,
                                                neighborhood_limits[layer])

            else:
                # This layer only perform pooling, no neighbors required
                conv_i = torch.zeros((0, 1), dtype=torch.int64)

            # Pooling neighbors indices
            # *************************
            # If end of layer is a pooling operation
            if 'pool' in block or 'strided' in block:

                # New subsampling length
                dl = 2 * r_normal / config.conv_radius
                # Subsampled points
                pool_p, pool_b = batch_grid_subsampling_kpconv(batched_points, batched_lengths, sampleDl=dl)
                # Radius of pooled neighbors
                if 'deformable' in block:
                    r = r_normal * config.deform_radius / config.conv_radius
                else:
                    r = r_normal

                # Subsample indices
                pool_i = batch_neighbors_kpconv(pool_p, batched_points, pool_b, batched_lengths, r,
                                                neighborhood_limits[layer])

                # Upsample indices (with the radius of the next layer to keep wanted density)
                up_i = batch_neighbors_kpconv(batched_points, pool_p, batched_lengths, pool_b, 2 * r,
                                            neighborhood_limits[layer])

            else:
                # No pooling in the end of this layer, no pooling indices required
                pool_i = torch.zeros((0, 1), dtype=torch.int64)
                pool_p = torch.zeros((0, 3), dtype=torch.float32)
                pool_b = torch.zeros((0,), dtype=torch.int64)
                up_i = torch.zeros((0, 1), dtype=torch.int64)

            # Updating input lists
            input_points += [batched_points.float()]
            input_neighbors += [conv_i.long()]
            input_pools += [pool_i.long()]
            input_upsamples += [up_i.long()]
            input_batches_len += [batched_lengths]

            # New points for next layer
            batched_points = pool_p
            batched_lengths = pool_b

            # Update radius and reset blocks
            r_normal *= 2
            layer += 1
            layer_blocks = []

        # coarse infomation
        coarse_level = coarse_level if coarse_level else config.coarse_level
        pts_num_coarse = input_batches_len[coarse_level].view(-1, 2)
        b_size = pts_num_coarse.shape[0]
        src_pts_max, tgt_pts_max = pts_num_coarse.amax(dim=0)
        coarse_pcd = input_points[coarse_level] # .numpy()
        coarse_matches= []
        coarse_flow = []
        src_ind_coarse_split= [] # src_feats shape :[b_size * src_pts_max]
        src_ind_coarse = []
        tgt_ind_coarse_split= []
        tgt_ind_coarse = []
        accumu = 0
        src_mask = torch.zeros([b_size, src_pts_max], dtype=torch.bool)
        tgt_mask = torch.zeros([b_size, tgt_pts_max], dtype=torch.bool)


        for entry_id, cnt in enumerate( pts_num_coarse ): #input_batches_len[-1].numpy().reshape(-1,2)) :

            n_s_pts, n_t_pts = cnt

            '''split mask for bottlenect feats'''
            src_mask[entry_id][:n_s_pts] = 1
            tgt_mask[entry_id][:n_t_pts] = 1

            '''split indices of bottleneck feats'''
            src_ind_coarse_split.append( torch.arange( n_s_pts ) + entry_id * src_pts_max )
            tgt_ind_coarse_split.append( torch.arange( n_t_pts ) + entry_id * tgt_pts_max )
            src_ind_coarse.append( torch.arange( n_s_pts ) + accumu )
            tgt_ind_coarse.append( torch.arange( n_t_pts ) + accumu + n_s_pts )

            '''get match at coarse level'''
            c_src_pcd_np = coarse_pcd[accumu : accumu + n_s_pts].numpy()
            c_src_pcd_np_rotated = ( batched_rot[entry_id].numpy() @ c_src_pcd_np.T).T
            c_tgt_pcd_np = coarse_pcd[accumu + n_s_pts: accumu + n_s_pts + n_t_pts].numpy()
                    
            if output_folder:
                if not os.path.exists(base + output_folder + feature_extractor + '_dataloader_ldmk'):
                    os.mkdir(base + output_folder + feature_extractor + '_dataloader_ldmk')

                src_pcd_o3d = o3d.geometry.PointCloud()
                src_pcd_o3d.points = o3d.utility.Vector3dVector(src_pcd)
                src_pcd_o3d.rotate(batched_rot[entry_id].numpy(), center=(0, 0, 0))
                o3d.io.write_point_cloud(base + output_folder + feature_extractor + '_dataloader_ldmk/' + 'src_pcd.ply', src_pcd_o3d)

                tgt_pcd_o3d = o3d.geometry.PointCloud()
                tgt_pcd_o3d.points = o3d.utility.Vector3dVector(tgt_pcd)
                o3d.io.write_point_cloud(base + output_folder + feature_extractor + '_dataloader_ldmk/' + 'tgt_pcd.ply', tgt_pcd_o3d)
                
                c_src_pcd = o3d.geometry.PointCloud()
                c_src_pcd.points = o3d.utility.Vector3dVector(np.array(c_src_pcd_np_rotated))
                o3d.io.write_point_cloud(base + output_folder  + feature_extractor + '_dataloader_ldmk/' + 'c_src_pcd.ply', c_src_pcd)

                c_tgt_pcd = o3d.geometry.PointCloud()
                c_tgt_pcd.points = o3d.utility.Vector3dVector(np.array(c_tgt_pcd_np))
                o3d.io.write_point_cloud(base + output_folder  + feature_extractor + '_dataloader_ldmk/' + 'c_tgt_pcd.ply', c_tgt_pcd)
            
            #interpolate flow
            f_src_pcd = batched_points_list[entry_id * 2]
            c_flow = blend_scene_flow( c_src_pcd_np, f_src_pcd, sflow_list[entry_id].numpy(), knn=3)
            c_src_pcd_deformed = c_src_pcd_np + c_flow
            s_pc_wrapped = ( batched_rot[entry_id].numpy() @ c_src_pcd_deformed.T  + batched_trn [entry_id].numpy() ).T
            
            if output_folder:
                s_pc_wrapped_pcd = o3d.geometry.PointCloud()
                s_pc_wrapped_pcd.points = o3d.utility.Vector3dVector(np.array(s_pc_wrapped))
                o3d.io.write_point_cloud(base + output_folder  + feature_extractor + '_dataloader_ldmk/' + 's_pc_wrapped_pcd.ply', s_pc_wrapped_pcd)
            
            coarse_match_gt = torch.from_numpy( multual_nn_correspondence(s_pc_wrapped , c_tgt_pcd_np , search_radius=config['coarse_match_radius'])  )# 0.1m scaled
            coarse_matches.append(coarse_match_gt)
            coarse_flow.append(torch.from_numpy(c_flow))
            accumu = accumu + n_s_pts + n_t_pts

            if output_folder:
                number_points_src = c_src_pcd_np.shape[0]
                correspondences = np.transpose(coarse_match_gt)
                correspondences[:, 1] += number_points_src
                total_points = np.concatenate((c_src_pcd_np_rotated, c_tgt_pcd_np), axis = 0)
                line_set = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(total_points),
                    lines=o3d.utility.Vector2iVector(correspondences),
                )
                o3d.io.write_line_set(base + output_folder + feature_extractor + '_dataloader_ldmk/' + 'dataloader_line_set.ply', line_set)
        
            vis=False # for debug
            if vis :
                viz_coarse_nn_correspondence_mayavi(c_src_pcd_np, c_tgt_pcd_np, coarse_match_gt, scale_factor=0.02)

        src_ind_coarse_split = torch.cat(src_ind_coarse_split)
        tgt_ind_coarse_split = torch.cat(tgt_ind_coarse_split)
        src_ind_coarse = torch.cat(src_ind_coarse)
        tgt_ind_coarse = torch.cat(tgt_ind_coarse)

    elif feature_extractor == 'fcgf':
        'Before the for loop'
        for ind, ( src_pcd, tgt_pcd, src_feats, tgt_feats, correspondences, rot, trn, s2t_flow, metric_index, depth_paths, cam_intrin, src_pcd_colors, src_feats_indices, tgt_feats_indices)  in enumerate(pairwise_data):
            print('ind : ', ind)
            b_size = 1
            coarse_level = config.coarse_level
            n_src_feats = src_feats.shape[0]
            n_tgt_feats = tgt_feats.shape[0]
            src_ind_coarse_split = torch.arange(n_src_feats)
            tgt_ind_coarse_split = torch.arange(n_tgt_feats)                         
            src_ind_coarse = torch.arange(n_src_feats)
            tgt_ind_coarse = torch.arange(n_src_feats, n_tgt_feats + n_src_feats)
            src_mask = torch.zeros([b_size, n_src_feats], dtype=torch.bool)
            src_mask[0][:n_src_feats] = 1
            tgt_mask = torch.zeros([b_size, n_tgt_feats], dtype=torch.bool)
            tgt_mask[0][:n_tgt_feats] = 1
            
            for i in range(4):
                input_points += [torch.tensor([])]
                input_batches_len += [torch.tensor([])]
                input_neighbors += [torch.tensor([])]
                input_pools += [torch.tensor([])]
                input_upsamples += [torch.tensor([])]
            
            src_coarse = src_pcd[src_feats_indices]
            src_coarse_rotated = (rot.numpy() @ src_coarse.T).T
            tgt_coarse = tgt_pcd[tgt_feats_indices]
            total_points = np.concatenate((src_coarse, tgt_coarse))
            input_points[coarse_level] = torch.tensor(total_points)
            input_batches_len[coarse_level] = torch.tensor([src_feats_indices.shape[0], tgt_feats_indices.shape[0]], dtype=torch.int32)
            inter = total_points.reshape(total_points.shape[0], 1, total_points.shape[1])
            dists = np.sqrt(np.einsum('ijk, ijk->ij', total_points-inter, total_points-inter))
            k  = 50
            input_neighbors[coarse_level] = torch.tensor(np.argpartition(dists, k, axis =- 1)[:, :k])
            
            c_flow = blend_scene_flow( src_coarse, src_pcd, s2t_flow, knn=3)
            c_src_pcd_deformed = src_coarse + c_flow
            s_pc_wrapped = (np.matmul(rot, c_src_pcd_deformed.T ) + trn).T
            coarse_match_gt = torch.from_numpy( multual_nn_correspondence(s_pc_wrapped , tgt_coarse , search_radius=config['coarse_match_radius']))
            coarse_matches.append(coarse_match_gt)
            coarse_flow.append(torch.from_numpy(c_flow))
            sflow_list.append( torch.from_numpy(s2t_flow).float())
            
            if output_folder:
                if not os.path.exists(base + output_folder + feature_extractor + '_dataloader_ldmk'):
                    os.mkdir(base + output_folder + feature_extractor + '_dataloader_ldmk')

                src_pcd_o3d = o3d.geometry.PointCloud()
                src_pcd_o3d.points = o3d.utility.Vector3dVector(src_pcd)
                src_pcd_o3d.rotate(rot.numpy(), center=(0, 0, 0))
                o3d.io.write_point_cloud(base + output_folder + feature_extractor + '_dataloader_ldmk/' + 'src_pcd.ply', src_pcd_o3d)

                tgt_pcd_o3d = o3d.geometry.PointCloud()
                tgt_pcd_o3d.points = o3d.utility.Vector3dVector(tgt_pcd)
                o3d.io.write_point_cloud(base + output_folder + feature_extractor + '_dataloader_ldmk/' + 'tgt_pcd.ply', tgt_pcd_o3d)
                
                c_src_pcd = o3d.geometry.PointCloud()
                c_src_pcd.points = o3d.utility.Vector3dVector(np.array(src_coarse_rotated))
                o3d.io.write_point_cloud(base + output_folder  + feature_extractor + '_dataloader_ldmk/' + 'c_src_pcd.ply', c_src_pcd)

                c_tgt_pcd = o3d.geometry.PointCloud()
                c_tgt_pcd.points = o3d.utility.Vector3dVector(np.array(tgt_coarse))
                o3d.io.write_point_cloud(base + output_folder  + feature_extractor + '_dataloader_ldmk/' + 'c_tgt_pcd.ply', c_tgt_pcd)
                        
                s_pc_wrapped_pcd = o3d.geometry.PointCloud()
                s_pc_wrapped_pcd.points = o3d.utility.Vector3dVector(np.array(s_pc_wrapped))
                o3d.io.write_point_cloud(base + output_folder  + feature_extractor + '_dataloader_ldmk/' + 's_pc_wrapped_pcd.ply', s_pc_wrapped_pcd)
    
    print('\n')
    print('Returned from collate_fn_4dmatch')
    print('len(src_pcd_list) : ', len(src_pcd_list))
    print('src_pcd_list[0].shape : ', src_pcd_list[0].shape)
    print('len(tgt_pcd_list) : ', len(tgt_pcd_list))
    print('tgt_pcd_list[0].shape : ', tgt_pcd_list[0].shape)

    print('len(input_points) : ', len(input_points))
    print('input_points[0].shape : ', input_points[0].shape)
    print('len(input_neighbors) : ', len(input_neighbors))
    print('input_neighbors[0].shape : ', input_neighbors[0].shape)

    print('len(input_pools) : ', len(input_pools))
    print('input_pools[0].shape : ', input_pools[0].shape)

    print('len(input_upsamples) : ', len(input_upsamples))
    print('input_upsamples[0].shape : ', input_upsamples[0].shape)

    print('batched_features.shape : ', batched_features.shape)

    print('len(input_batches_len) : ', len(input_batches_len))
    print('input_batches_len[0].shape : ', input_batches_len[0].shape)

    print('len(coarse_matches) : ', len(coarse_matches))
    print('coarse_matches[0].shape : ', coarse_matches[0].shape)

    print('len(coarse_flow) : ', len(coarse_flow))
    print('coarse_flow[0].shape : ', coarse_flow[0].shape)

    print('src_mask.shape : ', src_mask.shape)
    print('tgt_mask.shape : ', tgt_mask.shape)
    print('src_ind_coarse_split.shape : ', src_ind_coarse_split.shape)
    print('tgt_ind_coarse_split.shape : ', tgt_ind_coarse_split.shape)
    print('src_ind_coarse.shape : ', src_ind_coarse.shape)
    print('tgt_ind_coarse.shape : ', tgt_ind_coarse.shape)
    print('batched_rot.shape : ', batched_rot.shape)
    print('batched_trn.shape : ', batched_trn.shape)

    print('len(sflow_list) : ', len(sflow_list))
    print('sflow_list[0].shape : ', sflow_list[0].shape)

    print('len(correspondences_list) : ', len(correspondences_list))
    print('correspondences_list[0].shape : ', correspondences_list[0].shape)
           
    dict_inputs = {
        'src_pcd_list': src_pcd_list,
        'tgt_pcd_list': tgt_pcd_list,
        'points': input_points,
        'neighbors': input_neighbors,
        'pools': input_pools,
        'upsamples': input_upsamples,
        'features': batched_features.float(),
        'stack_lengths': input_batches_len,
        'coarse_matches': coarse_matches,
        'coarse_flow': coarse_flow,
        'src_mask': src_mask,
        'tgt_mask': tgt_mask,
        'src_ind_coarse_split': src_ind_coarse_split,
        'tgt_ind_coarse_split': tgt_ind_coarse_split,
        'src_ind_coarse': src_ind_coarse,
        'tgt_ind_coarse': tgt_ind_coarse,
        'batched_rot': batched_rot,
        'batched_trn': batched_trn,
        'sflow_list': sflow_list,
        'correspondences_list': correspondences_list,
        'depth_paths_list': depth_paths_list,
        'cam_intrin' : cam_intrin,
        'src_pcd_colors_list' : src_pcd_colors_list
    }

    return dict_inputs

def calibrate_neighbors(dataset, config, collate_fn, keep_ratio=0.8, samples_threshold=2000, feature_extractor = 'kpfcn'):

    # From config parameter, compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (config.deform_radius + 1) ** 3))
    neighb_hists = np.zeros((config.num_layers, hist_n), dtype=np.int32)

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):
        batched_input = collate_fn([dataset[i]], config, neighborhood_limits=[hist_n] * 5, output_folder=None, feature_extractor = feature_extractor)

        # update histogram
        counts = [torch.sum(neighb_mat < neighb_mat.shape[0], dim=1).numpy() for neighb_mat in batched_input['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighb_hists += np.vstack(hists)

        # if timer.total_time - last_display > 0.1:
        #     last_display = timer.total_time
        #     print(f"Calib Neighbors {i:08d}: timings {timer.total_time:4.2f}s")

        if np.min(np.sum(neighb_hists, axis=1)) > samples_threshold:
            break

    cumsum = np.cumsum(neighb_hists.T, axis=0)
    percentiles = np.sum(cumsum < (keep_ratio * cumsum[hist_n - 1, :]), axis=0)
    neighborhood_limits = percentiles
    print('\n')

    return neighborhood_limits

def get_datasets(config):
    if (config.dataset == '3dmatch'):
        train_set = _3DMatch(config, 'train', data_augmentation=True)
        val_set = _3DMatch(config, 'val', data_augmentation=False)
        test_set = _3DMatch(config, 'test', data_augmentation=False)
    elif(config.dataset == '4dmatch'):
        train_set = _4DMatch(config, 'train', data_augmentation=True)
        val_set = _4DMatch(config, 'val', data_augmentation=False)
        test_set = _4DMatch(config, 'test', data_augmentation=False)
    elif(config.dataset == '4dmatch_mv'):
        train_set = _4DMatch_Multiview(config, 'train', data_augmentation=True)
        val_set = _4DMatch_Multiview(config, 'val', data_augmentation=False)
        test_set = _4DMatch_Multiview(config, 'test', data_augmentation=False)
    else:
        raise NotImplementedError

    return train_set, val_set, test_set

def get_dataloader(dataset, config,  shuffle=True, neighborhood_limits=None, output_folder = None, base = None, coarse_level = None, feature_extractor = 'kpfcn'):

    print('feature extractor inside of get_dataloader : ', feature_extractor)
    collate_fn = collate_fn_4dmatch

    if neighborhood_limits is None:
        neighborhood_limits = calibrate_neighbors(dataset, config['kpfcn_config'], collate_fn=collate_fn, feature_extractor = feature_extractor)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=shuffle,
        num_workers=config['num_workers'],
        collate_fn=partial(collate_fn, config= config['kpfcn_config'], neighborhood_limits=neighborhood_limits, output_folder = output_folder, base = base, coarse_level = coarse_level, feature_extractor = feature_extractor),
        drop_last=False
    )

    return dataloader, neighborhood_limits

def get_multiview_dataloader(dataset, config,  shuffle=True, neighborhood_limits=None):

    collate_fn = collate_fn_4dmatch_multiview
    if neighborhood_limits is None:
        neighborhood_limits = calibrate_neighbors(dataset, config['kpfcn_config'], collate_fn=collate_fn)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=shuffle,
        num_workers=config['num_workers'],
        collate_fn=partial(collate_fn, config= config['kpfcn_config'], neighborhood_limits=neighborhood_limits ),
        drop_last=False
    )

    return dataloader, neighborhood_limits

if __name__ == '__main__':
    pass
