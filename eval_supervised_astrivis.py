from model.geometry import *
import os
import torch
import sys
sys.path.append("correspondence")

from tqdm import tqdm
import argparse

from model.registration import Registration
import  yaml
from easydict import EasyDict as edict
from model.loss import compute_flow_metrics

from utils.benchmark_utils import setup_seed
from utils.utils import AverageMeter
from utils.tiktok import Timers
from correspondence.landmark_estimator import Landmark_Model
from correspondence.datasets._astrivis_custom_single import _AstrivisCustomSingle
from correspondence.datasets.dataloader import get_dataloader
import h5py
import copy

def join(loader, node):
    seq = loader.construct_sequence(node)
    return '_'.join([str(i) for i in seq])
yaml.add_constructor('!join', join)

setup_seed(0)

path = '/home/aiday.kyzy/dataset/Synthetic/'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--s', type=str, help='Path to the src mesh.')
    parser.add_argument('--t', type=str, help='Path to the tgt mesh.')
    parser.add_argument('--s_feats', type=str, help='Path to the src feats.')
    parser.add_argument('--t_feats', type=str, help='Path to the tgt feats.')
    
    parser.add_argument('--output', type=str, help= 'Path to the file where to save source pcd after transformation.')
    parser.add_argument('--output_trans', type=str, help='Path to the final output transformation.')
    parser.add_argument('--matches', type=str, help='Path to ground truth matches')  
    parser.add_argument('--source_trans', type=str, help='Path to the source transformation')  
    parser.add_argument('--target_trans', type=str, help='Path to the target transformation')
        
    parser.add_argument('--config', type=str, help= 'Path to the config file.')
    parser.add_argument('--base', type=str, help= 'Base folder.')
    parser.add_argument('--w_cd', type=str, help= 'w_cd')
    parser.add_argument('--k0', type=str, help= 'k0 used in the positional encoding of the NDP layers')
    parser.add_argument('--w_reg', type=str, help= 'w_reg')
    parser.add_argument('--confidence_threshold', type=str, help= 'specifying the confidence threshold')
    parser.add_argument('--samples', type=str, help='number of samples to use')
    parser.add_argument('--levels', type=str, help='number of levels in NDP')
    parser.add_argument('--posenc_function', type=str, help='function type in the positional encoding')
    
    parser.add_argument('--reject_outliers', type=str, help= 'whether to use or not the outlier rejection network')
    parser.add_argument('--intermediate_output_folder', type=str, help='Where to place all the intermediate outputs')
    parser.add_argument('--preprocessing', type=str, help='Type of the preprocessing, can be single or mutual or none. By default mutual.')

    parser.add_argument('--coarse_level', type=str, help= 'coarse level')
    parser.add_argument('--index_coarse_feats', type=str, help='index at which to return coarse features in the lepard decoder')
    parser.add_argument('--number_centers', type=str, help='number of centers to use in the custom filtering')
    parser.add_argument('--average_distance_multiplier', type=str, help='multiplier in front of the average distance')
    parser.add_argument('--number_iterations_custom_filtering', type=str, help='number of iterations of the custom filtering')
    parser.add_argument('--inlier_outlier_thr', type=str, help='threshold which is used to determine the inliers and outliers')
    parser.add_argument('--mesh_path', type=str, help='path to initial source mesh which can be used in order to do poisson center sampling')
    parser.add_argument('--sampling', type=str, help='sampling strategy used')
    parser.add_argument('--max_ldmks',type=str, help='number of maximum landmarks in custom filtering')

    parser.add_argument('--indent', type=str, help='indent level used in order to access different files')
    parser.add_argument('--gt_thr', type=str, help='ground-truth threshold used to find the ground-truth correspondences')
    
    parser.add_argument('--edge_filtering_simple', action = 'store_true', help= 'wether to perform edge filtering using a simple neighborhood method')
    parser.add_argument('--edge_filtering_angle', action = 'store_true', help= 'wether to perform edge filtering using the angle criterion')
    parser.add_argument('--edge_filtering_shape', action = 'store_true', help= 'wether to perform edge filtering using the shape criterion')
    parser.add_argument('--edge_filtering_disc', action = 'store_true', help= 'wether to perform edge filtering using the disc criterion')
    parser.add_argument('--edge_filtering_mesh', action = 'store_true', help= 'wether to perform edge filtering using the mesh criterion')
    parser.add_argument('--min_dist_thr', type=str, help='threshold used for the edge filtering computation')
    parser.add_argument('--n_points_edge_filtering', type=str, help='number of points from the exterior edge filtering')
    
    parser.add_argument('--visualize', action = 'store_true', help= 'visualizing the point-clouds')
    parser.add_argument('--custom_filtering', action='store_true', help= 'custom filtering the correspondences')
    parser.add_argument('--print_keypoints', action = 'store_true', help= 'store the intermediate keypoints')
    parser.add_argument('--use_gt_ldmks', action = 'store_true', help= 'store the intermediate keypoints')
    parser.add_argument('--only_inference', action = 'store_true', help= 'only do the inference in order to find the landmarks')
    parser.add_argument('--knn_matching', action = 'store_true', help= 'whether to use knn matching instead of default matching')
    args = parser.parse_args()
    
    if args.base:
        path = args.base
        
    with open(args.config,'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    config['snapshot_dir'] = 'snapshot/%s/%s' % (config['folder'], config['exp_dir'])
    os.makedirs(config['snapshot_dir'], exist_ok=True)

    config = edict(config)

    # backup the experiment
    os.system(f'cp -r config {config.snapshot_dir}')
    os.system(f'cp -r data {config.snapshot_dir}')
    os.system(f'cp -r model {config.snapshot_dir}')
    os.system(f'cp -r utils {config.snapshot_dir}')

    if config.gpu_mode:
        config.device = torch.cuda.current_device()
    else:
        config.device = torch.device('cpu')

    if config.feature_extractor == 'kpfcn' or config.feature_extractor == 'fcgf':
        if args.indent:
            ldmk_model =  Landmark_Model(config_file = args.indent + config.ldmk_config, device=config.device, indent=args.indent, feature_extractor = config.feature_extractor)
        else:
            ldmk_model =  Landmark_Model(config_file = config.ldmk_config, device=config.device, feature_extractor = config.feature_extractor)
    else:
        raise Exception('Specify a valid feature extractor')

    config['kpfcn_config'] = ldmk_model.kpfcn_config

    model = Registration(config)
    timer = Timers()
    stats_meter = None
    
    test_set = _AstrivisCustomSingle(config, args.s, args.t, args.matches, args.source_trans, args.target_trans, args.base, source_feats = args.s_feats, target_feats = args.t_feats)
    
    coarse_level = int(args.coarse_level) if args.coarse_level else None
    if args.print_keypoints:
        test_loader, _ = get_dataloader(test_set, config, shuffle=False, coarse_level = coarse_level, output_folder=args.intermediate_output_folder, base = args.base, feature_extractor = config.feature_extractor)
    else:
        test_loader, _ = get_dataloader(test_set, config, shuffle=False, coarse_level = coarse_level, feature_extractor = config.feature_extractor)
        
    num_iter =  len(test_set)
    c_loader_iter = test_loader.__iter__()
    
    for c_iter in tqdm(range(num_iter)):

        inputs = c_loader_iter.next()
        for k, v in inputs.items():
            if type(v) == list:
                inputs [k] = [item.to(config.device) for item in v]
            elif type(v) in [dict, float, type(None), np.ndarray]:
                pass
            else:
                inputs [k] = v.to(config.device)

        """predict landmarks"""
        custom_filtering = True if args.custom_filtering else False
        index_coarse_feats = int(args.index_coarse_feats) if args.index_coarse_feats else 1
        intermediate_output_folder = args.intermediate_output_folder if args.intermediate_output_folder and args.print_keypoints else None
        preprocessing = args.preprocessing if args.preprocessing else 'mutual'
        number_centers = int(args.number_centers) if args.number_centers else 1000
        average_distance_multiplier = float(args.average_distance_multiplier) if args.average_distance_multiplier else 2
        number_iterations_custom_filtering = int(args.number_iterations_custom_filtering) if args.number_iterations_custom_filtering else 1
        matches_path = args.matches if args.matches else None
        inlier_outlier_thr = float(args.inlier_outlier_thr) if args.inlier_outlier_thr else 0.05
        sampling = args.sampling if args.sampling else 'linspace'
        mesh_path = args.mesh_path if args.mesh_path else None
        source_trans = args.source_trans if args.source_trans else None
        if args.reject_outliers:
            if args.reject_outliers == 'true':
                reject_outliers = True
            elif args.reject_outliers == 'false':
                reject_outliers = False
            else:
                raise Exception('specify valid value for reject_outliers')
        else:
            reject_outliers = config.reject_outliers
        gt_thr = float(args.gt_thr) if args.gt_thr else 0.01
        edge_filtering_simple = True if args.edge_filtering_simple else False
        edge_filtering_angle = True if args.edge_filtering_angle else False
        edge_filtering_shape = True if args.edge_filtering_shape else False
        edge_filtering_disc = True if args.edge_filtering_disc else False
        edge_filtering_mesh = True if args.edge_filtering_mesh else False
        min_dist_thr = float(args.min_dist_thr) if args.min_dist_thr else 0.05 # 1.0e-4
        print('min_dist_thr : ', min_dist_thr)
        knn_matching = True if args.knn_matching else False

        if isinstance(args.max_ldmks, str) and args.max_ldmks.isnumeric():
            max_ldmks = int(args.max_ldmks)
        else:
            max_ldmks = None

        if args.n_points_edge_filtering and args.n_points_edge_filtering != 'None':
            n_points_edge_filtering = int(args.n_points_edge_filtering)
        else:
            n_points_edge_filtering = None

        if not args.use_gt_ldmks:
            ldmk_s, ldmk_t, inlier_rate, inlier_rate_2 = ldmk_model.inference(inputs = inputs, mesh_path = mesh_path, source_trans = source_trans, sampling = sampling, inlier_outlier_thr = inlier_outlier_thr, matches_path = matches_path, custom_filtering = custom_filtering, number_iterations_custom_filtering = number_iterations_custom_filtering, average_distance_multiplier = average_distance_multiplier,  reject_outliers=reject_outliers, confidence_threshold = args.confidence_threshold, preprocessing = preprocessing, coarse_level = args.coarse_level, inlier_thr=config.inlier_thr, timer=timer, number_centers = number_centers, intermediate_output_folder = intermediate_output_folder, base = args.base, index_at_which_to_return_coarse_feats = index_coarse_feats, gt_thr = gt_thr, edge_filtering_simple = edge_filtering_simple, edge_filtering_angle = edge_filtering_angle, edge_filtering_shape = edge_filtering_shape, edge_filtering_disc = edge_filtering_disc, edge_filtering_mesh = edge_filtering_mesh, min_dist_thr = min_dist_thr, max_ldmks = max_ldmks, knn_matching = knn_matching, n_points_edge_filtering = n_points_edge_filtering)
            if args.only_inference:
                sys.exit(0)
        elif args.use_gt_ldmks and args.base:
            src_pcd = o3d.io.read_point_cloud(args.base + args.s)
            src_pcd = np.array(src_pcd.points).astype(np.float32)
            tgt_pcd = o3d.io.read_point_cloud(args.base + args.t)
            tgt_pcd = np.array(tgt_pcd.points).astype(np.float32)
            matches = np.load(args.base + args.matches)
            correspondences = np.array(matches['matches'])
            final_correspondences = np.empty((0,2), int)
            set_src_indices = set()
            for correspondence in correspondences:
                if correspondence[0] not in set_src_indices:
                    final_correspondences = np.append(final_correspondences, np.array(np.expand_dims(correspondence, axis=0)), axis=0)
                set_src_indices.add(correspondence[0])
            correspondence = final_correspondences
            indices_src = correspondences[:, 0]
            indices_tgt = correspondences[:, 1]
            overlap_s_points = src_pcd[indices_src]
            overlap_t_points = tgt_pcd[indices_tgt]
            ldmk_s = torch.tensor(overlap_s_points).to('cuda:0')
            ldmk_t = torch.tensor(overlap_t_points).to('cuda:0')
            
            if intermediate_output_folder:
                if not os.path.exists(args.base + intermediate_output_folder + config.feature_extractor + '_gt_ldmk'):
                    os.mkdir(args.base + intermediate_output_folder + config.feature_extractor + '_gt_ldmk')

                ldmk_s_pcd = o3d.geometry.PointCloud()
                ldmk_s_pcd.points = o3d.utility.Vector3dVector(overlap_s_points)
                o3d.io.write_point_cloud(args.base + intermediate_output_folder + config.feature_extractor + '_gt_ldmk/' + 'ldmk_s_pcd.ply', ldmk_s_pcd)

                ldmk_t_pcd = o3d.geometry.PointCloud()
                ldmk_t_pcd.points = o3d.utility.Vector3dVector(overlap_t_points)
                o3d.io.write_point_cloud(args.base + intermediate_output_folder + config.feature_extractor + '_gt_ldmk/' + 'ldmk_t_pcd.ply', ldmk_t_pcd)
                
                total_points = np.concatenate((overlap_s_points, overlap_t_points), axis = 0)
                number_points_src = overlap_s_points.shape[0]
                correspondences = [[i, i + number_points_src] for i in range(0, number_points_src)]
                line_set = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(total_points),
                    lines=o3d.utility.Vector2iVector(correspondences),
                )
                o3d.io.write_line_set(args.base + intermediate_output_folder + config.feature_extractor + '_gt_ldmk/' + 'gt_line_set.ply', line_set)
        else:
            raise Exception('Specify a valid combination')
        
        src_pcd, tgt_pcd = inputs["src_pcd_list"][0], inputs["tgt_pcd_list"][0]
        src_pcd_colors = inputs["src_pcd_colors_list"][0]
        copy_src_pcd = copy.deepcopy(src_pcd)
        copy_tgt_pcd = copy.deepcopy(tgt_pcd)
        
        src_pcd_o3d = o3d.geometry.PointCloud()
        src_pcd_o3d.points = o3d.utility.Vector3dVector(np.array(src_pcd.cpu()))
        
        target_pcd_o3d = o3d.geometry.PointCloud()
        target_pcd_o3d.points = o3d.utility.Vector3dVector(np.array(tgt_pcd.cpu()))
        
        s2t_flow = inputs['sflow_list'][0]
        rot, trn = inputs['batched_rot'][0],  inputs['batched_trn'][0]
        correspondence = inputs['correspondences_list'][0]

        """compute scene flow GT"""
        src_pcd_deformed = src_pcd
        src_pcd_deformed = src_pcd + s2t_flow
        
        src_pcd_deformed_o3d = o3d.geometry.PointCloud()
        src_pcd_deformed_o3d.points = o3d.utility.Vector3dVector(np.array(src_pcd_deformed.cpu()))

        s_pc_wrapped = ( rot @ src_pcd_deformed.T + trn ).T        
        src_pcd_wrapped_o3d = o3d.geometry.PointCloud()
        src_pcd_wrapped_o3d.points = o3d.utility.Vector3dVector(np.array(s_pc_wrapped.cpu()))
        
        s2t_flow = s_pc_wrapped - src_pcd
        flow_gt = s2t_flow.to(config.device)

        """compute overlap mask"""
        overlap = torch.zeros(len(src_pcd))
        overlap[correspondence[:, 0]] = 1
        overlap = overlap.bool()
        overlap =  overlap.to(config.device)

        model.load_pcds(copy_src_pcd, copy_tgt_pcd, landmarks=(ldmk_s, ldmk_t))
        
        w_cd = float(args.w_cd) if args.w_cd else None
        w_reg = float(args.w_reg) if args.w_reg else None
        print_keypoints = True if args.print_keypoints else False
        k0 = int(args.k0) if args.k0 else -8
        samples = int(args.samples) if args.samples else None
        levels = int(args.levels) if args.levels else None
        posenc_function = args.posenc_function if args.posenc_function else None
        warped_pcd, data, iter, timer = model.register(visualize=args.visualize, posenc_function = posenc_function, levels = levels, samples = samples, k0 = k0, intermediate_output_folder=args.intermediate_output_folder, timer = timer, base = path, print_keypoints = print_keypoints, w_cd = w_cd, w_reg = w_reg)
            
        final_transformation = np.identity(4)
        for i in range(0, levels if levels else 10):
            rot = data[i][2][-1].cpu()
            trans = data[i][3][-1].cpu().unsqueeze(1)
            se4_matrix = np.concatenate((rot, trans), axis=1)
            se4_matrix = np.concatenate((se4_matrix, np.array([[0,0,0,1]])), axis=0)
            final_transformation = se4_matrix@final_transformation
        
        f = h5py.File(path + args.output_trans, 'w')
        f.create_dataset('transformation', data=np.array(final_transformation))
        f.close()
        
        # warped_pcd is presumably the final pcd        
        final_pcd = o3d.geometry.PointCloud()
        final_pcd.points = o3d.utility.Vector3dVector(np.array(warped_pcd.cpu()))
        final_pcd.colors = o3d.utility.Vector3dVector(np.array(src_pcd_colors.cpu()))
        o3d.io.write_point_cloud(path + args.output, final_pcd)
        
        # Saving the line set
        ls2 = o3d.geometry.LineSet()
        total_points = np.concatenate((src_pcd.cpu(), np.array(warped_pcd.cpu())))
        ls2.points = o3d.utility.Vector3dVector(total_points)
        n_points = src_pcd.shape[0]
        total_lines = [[i, i + n_points] for i in range(0, n_points)]
        ls2.lines = o3d.utility.Vector2iVector(total_lines)
        
        flow = warped_pcd - model.src_pcd
        metric_info = compute_flow_metrics(flow, flow_gt, overlap=overlap, base = args.base, intermediate_output_folder = args.intermediate_output_folder)

        if stats_meter is None:
            stats_meter = dict()
            for key, _ in metric_info.items():
                stats_meter[key] = AverageMeter()
        for key, value in metric_info.items():
            stats_meter[key].update(value)

        message = f'{c_iter}/{len(test_set)}: '
        for key, value in stats_meter.items():
            message += f'{key}: {value.avg:.3f} \n'
        print(message)
