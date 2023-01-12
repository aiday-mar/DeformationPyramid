from .blocks import *
from .backbone_fcgf import FCGF
from .transformer import RepositioningTransformer
from .matching import Matching
from .procrustes import SoftProcrustesLayer
from .knn import find_knn_gpu
import numpy as np
import torch

class PipelineFCGF(nn.Module):

    def __init__(self, config):
        super(PipelineFCGF, self).__init__()
        self.config = config
        self.feature_extractor = config.feature_extractor
        self.backbone = FCGF(config['kpfcn_config'])
        self.pe_type = config['coarse_transformer']['pe_type']
        self.positioning_type = config['coarse_transformer']['positioning_type']
        self.coarse_transformer = RepositioningTransformer(config['coarse_transformer'])
        self.coarse_matching = Matching(config['coarse_matching'])
        self.soft_procrustes = SoftProcrustesLayer(config['coarse_transformer']['procrustes'])

    def forward(self, data, confidence_threshold = None, preprocessing = 'mutual', knn_matching = False, timers=None):

        print('knn_matching : ', knn_matching)

        self.timers = timers
        if self.timers: self.timers.tic('fcgf backbone encode')
        coarse_feats = self.backbone(data, phase="coarse")
        if self.timers: self.timers.toc('fcgf backbone encode')

        if self.timers: self.timers.tic('coarse_preprocess')
        src_feats, tgt_feats, s_pcd, t_pcd, src_mask, tgt_mask = self.split_feats(coarse_feats, data)
        data.update({ 's_pcd': s_pcd, 't_pcd': t_pcd })
        if self.timers: self.timers.toc('coarse_preprocess')

        if knn_matching is False:
            if self.timers: self.timers.tic('coarse feature transformer')
            src_feats, tgt_feats, src_pe, tgt_pe = self.coarse_transformer(src_feats, tgt_feats, s_pcd, t_pcd, src_mask, tgt_mask, data, preprocessing = preprocessing, confidence_threshold = confidence_threshold, feature_extractor = self.feature_extractor, timers=timers)
            if self.timers: self.timers.toc('coarse feature transformer')

            if self.timers: self.timers.tic('match feature coarse')
            conf_matrix_pred, coarse_match_pred = self.coarse_matching(src_feats, tgt_feats, src_pe, tgt_pe, src_mask, tgt_mask, data, preprocessing = preprocessing, confidence_threshold = confidence_threshold, pe_type = self.pe_type)
            
            data.update({'conf_matrix_pred': conf_matrix_pred, 'coarse_match_pred': coarse_match_pred })

            if self.timers: self.timers.toc('match feature coarse')

        else:

            if self.timers: self.timers.tic('coarse feature transformer')
            src_feats_i, tgt_feats_i, src_pe_i, tgt_pe_i = self.coarse_transformer(src_feats, tgt_feats, s_pcd, t_pcd, src_mask, tgt_mask, data, preprocessing = preprocessing, confidence_threshold = confidence_threshold, feature_extractor = self.feature_extractor, timers=timers)
            if self.timers: self.timers.toc('coarse feature transformer')

            if self.timers: self.timers.tic('match feature coarse')
            conf_matrix_pred, _ = self.coarse_matching(src_feats_i, tgt_feats_i, src_pe_i, tgt_pe_i, src_mask, tgt_mask, data, preprocessing = preprocessing, confidence_threshold = confidence_threshold, pe_type = self.pe_type)
            
            coarse_match_pred = find_knn_gpu(src_feats, tgt_feats, nn_max_n=20, knn=1,return_distance=False)
            n_src_coarse = coarse_match_pred.shape[0]
            src_indices = np.arange(n_src_coarse)
            src_indices = np.expand_dims(src_indices, axis=1)
            src_indices = torch.tensor(src_indices).to('cuda:0')
            # coarse_match_pred = torch.tensor(coarse_match_pred).to('cuda:0')
            coarse_match_pred = torch.cat((src_indices, coarse_match_pred), 1)
            coarse_match_pred = coarse_match_pred[None, :]
            data.update({'conf_matrix_pred': conf_matrix_pred, 'coarse_match_pred': coarse_match_pred })

            data['vec_6d'] = []
            ldmk_s = s_pcd
            ldmk_t_indices = coarse_match_pred[0][:, 1]
            print('ldmk_t_indices : ', ldmk_t_indices)
            ldmk_t = t_pcd[0][ldmk_t_indices]
            ldmk_t = ldmk_t[None, :]
            print('ldmk_s.shape : ', ldmk_s.shape)
            print('ldmk_t.shape : ', ldmk_t.shape)
            vec_6d = torch.cat((ldmk_s, ldmk_t), 1)
            data['vec_6d'].append(vec_6d)
            if self.timers: self.timers.toc('match feature coarse')

        if self.timers: self.timers.tic('procrustes_layer')
        R, t, _, _, _, _ = self.soft_procrustes(conf_matrix_pred, s_pcd, t_pcd, src_mask, tgt_mask)
        data.update({"R_s2t_pred": R, "t_s2t_pred": t})
        if self.timers: self.timers.toc('procrustes_layer')
        return data

    def split_feats(self, geo_feats, data):
   
        pcd = data['points'][self.config['kpfcn_config']['coarse_level']]
        src_mask = data['src_mask']
        tgt_mask = data['tgt_mask']

        src_ind_coarse_split = data['src_ind_coarse_split']
        tgt_ind_coarse_split = data['tgt_ind_coarse_split']

        src_ind_coarse = data['src_ind_coarse']
        tgt_ind_coarse = data['tgt_ind_coarse']

        b_size, src_pts_max = src_mask.shape
        tgt_pts_max = tgt_mask.shape[1]

        src_feats = torch.zeros([b_size * src_pts_max, geo_feats.shape[-1]]).type_as(geo_feats)
        tgt_feats = torch.zeros([b_size * tgt_pts_max, geo_feats.shape[-1]]).type_as(geo_feats)
        src_pcd = torch.zeros([b_size * src_pts_max, 3]).type_as(pcd)
        tgt_pcd = torch.zeros([b_size * tgt_pts_max, 3]).type_as(pcd)

        src_feats[src_ind_coarse_split] = geo_feats[src_ind_coarse]
        tgt_feats[tgt_ind_coarse_split] = geo_feats[tgt_ind_coarse]
        src_pcd[src_ind_coarse_split] = pcd[src_ind_coarse]
        tgt_pcd[tgt_ind_coarse_split] = pcd[tgt_ind_coarse]

        return src_feats.view( b_size , src_pts_max , -1), \
               tgt_feats.view( b_size , tgt_pts_max , -1), \
               src_pcd.view( b_size , src_pts_max , -1), \
               tgt_pcd.view( b_size , tgt_pts_max , -1), \
               src_mask, \
               tgt_mask