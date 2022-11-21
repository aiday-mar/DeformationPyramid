from .blocks import *
from .backbone import KPFCN
from .transformer import RepositioningTransformer
from .matching import Matching
from .procrustes import SoftProcrustesLayer
from .models import kpfcn_backbone

class Pipeline(nn.Module):

    def __init__(self, config):
        super(Pipeline, self).__init__()
        self.config = config
        self.config.kpfcn_config.architecture = kpfcn_backbone
        self.backbone = KPFCN(self.config.kpfcn_config)
        self.pe_type = config['coarse_transformer']['pe_type']
        self.positioning_type = config['coarse_transformer']['positioning_type']
        self.coarse_transformer = RepositioningTransformer(config['coarse_transformer'])
        self.coarse_matching = Matching(config['coarse_matching'])
        self.soft_procrustes = SoftProcrustesLayer(config['coarse_transformer']['procrustes'])

    def forward(self, data, confidence_threshold = None, preprocessing = 'mutual', coarse_level = None, timers=None):

        self.timers = timers

        if self.timers: self.timers.tic('kpfcn backbone encode')
        coarse_feats = self.backbone(data, phase="coarse")
        if self.timers: self.timers.toc('kpfcn backbone encode')

        if self.timers: self.timers.tic('coarse_preprocess')
        src_feats, tgt_feats, s_pcd, t_pcd, src_mask, tgt_mask = self.split_feats (coarse_feats, data, coarse_level)
        data.update({ 's_pcd': s_pcd, 't_pcd': t_pcd })
        if self.timers: self.timers.toc('coarse_preprocess')

        if self.timers: self.timers.tic('coarse feature transformer')
        src_feats, tgt_feats, src_pe, tgt_pe = self.coarse_transformer(src_feats, tgt_feats, s_pcd, t_pcd, src_mask, tgt_mask, data, timers=timers)
        if self.timers: self.timers.toc('coarse feature transformer')

        if self.timers: self.timers.tic('match feature coarse')
        conf_matrix_pred, coarse_match_pred = self.coarse_matching(src_feats, tgt_feats, src_pe, tgt_pe, src_mask, tgt_mask, data, preprocessing = preprocessing, confidence_threshold = confidence_threshold, pe_type = self.pe_type)
        data.update({'conf_matrix_pred': conf_matrix_pred, 'coarse_match_pred': coarse_match_pred })
        if self.timers: self.timers.toc('match feature coarse')

        if self.timers: self.timers.tic('procrustes_layer')
        R, t, _, _, _, _ = self.soft_procrustes(conf_matrix_pred, s_pcd, t_pcd, src_mask, tgt_mask)
        data.update({"R_s2t_pred": R, "t_s2t_pred": t})
        if self.timers: self.timers.toc('procrustes_layer')

        return data

    def split_feats(self, geo_feats, data, coarse_level):
        coarse_level = coarse_level if coarse_level else self.config['kpfcn_config']['coarse_level']
        coarse_level = int(coarse_level)
        print('coarse_level : ', coarse_level)
        pcd = data['points'][coarse_level]

        src_mask = data['src_mask']
        tgt_mask = data['tgt_mask']
        src_ind_coarse_split = data[ 'src_ind_coarse_split']
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