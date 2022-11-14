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



    def forward(self, data,  timers=None):
        
        print('Inside the forward method of the Pipeline')
        self.timers = timers

        if self.timers: self.timers.tic('kpfcn backbone encode')
        print('Before KPFCN backbone')
        coarse_feats = self.backbone(data, phase="coarse")
        if self.timers: self.timers.toc('kpfcn backbone encode')

        print('\n')
        print('Before splitting the features')
        if self.timers: self.timers.tic('coarse_preprocess')
        src_feats, tgt_feats, s_pcd, t_pcd, src_mask, tgt_mask = self.split_feats(coarse_feats, data)
        data.update({ 's_pcd': s_pcd, 't_pcd': t_pcd })
        if self.timers: self.timers.toc('coarse_preprocess')

        print('\n')
        print('Before the Repositioning Transformer')
        if self.timers: self.timers.tic('coarse feature transformer')
        src_feats, tgt_feats, src_pe, tgt_pe = self.coarse_transformer(src_feats, tgt_feats, s_pcd, t_pcd, src_mask, tgt_mask, data, timers=timers)
        if self.timers: self.timers.toc('coarse feature transformer')

        print('Before the matching')
        if self.timers: self.timers.tic('match feature coarse')
        conf_matrix_pred, coarse_match_pred = self.coarse_matching(src_feats, tgt_feats, src_pe, tgt_pe, src_mask, tgt_mask, data, pe_type = self.pe_type)
        data.update({'conf_matrix_pred': conf_matrix_pred, 'coarse_match_pred': coarse_match_pred })
        if self.timers: self.timers.toc('match feature coarse')

        print('Before the Soft Procrustes Layer')
        if self.timers: self.timers.tic('procrustes_layer')
        R, t, _, _, _, _ = self.soft_procrustes(conf_matrix_pred, s_pcd, t_pcd, src_mask, tgt_mask)
        data.update({"R_s2t_pred": R, "t_s2t_pred": t})
        if self.timers: self.timers.toc('procrustes_layer')

        return data




    def split_feats(self, geo_feats, data):
        print('Inside of split_feats')
        '''
        geo_feats.shape :  torch.Size([260, 528])
        src_mask.shape :  torch.Size([1, 132])
        tgt_mask.shape :  torch.Size([1, 128])
        src_ind_coarse_split.shape :  torch.Size([132])
        tgt_ind_coarse_split.shape :  torch.Size([128])
        src_ind_coarse.shape :  torch.Size([132])
        tgt_ind_coarse.shape :  torch.Size([128])
        src_feats.shape :  torch.Size([132, 528])
        tgt_feats.shape :  torch.Size([128, 528])
        src_feats.shape :  torch.Size([132, 528])
        tgt_feats.shape :  torch.Size([128, 528])
        src_pcd.shape :  torch.Size([132, 3])
        tgt_pcd.shape :  torch.Size([128, 3])
        '''
        print('geo_feats.shape : ', geo_feats.shape)
        pcd = data['points'][self.config['kpfcn_config']['coarse_level']]

        src_mask = data['src_mask']
        print('src_mask.shape : ', src_mask.shape)
        tgt_mask = data['tgt_mask']
        print('tgt_mask.shape : ', tgt_mask.shape)
        src_ind_coarse_split = data[ 'src_ind_coarse_split']
        print('src_ind_coarse_split.shape : ', src_ind_coarse_split.shape)
        tgt_ind_coarse_split = data['tgt_ind_coarse_split']
        print('tgt_ind_coarse_split.shape : ', tgt_ind_coarse_split.shape)
        src_ind_coarse = data['src_ind_coarse']
        print('src_ind_coarse.shape : ', src_ind_coarse.shape)
        tgt_ind_coarse = data['tgt_ind_coarse']
        print('tgt_ind_coarse.shape : ', tgt_ind_coarse.shape)

        b_size, src_pts_max = src_mask.shape
        tgt_pts_max = tgt_mask.shape[1]
        # multiply batch size by the max number of points in the source point-cloud
        # the second argument is the dimension of the geotransformer features
        src_feats = torch.zeros([b_size * src_pts_max, geo_feats.shape[-1]]).type_as(geo_feats)
        print('src_feats.shape : ', src_feats.shape)
        tgt_feats = torch.zeros([b_size * tgt_pts_max, geo_feats.shape[-1]]).type_as(geo_feats)
        print('tgt_feats.shape : ', tgt_feats.shape)
        src_pcd = torch.zeros([b_size * src_pts_max, 3]).type_as(pcd)
        tgt_pcd = torch.zeros([b_size * tgt_pts_max, 3]).type_as(pcd)

        src_feats[src_ind_coarse_split] = geo_feats[src_ind_coarse]
        tgt_feats[tgt_ind_coarse_split] = geo_feats[tgt_ind_coarse]
        src_pcd[src_ind_coarse_split] = pcd[src_ind_coarse]
        tgt_pcd[tgt_ind_coarse_split] = pcd[tgt_ind_coarse]
        
        print('src_feats.shape : ', src_feats.shape)
        print('tgt_feats.shape : ', tgt_feats.shape)
        print('src_pcd.shape : ', src_pcd.shape)
        print('tgt_pcd.shape : ', tgt_pcd.shape)
        
        return src_feats.view( b_size , src_pts_max , -1), \
               tgt_feats.view( b_size , tgt_pts_max , -1), \
               src_pcd.view( b_size , src_pts_max , -1), \
               tgt_pcd.view( b_size , tgt_pts_max , -1), \
               src_mask, \
               tgt_mask