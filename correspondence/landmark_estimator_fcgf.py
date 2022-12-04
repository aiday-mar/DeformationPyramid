import torch
import yaml
from easydict import EasyDict as edict


import sys
sys.path.append("")
from correspondence.lepard.pipeline_fcgf import Pipeline as Matcher
from correspondence.outlier_rejection.pipeline import   Outlier_Rejection
from correspondence.outlier_rejection.loss import   NeCoLoss



class Landmark_Model_FCGF():

    def __init__(self, config_file, device, indent=None ):

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
        self.matcher = Matcher(matcher_config).to(device)
        state = torch.load(indent + config.matcher_weights if indent else config.matcher_weights)
        self.matcher.load_state_dict(state['state_dict'])
        self.indent = indent
        
        # outlier model initialization
        self.outlier_model = Outlier_Rejection(outlier_rejection_config.model).to(device)
        state = torch.load(indent + config.outlier_rejection_weights if indent else config.outlier_rejection_weights)
        self.outlier_model.load_state_dict(state['state_dict'])

        self.device = device
        self.kpfcn_config = config['kpfcn_config']

    def inference(self, inputs, s_feats = None, t_feats = None, base = None, reject_outliers=True, inlier_thr=0.5, timer=None):

        self.matcher.eval()
        self.outlier_model.eval()
        with torch.no_grad():

            if timer: timer.tic("matcher")
            data = self.matcher(inputs, s_feats, t_feats, base, timers=None)
            if timer: timer.toc("matcher")

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


            return ldmk_s, ldmk_t, inlier_rate, inlier_rate_2