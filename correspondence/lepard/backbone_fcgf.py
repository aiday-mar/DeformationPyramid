import torch.nn.functional as F
import numpy as np
import os
import torch.nn as nn
import open3d as o3d
import subprocess
import torch

class FCGF(nn.Module):

    def __init__(self, config):
        super(FCGF, self).__init__()

    def forward(self, batch, s_feats = None, t_feats = None, base = None, phase = 'encode'):
        
        features_src = None
        features_tgt = None
        
        # TODO: Use pregenerated features
        if s_feats and t_feats:
            features_src = np.load(base + s_feats)
            features_tgt = np.load(base + t_feats)
        
            features_src = features_src['arr_0']
            features_tgt = features_tgt['arr_0']

        coarse_features = np.concatenate((features_src, features_tgt), axis=0)
        return torch.tensor(coarse_features).to('cuda:0')