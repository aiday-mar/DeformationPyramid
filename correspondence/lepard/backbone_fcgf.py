import torch.nn.functional as F
import numpy as np

class FCGF(nn.Module):

    def __init__(self, config):
        super(FCGF, self).__init__()

    def forward(self, batch, phase = 'encode'):
        print('Inside of FCGF forward')
        