'''
Branch to estimate uncertainty of samples
--------------------------------------
Author: Yang Qian
Email: yqian@aibee.com
'''
import torch.nn as nn
import torch

class Sigma(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.model = nn.Sequential(
                nn.Linear(n_input, n_input),
                nn.LeakyReLU(inplace=True),
                nn.Linear(n_input, n_output),
                nn.BatchNorm1d(n_output)
            )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 1e-4)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, s, t, l2_norm=True):
        if l2_norm:
            s = torch.nn.functional.normalize(s)
            t = torch.nn.functional.normalize(t)
        return self.model((s-t)*(s-t))