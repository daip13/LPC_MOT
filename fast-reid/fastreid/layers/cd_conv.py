import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class CDConv(nn.Module):
    '''
    Implement channel deformable convolution for a single kernel.
    Date: 2020/01/10
    Author: Yang Qian
    Steps:
        1. depth-wise convolution for data preparation
        2. calculate spatial offset and generate meshgrid for every channel
        3. sample on every channel
        4. point-wise convolution to merge information
    Usage:
        add one channel to current typical convolution layer to see the influence
    TODO: how to implement CDConv for multiple kernel efficiently, just like:
    https://stackoverflow.com/questions/59284752/how-to-run-sub-operators-in-parallel-by-contrast-with-nn-sequential-in-pytorch
    '''

    def __init__(self, inplanes, outplanes=1, kernel_size=3, stride=1, padding=1, group=1):
        super(CDConv, self).__init__()
        # assert outplanes==1
        self.depth_conv = nn.Conv2d(inplanes, inplanes, kernel_size=kernel_size, stride=stride, padding=padding, groups=inplanes, bias=False)
        self.point_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.offset = nn.Conv2d(inplanes, (inplanes//group)*2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.group = group

        self.grid = None
        self.sample_grid = None

    def forward(self, x):
        offset = F.tanh(self.offset(x))
        out = self.depth_conv(x)
        c, h, w = out.shape[1], out.shape[2], out.shape[3]
        if self.grid is None:
            grid_X, grid_Y = np.meshgrid(np.linspace(-1,1,w),np.linspace(-1,1,h))
            grid = np.concatenate((grid_X[np.newaxis, :, :], grid_Y[np.newaxis, :, :]), axis=0)
            grid = np.repeat(grid, [c//self.group, c//self.group], axis=0)
            self.grid = torch.FloatTensor(grid).unsqueeze(0).cuda()

        sample_grid = torch.clamp(self.grid+offset, -1, 1)
        sample_grid = sample_grid.permute(0, 2, 3, 1)
        self.sample_grid = sample_grid
        if self.group == 1:
            out = torch.cat([F.grid_sample(out[:, i, :, :].unsqueeze(1), sample_grid[:, :, :, i*2:i*2+2]) for i in range(c//self.group)], dim=1)
        else:
            out = torch.cat([F.grid_sample(out[:, i*self.group:(i+1)*self.group, :, :], sample_grid[:, :, :, i*2:i*2+2]) for i in range(c//self.group)], dim=1)
        out = self.point_conv(out)
        return out

class CDConvBlock(nn.Module):
    '''
    using a residule-style implementation, inspired by non-local neural network
    '''
    def __init__(self, inplanes, kernel_size=1, stride=1, padding=0, group=-1, bn_layer=False):
        super().__init__()
        if group == -1:
            group = inplanes
        self.conv = CDConv(inplanes=inplanes, outplanes=inplanes, kernel_size=kernel_size, stride=stride, padding=padding, group=group)
        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=inplanes, out_channels=inplanes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inplanes)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(in_channels=inplanes, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        return self.W(self.conv(x)) + x