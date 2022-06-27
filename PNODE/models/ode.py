import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from .fewshot import FewShotSeg
from collections import OrderedDict
from .vgg import Encoder

from torchdiffeq import odeint
# from torchdiffeq import odeint_adjoint as odeint

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)



class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEfunc(nn.Module):

    def __init__(self, dim, n_layers=3, sigma=0.1, noise_type="additive"):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.layers = []
        self.norm = norm(dim)
        for i in range(n_layers):
            self.layers.append(
                {
                    "conv": ConcatConv2d(dim, dim, 3, 1, 1),
                    "norm": norm(dim),
                }
            )
        lyrs = [l["conv"] for l in self.layers] + [l["norm"] for l in self.layers]
        self.layers_seq = torch.nn.Sequential(*lyrs)
        self.nfe = 0
        self.sigma = sigma
        self.noise_type = noise_type

    def forward(self, t, x):
        self.nfe += 1

        out = self.norm1(x)
        for i in range(len(self.layers)):
            out = self.relu(out)
            out = self.layers[i]["conv"](t, out)
            out = self.layers[i]["norm"](out)
        
        return out 




class ODEBlock(nn.Module):

    def __init__(self, odefunc, ode_time=1):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, ode_time]).float()
        self.tol = 1e-3

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=self.tol, atol=self.tol)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value




class ODENet(nn.Module):
    def __init__(self, in_channels, pretrained_path=None, ode_layers=3, ode_time=1, noise_type=None, sigma=None):
        super(ODENet, self).__init__()
        self.ode = ODEBlock(ODEfunc(in_channels, n_layers=ode_layers, noise_type=noise_type, sigma=sigma), ode_time=ode_time)

    def forward(self, x):
        return self.ode(x)



class FewShotSegOde(FewShotSeg):
    def __init__(self, in_channels=1, pretrained_path=None, pretrained_ode=False, ode_layers=3, ode_time=1, noise_type="None", sigma=None):
        super().__init__(in_channels=in_channels, pretrained_path=pretrained_path)
        ode_weights = pretrained_path if pretrained_ode else None
        # Encoder
        if ode_layers == 5:
            last_2_layers = 1
        elif ode_layers == 4:
            last_2_layers = 2
        else:
            last_2_layers = 3
        self.encoder = nn.Sequential(
            OrderedDict(
                [
                    ('backbone', Encoder(in_channels, self.pretrained_path, rem_last_layer=True, pretrained_ode=pretrained_ode, last_2_layers=last_2_layers)),
                    ('ode', ODENet(512, pretrained_path=ode_weights, ode_layers=ode_layers, ode_time=ode_time, noise_type=noise_type, sigma=sigma)), 
                ]
            )
        )