# Copyright 2020 University of Basel, Center for medical Image Analysis and Navigation
#
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.model_utils import get_activation
import numpy as np

from networks.DSGan.dsmc import Generator as GeneratorOriginal

sig = nn.Sigmoid()
ACTIVATION = nn.ReLU
#device = 'cuda'


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.reshape(x.size()[0], -1)


def crop_and_concat(upsampled, bypass, crop=False):
    if crop:
        c = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = F.pad(bypass, (-c, -c, -c, -c))

    return torch.cat((upsampled, bypass), 1)


def conv2d_bn_block(in_channels, out_channels, kernel=3, momentum=0.01, activation=ACTIVATION):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=1),
        nn.BatchNorm2d(out_channels, momentum=momentum),
        activation(),
    )


def deconv2d_bn_block(in_channels, out_channels, use_upsample=True, kernel=4, stride=2, padding=1, momentum=0.01,
                      activation=ACTIVATION):
    if use_upsample:
        up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        )
    else:
        up = nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=stride, padding=padding)
    return nn.Sequential(
        up,
        nn.BatchNorm2d(out_channels, momentum=momentum),
        activation(),
    )


def dense_layer_bn(in_dim, out_dim, momentum=0.01, activation=ACTIVATION):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim, momentum=momentum),
        activation()
    )


def conv2d_block(in_channels, out_channels, kernel=3, stride=1, padding=1, activation=ACTIVATION):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=padding),
        activation(),
    )


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.softmax(x1)
        return x * x1


class Generator(GeneratorOriginal):
    def __init__(self, n_channels=1, out_channels=1, nf=32, batch_norm=True, activation=ACTIVATION, final='tanh', mc=False):
        super(Generator, self).__init__(n_channels, out_channels, nf, batch_norm, activation, final, mc)

        self.attention3 = nn.Sequential(
            nn.Conv2d(8 * nf, 8 * nf // 8, kernel_size=1),
            nn.Conv2d(8 * nf // 8, 8 * nf, kernel_size=1),
            nn.Softmax(dim=-2)
        )

        self.attention2 = nn.Sequential(
            nn.Conv2d(4 * nf, 4 * nf // 8, kernel_size=1),
            nn.Conv2d(4 * nf // 8, 4 * nf, kernel_size=1),
            nn.Softmax(dim=-2)
        )

    def forward(self, xori, a=None):
        x = 1 * xori
        # c: (B, C)
        self.c_dim = 0
        if self.c_dim > 0:
            c = a
            c1 = c.view(c.size(0), c.size(1), 1, 1)
            c1 = c1.repeat(1, 1, x.size(2), x.size(3))  # (B, 2, H, W)
            x = torch.cat([x, c1], dim=1)

        x0 = self.down0(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)  # Dropout
        x3 = self.down3(x2)  # Dropout

        attn3 = self.attention3(x3)
        x3_attn = F.interpolate(attn3, size=x3.size()[2:], mode='bilinear', align_corners=False)
        x3 = x3 * attn3

        xu3 = self.up3(x3)
        cat3 = torch.cat([xu3, x2], 1)
        x5 = self.conv5(cat3)  # Dropout

        attn2 = self.attention2(x5)
        x5_attn = F.interpolate(attn2, size=x5.size()[2:], mode='bilinear', align_corners=False)
        x5 = x5 * attn2

        xu2 = self.up2(x5)
        cat2 = torch.cat([xu2, x1], 1)
        x6 = self.conv6(cat2)  # Dropout

        xu1 = self.up1(x6)
        x70 = self.conv7_k(xu1)
        x71 = self.conv7_g(xu1)

        return {'out0': x70, 'out1': x71, 'attn3': x3_attn, 'attn2': x5_attn}


if __name__ == '__main__':
    g = Generator(n_channels=3, batch_norm=False, final='tanh')
    #from torchsummary import summary
    from utils.data_utils import print_num_of_parameters
    print_num_of_parameters(g)

    out = g(torch.randn(1, 3, 256, 256))