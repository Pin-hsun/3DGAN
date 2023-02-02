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


def tile_like(x, target):  # tile_size = 256 or 4
    x = x.view(x.size(0), x.size(1), 1, 1)
    x = x.repeat(1, 1, target.size(2), target.size(3))
    return x


class Generator(nn.Module):
    def __init__(self, n_channels=1, out_channels=1, nf=32, batch_norm=True, activation=ACTIVATION, final='tanh', mc=False):
        super(Generator, self).__init__()

        conv_block = conv2d_bn_block if batch_norm else conv2d_block

        max_pool = nn.MaxPool2d(2)
        act = activation

        if mc:
            dropout = 0.5
        else:
            dropout = 0.0

        self.c_dim = 0

        self.down0 = nn.Sequential(
            #conv_block(n_channels + self.c_dim, nf, activation=act),
            # dsx
            nn.Conv2d(n_channels + self.c_dim, nf, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(nf, momentum=0.01),
            activation(),
            conv_block(nf, nf, activation=act)
        )
        self.down1 = nn.Sequential(
            max_pool,
            conv_block(nf, 2 * nf, activation=act),
            conv_block(2 * nf, 2 * nf, activation=act),
        )
        self.down2 = nn.Sequential(
            max_pool,
            conv_block(2 * nf, 4 * nf, activation=act),
            nn.Dropout(p=dropout, inplace=False),
            conv_block(4 * nf, 4 * nf, activation=act),

        )
        self.down3 = nn.Sequential(
            max_pool,
            conv_block(4 * nf, 8 * nf, activation=act),
            nn.Dropout(p=dropout, inplace=False),
            conv_block(8 * nf, 8 * nf, activation=act),
        )

        # DSX
        self.down4 = nn.Sequential(
            max_pool,
            conv_block(8 * nf, 16 * nf, activation=act),
            nn.Dropout(p=dropout, inplace=False),
            conv_block(16 * nf, 16 * nf, activation=act),
        )

        # DSX
        self.up4 = deconv2d_bn_block(16 * nf + 1, 8 * nf, activation=act)
        self.conv4 = nn.Sequential(
            conv_block(8 * nf, 8 * nf, activation=act),  # 8
            nn.Dropout(p=dropout, inplace=False),
            conv_block(8 * nf, 8 * nf, activation=act),
        )

        self.up3 = deconv2d_bn_block(8 * nf, 4 * nf, activation=act)
        self.conv5 = nn.Sequential(
            conv_block(4 * nf, 4 * nf, activation=act),  # 8
            nn.Dropout(p=dropout, inplace=False),
            conv_block(4 * nf, 4 * nf, activation=act),
        )

        self.up2 = deconv2d_bn_block(4 * nf, 2 * nf, activation=act)
        self.conv6 = nn.Sequential(
            conv_block(2 * nf, 2 * nf, activation=act),
            nn.Dropout(p=dropout, inplace=False),
            conv_block(2 * nf, 2 * nf, activation=act),
        )

        self.up1 = deconv2d_bn_block(2 * nf, nf, activation=act)

        final_layer = get_activation(final)

        self.conv7_k = nn.Sequential(
            conv_block(nf, out_channels, activation=final_layer),
        )

        self.conv7_g = nn.Sequential(
            conv_block(nf, out_channels, activation=final_layer),
        )

        #if NoTanh:
        #    self.conv7_k[-1] = self.conv7_k[-1][:-1]
        #    self.conv7_g[-1] = self.conv7_g[-1][:-1]

    def forward(self, xori, a=None):
        """
        concat in all three layers
        """
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
        x2 = self.down2(x1)   # Dropouta
        x3 = self.down3(x2)   # Dropout
        x40 = self.down4(x3)  # Dropout

        # injection

        # tiled_a = a * tile_like(torch.ones((x3.shape[0], 1)), x3).type_as(x3)
        Z = x40.shape[0] // a.shape[0]  # thickness = B*Z / B
        tiled_a = tile_like(a.unsqueeze(1).repeat(1, Z).view(-1, 1), x40).type_as(x40)

        x4 = torch.cat([x40, tiled_a], 1)
        xu4 = self.up4(x4)
        x3 = self.conv4(xu4)   # Dropout

        xu3 = self.up3(x3)
        x5 = self.conv5(xu3)   # Dropout

        # injection
        #tiled_a = a * tile_like(torch.ones((x5.shape[0], 1)), x5).type_as(x5)
        #tiled_a = tile_like(a.unsqueeze(1).repeat(1, Z).view(-1, 1), x5).type_as(x5)
        #x5 = torch.cat([x5, tiled_a], 1)

        xu2 = self.up2(x5)
        #cat2 = torch.cat([xu2, x1], 1)
        x6 = self.conv6(xu2)   # Dropout

        # injection
        #tiled_a = a * tile_like(torch.ones((x6.shape[0], 1)), x6).type_as(x6)
        #tiled_a = tile_like(a.unsqueeze(1).repeat(1, Z).view(-1, 1), x6).type_as(x6)
        #x6 = torch.cat([x6, tiled_a], 1)

        xu1 = self.up1(x6)
        xu1 = nn.Upsample(scale_factor=2)(xu1)
        x70 = self.conv7_k(xu1)
        x71 = self.conv7_g(xu1)

        return {'out0': x70, 'out1': x71, 'z': x40}


if __name__ == '__main__':
    g = Generator(n_channels=1, batch_norm=False, final='tanh')
    #from torchsummary import summary
    from utils.data_utils import print_num_of_parameters
    print_num_of_parameters(g)
    out = g(torch.rand(1, 1, 256, 256), a=torch.ones(1))
