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
import numpy as np
from networks.model_utils import get_activation
import tifffile

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


def conv2d_block(in_channels, out_channels, kernel=3, stride=1, padding=1, activation=ACTIVATION):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=padding),
        activation(),
    )


def conv2d_bn_block(in_channels, out_channels, kernel=3, momentum=0.01, activation=ACTIVATION):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=1),
        nn.BatchNorm2d(out_channels, momentum=momentum),
        activation(),
    )

def conv2d_in_block(in_channels, out_channels, kernel=3, momentum=0.01, activation=ACTIVATION):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=1),
        nn.InstanceNorm2d(out_channels, momentum=momentum),
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


def conv3d_block(in_channels, out_channels, kernel=3, stride=1, padding=1, activation=ACTIVATION):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel, stride=stride, padding=padding),
        activation(),
    )


def conv3d_bn_block(in_channels, out_channels, kernel=3, momentum=0.01, activation=ACTIVATION):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel, padding=1),
        nn.BatchNorm3d(out_channels, momentum=momentum),
        activation(),
    )

def conv3d_in_block(in_channels, out_channels, kernel=3, momentum=0.01, activation=ACTIVATION):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel, padding=1),
        nn.InstanceNorm3d(out_channels, momentum=momentum),
        activation(),
    )


def deconv3d_bn_block(in_channels, out_channels, use_upsample=True, kernel=4, stride=2, padding=1, momentum=0.01,
                      activation=ACTIVATION):
    if use_upsample:
        up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=1)
        )
    else:
        up = nn.ConvTranspose3d(in_channels, out_channels, kernel, stride=stride, padding=padding)
    return nn.Sequential(
        up,
        nn.BatchNorm3d(out_channels, momentum=momentum),
        activation(),
    )

def deconv3d_in_block(in_channels, out_channels, use_upsample=True, kernel=4, stride=2, padding=1, momentum=0.01,
                      activation=ACTIVATION):
    if use_upsample:
        up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=1)
        )
    else:
        up = nn.ConvTranspose3d(in_channels, out_channels, kernel, stride=stride, padding=padding)
    return nn.Sequential(
        up,
        nn.InstanceNorm3d(out_channels, momentum=momentum),
        activation(),
    )

def dense_layer_bn(in_dim, out_dim, momentum=0.01, activation=ACTIVATION):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim, momentum=momentum),
        activation()
    )


class Generator(nn.Module):
    def __init__(self, n_channels=1, out_channels=1, nf=32, norm='batch', activation=ACTIVATION, final='tanh', mc=False):
        super(Generator, self).__init__()
        if norm == 'batch':
            conv2_block = conv2d_bn_block
            conv3_block = conv3d_bn_block
        elif norm == 'instance':
            conv2_block = conv2d_in_block
            conv3_block = conv3d_in_block
        else:
            conv2_block = conv2d_block
            conv3_block = conv3d_block

        deconv3_block = deconv3d_in_block
        max2_pool = nn.MaxPool2d(2)
        self.max3_pool = nn.MaxPool3d(2)
        act = activation

        if mc:
            dropout = 0.5
        else:
            dropout = 0.0

        self.down0 = nn.Sequential(
            conv2_block(n_channels, nf, activation=act),
            conv2_block(nf, nf, activation=act)
        )
        self.down1 = nn.Sequential(
            #max2_pool,
            conv2_block(nf, 2 * nf, activation=act),
            conv2_block(2 * nf, 2 * nf, activation=act),
        )
        self.down2 = nn.Sequential(
            #max2_pool,
            conv2_block(2 * nf, 4 * nf, activation=act),
            nn.Dropout(p=dropout, inplace=False),
            conv2_block(4 * nf, 4 * nf, activation=act),

        )
        self.down3 = nn.Sequential(
            #max2_pool,
            conv2_block(4 * nf, 8 * nf, activation=act),
            nn.Dropout(p=dropout, inplace=False),
            conv2_block(8 * nf, 8 * nf, activation=act),
        )

        self.up3 = deconv3_block(8 * nf, 4 * nf, activation=act)

        self.conv5 = nn.Sequential(
            conv3_block(4 * nf, 4 * nf, activation=act),  # 8
            nn.Dropout(p=dropout, inplace=False),
            conv3_block(4 * nf, 4 * nf, activation=act),
        )
        self.up2 = deconv3_block(4 * nf, 2 * nf, activation=act)
        self.conv6 = nn.Sequential(
            conv3_block(2 * nf, 2 * nf, activation=act),
            nn.Dropout(p=dropout, inplace=False),
            conv3_block(2 * nf, 2 * nf, activation=act),
        )

        self.up1 = deconv3_block(2 * nf, nf, activation=act)

        final_layer = get_activation(final)

        self.conv7_k = nn.Sequential(
            conv3_block(nf, out_channels, activation=final_layer),
        )

        self.conv7_g = nn.Sequential(
            conv3_block(nf, out_channels, activation=final_layer),
        )

        #if NoTanh:
        #    self.conv7_k[-1] = self.conv7_k[-1][:-1]
        #    self.conv7_g[-1] = self.conv7_g[-1][:-1]

        self.encoder = nn.Sequential(self.down0, self.down1, self.down2, self.down3)
        self.decoder = nn.Sequential(
            # deconv3_block(24 * nf, 8 * nf, activation=act),
                                    self.up3, self.conv5, self.up2, self.conv6, self.up1)

    def forward(self, x, method=None, direction='xy'):
        if method != 'decode':
            # x (1, C, X, Y, Z)
            if direction == 'xy':
                x = x.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]  # (Z, C, X, Y)
            elif direction == 'yz':
                x = x.permute(2, 1, 3, 4, 0)[:, :, :, :, 0]  # (X, C, Y, Z)
            elif direction == 'xz':
                x = x.permute(3, 1, 2, 4, 0)[:, :, :, :, 0]  # (Y, C, X, Z)
            feat = []
            for i in range(len(self.encoder)):
                if i > 0:
                    x = x.permute(1, 2, 3, 0).unsqueeze(0)  # (1, C, X, Y, Z)
                    x = self.max3_pool(x)
                    x = x.squeeze(0).permute(3, 0, 1, 2)  # (Z, C, X, Y)
                x = self.encoder[i](x)
                if direction == 'xy':
                    feat.append(x.permute(1, 2, 3, 0).unsqueeze(0))
                if direction == 'yz':
                    feat.append(x.permute(1, 0, 2, 3).unsqueeze(0))
                if direction == 'xz':
                    feat.append(x.permute(1, 2, 0, 3).unsqueeze(0))
            if method == 'encode':
                return feat

            # flip to 3D same direction: x (1, C, X, Y, Z)
            if direction == 'xy':
                x = x.permute(1, 2, 3, 0).unsqueeze(0)  # (1, C, X, Y, Z)
            elif direction == 'yz':
                x = x.permute(1, 0, 2, 3).unsqueeze(0)
            elif direction == 'xz':
                x = x.permute(1, 2, 0, 3).unsqueeze(0)
        x = self.decoder(x)
        x70 = self.conv7_k(x)
        x71 = self.conv7_g(x)
        return {'out0': x70, 'out1': x71}


if __name__ == '__main__':
    g = Generator(n_channels=3, batch_norm=False, final='tanh')
    #from torchsummary import summary
    from utils.data_utils import print_num_of_parameters
    print_num_of_parameters(g)
    f = g(torch.rand(1, 3, 128, 128, 64), method='encode')