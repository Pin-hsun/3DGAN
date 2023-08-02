import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from networks.DeScarGan.descargan import conv2d_block, conv2d_bn_block, deconv2d_bn_block, conv3d_block, conv3d_bn_block, deconv3d_bn_block
from networks.model_utils import get_activation, Identity

class Generator(nn.Module):
    def __init__(self, n_channels=1, out_channels=1, nf=32, batch_norm=True, activation=nn.ReLU, final='tanh'
                 , mc=False, conv3D=True):
        super(Generator, self).__init__()
        if conv3D:
            print('use 3D conv')
            conv_block = conv3d_bn_block if batch_norm else conv3d_block
            max_pool = nn.MaxPool3d(2)
            deconv_block = deconv3d_bn_block
        else:
            conv_block = conv2d_bn_block if batch_norm else conv2d_block
            max_pool = nn.MaxPool2d(2)
            deconv_block = deconv2d_bn_block

        act = activation

        if mc:
            dropout = 0.5
        else:
            dropout = 0.0

        self.label_k = torch.tensor([0, 1]).half().cuda()
        self.c_dim = 0

        self.down0 = nn.Sequential(
            conv_block(n_channels + self.c_dim, nf, activation=act),
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

        self.up3 = deconv_block(8 * nf, 4 * nf, activation=act, use_upsample = False)

        self.conv5 = nn.Sequential(
            conv_block(8 * nf, 4 * nf, activation=act),  # 8
            nn.Dropout(p=dropout, inplace=False),
            conv_block(4 * nf, 4 * nf, activation=act),
        )
        self.up2 = deconv_block(4 * nf, 2 * nf, activation=act, use_upsample=False)
        self.conv6 = nn.Sequential(
            conv_block(4 * nf, 2 * nf, activation=act),
            nn.Dropout(p=dropout, inplace=False),
            conv_block(2 * nf, 2 * nf, activation=act),
        )

        self.up1 = deconv_block(2 * nf, nf, activation=act,use_upsample=False)

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
        x = 1 * xori
        # c: (B, C)
        self.c_dim = 0
        if self.c_dim > 0:
            c = a
            c1 = c.view(c.size(0), c.size(1), 1, 1, 1)
            c1 = c1.repeat(1, 1, x.size(2), x.size(3), x.size(4))  # (B, 2, H, W)
            x = torch.cat([x, c1], dim=1)

        x0 = self.down0(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)   # Dropout
        x3 = self.down3(x2)   # Dropout

        xu3 = self.up3(x3)
        cat3 = torch.cat([xu3, x2], 1)
        x5 = self.conv5(cat3)   # Dropout

        xu2 = self.up2(x5)
        cat2 = torch.cat([xu2, x1],1)
        x6 = self.conv6(cat2)   # Dropout
        xu1 = self.up1(x6)
        #cat1 = crop_and_concat(xu1, x0)

        #if self.label_k in c:
        x70 = self.conv7_k(xu1)
        #else:
        x71 = self.conv7_g(xu1)

        return {'out0': x70, 'out1': x71}

if __name__ == '__main__':
    g = Generator(n_channels=3, batch_norm=False, final='tanh')
    #from torchsummary import summary
    from utils.data_utils import print_num_of_parameters
    print_num_of_parameters(g)
