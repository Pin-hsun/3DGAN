# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
#
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

"""AttGAN, generator, and discriminator."""

import torch
import torch.nn as nn
from models.AttGAN.nn import LinearBlock, Conv2dBlock, ConvTranspose2dBlock

# This architecture is for images of 128x128s
# In the original AttGAN, slim.conv2d uses padding 'same'
MAX_DIM = 64 * 16  # 1024


class Att(nn.Module): #layers was 5
    def __init__(self, enc_dim=64, enc_layers=5, enc_norm_fn='batch', enc_acti_fn='lrelu',
                 dec_dim=64, dec_layers=5, dec_norm_fn='batch', dec_acti_fn='relu',
                 n_attrs=1, shortcut_layers=1, inject_layers=0, img_size=128, final='tanh'):
        super(Att, self).__init__()
        self.shortcut_layers = min(shortcut_layers, dec_layers - 1)
        self.inject_layers = min(inject_layers, dec_layers - 1)
        self.f_size = img_size // 2 ** enc_layers  # f_size = 4 for 128x128

        layers = []
        n_in = 3
        for i in range(enc_layers):
            n_out = min(enc_dim * 2 ** i, MAX_DIM)
            layers += [Conv2dBlock(
                n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=enc_norm_fn, acti_fn=enc_acti_fn
            )]
            n_in = n_out
        self.enc_layers = nn.ModuleList(layers)

        layers = []
        n_in = n_in + n_attrs  # 1024 + 13
        for i in range(dec_layers):
            if i < dec_layers - 1:
                n_out = min(dec_dim * 2 ** (dec_layers - i - 1), MAX_DIM)
                layers += [ConvTranspose2dBlock(
                    n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=dec_norm_fn, acti_fn=dec_acti_fn
                )]
                n_in = n_out
                n_in = n_in + n_in // 2 if self.shortcut_layers > i else n_in
                n_in = n_in + n_attrs if self.inject_layers > i else n_in
            else:
                layers += [ConvTranspose2dBlock(
                    n_in, 3, (4, 4), stride=2, padding=1, norm_fn='none', acti_fn=final
                )]
        self.dec_layers = nn.ModuleList(layers)

    def encode(self, x):
        z = x
        zs = []
        for layer in self.enc_layers:
            z = layer(z)
            zs.append(z)
        return zs

    def decode(self, zs, a):
        a_tile = a.view(a.size(0), -1, 1, 1).repeat(1, 1, self.f_size, self.f_size)
        z = torch.cat([zs[-1], a_tile], dim=1)
        for i, layer in enumerate(self.dec_layers):
            z = layer(z)
            if self.shortcut_layers > i:  # Concat 1024 with 512
                z = torch.cat([z, zs[len(self.dec_layers) - 2 - i]], dim=1)
            if self.inject_layers > i:
                a_tile = a.view(a.size(0), -1, 1, 1) \
                    .repeat(1, 1, self.f_size * 2 ** (i + 1), self.f_size * 2 ** (i + 1))
                z = torch.cat([z, a_tile], dim=1)
        return z

    def forward(self, x, a=None, mode='enc-dec'):
        if mode == 'enc-dec':
            return self.decode(self.encode(x), a),
        if mode == 'enc':
            return self.encode(x)
        if mode == 'dec':
            assert a is not None, 'No given attribute.'
            return self.decode(x, a)
        raise Exception('Unrecognized mode: ' + mode)


from models.DeScarGan.descargan import conv2d_block, conv2d_bn_block, deconv2d_bn_block, get_activation

class Generator(nn.Module):
    def __init__(self, n_channels=1, nf=32, batch_norm=True, activation=nn.ReLU, final='tanh'):
        super(Generator, self).__init__()

        conv_block = conv2d_bn_block if batch_norm else conv2d_block

        max_pool = nn.MaxPool2d(2)
        act = activation
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
            conv_block(4 * nf, 4 * nf, activation=act),

        )
        self.down3 = nn.Sequential(
            max_pool,
            conv_block(4 * nf, 8 * nf, activation=act),
            conv_block(8 * nf, 8 * nf, activation=act),
        )

        self.up3 = deconv2d_bn_block(8 * nf, 4 * nf, activation=act)

        self.conv5 = nn.Sequential(
            conv_block(8 * nf, 4 * nf, activation=act),  # 8
            conv_block(4 * nf, 4 * nf, activation=act),
        )
        self.up2 = deconv2d_bn_block(4 * nf, 2 * nf, activation=act)
        self.conv6 = nn.Sequential(
            conv_block(4 * nf, 2 * nf, activation=act),
            conv_block(2 * nf, 2 * nf, activation=act),
        )

        self.up1 = deconv2d_bn_block(2 * nf, nf, activation=act)

        final_layer = get_activation(final)

        self.conv7_k = nn.Sequential(
            conv_block(nf, n_channels, activation=final_layer),
        )

        self.conv7_g = nn.Sequential(
            conv_block(nf, n_channels, activation=final_layer),
        )

        self.encoder = nn.Sequential(self.down0, self.down1, self.down2, self.down3)

    def forward(self, x, a):
        # c: (B, C)
        if self.c_dim > 0:
            c = a
            c1 = c.view(c.size(0), c.size(1), 1, 1)
            c1 = c1.repeat(1, 1, x.size(2), x.size(3))  # (B, 2, H, W)
            x = torch.cat([x, c1], dim=1)

        x0 = self.down0(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        xu3 = self.up3(x3)
        xu2 = self.up2(xu3)
        xu1 = self.up1(xu2)

        x70 = self.conv7_k(xu1)
        x71 = self.conv7_g(xu1)

        return x70, x71


if __name__ == '__main__':
    g = Generator(n_channels=3, batch_norm=False, final='tanh').cuda()
    print(g(torch.rand(1, 3, 256, 256).cuda(), a=torch.ones(2, 2).cuda())[0].shape)
