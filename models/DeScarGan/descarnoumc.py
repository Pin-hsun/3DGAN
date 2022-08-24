import torch
import torch.nn as nn

from networks.DeScarGan.descargan import conv2d_block, conv2d_bn_block, deconv2d_bn_block, get_activation

class Generator(nn.Module):
    def __init__(self, n_channels=1, out_channels=1, nf=32, batch_norm=True, activation=nn.ReLU, final='tanh', mc=False):
        super(Generator, self).__init__()

        conv_block = conv2d_bn_block if batch_norm else conv2d_block

        max_pool = nn.MaxPool2d(2)
        act = activation
        self.label_k = torch.tensor([0, 1]).half().cuda()
        self.c_dim = 0

        if mc:
            dropout = 0.5
        else:
            dropout = 0.0

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

        self.up3 = deconv2d_bn_block(8 * nf, 4 * nf, activation=act)

        self.conv5 = nn.Sequential(
            conv_block(8 * nf, 4 * nf, activation=act),  # 8
            nn.Dropout(p=dropout, inplace=False),
            conv_block(4 * nf, 4 * nf, activation=act),
        )
        self.up2 = deconv2d_bn_block(4 * nf, 2 * nf, activation=act)
        self.conv6 = nn.Sequential(
            conv_block(4 * nf, 2 * nf, activation=act),
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

        self.encoder = nn.Sequential(self.down0, self.down1, self.down2, self.down3)

    def forward(self, x, a=None):
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
    print(g(torch.rand(1, 3, 128, 128).cuda(), a=torch.ones(2, 2).cuda())[0].shape)
