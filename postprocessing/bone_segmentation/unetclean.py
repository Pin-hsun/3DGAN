import torch.nn as nn
from torch.nn import functional as F
import torch
from torchvision import models


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch, coord=False):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch, coord=False):
        super(up_conv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            self.conv,
            #nn.ConvTranspose2d(out_ch, out_ch, 2, 2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


def get_encoder(self, name, depth):
    backbone = getattr(models, name)(pretrained=True)

    if name in ['resnet34', 'resnet50', 'resnet101']:
        resnet_dict = {
            'resnet34': [64, 64, 128, 256, 512],
            'resnet50': [64, 256, 512, 1024, 2048],
            'resnet101': [64, 256, 512, 1024, 2048],
        }
        enco_fil = resnet_dict[name]
        setattr(self, 'encoder0', nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu))
        setattr(self, 'encoder1', nn.Sequential(backbone.maxpool, backbone.layer1))
        for i in range(2, depth):
            setattr(self, 'encoder' + str(i), getattr(backbone, 'layer' + str(i)))

    if name in ['vgg11', 'vgg11_bn', 'vgg16', 'vgg16_bn', 'vgg19']:
        enco_fil = [64, 128, 256, 512, 512]
        vgg_dict = {
            'vgg11': [None, 2, 5, 10, 15, 20],
            'vgg11_bn': [None, 3, 7, 14, 21, 28],
            'vgg16': [None, 4, 9, 16, 23, 30],
            'vgg16_bn': [None, 6, 13, 23, 33, 43],
            'vgg19': [None, 4, 9, 18, 27, 36]
        }
        loc = vgg_dict[name]

        for i in range(depth):
            setattr(self, 'encoder' + str(i), backbone.features[loc[i]:loc[i + 1]])

    return enco_fil


class UNetClean(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """

    def __init__(self, output_ch=1, backbone='vgg11', depth=5):
        super(UNetClean, self).__init__()

        self.depth = depth
        self.skip = [1, 1, 1, 1]  # (len = depth-1)
        deco_fil = [x * 32 for x in [1, 2, 4, 8, 8]]

        self.coord = False
        enco_fil = get_encoder(self, backbone, depth)
        skip_fil = enco_fil

        # Up Path
        for i in range(depth - 3, -1, -1):  # [2, 1, 0]
            setattr(self, 'Up'+str(i), up_conv(deco_fil[i+1], deco_fil[i], coord=self.coord))
        # First up layer
        i = depth - 2
        setattr(self, 'Up'+str(i), up_conv(enco_fil[i+1], deco_fil[i], coord=self.coord))

        for i in range(depth - 2, -1, -1):  # [3, 2, 1, 0]
            setattr(self, 'Up_conv'+str(i),
                    conv_block(skip_fil[i] * self.skip[i] + deco_fil[i], deco_fil[i], coord=self.coord))

        # Final
        self.Conv = nn.Conv2d(deco_fil[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = x - x.min()
        x = x / x.max()
        # x (b, t, c, w, h)
        # Encoder Path
        B = x.shape[0]
        depth = self.depth
        e = [self.encoder0(x)]
        for i in range(1, depth):
            e.append(getattr(self, 'encoder'+str(i))(e[i-1]))  # (b*t, c, w, h)

        # Skip connection path
        for i in range(depth):
            e[i] = e[i]

        # Decoder path
        d = e[-1]
        for i in range(depth - 2, -1, -1):
            d = getattr(self, 'Up'+str(i))(d)
            if self.skip[i]:
                d = torch.cat((e[i], d), dim=1)
            d = getattr(self, 'Up_conv'+str(i))(d)

        d = self.Conv(d)
        d = d.view(B, d.shape[1], d.shape[2], d.shape[3])  # (b, t, c, w, h)

        return d,
