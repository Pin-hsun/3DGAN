import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        assert(n_blocks >= 0)
        super(Generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light
        self.max3_pool = nn.MaxPool3d(2)

        DownBlock = []
        DownBlock += [nn.ReflectionPad2d(3),  # changed padding 3>1
                      nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=False),  # changed kernel 7>3
                      nn.InstanceNorm2d(ngf),
                      nn.ReLU(True)]

        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=False),
                          nn.InstanceNorm2d(ngf * mult * 2),
                          nn.ReLU(True)]
                          # Downsample(ngf * mult * 2)]

        # Down-Sampling Bottleneck
        mult = 2**n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        # Up-Sampling
        UpBlock2 = []
        n_downsampling = 2
        mult = 2 ** n_downsampling

        for i in range(n_blocks):
            UpBlock2 += [ResnetBlock3D(ngf * mult, use_bias=False)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            UpBlock2 += [
                            Upsample2(scale_factor=2),
                            # Upsample(ngf * mult),
                            nn.Conv3d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=1,
                                         padding=1, bias=False),
                            # nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2),
                            #              kernel_size=3, stride=2,
                            #              padding=1, output_padding=1,
                            #              bias=False),
                            nn.InstanceNorm3d(int(ngf * mult / 2)),
                            nn.ReLU(True)
                            ]
        UpBlock2 += [nn.ReflectionPad3d(3),
                     nn.Conv3d(ngf, output_nc, kernel_size=7, padding=0),
                     nn.Tanh()]
                     # nn.Sigmoid()]  # changed tanh > sigmoid

        self.DownBlock = nn.Sequential(*DownBlock)
        self.UpBlock2 = nn.Sequential(*UpBlock2)

    def forward(self, x, method=None, nce_layers=[3,6,10,13], direction='xy'):
        if method != 'decode':
            if direction == 'xy':
                x = x.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]  # (Z, C, X, Y)
            elif direction == 'yz':
                x = x.permute(2, 1, 3, 4, 0)[:, :, :, :, 0]  # (X, C, Y, Z)
            elif direction == 'xz':
                x = x.permute(3, 1, 2, 4, 0)[:, :, :, :, 0]  # (Y, C, X, Z)
            feats = []
            for layer_id, layer in enumerate(self.DownBlock):
                if layer_id in [4, 8]:
                    x = x.permute(1, 2, 3, 0).unsqueeze(0)  # (1, C, X, Y, Z)
                    x = self.max3_pool(x)
                    x = x.squeeze(0).permute(3, 0, 1, 2)  # (Z, C, X, Y)
                x = layer(x)
                if layer_id in nce_layers:
                    # print(layer_id, layer, x.shape)
                    if direction == 'xy':
                        feats.append(x.permute(1, 2, 3, 0).unsqueeze(0))
                    if direction == 'yz':
                        feats.append(x.permute(1, 0, 2, 3).unsqueeze(0))
                    if direction == 'xz':
                        feats.append(x.permute(1, 2, 0, 3).unsqueeze(0))
            if method == 'encode':
                return feats
            # flip to 3D same direction: x (1, C, X, Y, Z)
            if direction == 'xy':
                x = x.permute(1, 2, 3, 0).unsqueeze(0)  # (1, C, X, Y, Z)
            elif direction == 'yz':
                x = x.permute(1, 0, 2, 3).unsqueeze(0)
            elif direction == 'xz':
                x = x.permute(1, 2, 0, 3).unsqueeze(0)
        # print(x.shape) # torch.Size([32, 128, 32, 32])
        for layer_id, layer in enumerate(self.UpBlock2):
            x = layer(x)

        return {'out0': x}

class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ResnetBlock3D(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetBlock3D, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad3d(1),
                       nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm3d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad3d(1),
                       nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm3d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = nn.ReplicationPad2d(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class Upsample2(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()
        self.factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=self.factor, mode=self.mode)


class Upsample(nn.Module):
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride**3)
        print(filt.shape)
        self.register_buffer('filt', filt[None, None, :, :, :].repeat((self.channels, 1, 1, 1, 1)))

        self.pad = nn.ReflectionPad3d([1, 1, 1, 1, 1, 1])

    def forward(self, inp):
        print(self.pad(inp).shape)
        ret_val = F.conv_transpose3d(self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size, groups=inp.shape[1])[:, :, 1:, 1:]
        if(self.filt_odd):
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1, -1]

def get_filter(filt_size=3):
    if(filt_size == 1):
        a = np.array([1., ])
    elif(filt_size == 2):
        a = np.array([1., 1.])
    elif(filt_size == 3):
        a = np.array([1., 2., 1.])
    elif(filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif(filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)

    return filt
