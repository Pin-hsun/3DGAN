"""Network components."""

import torch.nn as nn
from models.AttGAN.switchable_norm import SwitchNorm1d, SwitchNorm2d


def add_normalization_1d(layers, fn, n_out):
    if fn == 'none':
        pass
    elif fn == 'batch':
        layers.append(nn.BatchNorm1d(n_out))
    elif fn == 'instance':
        layers.append(Unsqueeze(-1))
        layers.append(nn.InstanceNorm1d(n_out, affine=True))
        layers.append(Squeeze(-1))
    elif fn == 'switch':
        layers.append(SwitchNorm1d(n_out))
    else:
        raise Exception('Unsupported normalization: ' + str(fn))
    return layers


def add_normalization_2d(layers, fn, n_out):
    if fn == 'none':
        pass
    elif fn == 'batch':
        layers.append(nn.BatchNorm2d(n_out))
    elif fn == 'instance':
        layers.append(nn.InstanceNorm2d(n_out, affine=True))
    elif fn == 'switch':
        layers.append(SwitchNorm2d(n_out))
    else:
        raise Exception('Unsupported normalization: ' + str(fn))
    return layers


def add_activation(layers, fn):
    if fn == 'none':
        pass
    elif fn == 'relu':
        layers.append(nn.ReLU())
    elif fn == 'lrelu':
        layers.append(nn.LeakyReLU(0.01))  # pix2pix use 0.2
    elif fn == 'sigmoid':
        layers.append(nn.Sigmoid())
    elif fn == 'tanh':
        layers.append(nn.Tanh())
    else:
        raise Exception('Unsupported activation function: ' + str(fn))
    return layers


class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class LinearBlock(nn.Module):
    def __init__(self, n_in, n_out, norm_fn='none', acti_fn='none'):
        super(LinearBlock, self).__init__()
        layers = [nn.Linear(n_in, n_out, bias=(norm_fn == 'none'))]
        layers = add_normalization_1d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Conv2dBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride=1, padding=0,
                 norm_fn=None, acti_fn=None):
        super(Conv2dBlock, self).__init__()
        layers = [nn.Conv2d(n_in, n_out, kernel_size, stride=stride, padding=padding, bias=(norm_fn == 'none'))]
        layers = add_normalization_2d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvTranspose2dBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride=1, padding=0,
                 norm_fn=False, acti_fn=None):
        super(ConvTranspose2dBlock, self).__init__()
        layers = [
            nn.ConvTranspose2d(n_in, n_out, kernel_size, stride=stride, padding=padding, bias=(norm_fn == 'none'))]
        layers = add_normalization_2d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)