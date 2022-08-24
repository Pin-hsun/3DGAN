import torch.nn as nn

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_activation(fn):
    if fn == 'none':
        return Identity
    elif fn == 'relu':
        return nn.ReLU
    elif fn == 'lrelu':
        return nn.LeakyReLU(0.01)  # pix2pix use 0.2
    elif fn == 'sigmoid':
        return nn.Sigmoid
    elif fn == 'tanh':
        return nn.Tanh
    else:
        raise Exception('Unsupported activation function: ' + str(fn))

