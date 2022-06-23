import torch
import torch.nn as nn


def discriminator_block(in_filters, out_filters, normalize=True):
    """Returns downsampling layers of each discriminator block"""
    layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
    if normalize:
        layers.append(nn.InstanceNorm2d(out_filters))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        assert patch in [4, 8, 16]
        print('Use ' + str(patch) + ' patch discriminator')

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        out = self.model(img)
        return out,