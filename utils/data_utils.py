import torch
import numpy as np
from PIL import Image
from skimage import data, io
import matplotlib.pyplot as plt


def print_num_of_parameters(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    print('Number of parameters: ' + str(sum([np.prod(p.size()) for p in model_parameters])))


def norm_01(x):
    """
    normalize to 0 - 1
    """
    x = x - x.min()
    x = x / x.max()
    return x


def to_8bit(x):
    if type(x) == torch.Tensor:
        x = (x / x.max() * 255).numpy().astype(np.uint8)
    else:
        x = (x / x.max() * 255).astype(np.uint8)

    if len(x.shape) == 2:
        x = np.concatenate([np.expand_dims(x, 2)]*3, 2)
    return x


def imagesc(x, show=True, save=None):
    # switch
    if (len(x.shape) == 3) & (x.shape[0] == 3):
        x = np.transpose(x, (1, 2, 0))

    x = x - x.min()
    x = Image.fromarray(to_8bit(x))

    if show:
        io.imshow(np.array(x))
        plt.show()
    if save:
        x.save(save)


def purge_logs():
    import glob, os
    list_version = sorted(glob.glob('logs/default/*/'))
    list_checkpoint = sorted(glob.glob('logs/default/*/checkpoints/*'))

    checkpoint_epochs = [0] * len(list_version)
    for c in list_checkpoint:
        checkpoint_epochs[list_version.index(c.split('checkpoints')[0])] = int(c.split('epoch=')[-1].split('.')[0])

    for i in range(len(list_version)):
        if checkpoint_epochs[i] < 60:
            os.system('rm -rf ' + list_version[i])



