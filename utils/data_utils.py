import torch
import numpy as np
from PIL import Image
from skimage import data, io
import matplotlib.pyplot as plt
import tifffile as tiff
from torch import nn

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

def three2twoD(tensor): #torch.Size([32, C, 32, 32])
    # intput(B,C,x,y,z)
    # ls = []
    # for i in range(tensor.shape[0]):
    #     cube = tensor[i,:,:,:,:] #size(C,x,y,z)
    #     # let z become B
    #     out = cube.permute(0,3,2,1)
    #     ls.append(out)
    # out = torch.cat(ls, 0)
    # out = out.permute(3, 0, 2, 1)
    # out = out.reshape(out.shape[0]*out.shape[1], 1, out.shape[2], out.shape[3])
    out = tensor.reshape(tensor.shape[0] * tensor.shape[2],  tensor.shape[1], tensor.shape[3], tensor.shape[4])
    return out

def save_tif(tensor, dest):
    tensor = tensor.detach().cpu()
    np = tensor.numpy()
    tiff.imsave(dest, np)

def interpolation(input, N):
    # m = nn.Upsample(scale_factor=(N, 1, 1), mode='trilinear')
    m = nn.Upsample(scale_factor=(N, 1, 1))
    out = m(input) #torch.Size([1, 1, 318, 444, 444])

    return out

def slice_cube(tensor, N, size, start=5):
    step_size = int(size / N)
    selected_slices = []
    # Iterate through the tensor and select slices
    for i in range(start, size-start, step_size):
        selected_slices.append(tensor[:, :, :, :, i].unsqueeze(0).permute(0, 1, 4, 3, 2))

    out = torch.cat(selected_slices, 1)
    out = out.squeeze(0).permute(3, 0, 1, 2)  #torch.Size([4, 1, 256, 256])
    return out