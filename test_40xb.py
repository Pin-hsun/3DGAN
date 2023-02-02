from __future__ import print_function
import torch
import matplotlib.pyplot as plt
from dataloader.data_multi import PairedDataTif
from utils.data_utils import imagesc
import tifffile as tiff
import os, glob
import numpy as np
import torch.nn as nn


def test_40xb():
    cm = plt.get_cmap('viridis')

    test_set = PairedDataTif(root='/home/ubuntu/Data/Dataset/paired_images/40xhan/',
                             directions='xyft005_xyori05', permute=(0, 1, 2),
                             crop=[0, 512, 0, 896, 0, 1024], trd=500)#)[1890-1792, 1890, 1024, 2048, 0, 1024])
    x = test_set.__getitem__(0)['img']

    x = [y.permute(1, 0, 2, 3) for y in x]

    net = torch.load('/home/ubuntu/Data/logs/40xhan/cyc4b/'
                     '40xB/checkpoints/netGXY_model_epoch_200.pth').cuda()
    net.train()

    input = [i[400:401, :, ::].cuda() for i in x]
    out0, out1 = net(torch.cat(input, 1), a=None)

    #.detach().cpu()

    output = [i.detach().cpu() for i in [out0, out1]]

    imagesc(input[0][0, 0, -512:, -512:].cpu())
    imagesc(input[1][0, 0, -512:, -512:].cpu())
    imagesc(output[0][0, 0, -512:, -512:].cpu())
    imagesc(output[1][0, 0, -512:, -512:].cpu())
    tiff.imsave('temp.tif', output[0][0, 0, -512:, -512:].cpu().numpy())


def test_henry():
    net = torch.load('/home/ubuntu/Data/logs/Henry221215/0/checkpoints/segnet_model_epoch_100.pth').cuda()
    l = sorted(glob.glob('/home/ubuntu/Data/Dataset/paired_images/Henry221215/full/ori/*'))
    x = tiff.imread(l[5888-1])
    x = x - x.min()
    x = x / x.max()
    #x = (x - 0.5) * 2

    out = net(torch.from_numpy(x).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor).cuda(), a=None)
    imagesc(x)
    imagesc(out[0][0, 1, ::].detach().cpu())
    imagesc(torch.argmax(out[0], 1)[0, ::].detach().cpu())


def test_henry_stack():
    net = torch.load('/home/ubuntu/Data/logs/Henry221215/0/checkpoints/segnet_model_epoch_160.pth',
                     map_location='cuda:0').cuda()
    l = sorted(glob.glob('/home/ubuntu/Data/Dataset/BRC/Henry/221215/Crop_Original_Data/*'))
    dx = 128

    for name in l[:]:
        print(name)
        x0 = tiff.imread(name)
        print(x0.dtype)
        x = 1 * x0
        x[x == 0] = x.mean()
        x = x - x.min()
        x = x / x.max()
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor)
        osize = x.shape[2:4]

        H = int(dx*np.rint(osize[0] / 128))
        W = int(dx*np.rint(osize[1] / 128))

        x = torch.nn.Upsample(size=(H, W))(x)

        hwrange = [[(0, H//2, 0, W//2), (H//2, H, 0, W//2)], [(0, H//2, W//2, W), (H//2, H, W//2, W)]]

        seg_all = []
        for hw in hwrange:
            seg_col = []
            for h in hw:
                patch = x[:, :, h[0]:h[1], h[2]:h[3]]
                out = net(patch.cuda())
                out = nn.Softmax(dim=1)(out[0])
                seg = out[0, 1, ::].detach().cpu()
                seg_col.append(seg)
                del out
            seg_all.append(seg_col)

        seg_all = [torch.cat(x, 0) for x in seg_all]
        seg_all = torch.cat(seg_all, 1)
        seg_all = torch.nn.Upsample(size=osize)(seg_all.unsqueeze(0).unsqueeze(0))[0, 0, ::]

        seg_all = seg_all.numpy()
        seg_all = (seg_all * 255).astype(np.uint8)

        tiff.imwrite(name.replace('/Crop_Original_Data/', '/seg/'), seg_all)


if __name__ == '__main__':
    test_henry_stack()
    print('x')
