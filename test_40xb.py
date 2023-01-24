from __future__ import print_function
import torch
import matplotlib.pyplot as plt
from dataloader.data_multi import PairedDataTif
from utils.data_utils import imagesc
import tifffile as tiff
import os, glob


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
    net = torch.load('/home/ubuntu/Data/logs/Henry221215/0/checkpoints/segnet_model_epoch_100.pth').cuda()
    l = sorted(glob.glob('/home/ubuntu/Data/Dataset/BRC/Henry/221215/Crop_Original_Data/*'))

    for name in l[:10]:
        x = tiff.imread(name)
        x = x - x.min()
        x = x / x.max()
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor)
        x = torch.nn.Upsample(size=(128*37, 50*128))(x)
        out = net(x.cuda())
        seg = out[0][0, 1, ::].detach().cpu()
        tiff.imwrite(name.repalce('/Crop_Original_data/', 'seg'), seg)


if __name__ == '__main__':
    test_henry_stack()
