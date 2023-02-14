from __future__ import print_function
import torch
import matplotlib.pyplot as plt
from dataloader.data_multi import PairedDataTif
from utils.data_utils import imagesc

cm = plt.get_cmap('viridis')

test_set = PairedDataTif(root='/media/ExtHDD01/Dataset/paired_images/Fly0B',
                         directions='xyzori025_xyzori025csbskel', permute=(0, 1, 2),
                         crop=None, trd=2000)#)[1890-1792, 1890, 1024, 2048, 0, 1024])
x = test_set.__getitem__(0)['img']

x = [y.permute(1, 0, 2, 3) for y in x]


# 20 with unet 32 is quite good
#net = torch.load('/media/ExtHDD01/logs/Fly0B/csb/1/checkpoints/net_gYX_model_epoch_20.pth').cuda()
#net = torch.load('/media/ExtHDD01/logs/Fly0B/csb/unet64ori/checkpoints/net_gYX_model_epoch_120.pth').cuda()
#net = torch.load('/media/ExtHDD01/logs/Fly0B/csb/dsmcori/checkpoints/net_gYX_model_epoch_40.pth').cuda()
net = torch.load('/media/ExtHDD01/logs/Fly0B/csb/025/dsmcOS/checkpoints/net_gYX_model_epoch_80.pth').cuda()
net.train()

#input = x[1][1000:1001, :, -512:-512+256, -512:-512+256].cuda()
#input = x[1][1200:1201, :, 512:256+512, 256:256+256].cuda()
input = x[1][1200:1201, :, :, :].cuda()

#input = (input >= -0.9) / 1
#input = (2 * input) - 1

out0 = net(input)[0]

#.detach().cpu()

imagesc(input[0, 0, ::].detach().cpu())
imagesc(out0[0, 0, ::].detach().cpu())


