from __future__ import print_function
import torch
import matplotlib.pyplot as plt
from dataloader.data_multi import PairedDataTif
from utils.data_utils import imagesc

cm = plt.get_cmap('viridis')

test_set = PairedDataTif(root='/home/ubuntu/Data/Dataset/paired_images/40x/',
                         directions='original40xdown2', permute=(0, 1, 2),
                         crop=[0, 512, 0, 896, 0, 1024], trd=500)#)[1890-1792, 1890, 1024, 2048, 0, 1024])
x = test_set.__getitem__(0)['img']

x = [y.permute(1, 0, 2, 3) for y in x]

net = torch.load('/home/ubuntu/Data/logs/40x/cyc/test1/checkpoints/netGXY_model_epoch_40.pth').cuda()

input = x[0][400:401, :, ::]
out = net(input.cuda())[0].detach().cpu()

imagesc(input[0, 0,:256,:256])
imagesc(out[0,0,:256, :256])