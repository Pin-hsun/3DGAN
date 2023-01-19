from __future__ import print_function
import torch
import matplotlib.pyplot as plt
from dataloader.data_multi import PairedDataTif
from utils.data_utils import imagesc

cm = plt.get_cmap('viridis')

test_set = PairedDataTif(root='/media/ExtHDD01/Dataset/paired_images/Fly0B',
                         directions='xysb_xyd100', permute=(0, 1, 2),
                         crop=None, trd=0)#)[1890-1792, 1890, 1024, 2048, 0, 1024])
x = test_set.__getitem__(0)['img']

x = [y.permute(1, 0, 2, 3) for y in x]

net = torch.load('/media/ExtHDD01/logs/Fly0B/csb/sbd100/checkpoints/net_gYX_model_epoch_80.pth').cuda()
net.train()

input = x[1][200:201, :, -512:, :512].cuda()
out0, = net(input)

#.detach().cpu()

imagesc(input[0, 0, ::].detach().cpu())
imagesc(out0[0, 0, ::].detach().cpu())


