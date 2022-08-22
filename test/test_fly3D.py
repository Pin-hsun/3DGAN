from __future__ import print_function
import argparse, json
import os
from utils.data_utils import imagesc
import torch
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from utils.data_utils import norm_01
from skimage import io
import torchvision.transforms as transforms
import torch.nn as nn
from engine.base import combine
import tifffile as tiff
from dataloader.data_multi import MultiData as Dataset

import numpy as np
import matplotlib.pyplot as plt
cm = plt.get_cmap('viridis')


def get_sobel():
    SOBEL = nn.Conv2d(1, 2, 1, padding=1, padding_mode='replicate', bias=False)
    SOBEL.weight.requires_grad = False
    ww = torch.Tensor([[
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]],
        [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]
         ]]).reshape(2, 1, 3, 3)
    SOBEL.weight.set_(ww).cuda()
    return SOBEL


class Pix2PixModel:
    def __init__(self, args):
        self.args = args
        self.net_g = None
        self.dir_checkpoints = os.environ.get('LOGS')
        if not self.args.tif:
            self.test_set = Dataset(root=os.environ.get('DATASET') + args.testset,
                                    path=args.direction,
                                    opt=args, mode='test', filenames=True)
        else:
            from dataloader.data_multi import PairedDataTif
            self.test_set = PairedDataTif(root='/media/ExtHDD01/Dataset/paired_images/Fly0B/',
                                          directions='xyzweak_xyzsb', permute=(0, 2, 1),
                                          crop=[0, 1890, 1024+512, 1024+512+32, 0, 1024])
        print(len(self.test_set))

        self.device = torch.device("cuda:0")

    def get_model(self,  epoch, eval=False):
        model_path = os.path.join(self.dir_checkpoints, self.args.dataset, self.args.prj, 'checkpoints') + \
               ('/' + self.args.netg + '_model_epoch_{}.pth').format(epoch)
        print(model_path)
        net = torch.load(model_path).to(self.device)
        if eval:
            net.eval()
        else:
            net.train()
        net = net.train()
        self.net_g = net

    def get_one_output(self, i, alpha=None):
        alpha = alpha / 100
        # inputs
        x = self.test_set.__getitem__(i)
        x = x['img']
        print(x[0].shape)

        for b in range(len(x)):
            x[b] = x[b].to(self.device)

            if len(x[b].shape) == 3:
                x[b] = x[b].unsqueeze(0)
            else:
                x[b] = x[b].permute(3, 0, 1, 2)
            print(x[b].shape)

        # test method
        engine = args.engine
        test_method = getattr(__import__('engine.' + engine), engine).GAN.test_method

        output = []
        batch = self.args.batch_size
        for s in range(0, x[0].shape[0], batch):
            output.append(test_method(self, self.net_g, [y[s : s+batch, :, :, :] for y in x]).detach().cpu())
        output = torch.cat(output, 0)

        x[0] = x[0].detach().cpu()
        x[1] = x[1].detach().cpu()
        output = output.detach().cpu()

        if self.args.gray:
            x[0] = x[0].repeat(1, 3, 1, 1)
            x[1] = x[1].repeat(1, 3, 1, 1)
            output = output.repeat(1, 3, 1, 1)

        return x[0], x[1], output



# Command Line Argument
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--env', type=str, default=None, help='environment_to_use')
parser.add_argument('--jsn', type=str, default='womac3', help='name of ini file')
parser.add_argument('--engine', dest='engine', type=str, help='use which engine')
parser.add_argument('--dataset', help='name of training dataset')
parser.add_argument('--testset', help='name of testing dataset if different than the training dataset')
parser.add_argument('--prj', type=str, help='N01/DescarMul')
parser.add_argument('--direction', type=str, help='a_b')
parser.add_argument('--netg', type=str)
parser.add_argument('--resize', type=int)
parser.add_argument('-b', dest='batch_size', default=1, type=int)
parser.add_argument('--cropsize', type=int)
parser.add_argument('--bysubject', action='store_true', dest='bysubject', default=False)
parser.add_argument('--t2d', action='store_true', dest='t2d', default=False)
parser.add_argument('--cmb', type=str, help='way to combine output to the input')
parser.add_argument('--n01', action='store_true', dest='n01')
parser.add_argument('--gray', action='store_true', dest='gray', default=False, help='dont copy img to 3 channel')
parser.add_argument('--flip', action='store_true', dest='flip')
parser.add_argument('--eval', action='store_true', dest='eval')
parser.add_argument('--all', action='store_true', dest='all', default=False)
parser.add_argument('--nepochs', default=(20, 30, 10), nargs='+', help='which checkpoints to be interfered with', type=int)
parser.add_argument('--nalpha', default=(0, 100, 1), nargs='+', help='range of additional input parameter for generator', type=int)
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--port', type=str, default='dummy')
parser.add_argument('--mc', action='store_true', dest='mc')
parser.add_argument('--sfx', dest='suffix', type=str, default='')
parser.add_argument('--tif', action='store_true', dest='tif')

with open('outputs/jsn/' + parser.parse_args().jsn + '.json', 'rt') as f:
    t_args = argparse.Namespace()
    t_args.__dict__.update(json.load(f))
    args = parser.parse_args(namespace=t_args)

# environment file
if args.env is not None:
    load_dotenv('env/.' + args.env)
else:
    load_dotenv('env/.t09')

if len(args.nepochs) == 1:
    args.nepochs = [args.nepochs[0], args.nepochs[0]+1, 1]
if len(args.nalpha) == 1:
    args.nalpha = [args.nalpha[0], args.nalpha[0]+1, 1]

test_unit = Pix2PixModel(args=args)
print(len(test_unit.test_set))

for epoch in range(*args.nepochs):
    test_unit.get_model(epoch, eval=args.eval)

    if args.all:
        iirange = range(len(test_unit.test_set))[:]
    else:
        iirange = range(1)

    for ii in iirange:
        if args.all:
            args.irange = [ii]
        seg0_all = []
        seg1_all = []
        out_all = []
        for alpha in np.linspace(*args.nalpha)[:]:
            x0, x1, output = test_unit.get_one_output(args.irange[0], alpha)
            out_all.append(output)

        output = torch.cat([x.unsqueeze(4) for x in out_all], 4)
        output_mean = output.mean(4)
        output_var = output.var(4)

        if not args.tif:
            os.makedirs('outputs/' + args.dataset + '/' + args.prj + '/', exist_ok=True)
            tiff.imsave('outputs/' + args.dataset + '/' + args.prj + '/' + str(epoch) + '_' + str(args.nalpha[2]) + args.suffix + '.tif', output_mean[:, 0, :, :].detach().cpu().numpy())
            if args.nalpha[2] > 1:
                tiff.imsave('outputs/' + args.dataset + '/' + args.prj + '/' + str(epoch) + '_' + str(args.nalpha[2]) + args.suffix + 'v.tif', output_var[:, 0, :, :].detach().cpu().numpy())
        else:
            tiff.imsave('/home/ghc/Desktop/temp0/' + args.suffix + '.tif', output_mean[:, 0, ::].detach().cpu().numpy())
            if args.nalpha[2] > 1:
                tiff.imsave('/home/ghc/Desktop/temp0/' + args.suffix + 'v.tif', output_var[:, 0, ::].detach().cpu().numpy())

# USAGE
# CUDA_VISIBLE_DEVICES=0 python test.py --jsn default --dataset pain --nalpha 0 100 2  --prj VryAtt
# CUDA_VISIBLE_DEVICES=0 python test.py --jsn default --dataset TSE_DESS --nalpha 0 100 1  --prj VryCycleUP --net netGXY
# python testoai.py --jsn womac3 --direction a_b --prj N01/DescarMul/ --cropsize 384 --n01 --cmb mul
# python testrefactor.py --jsn womac3 --direction b_a --prj N01/DescarMul/ --cropsize 384 --n01 --cmb mul --nepoch 20

# CUDA_VISIBLE_DEVICES=1 python test_fly3D.py --env a6k --jsn FlyZWpWn --direction zyweak512_zyorisb512 --prj wnwp3d/cyc3/GdenuWSmcYL10 --engine cyclegan23dwo --nalpha 0 20 20 --nepochs 0 201 20

# CUDA_VISIBLE_DEVICES=1 python test_fly3D.py --jsn FlyZWpWn --direction zyweak512_zyori512 --prj wnwp3d/cyc2l1/0 --nepochs 60 --engine cyclegan23d --cropsize 512

# single
# CUDA_VISIBLE_DEVICES=1 python test_fly3D.py --jsn FlyZWpWn --direction zyweak5_zysb5 --prj wnwp3d/cyc4z/GdeWOmc --nepochs 140 --engine cyc4 --cropsize 512 --nalpha 0 20 20 -b 16