from __future__ import print_function
import argparse, json
import os,glob
from utils.data_utils import imagesc
import torch
from torchvision.utils import make_grid
import torchvision

import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from utils.data_utils import norm_01
from skimage import io
import torchvision.transforms as transforms
import torch.nn as nn
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
            self.test_set = PairedDataTif(root='/media/ExtHDD01/Dataset/paired_images/Fly3D/',
                                          directions='xyzweak_xyzsb', permute=None,
                                          crop=None)

        self.device = torch.device("cuda:0")

    def get_model(self,  epoch, eval=False):
        model_path = os.path.join(self.dir_checkpoints, self.args.dataset.split('/')[0], self.args.prj, 'checkpoints') + \
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
parser.add_argument('--jsn', type=str, default='wnwp3d', help='name of ini file')
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

with open('env/jsn/' + parser.parse_args().jsn + '.json', 'rt') as f:
    t_args = argparse.Namespace()
    t_args.__dict__.update(json.load(f)['test'])
    args = parser.parse_args(namespace=t_args)

args.dataset = 'Fly0B'
args.direction = 'zyweak1024_zyori1024'
args.cropsize = 1024

# environment file
if args.env is not None:
    load_dotenv('env/.' + args.env)
else:
    load_dotenv('env/.t09')

test_unit = Pix2PixModel(args=args)


def make_rotation_3d():
    if 0:
        net = torch.load('/media/ExtHDD01/logs/Fly3D/wnwp3d/cyc4/GdenuWO/checkpoints/netGXY_model_epoch_160.pth').cuda()
        x = test_unit.test_set.__getitem__(0)['img']
        print(len(test_unit.test_set))
    else:
        net = torch.load('/media/ExtHDD01/logs/Fly0B/wnwp3d/cyc4/GdenuWBmc/checkpoints/netGXY_model_epoch_100.pth').cuda()
        #net = torch.load('submodels/1.pth').cuda()  #  mysterious Resnet model with ResnetAdaILNBlock (ugatit?)
        from dataloader.data_multi import PairedDataTif
        test_set = PairedDataTif(root='/media/ExtHDD01/Dataset/paired_images/Fly0B/',
                                           directions='xyzweak_xyzsb', permute=(0, 1, 2),
                                           crop=[1890-1792, 1890, 1024, 2048, 0, 1024])
        x = test_set.__getitem__(0)['img']

    w = x[0][0, ::]  #(z, x, y)
    b = x[1][0, ::]  #(z, x, y)
    del x

    #w = w.permute(0, 2, 1)  #(z, x, y)
    #b = b.permute(0, 2, 1)

    #w = ((w > 0) / 1) * 2 - 1

    for angle in list(range(0, 360, 10)):
        print(angle)
        wp = transforms.functional.rotate(w.unsqueeze(1), angle=angle,  #(z, 1, x, y)
                                          interpolation=torchvision.transforms.functional.InterpolationMode.BILINEAR, fill=-1)
        bp = transforms.functional.rotate(b.unsqueeze(1), angle=angle,
                                          interpolation=torchvision.transforms.functional.InterpolationMode.BILINEAR, fill=-1)

        wall = []
        ball = []
        for i in range(wp.shape[2]):
            print(i)
            w_slice = wp[:, 0, i, :].unsqueeze(0).unsqueeze(0).cuda()
            b_slice = bp[:, 0, i, :].unsqueeze(0).unsqueeze(0).cuda()

            wout, bout = net(torch.cat((w_slice, b_slice), 1))#, a=None)
            wall.append(wout.detach().cpu())
            ball.append(bout.detach().cpu())

        del wp
        del bp

        wall = torch.cat(wall, 0)  #(x, 1, z, y)
        wall = wall.permute(2, 1, 0, 3)
        ball = torch.cat(ball, 0)  #(x, 1, z, y)
        ball = ball.permute(2, 1, 0, 3)

        wall = transforms.functional.rotate(wall, angle=-angle,  #(z, 1, x, y)
                                           interpolation=torchvision.transforms.functional.InterpolationMode.BILINEAR, fill=-1)
        ball = transforms.functional.rotate(ball, angle=-angle,  #(z, 1, x, y)
                                           interpolation=torchvision.transforms.functional.InterpolationMode.BILINEAR, fill=-1)

        mask = (w > -1).unsqueeze(1)

        #wall[mask == 0] = -1
        #ball[mask == 0] = -1

        wall = wall.numpy()[:, 0, ::]
        ball = ball.numpy()[:, 0, ::]

        wall = wall[:, 256:-256, 256:-256]
        ball = ball[:, 256:-256, 256:-256]

        tiff.imsave('/media/ghc/GHc_data2/allmbm/' + str(angle).zfill(3) + '.tif', wall)
        tiff.imsave('/media/ghc/GHc_data2/allmbb/' + str(angle).zfill(3) + '.tif', ball)


def sum_all():
    files = sorted(glob.glob('/media/ghc/GHc_data2/allbbb/*'))

    if 1: # calculate average
        all = tiff.imread(files[0])
        for f in files[1:]:
            print(f)
            all = all + tiff.imread(f)
        m = all / len(files)
    else:
        mean = tiff.imread('/media/ghc/GHc_data2/wg.tif')
        var = np.square(tiff.imread(files[0]) - mean)
        for f in files[1:]:
            print(f)
            var = var + np.square(tiff.imread(f) - mean)
        var = var / len(files)


if 0:
    wp = w.unsqueeze(1)
    wp = wp[:, :, 256:-256, 256:-256]
    bp = b.unsqueeze(1)
    bp = bp[:, :, 256:-256, 256:-256]
    ball = []
    for i in range(bp.shape[2]):
        print(i)
        b_slice = bp[:, 0, i, :].unsqueeze(0).unsqueeze(0).cuda()
        #w_slice = wp[:, 0, i, :].unsqueeze(0).unsqueeze(0).cuda()

        bout = net(b_slice)[0]
        #wout, bout = net(torch.cat((w_slice, b_slice), 1), a=None)
        ball.append(bout.detach().cpu())

    ball = torch.cat(ball, 0)  # (x, 1, z, y)
    ball = ball.permute(2, 1, 0, 3)
    ball = ball.numpy()[:, 0, ::]
    #ball = ball[:, 256:-256, 256:]
    tiff.imsave('gut.tif', ball)

# USAGE
# CUDA_VISIBLE_DEVICES=0 python test.py --jsn default --dataset pain --nalpha 0 100 2  --prj VryAtt
# CUDA_VISIBLE_DEVICES=0 python test.py --jsn default --dataset TSE_DESS --nalpha 0 100 1  --prj VryCycleUP --net netGXY
# python testoai.py --jsn womac3 --direction a_b --prj N01/DescarMul/ --cropsize 384 --n01 --cmb mul
# python testrefactor.py --jsn womac3 --direction b_a --prj N01/DescarMul/ --cropsize 384 --n01 --cmb mul --nepoch 20

# CUDA_VISIBLE_DEVICES=1 python test_fly3D.py --env a6k --jsn FlyZWpWn --direction zyweak512_zyorisb512 --prj wnwp3d/cyc3/GdenuWSmcYL10 --engine cyclegan23dwo --nalpha 0 20 20 --nepochs 0 201 20

# CUDA_VISIBLE_DEVICES=1 python test_fly3D.py --jsn FlyZWpWn --direction zyweak512_zyori512 --prj wnwp3d/cyc2l1/0 --nepochs 60 --engine cyclegan23d --cropsize 512

# single
# CUDA_VISIBLE_DEVICES=1 python test_fly3D.py --jsn FlyZWpWn --direction zyweak5_zysb5 --prj wnwp3d/cyc4z/GdeWOmc --nepochs 140 --engine cyc4 --cropsize 512 --nalpha 0 20 20 -b 16