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


class Pix2PixModel:
    def __init__(self, args):
        self.args = args
        self.net_g = None
        self.dir_checkpoints = os.environ.get('LOGS')
        self.test_set = Dataset(root=os.environ.get('DATASET') + args.testset,
                                path=args.direction,
                                opt=args, mode='test', filenames=True)

        #self.seg_model = torch.load(os.environ.get('model_seg')).cuda()
        #self.seg_cartilage = torch.load('submodels/oai_cartilage_384.pth')#model_seg_ZIB.pth')
        self.seg_cartilage = torch.load('submodels/model_seg_ZIB_res18_256.pth')
        self.seg_bone = torch.load('submodels/model_seg_ZIB.pth')
        #self.cartilage = torch.load('submodels/femur_tibia_fc_tc.pth').cuda()
        self.netg_t2d = torch.load('submodels/tse_dess_unet32.pth')

        self.netg_t2d.eval()
        self.seg_cartilage.eval()
        self.seg_bone.eval()

    def get_model(self,  epoch, eval=False):
        model_path = os.path.join(self.dir_checkpoints, self.args.dataset, self.args.prj, 'checkpoints') + \
               ('/' + self.args.netg + '_model_epoch_{}.pth').format(epoch)
        print(model_path)
        net = torch.load(model_path).cuda()
        if eval:
            net.eval()
            print('eval mode')
        else:
            net.train()
            print('train mode')
        self.net_g = net

    def get_one_output(self, i, alpha=None):
        # inputs
        x, name = self.test_set.__getitem__(i)

        if args.bysubject:
            x = [y.permute(3, 0, 1, 2) for y in x]
        else:
            x = [y.unsqueeze(0) for y in x]

        oriX = x[0].cuda()
        oriY = x[1].cuda()

        alpha = alpha / 100

        output, output1 = self.net_g(oriX, alpha * torch.ones(1, 2).cuda())

        if self.args.gray:
            oriX = oriX.repeat(1, 3, 1, 1)
            oriY = oriY.repeat(1, 3, 1, 1)
            output = output.repeat(1, 3, 1, 1)

        if args.cmb != "None":
            output = combine(output, oriX, args.cmb)
            output1 = combine(output1, oriX, args.cmb)

        oriX = oriX.detach().cpu()
        oriY = oriY.detach().cpu()
        output = output.detach().cpu()
        output1 = output1.detach().cpu()

        return oriX, oriY, output, output1, name

    def get_segmentation(self, x0):
        # normalize
        x = 1 * x0
        if self.args.n01:
            x = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x)
        else:
            x = x0
        if self.args.t2d:
            x = self.netg_t2d(x.cuda())[0]

        seg = self.seg_cartilage(x)
        seg = torch.argmax(seg, 1).detach().cpu()

        return seg


def seperate_by_seg(x0, seg, masked, absolute, threshold, rgb):
    x = 1 * x0
    for c in masked:
        x[(seg == c).unsqueeze(1).repeat(1, 3, 1, 1)] = 0
    if absolute:
        x[x < 0] = 0
    if threshold > 0:
        x[x > threshold] = threshold
    if rgb:
        x = x.numpy()
        out = []
        for i in range(x.shape[0]):
            # normalize by subject
            xi = x[i, 0, ::]
            xi[0, 0] = 0.2
            xi = xi / xi.max()
            #xi = xi - xi.min()
            #xi = xi / xi.max()
            out.append(np.transpose(cm(xi)[:, :, :3], (2, 0, 1)))
        out = np.concatenate([np.expand_dims(x, 0) for x in out], 0)
        out = torch.from_numpy(out)
    else:
        out = x
    return out


def to_print(to_show, save_name):
    os.makedirs(os.path.join("outputs/results", args.dataset, args.prj), exist_ok=True)

    show = []
    for x in to_show:
        x = x - x.min()
        x = x / x.max()
        x = torch.permute(x, (1, 2, 0, 3))
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        show.append(x)
    show = torch.cat(show, 2).numpy()#.astype(np.float16)
    #imagesc(show, show=False, save=save_name)
    tiff.imsave(save_name.replace('.jpg', '.tif'), show[0, :, :])


# Command Line Argument
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--env', type=str, default=None, help='environment_to_use')
parser.add_argument('--jsn', type=str, default='womac3', help='name of ini file')
parser.add_argument('--dataset', help='name of training dataset')
parser.add_argument('--testset', help='name of testing dataset if different than the training dataset')
parser.add_argument('--prj', type=str, help='N01/DescarMul')
parser.add_argument('--direction', type=str, help='a_b')
parser.add_argument('--netg', type=str)
parser.add_argument('--resize', type=int)
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

def to_heatmap(x):
    x[0, 0] = 10
    x = x / x.max()
    return np.transpose(cm(x)[:, :, :3], (2, 0, 1))


def get_iirange():
    #iirange = [14, 24, 190, 270, 283, 459, 503, 149, 244, 338, 437, 442, 618, 628, 658]
    #iirange = [29, 39, 119, 139, 212]
    #iirange = [222, 236, 250, 279, 336]
    iirange = [39, 57, 76, 112, 118, 124, 139, 162, 212, 234, 247, 390, 482, 496, 542, 568, 591, 598, 607, 623, 695, 718, 779, 787, 792]

    import pandas as pd
    df = pd.read_csv('/home/ghc/Dropbox/TheSource/scripts/OAI_pipelines/meta/subjects_unipain_womac3.csv')
    train_index = [x for x in range(df.shape[0]) if not df['has_moaks'][x]]
    eval_index = [x for x in range(df.shape[0]) if df['has_moaks'][x]]

    ti = [list(range(x*23, x*23+23)) for x in eval_index[:]]
    iirange = []
    for l in ti:
        iirange = iirange + l

    #iirange = [iirange[x] for x in [8, 23, 25, 141, 163, 176, 186, 187, 206, 212, 279, 280]]
    return iirange


for epoch in range(*args.nepochs):
    test_unit.get_model(epoch, eval=args.eval)

    if args.all:
        iirange = range(len(test_unit.test_set))[:]
    else:
        iirange = range(1)

    iirange = range(len(test_unit.test_set))

    for ii in iirange[:1000]:
        if args.all:
            args.irange = [ii]

        diffseg0_all = []
        diffseg1_all = []
        for alpha in np.linspace(*args.nalpha)[:]:
            outputs = list(map(lambda v: test_unit.get_one_output(v, alpha), args.irange))
            [imgX, imgY, imgXY, imgXY1, names] = list(zip(*outputs))

            imgX = torch.cat(imgX, 0)
            imgY = torch.cat(imgY, 0)
            imgXY = torch.cat(imgXY, 0)
            imgXY1 = torch.cat(imgXY1, 0)

            imgXseg = test_unit.get_segmentation(imgX)
            imgYseg = test_unit.get_segmentation(imgY)
            imgXYseg = test_unit.get_segmentation(imgXY)
            imgXY1seg = test_unit.get_segmentation(imgXY1)

            diff_XY = imgX - imgXY
            diff_XY1 = imgX - imgXY1

            tag = False
            diffseg0 = seperate_by_seg(x0=diff_XY, seg=imgXYseg, masked=[0, 2, 4], absolute=tag, threshold=0, rgb=tag)
            diffseg1 = seperate_by_seg(x0=diff_XY, seg=imgXYseg, masked=[1, 3], absolute=tag, threshold=0, rgb=tag)

            diff1seg0 = seperate_by_seg(x0=diff_XY1, seg=imgXY1seg, masked=[0, 2, 4], absolute=tag, threshold=0, rgb=tag)
            diff1seg1 = seperate_by_seg(x0=diff_XY1, seg=imgXY1seg, masked=[1, 3], absolute=tag, threshold=0, rgb=tag)

            #seg1 = ((imgXYseg == 1) / 1 + (imgXYseg == 3) / 1).unsqueeze(1)

            # monte carlo
            diffseg0_all.append(diffseg0.unsqueeze(4))
            diffseg1_all.append(diffseg1.unsqueeze(4))

        # mponte carlo end
        diffseg0_all = torch.cat(diffseg0_all, 4)
        diffseg1_all = torch.cat(diffseg1_all, 4)

        diffseg0_mean = diffseg0_all.mean(4)[:, 0, :, :]
        diffseg0_std = diffseg0_all.std(4)[:, 0, :, :]
        diffseg0_unc = torch.div(diffseg0_mean, diffseg0_std+0.0001)

        diffseg1_mean = diffseg1_all.mean(4)[:, 0, :, :]
        diffseg1_std = diffseg1_all.std(4)[:, 0, :, :]
        diffseg1_unc = torch.div(diffseg1_mean, diffseg1_std+0.0001)

        diffseg0_unc[diffseg0_mean == 0] = 0
        diffseg1_unc[diffseg1_mean == 0] = 0

        d0 = diffseg0_mean
        d1 = diffseg1_mean

        u0 = diffseg0_unc
        u1 = diffseg1_unc

        if 1:
            m = 20
            u0[:, 0, 0] = m
            u1[:, 0, 0] = m
            u0[u0 >= m] = m
            u1[u1 >= m] = m

        if 1:
            m = 1
            d0[:, 0, 0] = m
            d1[:, 0, 0] = m
            d0[d0 >= m] = m
            d1[d1 >= m] = m

        imgXY[imgXY < 0] = 0

        to_show = [imgX,
                   imgXY,
                   torch.cat([d0.unsqueeze(0).unsqueeze(0)] * 3, 1),
                   torch.cat([d1.unsqueeze(0).unsqueeze(0)] * 3, 1),
                   torch.cat([u0.unsqueeze(0).unsqueeze(0)] * 3, 1),
                   torch.cat([u1.unsqueeze(0).unsqueeze(0)] * 3, 1),
                   # diff1seg0,
                   # diff1seg1
                   ]

        def get_significance(x0, y0, xt, yt):
            x = 1 * x0
            x[x >= xt] = xt
            y = 1 * y0
            y = (y >= yt) / 1
            z = np.multiply(x, y)
            z = z / z.max()
            # z = cm(z)
            return z

        to_print = get_significance(x0=d0.numpy(), y0=u0.numpy(), xt=0.2, yt=0.2).astype(np.float32)
        destination = '/media/ExtHDD01/Dataset/paired_images/womac3/full/abml2/'
        os.makedirs(destination, exist_ok=True)
        print(to_print.shape)
        if args.bysubject:
            for b in range(to_print.shape[0]):
                tiff.imsave(destination + names[0][b].split('/')[-1], to_print[b, ::])
        else:
            tiff.imsave(destination + names[0][0].split('/')[-1], to_print)

        to_print = get_significance(x0=d1.numpy(), y0=u1.numpy(), xt=0.9, yt=0.5).astype(np.float32)
        destination = '/media/ExtHDD01/Dataset/paired_images/womac3/full/aeff2/'
        os.makedirs(destination, exist_ok=True)
        print(to_print.shape)
        if args.bysubject:
            for b in range(to_print.shape[0]):
                tiff.imsave(destination + names[0][b].split('/')[-1], to_print[b, ::])
        else:
            tiff.imsave(destination + names[0][0].split('/')[-1], to_print)


# CUDAVISIBLE_DEVICES=1 python test2seg.py --jsn womac3 --direction a_b --prj mcfix/descar2/Gdescarsmc_index2 --cropsize 384 --n01 --nalpha 0 20 21 --all --nepochs 200 --gray --testset womac3/full/

# USAGE
# CUDA_VISIBLE_DEVICES=0 python test.py --jsn default --dataset pain --nalpha 0 100 2  --prj VryAtt
# CUDA_VISIBLE_DEVICES=0 python test.py --jsn default --dataset TSE_DESS --nalpha 0 100 1  --prj VryCycleUP --net netGXY
# python testoai.py --jsn womac3 --direction a_b --prj N01/DescarMul/ --cropsize 384 --n01 --cmb mul
# python testrefactor.py --jsn womac3 --direction b_a --prj N01/DescarMul/ --cropsize 384 --n01 --cmb mul --nepoch 20


# CUDA_VISIBLE_DEVICES=1 python test2.py --jsn womac3 --direction a_b --prj bysubjectright/mc/descar2/Gunet128 --cropsize 384 --n01 --nalpha 0 100 101 --all --nepochs 80

# CUDA_VISIBLE_DEVICES=1 python test2seg.py --jsn womac3 --direction a_b --prj mcfix/descar2/Gdescarsmc_index2 --cropsize 384 --n01 --nalpha 0 100 101 --all --nepochs 100 --gray --testset womac3/full/