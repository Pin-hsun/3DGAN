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

        #self.magic256 = torch.load('/media/ExtHDD01/checkpoints/FlyZ_WpOp/netG_model_epoch_170.pth').cuda()
        #self.magic286 = torch.load('/media/ExtHDD01/checkpoints/FlyZ_WpOp/netG_model_epoch_170.pth').cuda()
        #self.magic286 = torch.load('/media/ExtHDD01/checkpointsold/FlyZ_WpOp286Mask/netG_model_epoch_10.pth').cuda()

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
        self.net_g = net

    def get_one_output(self, i, alpha=None):
        # inputs
        x, name = self.test_set.__getitem__(i)
        oriX = x[0].unsqueeze(0).to(self.device)
        oriY = x[1].unsqueeze(0).to(self.device)
        in_img = oriX
        out_img = oriY

        alpha = alpha / 100

        try:
            output, output1 = self.net_g(in_img, alpha * torch.ones(1, 2).cuda())
        except:
            try: ## descargan
                output = self.net_g(in_img, alpha * torch.ones(1, 2).cuda())[0]
            except:
                try: ## attgan
                    self.net_g.f_size = args.cropsize // 32
                    output = self.net_g(in_img, alpha * torch.ones(1, 1).cuda())[0]
                except:
                    output = self.net_g(in_img)[0]

        if self.args.gray:
            in_img = in_img.repeat(1, 3, 1, 1)
            out_img = out_img.repeat(1, 3, 1, 1)
            output = output.repeat(1, 3, 1, 1)

        if args.cmb is not False:
            output = combine(output, in_img, args.cmb)

        in_img = in_img.detach().cpu()
        out_img = out_img.detach().cpu()
        output = output.detach().cpu()
        return in_img, out_img, output, name

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
    show = torch.cat(show, 1).numpy().astype(np.float16)
    imagesc(show, show=False, save=save_name)


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

with open('outputs/' + parser.parse_args().jsn + '.json', 'rt') as f:
    t_args = argparse.Namespace()
    t_args.__dict__.update(json.load(f))
    args = parser.parse_args(namespace=t_args)

# environment file
if args.env is not None:
    load_dotenv('.' + args.env)
else:
    load_dotenv('.env')

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

        for alpha in np.linspace(*args.nalpha)[:]:
            outputs = list(map(lambda v: test_unit.get_one_output(v, alpha), args.irange))
            [imgX, imgY, imgXY, names] = list(zip(*outputs))

            imgX = torch.cat(imgX, 0)
            imgY = torch.cat(imgY, 0)
            imgXY = torch.cat(imgXY, 0)

            imgXseg = test_unit.get_segmentation(imgX)
            imgYseg = test_unit.get_segmentation(imgY)
            imgXYseg = test_unit.get_segmentation(imgXY)

            diff_XY = imgX - imgXY

            tag = True
            diff = seperate_by_seg(x0=diff_XY, seg=imgXYseg, masked=[], absolute=tag, threshold=0, rgb=tag)
            diffseg0 = seperate_by_seg(x0=diff_XY, seg=imgXYseg, masked=[0, 2, 4], absolute=tag, threshold=0, rgb=tag)
            diffseg1 = seperate_by_seg(x0=diff_XY, seg=imgXYseg, masked=[1, 3], absolute=tag, threshold=0, rgb=tag)

            to_show = [imgX,
                       imgXY,
                       #diff,
                       diffseg0,
                       diffseg1
                       ]

            to_print(to_show, save_name=os.path.join("outputs/results/", args.dataset, args.prj,
                                                     str(epoch) + '_' + str(alpha) + '_' + str(ii).zfill(4) + '.jpg'))

            if 0:#args.all:
                result = diffseg0
                destination = 'outputs/results/seg0b/'
                os.makedirs(destination, exist_ok=True)
                if tag:
                    to_save = result[0, ::].permute(1, 2, 0).numpy().astype(np.float16)
                else:
                    to_save = result[0, 0, ::].numpy().astype(np.float32)
                tiff.imsave(os.path.join(destination,  names[0][0].split('/')[-1]), to_save)

                result = diffseg1
                destination = 'outputs/results/seg1b/'
                os.makedirs(destination, exist_ok=True)
                if tag:
                    to_save = result[0, ::].permute(1, 2, 0).numpy().astype(np.float16)
                else:
                    to_save = result[0, 0, ::].numpy().astype(np.float32)
                tiff.imsave(os.path.join(destination,  names[0][0].split('/')[-1]), to_save)




# USAGE
# CUDA_VISIBLE_DEVICES=0 python test.py --jsn default --dataset pain --nalpha 0 100 2  --prj VryAtt
# CUDA_VISIBLE_DEVICES=0 python test.py --jsn default --dataset TSE_DESS --nalpha 0 100 1  --prj VryCycleUP --net netGXY
# python testoai.py --jsn womac3 --direction a_b --prj N01/DescarMul/ --cropsize 384 --n01 --cmb mul
# python testrefactor.py --jsn womac3 --direction b_a --prj N01/DescarMul/ --cropsize 384 --n01 --cmb mul --nepoch 20


