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
from models.base import combine
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

cm = plt.get_cmap('viridis')


def to_heatmap(x):
    return cm(x)


def overlap_red(x0, y0):
    y = 1 * y0
    x = 1 * x0
    x = x - x.min()
    x = x / x.max()

    c = 0
    x[1, y == c] = 0.1 * x[1, y == c]
    x[2, y == c] = 0.1 * x[2, y == c]

    c = 1
    x[0, y == c] = 0.1 * x[0, y == c]
    x[2, y == c] = 0.1 * x[2, y == c]

    c = 3
    x[0, y == c] = 0.1 * x[0, y == c]
    x[2, y == c] = 0.1 * x[2, y == c]

    c = 2
    x[0, y == c] = 0.1 * x[0, y == c]
    x[1, y == c] = 0.1 * x[1, y == c]

    c = 4
    x[0, y == c] = 0.1 * x[0, y == c]
    x[1, y == c] = 0.1 * x[1, y == c]
    return x


class Pix2PixModel:
    def __init__(self, args):
        self.args = args
        self.net_g = None
        self.dir_checkpoints = os.environ.get('LOGS')
        from dataloader.data_multi import MultiData as Dataset

        self.test_set = Dataset(root=os.environ.get('DATASET') + args.testset,
                                path=args.direction,
                                opt=args, mode='test', filenames=True)

        self.seg_cartilage = torch.load('submodels/model_seg_ZIB_res18_256.pth')
        self.seg_bone = torch.load('submodels/model_seg_ZIB.pth')
        self.netg_t2d = torch.load('submodels/tse_dess_unet32.pth')
        self.eff = torch.load('submodels/model_seg_eff.pth')

        self.netg_t2d.eval()
        self.seg_cartilage.eval()
        self.seg_bone.eval()

        self.device = torch.device("cuda:0")

    def get_model(self,  epoch, eval=False):
        model_path = os.path.join(self.dir_checkpoints, self.args.dataset, self.args.prj, 'checkpoints') + \
               ('/' + self.args.netg + '_model_epoch_{}.pth').format(epoch)
        print(model_path)
        net = torch.load(model_path, map_location='cpu').cuda()
        if eval:
            net.eval()
        else:
            net.train()
        self.net_g = net

    def get_one_output(self, i, xy='x', alpha=None):
        # inputs
        x = self.test_set.__getitem__(i)
        name = x['filenames']
        img = x['img']
        img = [x.unsqueeze(0).to(self.device) for x in img]
        in_img = img[0]
        out_img = img[1]

        ###
        self.net_g.train()

        # test_method
        engine = args.engine
        test_method = getattr(__import__('engine.' + engine), engine).GAN.test_method

        output = test_method(self, self.net_g, img)
        combined = combine(output, in_img, args.cmb)

        in_img = in_img.detach().cpu()[0, ::]
        out_img = out_img.detach().cpu()[0, ::]
        output = output[0, ::].detach().cpu()
        combined = combined[0,::].detach().cpu()

        return in_img, out_img, combined, output, name

    def get_t2d(self, ori):
        t2d = self.netg_t2d(ori.cuda().unsqueeze(0))[0][0, ::].detach().cpu()
        return t2d

    def get_seg(self, ori):
        bone = self.seg_bone(ori.cuda().unsqueeze(0))
        bone = torch.argmax(bone, 1)[0,::].detach().cpu()
        seg = self.seg_cartilage(ori.cuda().unsqueeze(0))
        seg = torch.argmax(seg, 1)[0,::].detach().cpu()
        return seg

    def get_eff(self, x0):
        x = 1 * x0
        x = torch.cat([x.unsqueeze(0)]*3, 1)
        x = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x)
        eff_prob = self.eff(x.cuda())
        eff_seg = torch.argmax(eff_prob, 1)[0, ::]
        return eff_seg.cpu().detach().numpy()

    def get_magic(self, ori):
        magic = self.magic286(ori.cuda().unsqueeze(0))
        magic = magic[0][0, 0, ::].detach().cpu()
        return magic

    def get_all_seg(self, input_ori):
        if self.args.n01 or self.args.gray:
            input =[]
            for i in range(len(input_ori)):
                temp = []
                for j in range(len(input_ori[0])):
                    o = input_ori[i][j]
                    if self.args.gray:
                        o = torch.cat([o]*3, 0)
                    if self.args.n01:
                        o = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(o)
                    temp.append(o)
                input.append(temp)
        else:
            input = input_ori
        print(self.args.t2d)
        if self.args.t2d:
            input = list(map(lambda k: list(map(lambda v: self.get_t2d(v), k)), input))  # (3, 256, 256) (-1, 1)
        list_seg = list(map(lambda k: list(map(lambda v: self.get_seg(v), k)), input))
        return list_seg


def to_print(to_show, save_name):
    os.makedirs(os.path.join("outputs/results", args.dataset, args.prj), exist_ok=True)
    to_show = [torch.cat(x, len(x[0].shape) - 1) for x in to_show]

    for i in range(len(to_show)):
        to_show[i] = to_show[i] - to_show[i].min()
        if to_show[i].max() > 0:
            to_show[i] = to_show[i] / to_show[i].max()

    for i in range(len(to_show)):
        if to_show[i].shape[0] == 1:
            to_show[i] = torch.cat([to_show[i]] * 3, 0)

    to_print = np.concatenate(to_show, 1).astype(np.float16)
    if args.all:
        tiff.imsave(save_name + '.tif', to_print)
    else:
        imagesc(to_print, show=False, save=save_name + '.png')


def to_rgb(a):
    a = np.transpose(cm(a[0, ::]), (2, 0, 1))[:3, ::]
    a = torch.from_numpy(a)
    return a


def seperate_by_seg(x, seg_used, masked, if_absolute):
    out = []
    for n in range(len(x)):
        a = 1 * x[n]
        if if_absolute:
            a[a < 0] = 0
        for c in masked:
            a[:, seg_used[n] == c] = 0
        if a.max() > 0:
            a = a / a.max()
        out.append(a)
    return out


# Command Line Argument
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--env', type=str, default=None, help='environment_to_use')
parser.add_argument('--jsn', type=str, default='womac3', help='name of ini file')
parser.add_argument('--dataset', help='name of training dataset')
parser.add_argument('--engine', dest='engine', type=str, help='use which engine')
parser.add_argument('--testset', help='name of testing dataset if different than the training dataset')
parser.add_argument('--bysubject', action='store_true', dest='bysubject')
parser.add_argument('--gray', action='store_true', dest='gray', default=False, help='dont copy img to 3 channel')
parser.add_argument('--prj', type=str, help='model')
parser.add_argument('--direction', type=str, help='a_b')
parser.add_argument('--netg', type=str)
parser.add_argument('--resize', type=int)
parser.add_argument('--cropsize', type=int)
parser.add_argument('--t2d', action='store_true', dest='t2d', default=False)
parser.add_argument('--cmb', type=str, default=None, help='way to combine output to the input')
parser.add_argument('--trd', type=float, dest='trd', help='threshold of images')
parser.add_argument('--n01', dest='n01', action='store_true')
parser.add_argument('--n11', dest='n01', action='store_false')
parser.set_defaults(n01=False)
parser.add_argument('--eval', action='store_true', dest='eval')
parser.add_argument('--nepochs', nargs='+', help='which checkpoints to be interfered with', type=int)
parser.add_argument('--nalpha', nargs='+', help='range of additional input parameter for generator', type=int)
parser.add_argument('--all', action='store_true', dest='all', default=False)
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--port', type=str, default='dummy')

with open('env/jsn/' + parser.parse_args().jsn + '.json', 'rt') as f:
    t_args = argparse.Namespace()
    t_args.__dict__.update(json.load(f)['test'])
    args = parser.parse_args(namespace=t_args)


# try cam
df = pd.read_csv('env/subjects_unipain_womac3.csv')
train_index = [x for x in range(df.shape[0]) if not df['has_moaks'][x]]
test_index = [x for x in range(df.shape[0]) if df['has_moaks'][x]]



args.nepochs = [160, 161, 1]
args.prj = '3D/descar3/GdsmcDboatch16'
args.cropsize = 384
args.n01 = True
args.cmb = 'not'
args.gray = True
args.nalpha = [0, 20, 20]
args.env = 'a6k'
args.engine = 'descar3'


netd = torch.load('/home/ubuntu/Data/logs/womac3/3D/descar3/GdsmcDboatch16/checkpoints/netD_model_epoch_160.pth').cuda()
classifier = torch.load('/home/ubuntu/Data/logs/womac3/3D/descar3/GdsmcDboatch16/checkpoints/classifier_model_epoch_160.pth').cuda()



def get_cam(x, y, netd, classifier):
    weight = classifier.weight[:, :, 0, 0]
    adv_x, cls_x = netd(torch.cat([x.unsqueeze(1)]*2, 1).cuda())
    adv_y, cls_y = netd(torch.cat([y.unsqueeze(1)]*2, 1).cuda())
    cam = torch.zeros((cls_x.shape[2], cls_x.shape[3])).cuda()
    for f in range(cls_x.shape[1]):
        cam = cam + weight[0, f] * (cls_x[0, f, :, :] - cls_y[0, f, :, :])
    cam = -1 * torch.nn.Upsample((384, 384), mode='bicubic', align_corners=True)(cam.unsqueeze(0).unsqueeze(0))[0, ::]
    return cam.detach().cpu()


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


for subject in test_index:
    args.irange = range(23*subject, 23*(subject+1))
    for epoch in range(*args.nepochs):
        test_unit.get_model(epoch, eval=args.eval)

        if args.all:
            iirange = range(len(test_unit.test_set))
        else:
            iirange = range(1)

        for ii in iirange:
            if args.all:
                args.irange = [ii]

            # MC
            results = dict()
            results_keys = ['diff', 'combined', 'output', 'camX', 'camXd']
            for k in results_keys:
                results[k] = []

            for alpha in np.linspace(*args.nalpha):
                out_xy = list(map(lambda v: test_unit.get_one_output(v, 'x', alpha), args.irange))  # N(subjects) list of O(outputs)

                [imgX, imgY, combined, output, names] = list(zip(*out_xy))  # O(outputs) list of N(subjects)

                diff = [(x[1] - x[0]) for x in list(zip(combined, imgX))]

                camX = [get_cam(x, y, netd, classifier) for x, y in zip(imgX, combined)]

                camXd = [torch.multiply((x)/1, y) for x, y in zip((diff), camX)]

                # MC
                for k in results_keys:
                    results[k].append([x.unsqueeze(0) for x in eval(k)])  # A(alphas) list of N(subjects)

            # MC
            for k in results_keys:
                results[k] = list(zip(*results[k]))  # N(subjects) list of A(alphas)
                results[k] = [torch.cat(x, 0) for x in results[k]]
                results[k] = [x.mean(0) for x in results[k]]

            # average seg
            a = test_unit.get_all_seg([results['combined']])[0]
            # Segmentation
            mask_bone = [0, 2, 4]
            mask_eff = [1, 3]

            tag = False
            diffseg0 = seperate_by_seg(x=results['diff'], seg_used=a, masked=mask_bone, if_absolute=True)
            diffseg1 = seperate_by_seg(x=results['diff'], seg_used=a, masked=mask_eff, if_absolute=True)

            # Print
            if args.all:
                destination = '/media/ExtHDD01/Dataset/paired_images/womac3/full/moaks/abml2/'
                os.makedirs(destination, exist_ok=True)
                tiff.imwrite(destination + names[0][0].split('/')[-1], diffseg0[0][0,::].numpy().astype(np.float32))

                destination = '/media/ExtHDD01/Dataset/paired_images/womac3/full/moaks/aeff2/'
                os.makedirs(destination, exist_ok=True)
                tiff.imwrite(destination + names[0][0].split('/')[-1], diffseg1[0][0,::].numpy().astype(np.float32))
            else:
                to_show = [imgX, combined, diffseg0, diffseg1, camX, camXd]
                #to_show = [imgX, combined, [to_rgb(x) for x in diffseg0], [to_rgb(x) for x in diffseg1]]
                to_print(to_show, save_name=os.path.join("outputs/results", args.dataset, args.prj,
                                                         str(epoch) + '_' + str(subject) + '_' + str(ii).zfill(4) + 'm'))




# USAGE
# CUDA_VISIBLE_DEVICES=1 python testoai.py --jsn womac3 --direction a_b --prj Gds/descar2/Gdsmc --engine descar2 --cropsize 384 --n01 --cmb mul --gray --nepochs 80 81 20 --nalpha 0 20 20

