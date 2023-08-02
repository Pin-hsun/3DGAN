from __future__ import print_function
import argparse, json
import networks, models
import os, sys
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
cm = plt.get_cmap('viridis')
sys.path.insert(0, './models')

def norm_11(x):
    x = x - x.min()
    x = x / x.max()
    x = 2 * x - 1
    return x


def to_heatmap(x):
    return cm(x)


class TestModel:
    """Test model class"""
    def __init__(self, args):
        self.args = args
        self.net_g = None
        self.dir_checkpoints = os.environ.get('LOGS')
        from dataloader.data_multi import MultiData as Dataset

        self.test_set = Dataset(root=root_data,
                                path=args.direction,
                                opt=args, mode='test', filenames=True)

        sys.modules['models'] = networks
        #self.seg_cartilage = torch.load('submodels/model_seg_ZIB_res18_256.pth')
        self.seg_cartilage = torch.load('submodels/78.pth')
        #self.seg_bone = torch.load('submodels/model_seg_ZIB.pth')
        self.netg_t2d = torch.load('/media/ExtHDD01/logs/t2d/1/checkpoints/net_g_model_epoch_200.pth').cuda()
        #self.eff = torch.load('submodels/model_seg_eff.pth')
        sys.modules['models'] = models

        self.netg_t2d.eval()
        self.seg_cartilage.eval()

        self.device = torch.device("cuda:0")

    def get_model(self,  epoch, eval=False):
        model_path = os.path.join(self.dir_checkpoints, self.args.dataset, self.args.prj, 'checkpoints') + \
               ('/' + self.args.netg + '_model_epoch_{}.pth').format(epoch)
        print('Loading model: ' + model_path)
        try:
            net = torch.load(model_path, map_location='cpu').cuda()
        except:
            sys.modules['models'] = networks
            net = torch.load(model_path, map_location='cpu').cuda()
            sys.modules['models'] = models
        print(type(net))
        if eval:
            net.eval()
        else:
            net.train()
        self.net_g = net

    def get_one_output(self, i, alpha=None):
        # get the test method
        engine = args.engine
        test_method = getattr(__import__('models.' + engine), engine).GAN.test_method

        # inputs
        x = self.test_set.__getitem__(i)
        img = x['img']
        imgs = [x.unsqueeze(0).to(self.device) for x in img]

        output = test_method(self.net_g, imgs, args=args, a=alpha/100)
        output['imgX'] = img[0]
        output['imgY'] = img[1]
        output['name'] = x['filenames']

        return output

    def tse_segmentation(self, x, t2d=True):
        # netg_t2d.eval()
        self.seg_cartilage.eval()
        if t2d:
            x = preprocess_tensor(x, normalization='-11')
            x, = self.netg_t2d(x.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1).type(torch.FloatTensor).cuda())['out0']
        x = preprocess_tensor(x, normalization='-11')
        s = self.seg_cartilage(x.unsqueeze(0).repeat(1, 3, 1, 1).cuda())
        s = torch.nn.Softmax(dim=1)(s).detach().cpu()
        seg = torch.argmax(s, dim=1)[0, ::]
        return seg


def to_print(to_show, save_name):
    os.makedirs(os.path.join("outputs/results", args.dataset, args.prj), exist_ok=True)
    # combine each row
    to_show = [torch.cat(x, len(x[0].shape) - 1) for x in to_show]

    # normalize each row
    for i in range(len(to_show)):
        to_show[i] = to_show[i] - to_show[i].min()
        if to_show[i].max() > 0:
            to_show[i] = to_show[i] / to_show[i].max()

    for i in range(len(to_show)):
        if to_show[i].shape[0] == 1:
            to_show[i] = torch.cat([to_show[i]] * 3, 0)

    to_print = np.concatenate(to_show, 1).astype(np.float16)
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


def preprocess_tensor(x, normalization='01', duplicate_channels=False):
    if duplicate_channels and x.size(0) == 1:
        x = torch.cat([x, x, x], dim=0)

    if normalization == '01':
        max_val = x.max()
        min_val = x.min()
        x = (x - min_val) / (max_val - min_val)
    elif normalization == '-11':
        max_val = x.max()
        min_val = x.min()
        x = (x - (max_val + min_val) / 2.0) / ((max_val - min_val) / 2.0)

    return x


# Command Line Argument
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--env', type=str, default=None, help='environment_to_use')
parser.add_argument('--jsn', type=str, default='womac3', help='name of ini file')
parser.add_argument('--dataset', help='name of training dataset', default='womac4')
parser.add_argument('--engine', dest='engine', type=str, help='use which engine')
parser.add_argument('--testset', help='name of testing set if different than the training set', default='womac3/full/')
parser.add_argument('--load3d', action='store_true', dest='load3d')
parser.add_argument('--gray', action='store_true', dest='gray', help='dont copy img to 3 channel')
parser.add_argument('--notgray', action='store_false', dest='gray', help='dont copy img to 3 channel')
parser.add_argument('--prj', type=str, help='model', default='3D/test4ano/0v/')
parser.add_argument('--direction', type=str, help='a_b')
parser.add_argument('--netg', type=str)
parser.add_argument('--resize', type=int)
parser.add_argument('--cropsize', type=int)
parser.add_argument('--t2d', action='store_true', dest='t2d', default=False)
parser.add_argument('--trd', type=float, dest='trd', help='threshold of images')
parser.add_argument('--nm', dest='nm')
parser.set_defaults(n01=False)
parser.add_argument('--eval', action='store_true', dest='eval')
parser.add_argument('--nepochs', nargs='+', help='which ckpt to be interfered with', type=int, default=[120, 121, 1])
parser.add_argument('--nalpha', nargs='+', help='additional input par. for generator', type=int, default=[0, 100, 11])
parser.add_argument('--all', type=int, nargs='+', dest='all', default=None)
parser.add_argument('--mc', type=int, dest='mc', default=1, help='monte carlo sampling')
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--port', type=str, default='dummy')
parser.add_argument('--item', nargs='+', help='item to print out', type=str, default=[['diff_bone', 'diff_soft']])
parser.add_argument('--sfx', type=str, dest='suffix', default='', help='suffix of output')
parser.add_argument('--dest', type=str, dest='destination', default='gan_output', help='suffix of output')
parser.add_argument('--fix', action='store_true', dest='fix', default=False)
parser.add_argument('--cmb', action='store_true', dest='cmb', default=False)


with open('env/jsn/' + parser.parse_args().jsn + '.json', 'rt') as f:
    t_args = argparse.Namespace()
    t_args.__dict__.update(json.load(f)['test'])
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

args.destination = 'gan/' + args.prj

root_data = os.environ.get('DATASET') + args.testset

test_unit = TestModel(args=args)
print(len(test_unit.test_set))


def concatenate_list_of_dicts(list_of_dicts):
    """Concatenate list of dicts into one dict"""
    concatenated_dict = {}
    for keys in list_of_dicts[0].keys():
        concatenated_dict[keys] = []
        for d in list_of_dicts:
            concatenated_dict[keys].append(d[keys])
    return concatenated_dict


if args.all is not None:
    if len(args.all) == 3:
        iirange = range(len(test_unit.test_set))[args.all[0]:args.all[1]:args.all[2]]
    else:
        iirange = args.all
else:
    iirange = range(1)


for epoch in range(*args.nepochs):
    test_unit.get_model(epoch, eval=args.eval)

    for ii in iirange:
        if args.all is not None:
            args.irange = [ii]

        for alpha in np.linspace(*args.nalpha):

            mc_diff = []
            mc_combined = []

            for m in range(args.mc): # Monte Carlo
                out = list(map(lambda v: test_unit.get_one_output(v, alpha), args.irange))
                out = concatenate_list_of_dicts(out)

                #[imgX, imgY, output, names] = list(zip(*out_xy))

                # decide to assign combined as XY or XX
                if args.fix:
                    if alpha == 0:
                        print('XX')
                        out['combined'] = out['combinedXX']
                    else:
                        out['combined'] = out['combinedXY']
                else:
                    out['combined'] = out['combinedXY']

                # processing the rest
                out['diff_xy'] = [(x[1] - x[0]) for x in list(zip(out['combined'], out['imgX']))]
                # segmentation
                out['imgX_seg'] = [test_unit.tse_segmentation(x[0, ::], t2d=True).unsqueeze(0) for x in out['imgX']]
                out['combined_seg'] = [test_unit.tse_segmentation(x[0, ::], t2d=True).unsqueeze(0) for x in out['combined']]
                out['diff_bone'] = [torch.multiply(x, (y == 1) + (y == 3)) for x, y in list(zip(out['diff_xy'], out['imgX_seg']))]
                out['diff_soft'] = [torch.multiply(x, (y == 0)) for x, y in list(zip(out['diff_xy'], out['imgX_seg']))]

                #mc_diff.append([((1 - x) > 0.7)/1 for x in out['imgXY']])
                #mc_diff.append([1 - x for x in out['imgXY']])

                mc_diff.append(out['diff_xy'])
                mc_combined.append(out['combined'])

            mc_diff = list(zip(*mc_diff))
            mc_combined = list(zip(*mc_combined))

            out['mean_diff'] = []
            out['var_diff'] = []
            out['sig_diff'] = []
            out['mean_combined'] = []
            out['var_combined'] = []
            out['sig_combined'] = []

            for s in range(len(mc_diff)):
                mc_diff[s] = torch.cat(mc_diff[s], dim=0)
                mc_combined[s] = torch.cat(mc_combined[s], dim=0)
                out['mean_diff'].append(torch.mean(mc_diff[s], dim=0, keepdim=True))
                out['var_diff'].append(torch.var(mc_diff[s], dim=0, keepdim=True))
                out['sig_diff'].append(torch.divide(out['mean_diff'][s], torch.sqrt(out['var_diff'][s]) + 0.00))
                out['mean_combined'].append(torch.mean(mc_combined[s], dim=0, keepdim=True))
                out['var_combined'].append(torch.var(mc_combined[s], dim=0, keepdim=True))
                out['sig_combined'].append(torch.divide(out['mean_combined'][s], torch.sqrt(out['var_combined'][s]) + 0.00))


            # print
            if args.all:
                for item in args.item:
                    os.makedirs(os.path.join(root_data, args.destination, item + args.suffix), exist_ok=True)
                    tiff.imwrite(os.path.join(root_data, args.destination, item + args.suffix,
                                              out['name'][0][0].split('/')[-1].split('.')[0] + '_' + str(alpha).zfill(3) + '.tif'),
                                 out[item][0][0, ::].numpy())
            else:
                to_show = [out['var']]#[out['imgX'], out['combined_seg'], out['combined'], out['diff_bone'], out['diff_soft'], mc['mean'], mc['var']]
                to_print(to_show, save_name=os.path.join("outputs/results", args.dataset, args.prj,
                                                     str(epoch) + '_' + str(alpha) + '_' + str(ii).zfill(4) + 'm'))

            if 1:   # print all the mc combined
                item = 'mc'
                os.makedirs(os.path.join(root_data, args.destination, item + args.suffix), exist_ok=True)
                for s in range(len(mc_combined)):
                    for i in range(mc_combined[s].shape[0]):
                        tiff.imwrite(os.path.join(root_data, args.destination, item + args.suffix, str(i).zfill(3) +
                                                 '.tif'), mc_combined[s][i, ::].numpy())


import glob

root = os.path.join(root_data, args.destination)

os.makedirs(os.path.join(root, 'diff_sig'), exist_ok=True)
m = [tiff.imread(x)for x in sorted(glob.glob(os.path.join(root, 'mean_combinedA/*')))]
v = [tiff.imread(x) for x in sorted(glob.glob(os.path.join(root, 'var_combinedA/*')))]

m0 = [tiff.imread(x.replace(x.split('_')[-1], '0.0.tif')) for x in sorted(glob.glob(os.path.join(root, 'mean_combinedA/*')))]
v0 = [tiff.imread(x.replace(x.split('_')[-1], '0.0.tif')) for x in sorted(glob.glob(os.path.join(root, 'var_combinedA/*')))]

significant_new = []
for i in range(len(m)):
    significant_new.append((m0[i] - m[i]) / np.sqrt(v[i] + v0[i]))
    tiff.imsave(os.path.join(root, 'diff_sig', str(i).zfill(3) + '.tif'), significant_new[-1])







# USAGE
# CUDA_VISIBLE_DEVICES=1 python -m test.test_oaiinj.py --jsn womac3 --direction a_b --prj 3D/test4ano/0v/ --engine descar4 --cropsize 384 --nm 01 --cmb not --gray --nepochs 120 121 40 --nalpha 0 100 11 --dataset womac4 --testset womac3/full/


# CUDA_VISIBLE_DEVICES=0 python -m test.test_oaiinj.py --jsn womac3 --direction a_b --testset womac4/full/ --item mean sig --prj 3D/test4mcVgg10/ --all 0 1
# 00 1 --mc 100 --sfx A --dest out0324 --nepochs 100 101 1

 # CUDA_VISIBLE_DEVICES=0 python -m test.test_oaiinj.py --jsn womac3 --direction a_b --testset womac4/full/ --item combined mean_diff var_diff sig_diff mean_combined var_combined sig_combined --prj 3D/test4mcVgg10/ --all 1223 2698 6909 9351 9528 5591 925 --mc 100 --sfx A  --nepochs 100 101 1 --engine descar4ab --nalpha 0 101 2
