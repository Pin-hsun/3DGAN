from __future__ import print_function
import argparse, json
import os, glob, sys
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
import os, glob
import numpy as np
import matplotlib.pyplot as plt
cm = plt.get_cmap('viridis')

source='/media/ghc/GHc_data2/N3D/F0OTrd2k/xyzft0/'


def sum_all(source):
    files = sorted(glob.glob(source + '*0.tif'))
    tifs = []
    for f in files:
        tifs.append(tiff.imread(f))

    tifs = np.stack(tifs, 3)
    tiff.imwrite(source + 'mean.tif', np.mean(tifs, 3))
    tiff.imwrite(source + 'var.tif', np.var(tifs, 3))


def calculate_fft():
    root = os.path.join('/home/ubuntu/Data/Dataset/paired_images', args.dataset)
    from dataloader.data_multi import PairedDataTif
    test_set = PairedDataTif(root=root,
                             directions=args.direction)
    x = test_set.__getitem__(0)['img']


def Global_Filter(x):
    x = x.permute(0, 2, 3, 1).contiguous()
    x = torch.fft.fft(x, dim=3, norm='ortho')
    out = torch.unsqueeze(x[:, :, :, 0], -1).permute(0, 3, 1, 2).contiguous()
    out = out.type(torch.cuda.FloatTensor)
    return out


def make_rotation_3d(stepx, stepy):
    stepx = 2
    stepy = 0
    # get model
    import networks, models
    sys.modules['models'] = networks
    #net = torch.load('/media/ExtHDD01/logs/Fly0B/wnwp3d/cyc/GdenuBmc/checkpoints/netGXY_model_epoch_100.pth')
    #net = torch.load('submodels/1.pth').cuda()  #  mysterious Resnet model with ResnetAdaILNBlock (ugatit?)

    args.epoch = 20
    logs = '/home/ubuntu/Data/logs' #os.environ.get('LOGS')
    root = os.path.join('/home/ubuntu/Data/Dataset/paired_images', args.dataset)

    net = torch.load(os.path.join(logs, args.dataset, args.prj, 'checkpoints',
                                  args.netg + '_model_epoch_' + str(args.epoch) + '.pth'),
                     map_location=torch.device('cpu')).cuda()

    # get a 3D volume
    sys.modules['models'] = models
    from dataloader.data_multi import PairedDataTif
    test_set = PairedDataTif(root=root,
                             directions=args.direction, permute=(0, 1, 2), trd=args.trd,
                             crop=[1890-1792, 1890, 512 * stepx, 512 * stepx + 1024, 512 * stepy, 512 * stepy + 1024])
                             #crop=[1890-1792, 1890, 1024, 2048, 0, 1024])
    x = test_set.__getitem__(0)['img']

    for angle in [0]:#list(range(0, 331, 30)):
        xp = [y for y in x]

        #  testing over slices
        all = []
        for p in range(len(xp)):
            all.append([])

        for i in range(xp[0].shape[2]):
            print(i)

            slices = []
            for p in range(len(xp)):
                slices.append(xp[p][:, :, i, :].unsqueeze(0).cuda())

            fft = Global_Filter(torch.cat([slices[0], slices[1], slices[2], slices[3]], 1))
            out = net(torch.cat((fft, slices[4]), 1))  # , a=None)
            out = [out['out0'], out['out1']]
            out = [y.detach().cpu() for y in out]

            for p in range(len(out)):
                all[p].append(out[p])
        del xp

        for j in range(len(all)):
            all[j] = torch.cat(all[j], 0)  #(x, 1, z, y)
            all[j] = all[j].permute(2, 1, 0, 3)

        for j in range(len(all)):
            if angle != 0:
                all[j] = transforms.functional.rotate(all[j], angle=-angle,  #(z, 1, x, y)
                                                      interpolation=torchvision.transforms.functional.InterpolationMode.BILINEAR,
                                                      fill=-1)
            all[j] = all[j].numpy()[:, 0, ::]
            all[j] = all[j][:, 256:-256, 256:-256].astype(np.float16)

        destination = args.destination + str(stepx) + str(stepy) + '/'
        dirs = args.direction.split('_')

        for j in range(len(all)):
            os.makedirs(os.path.join(destination, dirs[j]), exist_ok=True)
            tiff.imwrite(os.path.join(destination, dirs[j], str(angle).zfill(3) + '.tif'), all[j])
        del all


def calculate_variance():
    root = '/media/ghc/GHc_data2/N3D/F0OTrd2k/xyzft0/'
    x = sorted(glob.glob(root + '*0.tif'))
    t = [tiff.imread(y) for y in x]


def ori_var_overlap():
    root = '/media/ghc/GHc_data2/N3D/'
    ori = tiff.imread(root + 'ori.tif')
    var = tiff.imread('/media/ghc/GHc_data2/N3D/F0OTrd2k/xyzft0/var2.tif')
    var = var / var.max()

    for i in range(300, 1792, 1):
        r = ori[i, :, :] * 1
        g = np.multiply(ori[i, :, :], 1 - 0.3 * var[i, :, :])
        b = np.multiply(ori[i, :, :], 1 - 0.3 * var[i, :, :])
        o = np.stack([r, g, b], 0)
        imagesc(o, show=False, save=root + 'F0OTrd2k/xyzft0/overlapall/' + str(i).zfill(4) + '.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
    parser.add_argument('--dataset', type=str, default='Fly0B', help='environment_to_use')
    parser.add_argument('--prj', type=str, default='wnwp3d/cyc4/GdenuF0Bmc', help='environment_to_use')
    parser.add_argument('--netg', default='netGXY', type=str)
    parser.add_argument('--env', default=None, type=str)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--direction', default='xyzft0_xyzsb', type=str)
    parser.add_argument('--trd', default=0, type=int)
    parser.add_argument('--destination', default='/media/ghc/GHc_data2/N3D/F0B/', type=str)
    parser.add_argument('--mode', type=str, default='dummy')
    parser.add_argument('--port', type=str, default='dummy')
    parser.add_argument('--host', type=str, default='dummy')
    args = parser.parse_args()

    #args.env = 'a6k'

    # environment file
    if args.env is not None:
        load_dotenv('env/.' + args.env)
    else:
        load_dotenv('env/.t09')

    if 0:
        args.prj = 'wnwp3d/cyc/GdenuBmc'
        args.direction = 'xyzsb'
        args.destination = '/media/ghc/GHc_data2/N3D/B/'
        args.trd = 0
    elif 1:
        args.prj = 'cyc4fft/0'
        args.direction = 'xyzft0_xyzori'
        args.destination = '/media/ghc/GHc_data2/N3D/F0OmcX/'
        args.trd = 2000

    #args.destination = '/home/ubuntu/Data/N3D/F0OTrd2k'

    #make_rotation_3d(stepx=2, stepy=0)

# USAGE
# CUDA_VISIBLE_DEVICES=0 python test.py --jsn default --dataset pain --nalpha 0 100 2  --prj VryAtt
# CUDA_VISIBLE_DEVICES=0 python test.py --jsn default --dataset TSE_DESS --nalpha 0 100 1  --prj VryCycleUP --net netGXY
# python testoai.py --jsn womac3 --direction a_b --prj N01/DescarMul/ --cropsize 384 --n01 --cmb mul
# python testrefactor.py --jsn womac3 --direction b_a --prj N01/DescarMul/ --cropsize 384 --n01 --cmb mul --nepoch 20

# CUDA_VISIBLE_DEVICES=1 python test_fly3D.py --env a6k --jsn FlyZWpWn --direction zyweak512_zyorisb512 --prj wnwp3d/cyc3/GdenuWSmcYL10 --engine cyclegan23dwo --nalpha 0 20 20 --nepochs 0 201 20

# CUDA_VISIBLE_DEVICES=1 python test_fly3D.py --jsn FlyZWpWn --direction zyweak512_zyori512 --prj wnwp3d/cyc2l1/0 --nepochs 60 --engine cyclegan23d --cropsize 512

# single
# CUDA_VISIBLE_DEVICES=1 python test_fly3D.py --jsn FlyZWpWn --direction zyweak5_zysb5 --prj wnwp3d/cyc4z/GdeWOmc --nepochs 140 --engine cyc4 --cropsize 512 --nalpha 0 20 20 -b 16