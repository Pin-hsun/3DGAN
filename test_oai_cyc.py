import torch
import tifffile as tiff
import os, glob
import numpy as np
from dotenv import load_dotenv
import argparse

def to_8_bit(img):
    img = (np.array((img - img.min()) / (img.max() - img.min())) * 255).astype(np.uint8)
    return img
def transform3d(x, cropsize):
    x[x >= 800] = 800
    print(x.shape)
    x = (x - x.min()) / (x.max() - x.min())
    x = (x - 0.5) * 2
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().permute(0, 1, 3, 4, 2)
    print(x.shape)
    upsample = torch.nn.Upsample(size=(384, 384, x.shape[4] * 8), mode='trilinear')
    x = upsample(x)
    crop = int((384-cropsize)/2)
    x = x[:, :, crop:384-crop, crop:384-crop, :]

    return x

def transform2d(x, trd):
    x[x >= trd] = trd

    x = (x - x.min()) / (x.max() - x.min())
    x = (x - 0.5) * 2

    x = torch.from_numpy(x).float().unsqueeze(1)

    cropsize = 384
    x = x[int(x.shape[0] / 2 - 16):int(x.shape[0] / 2 + 16), :,
            int(x.shape[2] / 2 - cropsize / 2):int(x.shape[2] / 2 + cropsize / 2),
            int(x.shape[3] / 2 - cropsize / 2):int(x.shape[3] / 2 + cropsize / 2)]

    return x

load_dotenv('env/.t09')
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
# Data
parser.add_argument('--prj', type=str, default='test')
parser.add_argument('--dataset', type=str, default='womac4min0/raw3D/')
parser.add_argument('--bysubject', action='store_true', dest='bysubject', default=False)
parser.add_argument('--direction', type=str, default='SagIwTSE', help='a2b or b2a')
parser.add_argument('--flip', action='store_true', dest='flip', default=False, help='image flip left right')
parser.add_argument('--resize', type=int, default=0)
parser.add_argument('--cropsize', type=int, default=384)
parser.add_argument('--cropz', type=int, default=0)
parser.add_argument('--n01', action='store_true', dest='n01', default=False)
parser.add_argument('--nm', type=str, default='11', help='normalization method for dataset')
parser.add_argument('--gray', action='store_true', dest='gray', default=False, help='only use 1 channel')
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--port', type=str, default='dummy')
parser.add_argument('--trd', type=int, dest='trd', nargs='+',  help='threshold of images')
parser.add_argument('--permute', action='store_true', dest='permute', default=True, help='do interpolation and permutation')
parser.add_argument('--load_3D', action='store_true', dest='load_3D', default=False, help='load 3D cube')
opt = parser.parse_args()

root = os.environ.get('DATASET') + opt.dataset
os.makedirs('out/{}'.format(opt.prj), exist_ok=True)
#
if opt.direction == 'SagIwTSE':
    print('load SagIWTSE')
    x = tiff.imread('/media/ziyi/glory/OAIDataBase/womac4min0/raw3D/SagIwTSE/9573975_00_LEFT.tif')
    tiff.imsave('out/{}/SAG.tif'.format(opt.prj), x)
else:
    print('load CorIwTSE')
    x = tiff.imread('/media/ziyi/glory/OAIDataBase/womac4min0/raw3D/CorIwTSE/9573975_00_LEFT.tif')
y = tiff.imread('/media/ziyi/glory/OAIDataBase/womac4min0/raw3D/SagIwTSE/9573975_00_LEFT.tif')

tiff.imsave('out/{}/ori.tif'.format(opt.prj), x)

if opt.load_3D:
    print('load 3D')
    x = transform3d(x, opt.cropsize)
else:
    print('load 2D')
    x = transform2d(x, 1600)
    y = transform2d(y, 600)
    tiff.imsave('out/{}/b.tif'.format(opt.prj), y.numpy())

tiff.imsave('out/{}/a.tif'.format(opt.prj), x.numpy())

for ep in range(10, 210, 10):
    if opt.direction == 'SagIwTSE':
        model = torch.load('/media/ziyi/glory/logs_pin/womac4min0/raw3D/{prj}/checkpoints/net_g_model_epoch_{ep}.pth'
                           .format(prj=opt.prj, ep=ep),map_location='cpu')
    else:
        model = torch.load('/media/ziyi/glory/logs_pin/womac4min0/raw3D/{prj}/checkpoints/net_gYX_model_epoch_{ep}.pth'
                           .format(prj=opt.prj, ep=ep), map_location='cpu')
    # model.eval()
    o = model(x)['out0']
    o = to_8_bit(o.detach())
    # o1 = o.permute(2, 1, 4, 3, 0) #(X, C, Z, Y)
    # o2 = o.permute(3, 1, 4, 2, 0) # (Y, C, Z, X)
    os.makedirs('out/{}'.format(opt.prj), exist_ok=True)
    tiff.imsave("out/{prj}/ep{ep}_out.tif".format(prj=opt.prj, ep=ep), o)
    # tiff.imsave("out/{prj}/ep{ep}_outzy.tif".format(prj=opt.prj, ep=ep), o1.detach().numpy())
    # tiff.imsave("out/{prj}/ep{ep}_outzx.tif".format(prj=opt.prj, ep=ep), o2.detach().numpy())

# import tifffile as tiff
# import torch
# o = tiff.imread('out/0719_cut_3enc/ep80_out.tif')
# o = torch.from_numpy(o).float()
# o1 = o.permute(0,1,4,2,3)
# tiff.imsave('out/0719_cut_3enc/xy_ep80.tif',o1.numpy())
# seg_model = torch.load('submodels/seg_atten.pth', map_location=torch.device('cpu')).eval()
# img_a = o[0].permute(3, 0, 1, 2)
# pred_a = torch.argmax(seg_model(img_a), 1, True)

# python test_oai_cyc.py --direction SagIwTSE --load_3D --prj
# python test_oai_cyc.py --direction CorIwTSE --prj