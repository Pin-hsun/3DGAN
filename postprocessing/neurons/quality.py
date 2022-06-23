import torch
from piq import FID
import piq
import torchvision
import tifffile as tiff
from utils.data_utils import imagesc


def get_features(x):
    out = []
    for i in range(x.shape[0] // 32):
        f = net(x[i*32:(i+1)*32, ::].cuda()).detach().cpu()
        out.append(f)
    out = torch.cat(out, 0)
    return out


def side_patches(x):
    x = torch.from_numpy(x).permute(1, 0, 2)
    x = x.reshape(512, x.shape[1] // 512, 512, 512)
    x = x.reshape(512 * x.shape[1], 512, 512)
    x = x.unsqueeze(1).repeat(1, 3, 1, 1)
    return x

def save_to_imgs(x, dest):
    for i in range(x.shape[0]):
        slice = x[i, ::]
        imagesc(slice, show=False, save=dest+str(i).zfill(4)+'.png')

def n01(x0):
    x = 1 * x0
    x = x - x.min()
    x = x / x.max()
    return x


net = torchvision.models.inception_v3(pretrained=True).cuda()
net.eval()
root = '/media/ghc/GHc_data2/'
w = tiff.imread(root + 'b.tif')
wg = tiff.imread(root + 'bg.tif')

w = w[-512*3:, ::]
wg = wg[-512*3:, ::]

wtop = torch.from_numpy(w).unsqueeze(1).repeat(1, 3, 1, 1)
wgtop = torch.from_numpy(wg).unsqueeze(1).repeat(1, 3, 1, 1)

wside1 = side_patches(w)
wgside1 = side_patches(wg)

wtopf = get_features(wtop)

f = [get_features(x[:, :, :, :]) for x in [wtop, wgtop, wside1, wgside1]]

metric = FID()



w = tiff.imread(root + 'w.tif')
wg = tiff.imread(root + 'wg.tif')

w = w[-512:, ::]
wg = wg[-512:, ::]


def poros(x):
    p = []
    for s in x.shape[0]:
        p.