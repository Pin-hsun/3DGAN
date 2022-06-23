import tifffile as tiff
import torch

from utils.data_utils import imagesc
from skimage.measure import label
from skimage.morphology import skeletonize, skeletonize_3d
import scipy.ndimage
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import skimage.measure
from skimage.metrics import structural_similarity as ssim

def show_threshold_seg(oo):
    rr = ((oo > 2000) & (scipy.ndimage.distance_transform_edt(oo > 2000) <= 3)) / 1
    gg = ((oo > 1200) & (scipy.ndimage.distance_transform_edt(oo > 1200) <= 3)) / 1

    ooo = np.concatenate([np.expand_dims(oo, 2)]*3, 2)

    ooo[:, :, 1] = np.multiply(ooo[:, :, 1], 1 - 0.5 * rr)
    ooo[:, :, 2] = np.multiply(ooo[:, :, 2], 1 - 0.5 * rr)

    ooo[:, :, 0] = np.multiply(ooo[:, :, 0], 1 - 0.5 * gg)
    ooo[:, :, 2] = np.multiply(ooo[:, :, 2], 1 - 0.5 * gg)

    imagesc(ooo)


def seg_to_skel(x, dim):
    if dim == 3:
        x3d = 0 * x
        for i in range(1, x.shape[0]-1):
            x3d[i, :, :] = skeletonize_3d(x[i-1:i+2, :, :])[1, :, :]
        x3d = x3d // x3d.max()
        x3d = x3d.astype(np.uint8)
        return x3d
    elif dim == 0:
        x2d = 0 * x
        for i in range(x.shape[0]):
            x2d[i, :, :] = skeletonize(x[i, :, :] > 0)
        x2d = x2d.astype(np.uint8)
        return x2d
    elif dim == 1:
        x2d = 0 * x
        for i in range(x.shape[1]):
            x2d[:, i, :] = skeletonize(x[:, i, :] > 0)
        x2d = x2d.astype(np.uint8)
        return x2d
    elif dim == 2:
        x2d = 0 * x
        for i in range(x.shape[2]):
            x2d[:, :, i] = skeletonize(x[:, :, i] > 0)
        x2d = x2d.astype(np.uint8)
        return x2d

def print_skel_overlap(x, ori, destination):
    for i in range(x.shape[2]):
        if not os.path.isdir(destination):
            os.mkdir(destination)
        a = Image.fromarray(x[:, :, i])
        #a = a.resize((1024, 1024), resample=Image.BILINEAR)
        a = np.array(a)
        #a = (a > 0) / 1
        b = ori[:, :, i]
        c = np.zeros((1024, 1024, 3))
        c[:, :, 0] = b#np.multiply(b, a)
        c[:, :, 1] = np.multiply(b, 1 - a)
        c[:, :, 2] = np.multiply(b, 1 - a)
        c[0, 0, 0] = ori.max()
        imagesc(c, show=False, save=destination + str(i).zfill(3) + '.png')


def distance_transform_2d(x, dim):
    if dim == 0:
        x = np.transpose(x, (0, 1, 2))
    elif dim == 1:
        x = np.transpose(x, (1, 0, 2))
    elif dim == 2:
        x = np.transpose(x, (2, 0, 1))
    d3d = []
    for i in range(x.shape[0]):
        d = scipy.ndimage.distance_transform_edt(x[i, ::])
        d = np.expand_dims(d, dim)
        d3d.append(d)
    d3d = np.concatenate(d3d, dim)
    return d3d

def neuron_scale_moving_average(d, skel):
    s = d.shape[0]
    sdiv = list(range(0, s, s//10))
    ddiv = []
    sval = []
    for i in range(len(sdiv) - 1):
        d_one_div = (d[sdiv[i]:sdiv[i+1], ::])[skel[sdiv[i]:sdiv[i+1], ::] > 0]
        ddiv.append(d_one_div)
        sval.append(np.ones(d_one_div.shape) * i)

    ddiv = np.concatenate(ddiv, 0)
    sval = np.concatenate(sval, 0)

    return np.concatenate([np.expand_dims(x, 1) for x in [ddiv, sval]], 1)


def neuron_distance(x):
    #d3d = scipy.ndimage.distance_transform_edt(x)
    d0 = distance_transform_2d(x, dim=0)
    d1 = distance_transform_2d(x, dim=1)
    d2 = distance_transform_2d(x, dim=2)
    return [d0,  d1, d2]


def neuron_skel(x):
    skel0 = seg_to_skel(x, 0)
    skel1 = seg_to_skel(x, 1)
    skel2 = seg_to_skel(x, 2)
    skel3 = skeletonize_3d(x)
    skel3 = (skel3 > 0).astype(np.uint8)
    return [skel0, skel1, skel2, skel3]


def neuron_scales():
    root = '/media/ghc/GHc_data2/'
    wg = tiff.imread(root + 'wg.tif')
    wgvar = tiff.imread(root + 'wgvar.tif')
    w = tiff.imread(root + 'w.tif')

    wg_distance = neuron_distance(wg > 0)
    wg_skel = neuron_skel(wg > 0)

    w_distance = neuron_distance(w > 0)
    w_skel = neuron_skel(w > 0)

    wgv_distance = neuron_distance((wg > 0) & (wgvar < 0.2))
    wgv_skel = neuron_skel((wg > 0) & (wgvar < 0.2))

    ## relative surface area
    wgsurface = np.divide((wg_distance[-1] == 1).sum(0).sum(0), (wg_distance[-1] > 0).sum(0).sum(0))
    wsurface = np.divide((w_distance[-1] == 1).sum(0).sum(0), (w_distance[-1] > 0).sum(0).sum(0))


def deconv():  # plot point spreading function
    from scipy.ndimage.filters import gaussian_filter
    if 1:
        net = torch.load('/media/ExtHDD01/logs/Fly3D/wnwp3d/cyc4z/GdeWOz4/checkpoints/netGYX_model_epoch_100.pth').cuda()
        z = torch.FloatTensor([1])
        a = -1 * z.view(1, 1, 1, 1).repeat(1, 1, 32//2, 32//2).cuda()
    else:
        net = torch.load('/media/ExtHDD01/logs/Fly3D/wnwp3d/cyc4z/GdeWOmc/checkpoints/netGYX_model_epoch_200.pth').cuda()
        a = None

    p_all = []
    for i in range(100):
        x = np.zeros((128, 128))
        x[63:65, 63:65] = 1
        x = gaussian_filter(x, sigma=1)
        x = x / x.max()
        x = x.astype(np.float32)
        x = x - 0.5
        x = x * 2
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).cuda()
        o = net(torch.cat([x, x], 1), a=a)
        p = o[1][0, ::].detach().cpu()
        p_all.append(p)
    p_all = torch.cat(p_all, 0)
    imagesc(p_all.mean(0))
    imagesc(p_all.var(0))


def skel_connect_2d(skel, dim):
    skel = (skel > 0).astype(np.uint8)
    if dim == 1:
        skel = np.transpose(skel, (1, 0, 2))
    elif dim == 2:
        skel = np.transpose(skel, (2, 0, 1))

    coors = np.nonzero(skel)
    connect = 0 * skel

    for z, x, y in zip(coors[0], coors[1], coors[2]):
        voxel = skel[z, x-1:x+2, y-1:y+2]
        connect[z, x, y] = voxel.sum()

    rgb = np.concatenate([0 * np.expand_dims(connect, 3)] * 3, 3)

    endpt = (connect == 2)
    node = (connect >= 4)

    all = []
    for s in range(connect.shape[0]):
        endpt_slice = endpt[s, ::]
        node_slice = node[s, ::]

        dis_to_node = scipy.ndimage.distance_transform_edt(1 - node_slice)
        endpt_dst_to_node = np.multiply(endpt_slice, dis_to_node)
        all.append(np.expand_dims(endpt_dst_to_node, 0))
    all = np.concatenate(all, 0)


def skel_to_connect(xskel):

    skel = (xskel > 0).astype(np.uint8)

    coors = np.nonzero(skel)
    neighbor = []
    zxy = []
    for z, x, y in zip(coors[0], coors[1], coors[2]):
        voxel = skel[z-1:z+2, x-1:x+2, y-1:y+2]
        if voxel.shape == (3, 3, 3):
            voxel = np.reshape(voxel, (27, 1))
            neighbor.append(voxel)
            zxy.append(np.array([z, x, y]))

    nei = np.concatenate(neighbor, 1)
    zxy = np.concatenate([np.expand_dims(x, 1) for x in zxy], 1)

    skelrgb = 0 * np.concatenate([np.expand_dims(skel, 3)]*3, 3)

    kind = nei.sum(0)
    print(np.unique(nei.sum(0), return_counts=True))

    for i in range(len(kind)):
        if kind[i] == 2:
            skelrgb[zxy[0, i],  zxy[1, i], zxy[2, i], 0] = 1
            #skelrgb[zxy[0, i],  zxy[1, i], zxy[2, i], 2] = 0
        #if kind[i] == 3:
        #    skelrgb[zxy[0, i],  zxy[1, i], zxy[2, i], 0] = 0
        #    skelrgb[zxy[0, i],  zxy[1, i], zxy[2, i], 2] = 0
        if kind[i] >= 4:
            skelrgb[zxy[0, i],  zxy[1, i], zxy[2, i], 2] = 1
            #skelrgb[zxy[0, i],  zxy[1, i], zxy[2, i], 1] = 0

    tiff.imsave('skelrgb.tif', skelrgb)


    p2 = (skelrgb[:, :, :, 1]==0) &  (skelrgb[:, :, :, 0]==1)
    p4 = (skelrgb[:, :, :, 0]==0) &  (skelrgb[:, :, :, 2]==1)