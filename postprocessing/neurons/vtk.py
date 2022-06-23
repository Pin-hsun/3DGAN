import numpy as np
import glob, os
import os
import distutils
import numpy as np
from PIL import Image
import tifffile as tiff
from skimage import io
import matplotlib.pyplot as plt
import pyvista as pv
from skimage.measure import label
from skimage.morphology import skeletonize, skeletonize_3d
import torch
import torch.nn as nn


def show_compare(x):
    folder_name = x
    seg_name = x.replace(x.split('/')[-2], 'seg-' + x.split('/')[-2])
    x0 = glob.glob(folder_name + '*')
    x0.sort()
    s0 = glob.glob(seg_name + '*')
    s0.sort()
    compare_name = x.replace(x.split('/')[-2], 'compare-' + x.split('/')[-2])
    if not os.path.isdir(compare_name):
        os.mkdir(compare_name)

    for i in range(len(x0)):
        a = np.array(Image.open(x0[i]))
        if i == 0:
            amax = 10000#a.max()
        a = a / amax
        b = np.array(Image.open(s0[i]))

        print(x0[i])
        print(s0[i])

        imagesc(np.concatenate([a, b], 1), show=False, save=compare_name + str(i).zfill(3) + '.png')


def npy_to_tiffs(npy, destination, intensity=40, subsampling=1):
    if not os.path.isdir(destination):
        os.mkdir(destination)
    for i in range(npy.shape[0]):
        x = 1 * npy[i, ::subsampling, ::subsampling]
        x = x.astype(np.uint16)
        x = intensity * x
        tiff.imsave(destination + str(i).zfill(3) + '.tif', x)


def tiff_to_slices(tiffnpy, destination, fmt='.tif', subsampling=1):
    if not os.path.isdir(destination):
        os.mkdir(destination)
    for i in range(tiffnpy.shape[0]):
        x = 1 * tiffnpy[i, ::subsampling, ::subsampling]
        if fmt == '.tif':
            tiff.imsave(destination + str(i).zfill(3) + '.tif', x)
        elif fmt == '.png':
            imagesc(x, show=False, save=destination + str(i).zfill(3) + '.png')


def upsample_to_square(x, size, mode):
    x = nn.Upsample(size=size, mode=mode)(torch.from_numpy(x/1).unsqueeze(0).unsqueeze(0))
    return x.squeeze().numpy().astype(np.uint16)


def npy_to_vtk(npy, vtkname, subsample=(1, 1, 1)):
    vtk = pv.wrap(npy[::subsample[0], ::subsample[1], ::subsample[2]])
    # vtk.origin = [(401 - 120) * xyz[2] / rate, (512 - 75) * xyz[0] / rate,
    #              (512 - 75) * xyz[1] / rate]
    vtk.origin = [0, 0, 0]
    vtk.spacing = [1, 1, 1]
    # [coordinates[, coordinates[0] / downsample_rate, coordinates[1] / downsample_rate]
    vtk.save(vtkname)


def load_imgs_to_stack(source, resize=False):
    l = sorted(glob.glob(source + '*'))

    x = []  # segmented
    for i in l:
        s = Image.open(i)
        if resize:
            s = s.resize(resize, resample=Image.BILINEAR)
        x.append(np.expand_dims(np.array(s), 0))
    x = np.concatenate(x, 0)
    return x


def seg_to_skel(x, dim):
    if dim == 2:
        x2d = 0 * x
        for i in range(x.shape[0]):
            x2d[i, :, :] = skeletonize(x[i, :, :] > 0)
        x2d = x2d.astype(np.uint8)
        return x2d
    if dim == 3:
        x3d = 0 * x
        for i in range(1, x.shape[0]-1):
            x3d[i, :, :] = skeletonize_3d(x[i-1:i+2, :, :])[1, :, :]
        x3d = x3d // x3d.max()
        x3d = x3d.astype(np.uint8)
        return x3d


def print_skel_overlap(x, ori, destination):
    overlap = []
    for i in range(x.shape[0]):
        if not os.path.isdir(destination):
            os.mkdir(destination)
        a = Image.fromarray(x[i, :, :])
        #a = a.resize((1024, 1024), resample=Image.BILINEAR)
        a = np.array(a)
        a = a / a.max()
        #a = (a > 0) / 1
        b = ori[i, :, :]
        c = np.zeros((ori.shape[1], ori.shape[2], 3))
        c[:, :, 0] = a#np.multiply(b, a)
        c[:, :, 1] = np.multiply(b, 1 - a)
        c[:, :, 2] = np.multiply(b, 1 - a)
        #c[0, 0, 0] = ori.max()
        imagesc(c, show=False, save=destination + str(i).zfill(3) + '.png')
        overlap.append(np.expand_dims(c, 0))
    return np.concatenate(overlap, 0)



def sample_random_cubes(x, locator, locator_val_min, random_range, cropsize, n_cubes, destination):
    n = 0
    while n <= n_cubes:
        z0 = np.random.randint(*random_range[0])
        x0 = np.random.randint(*random_range[1])
        y0 = np.random.randint(*random_range[2])

        a = np.squeeze(x[z0:z0+cropsize[0], x0:x0+cropsize[1], y0:y0+cropsize[2]])
        locator_val = locator[z0+cropsize[0], x0:x0+cropsize[1], y0:y0+cropsize[2]]
        b = upsample_to_square(a[::8, :], size=(max(a.shape), max(a.shape)), mode='bicubic')
        print(locator_val.mean())
        if locator_val.mean() >= locator_val_min:
            imagesc(a, show=False, save=destination + 'a/' + str(n) + '.png')
            imagesc(b, show=False, save=destination + 'b/' + str(n) + '.png')
            n = n + 1


file_list = sorted(glob.glob('/media/ghc/GHc_data1/BRC/64246_VMAT_complete_560/*.tif'))
ori = io.imread(file_list[3])
seg = load_imgs_to_stack('/media/ghc/GHc_data1/BRC/processed/seg-ori_3_x1/', resize=(2048, 2048))
#skel = seg_to_skel((seg > 0.3)/1, dim=3)
overlap = print_skel_overlap(seg, ori[:, ::1, ::1], '/media/ghc/GHc_data1/BRC/brc_segmentation/overlap_3_seg/')

#
ori = io.imread(file_list[4])
ori[ori>=5000] = 5000
ori[:, 0, 0] = 5000
ori = ori/ori.max()

seg = load_imgs_to_stack('/media/ghc/GHc_data1/BRC/64246_VMAT_complete_560/3_cropped_x0.5_deconv/')
seg = seg/seg.max()
print_skel_overlap(seg, ori, '/media/ghc/GHc_data1/BRC/64246_VMAT_complete_560/3_cropped_x0.5_deconv_overlap/')





ori = ori / ori.max()
skel = skel / skel.max()
for i in range(ori.shape[0]):
    ori_rgb = np.concatenate([np.expand_dims(x, 2) for x in [ori[i, ::2, ::2]]] * 3, 2)
    compare = np.concatenate([ori_rgb, overlap[i, :, :, :]], 1)
    imagesc(compare, show=False, save= '/media/ghc/GHc_data1/BRC/brc_segmentation/compare_0/' + str(i).zfill(3) + '.png')

if 0:
    # subsample on the x-y plane
    sample_random_cubes(x=ori,
                        locator=(seg > 0.3)/1, locator_val_min=0.1,
                        random_range=[(0, 200), (0, 1024), (0, 1024)],
                        cropsize=[1, 256, 256],
                        n_cubes=500, destination='/media/ghc/GHc_data1/paired_images/flybrain/test/')

if 0:
    # subsample from the sides
    sample_random_cubes(x=ori,
                        locator=(seg > 0.3)/1, locator_val_min=0.2,
                        random_range=[(0, 200), (0, 1024), (0, 1024)],
                        cropsize=[64, 1, 256],
                        n_cubes=100, destination='/media/ghc/GHc_data1/paired_images/flybrain/test/')

if 0:
    x = ori
    for i in range(0, x.shape[1]):
        imagesc(upsample_to_square(x[-256:, i, :], size=(2048, 2048), mode='bicubic'),
                show=False, save='/media/ghc/GHc_data1/paired_images/flybrain/test/side1/' + str(i).zfill(3) + '.png')


if 0:
    tiff_to_slices(tiffnpy=tiffnpy,
                   destination='/media/ghc/GHc_data1/BRC/processed/ori_0_x4/',
                   fmt='.png',
                   subsampling=4)



