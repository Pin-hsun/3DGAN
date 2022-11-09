import glob, os
import cv2
import numpy as np
import tifffile as tiff
import torchvision.transforms as transforms
import torch.nn as nn
import torch
from utils.data_utils import imagesc
import scipy.ndimage
import pandas as pd

def quick_compare(x0, y0):
    x = 1 * x0
    y = 1 * y0
    x = x / x.max()
    y = y / y.max()
    z = np.concatenate([np.expand_dims(x, 2)] * 3, 2)
    z[:, :, 1] = y
    return z


def linear_registration(im1, im2, warp_mode, steps):
    # try registration using open CV
    # use cv2.findTransformECC

    #im1 = to_8bit(im1)[:, :, 0]#cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    #im2 = to_8bit(im2)[:, :, 0]#cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    if warp_mode == 3:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    number_of_iterations = steps
    termination_eps = 1e-10
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    (cc, warp_matrix) = cv2.findTransformECC(im1, im2, warp_matrix, warp_mode, criteria=criteria)

    sz = im1.shape
    im2_aligned = apply_warp(sz, im2, warp_matrix, warp_mode)
    return im2_aligned, warp_matrix


def apply_warp(sz, im2, warp_matrix, warp_mode):
    if warp_mode == 3:
        im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return im2_aligned


def quick_compare_by_subject():
    root = '/media/ExtHDD01/Dataset/paired_images/t2d/full/'
    list_a = sorted(glob.glob(root + 'tres/*'))

    subjects = sorted(list(set([x.split('/')[-1].split('_')[0] for x in list_a])))
    for sub in subjects:
        slice_a = sorted(glob.glob(root + 'tres/' + sub + '*'))

        all_z = []
        for a in slice_a:
            atif = tiff.imread(a)
            btif = tiff.imread(a.replace('/tres/', '/d/'))
            z = np.concatenate([np.expand_dims(atif, 2)] * 3, 2)
            z[:, :, 1] = btif
            all_z.append(np.expand_dims(z, 3))

        all_z += [0 * all_z[-1]] * (24 - len(slice_a))
        all_z = np.concatenate(all_z, 3)
        all_z = np.reshape(all_z, (384, 384, 3, 6, 4))

        all = []
        for i in range(4):
            row = []
            for j in range(6):
                row.append(all_z[:, :, :, j, i])
            row = np.concatenate(row, 1)
            all.append(row)
        all = np.concatenate(all, 0)
        #tiff.imsave(root + 'check0/' + sub + '.tif', all)
        imagesc(all, show=False, save=root + 'check0/' + sub + '.png')


from scipy import interpolate

def matrix_interp(m):
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            f = interpolate.interp1d(list(range(23)), m[i, j, :], kind='quadratic')
            m[i, j, :] = f(list(range(23)))
    return m



if __name__ == '__main__':
    #warp_mode = cv2.MOTION_AFFINE#
    warp_mode = cv2.MOTION_EUCLIDEAN

    # data
    root = '/media/ExtHDD01/Dataset/paired_images/womac4/full/'
    list_a = sorted(glob.glob(root + 'b/*'))
    list_b = sorted(glob.glob(root + 'a/*'))

    list_a = np.reshape(np.array(list_a), (-1, 23))
    list_b = np.reshape(np.array(list_b), (-1, 23))

    # labels
    x = pd.read_csv('/home/ghc/Dropbox/TheSource/scripts/OAI_API/meta/womac4min0.csv')
    labels = x.loc[x['SIDE'] == 'RIGHT']['V$$WOMKPR'] > x.loc[x['SIDE'] == 'RIGHT']['V$$WOMKPL']
    labels = np.array(labels)
    #labels = [(x for x in labels]
    #labels = np.concatenate([[x[0]] * 23 for x in labels], 0)

    #
    os.makedirs(os.path.join(root, 'ap'), exist_ok=True)
    os.makedirs(os.path.join(root, 'bp'), exist_ok=True)
    os.makedirs(os.path.join(root, 'check'), exist_ok=True)

    for i in range(10):#list_a.shape[0])[:1]:
        all_warp_matrix = []
        for j in range(23):
            print([i, j])
            if labels[i] == 1:  # PAIN RIGHT
                name_s = list_b[i, j]  # LEFT
                name_t = list_a[i, j]  # RIGHT
            else:  # PAIN LEFT
                name_s = list_a[i, j]  # LEFT
                name_t = list_b[i, j]  # RIGHT

            s = tiff.imread(name_s)
            t = tiff.imread(name_t)

            smax = 1#s.max()
            s = s / smax
            t = t / smax
            s = s.astype(np.float32)
            t = t.astype(np.float32)
            sc = s.copy()
            tc = t.copy()

            try:
                _, warp_matrix = linear_registration(im1=tc, im2=sc, warp_mode=warp_mode, steps=500)
            except:
                if j == 0:
                    warp_matrix = np.array([[1, 0, 0], [0, 1, 0]])
                else:
                    warp_matrix = all_warp_matrix[-1]
            all_warp_matrix.append(warp_matrix)

        all_warp_matrix = np.stack(all_warp_matrix, 2)
        all_warp_matrix = matrix_interp(all_warp_matrix)

        for j in range(23):
            if labels[i] == 1:  # PAIN RIGHT
                name_s = list_b[i, j]  # LEFT
                name_t = list_a[i, j]  # RIGHT
            else:  # PAIN LEFT
                name_s = list_a[i, j]  # LEFT
                name_t = list_b[i, j]  # RIGHT

            s = tiff.imread(name_s)
            t = tiff.imread(name_t)

            smax = 1#s.max()
            s = s / smax
            t = t / t.max()
            s = s.astype(np.float32)
            t = t.astype(np.float32)

            sprime = apply_warp(s.shape, s, all_warp_matrix[:, :, j], warp_mode)
            sprime = (sprime * smax).astype(np.uint16)

            name = list_a[i, j].split('/')[-1]

            if labels[i] == 1:  # PAIN RIGHT
                tiff.imsave(os.path.join(root, 'ap', name), tiff.imread(list_a[i, j]))  # RIGHT TARGET a
                tiff.imsave(os.path.join(root, 'bp', name), sprime)  # LEFT SOURCE b
            else:  # PAIN LEFT
                tiff.imsave(os.path.join(root, 'bp', name), tiff.imread(list_b[i, j]))  # RIGHT TARGET b
                tiff.imsave(os.path.join(root, 'ap', name), sprime)  # LEFT SOURCE a

            z = quick_compare(sprime, t)
            imagesc(z, show=False, save=os.path.join(root, 'check', name))


