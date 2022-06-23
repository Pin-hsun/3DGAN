import glob, os
import cv2
import numpy as np
import tifffile as tiff
import torchvision.transforms as transforms
import torch.nn as nn
import torch
from utils.data_utils import imagesc

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


if __name__ == '__main__':

    netg_t2d = torch.load('submodels/tse_dess_unet32.pth')
    netg_t2d.eval()

    list_a = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/t2d/full/t/*'))[1::2]
    list_b = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/t2d/full/d/*'))[1::2]

    warp_mode = cv2.MOTION_HOMOGRAPHY

    for (name_a, name_b) in zip(list_a, list_b):
        print(name_a)
        try:
            a = tiff.imread(name_a)
            b = tiff.imread(name_b)

            amax = a.max()

            a = a / a.max()
            b = b / b.max()

            a = a.astype(np.float32)
            b = b.astype(np.float32)

            x = 1 * a
            x = transforms.Normalize((0.5), (0.5))(torch.from_numpy(x).unsqueeze(0).unsqueeze(0))
            x = netg_t2d(x.repeat(1, 3, 1, 1).cuda())
            x = x[0].detach().cpu().numpy()
            x = x[0, 0, ::]

            warp_mode = cv2.MOTION_HOMOGRAPHY
            bprime, warp_matrix = linear_registration(im1=b, im2=x, warp_mode=warp_mode, steps=500)

            aprime = apply_warp(a.shape, a, warp_matrix, warp_mode)
            aprime = (aprime * amax).astype(np.uint16)

            tiff.imsave(name_a.replace('/t/', '/tres/'), aprime)
            z = quick_compare(aprime, b)
            imagesc(z, show=False, save=name_a.replace('/t/', '/check/'))

        except:
            print(name_a + '  failed')

    if 0:
        for i in range(19):
            num_of_slice = meta.loc[(meta['ID'] == ID_list[i]) & (meta['sequences'] == 'TSE')].shape[0]
            for s in range(num_of_slice):
                warp_mode = cv2.MOTION_HOMOGRAPHY
                ###################################################
                # Add the function where you auto-match a DESS slice
                ###################################################
                t, d = quick_compare(i, s, npy_folder, show=False)
                t_aligned, ssim0, ssim1 = linear_registration(im1=d, im2=t, warp_mode=warp_mode, show=False)
                savename = ('outputs/registration/' + ID_list[i] + '_' + str(s) + '_{:.3f}_{:.3f}.jpg').format(ssim0, ssim1)
                imagesc([d, make_compare(d, t_aligned), t_aligned, t], show=False, save=savename)