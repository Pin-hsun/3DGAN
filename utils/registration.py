import glob, os
import cv2
import numpy as np
import tifffile as tiff
import torchvision.transforms as transforms
import torch.nn as nn
import torch
from utils.data_utils import imagesc
import scipy.ndimage

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
    netg_bone = torch.load('submodels/model_seg_ZIB.pth')
    netg_bone.eval()
    netg_t2d = torch.load('submodels/tse_dess_unet32.pth')
    if 0:  # t2d
        list_a = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/t2d/full/t/*'))[1::2]
        list_b = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/t2d/full/d/*'))[1::2]

    root  = '/media/ExtHDD01/Dataset/paired_images/womac3/full/'
    list_a = sorted(glob.glob(root + 'b/*'))
    list_b = sorted(glob.glob(root + 'a/*'))
    list_breg = sorted(glob.glob(root + 'breg/*'))

    old_name = '/b/'
    new_name = '/breg/'

    warp_mode = cv2.MOTION_EUCLIDEAN

    failed = sorted(set([x.split('/')[-1] for x in list_a]) - set([x.split('/')[-1] for x in list_breg]))

    list_s = [root + 'b/' + x for x in failed]
    list_t = [root + 'a/' + x for x in failed]

    for i in range(len(list_s)):
        name_s = list_s[i]
        name_t = list_t[i]
        #i = 26
        #name_s, name_t = list(zip(list_s, list_t))[i]

        s = tiff.imread(name_s)
        t = tiff.imread(name_t)

        smax = s.max()

        s = s / smax
        t = t / smax

        s = s.astype(np.float32)
        t = t.astype(np.float32)

        try:
            sc = s.copy()
            tc = t.copy()

            bprime, warp_matrix = linear_registration(im1=tc, im2=sc, warp_mode=warp_mode, steps=500)
            sprime = apply_warp(s.shape, s, warp_matrix, warp_mode)
            sprime = (sprime * smax).astype(np.uint16)

            tiff.imsave(name_s.replace(old_name, new_name), sprime)
            z = quick_compare(sprime, t)
            imagesc(z, show=False, save=name_s.replace(old_name, '/check/'))

        except:
            sc = s.copy()
            tc = t.copy()

            sc = 2 * sc - 1
            tc = 2 * tc - 1

            sc = transforms.Normalize((0.5), (0.5))(torch.from_numpy(sc).unsqueeze(0).unsqueeze(0))
            sc = netg_t2d(sc.repeat(1, 3, 1, 1).cuda())
            sc = (torch.argmax(netg_bone(sc[0])[0], 0),)
            sc = sc[0].detach().cpu().numpy()
            sc = sc.astype(np.uint8)
            # sc = sc[0, 0, ::]

            tc = transforms.Normalize((0.5), (0.5))(torch.from_numpy(tc).unsqueeze(0).unsqueeze(0))
            tc = netg_t2d(tc.repeat(1, 3, 1, 1).cuda())
            tc = (torch.argmax(netg_bone(tc[0])[0], 0),)
            tc = tc[0].detach().cpu().numpy()
            tc = tc.astype(np.uint8)
            try:
                bprime, warp_matrix = linear_registration(im1=tc, im2=sc, warp_mode=warp_mode, steps=1000)
                sprime = apply_warp(s.shape, s, warp_matrix, warp_mode)
                sprime = (sprime * smax).astype(np.uint16)

                tiff.imsave(name_s.replace(old_name, new_name), sprime)
                z = quick_compare(sprime, t)
                imagesc(z, show=False, save=name_s.replace(old_name, '/check2/'))
            except:
                # center of mass
                com_tc = scipy.ndimage.measurements.center_of_mass(tc)
                com_sc = scipy.ndimage.measurements.center_of_mass(sc)
                warp_matrix = np.array([[1,0, com_sc[1]-com_tc[1]],[0,1, com_sc[0]-com_tc[0]]])
                sprime = apply_warp(s.shape, s, warp_matrix, warp_mode)
                sprime = (sprime * smax).astype(np.uint16)
                z = quick_compare(sprime, t)
                imagesc(z, show=False, save=name_s.replace(old_name, '/check2/'))
                tiff.imsave(name_s.replace(old_name, new_name), sprime)


