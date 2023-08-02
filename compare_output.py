import glob
import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

do = 'compare seg'

if do == 'compare output':
    prj = ['0726_oai3d_2_cut_resnet', '0729_cut_3enc']
    ep = ['40', '50']

    for p, e in zip(prj, ep):
        img = tiff.imread(f'/home/glory/projects/230510_GAN/out/{p}/ep{e}_out.tif') #B,C,Y,X,Z
        img_a = img[0, 0, ::30, :, :]
        img_s = img.transpose(0, 1, 4, 2, 3)[0, 0, ::25, :, 32:328]
        img_c = img.transpose(0, 1, 3, 2, 4)[0, 0, ::30, :, :]
        print(img_a.shape, img_s.shape, img_c.shape)
        img = np.concatenate([img_s, img_a, img_c], axis=1)
        img = np.concatenate([img[i, :, :] for i in range(2,10)], axis=1)
        plt.imsave(f'/home/glory/projects/230510_GAN/out/{p}.png', img, cmap='gray')

if do == 'compare seg':
    prj = '0728_cyc_oai3d_coherant_ep50'
    img = tiff.imread(glob.glob(f'/home/glory/projects/230510_GAN/inference_seg/{prj}/9464295_00_RIGHT/Saggital*')[0])
    print(img.shape)
    img = img[5:35:5, :, :]
    print(img.shape)
    img = np.concatenate([img[i, 0, :, :] for i in range(6)], axis=1)
    plt.imsave(f'/home/glory/projects/230510_GAN/out/{prj}_seg.png', img, cmap='gray')



