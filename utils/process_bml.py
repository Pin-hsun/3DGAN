import numpy as np
import tifffile as tiff
import os, glob
from scipy import stats
import pandas as pd
from utils.data_utils import imagesc
from PIL import Image

def old_shit():
    aa = sorted(glob.glob('/media/ExtHDD01/logs/segA0/*'))
    x = []
    for a in aa:
        x.append(np.expand_dims(np.array(tiff.imread(a))[:, :], 2))
    x = np.concatenate(x, 2)
    s = stats.tstd(x, axis=2)
    m = np.mean(x, 2)
    d = np.divide(m, s)
    d[np.isnan(d)] = 0
    np.save('d1.npy', d)

    # pain score
    df = pd.read_csv('/media/ExtHDD01/OAI/OAI_extracted/OAI00womac3/OAI00womac3.csv')
    labels = [(x,) for x in df.loc[df['SIDE'] == 1, 'P01KPN#EV'].astype(np.int8)]

    PR = df.loc[df['SIDE'] == 1, ['V00WOMKP#']].values[497:, 0]
    PL = df.loc[df['SIDE'] == 2, ['V00WOMKP#']].values[497:, 0]
    PD = np.abs(PR-PL)

    # bml image
    #source = '/media/ExtHDD01/Dataset/paired_images/womac3/test/'
    source = 'outputs/results/'
    alist = sorted(glob.glob(source + 'seg1/*'))
    blist = sorted(glob.glob(source + 'seg1b/*'))

    abml = []
    a3d = []
    for a in alist:
        a = tiff.imread(a)
        #a[a < 0.5] = 0
        a = (a >= 0.5)
        abml.append(a.sum())
        a3d.append(np.expand_dims(a, 2))

    bbml = []
    b3d = []
    for b in blist:
        b = tiff.imread(b)
        #b[b < 0] = 0
        b = (b >= 0.5)
        bbml.append(b.sum())
        b3d.append(np.expand_dims(b, 2))

    abml = np.array(abml)
    bbml = np.array(bbml)

    bbml = np.reshape(bbml, (4899//23, 23)).sum(1)
    abml = np.reshape(abml, (4899//23, 23)).sum(1)

    print(stats.ttest_rel(abml, bbml))

    if 1:
        a33d = np.concatenate(a3d, 2)
        b33d = np.concatenate(b3d, 2)
        a33d = np.reshape(a33d, (384, 384, 4899//23, 23))
        b33d = np.reshape(b33d, (384, 384, 4899//23, 23))

        np.save('abml', abml)
        np.save('abml', abml)
        np.save('bbml', bbml)
        np.save('PR', PR)
        np.save('PL', PL)
        np.save('PD', PD)

def get_significant(x0, y0, xt, yt):
    x = 1 * x0
    x[x >= xt] = xt
    y = 1 * y0
    y = (y >= yt) / 1
    z = np.multiply(x, y)
    z = z / z.max()
    #z = cm(z)
    return z

def compareab():

    xx = ['a_0', 'a_1', 'a_2', 'b_0', 'b_1', 'b_2', 'c_0', 'c_1', 'c_2', 'd_0', 'd_1', 'd_2', 'e_0', 'e_1', 'e_2']
    xx = np.reshape(np.array(xx), (5, 3))

    source = '/media/ExtHDD01/Dataset/paired_images/womac3/full/'
    abml0 = np.load(source+'abml.npy')
    bbml0 = np.load(source+'bbml.npy')
    aeff0 = np.load(source+'aeff.npy')
    beff0 = np.load(source+'beff.npy')

    abml = abml0.sum(0).sum(0)
    bbml = bbml0.sum(0).sum(0)
    aeff = (aeff0).sum(0).sum(0)
    beff = (beff0).sum(0).sum(0)

    [abml, bbml, aeff, beff] = [x.sum(0).sum(0) for x in [abml, bbml, aeff, beff]]
    [abml, bbml, aeff, beff] = [np.reshape(x, (710, 23)) for x in [abml, bbml, aeff, beff]]
    [abml, bbml, aeff, beff] = [x.sum(1) for x in [abml, bbml, aeff, beff]]

    df = pd.read_csv('/home/ghc/Dropbox/TheSource/scripts/OAI_pipelines/meta/subjects_unipain_womac3.csv')
    pdiff = (df['V00WOMKPR'] - df['V00WOMKPL']).abs()
    [q0, q1, q2, q3, q4] = np.quantile(pdiff, [0, 0.25, 0.5, 0.75, 1])

    if 0:
        paina = df[['V00WOMKPR', 'V00WOMKPL']].max(1).astype(np.float16)
        painb = df[['V00WOMKPR', 'V00WOMKPL']].min(1).astype(np.float16)
        pdiff = np.divide(painb, paina)
        [q0, q1, q2, q3, q4] = np.quantile(pdiff, [0, 0.25, 0.5, 0.75, 1])

    atar = aeff
    btar = beff

    [qa, qb, qc] = [q0, q1, q2]

    ta = (np.divide(btar[(pdiff > qb) & (pdiff <= qc)], atar[(pdiff > qb) & (pdiff <= qc)]))
    tb = (np.divide(btar[(pdiff > qa) & (pdiff <= qb)], atar[(pdiff > qa) & (pdiff <= qb)]))

    #ta = atar[(pdiff > qb) & (pdiff <= qc)] - btar[(pdiff > qb) & (pdiff <= qc)]
    #tb = atar[(pdiff > qa) & (pdiff <= qb)] - btar[(pdiff > qa) & (pdiff <= qb)]

    print(stats.ttest_ind(ta, tb))

def read_tif(t):
    x = tiff.imread(t)
    m0 = x[:, 0 * dx:1 * dx]
    m1 = x[:, 1 * dx:2 * dx]
    d0 = x[:, 2 * dx:3 * dx]
    d1 = x[:, 3 * dx:4 * dx]
    u0 = x[:, 4 * dx:5 * dx]
    u1 = x[:, 5 * dx:6 * dx]
    z0 = get_significant(d0, u0, xt=0.2, yt=0.2)[:, :]
    z1 = get_significant(d1, u1, xt=0.9, yt=0.5)[:, :]
    m0 = np.concatenate([np.expand_dims(m0, 2)]*3, 2)
    d0 = np.concatenate([np.expand_dims(d0, 2)]*3, 2)
    d1 = np.concatenate([np.expand_dims(d1, 2)]*3, 2)
    z = np.concatenate([m0, d0, d1, cm(z0)[:, :, :3], cm(z1)[:, :, :3]], 1)
    #z = np.concatenate([z0, z1], 1)
    return z

import matplotlib.pyplot as plt
cm = plt.get_cmap('viridis')

#source = '/home/ghc/Dropbox/TheSource/scripts/lightning_pix2pix/outputs/results/womac3/mcfix/NSsegsobel/Gdescarsmc_index2/'
#source = '/home/ghc/Dropbox/TheSource/scripts/lightning_pix2pix/outputs/results/womac3/mcfix/descar2/Gdescarsmc_index2_100/'
source = '/media/ExtHDD01/Dataset/paired_images/womac3/full/beff/'
destination = '/media/ExtHDD01/Dataset/paired_images/womac3/full/beff.npy'
tifs = sorted(glob.glob(source + '*'))
from utils.data_utils import imagesc

dx = 384
all_z = []
for i in range(len(tifs))[:]:
    z = tiff.imread(tifs[i])
    all_z.append(np.expand_dims(z, 2))
all_z = np.concatenate(all_z, 2)
np.save(destination, all_z)