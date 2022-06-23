import os, glob
import pandas as pd
import tifffile as tiff
import numpy as np

def womac3_with_moaks_tif():
    df = pd.read_csv('/home/ghc/Dropbox/TheSource/scripts/OAI_pipelines/meta/subjects_unipain_womac3.csv')
    df = df.loc[df['has_moaks']]

    source = '/media/ExtHDD01/Dataset/paired_images/womac3/full/b/'
    destination = '/media/ExtHDD01/Dataset/paired_images/womac3/full/b_moaks/'

    for i in range(df.shape[0]):
        ID = df.iloc[i]['ID']
        SIDE = df.iloc[i]['SIDE']
        tifs = sorted(glob.glob(source + str(ID) + '*.tif'))
        npys = []
        for t in tifs:
            npys.append(np.expand_dims(tiff.imread(t), 0))
        npys = np.concatenate(npys, 0)
        tiff.imsave(destination + str(ID) + '_' + SIDE + '.tif', npys)