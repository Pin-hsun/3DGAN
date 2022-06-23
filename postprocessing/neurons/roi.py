import tifffile as tiff
import numpy as np
from utils.data_utils import imagesc

roi = []
roi.append([90, 276, 285, 454, 466])
roi.append([1023, 8, 174, 194, 370])
roi.append([964, 82, 320, 234,  476])
roi.append([853, 50, 101, 198, 253])
roi.append([755, 264, 184, 429, 350])

root = 'forpaper/'
tiflist = ['oroi', 'wm', 'wv', 'wroilabel', 'wmlabel']

for idx in range(5):
    to_print = []
    for t in tiflist:
        tif = tiff.imread(root + t + '.tif')
        if t == 'wroilabel':
            tif[tif == 2] = 1
            tif[tif > 2] = 2
        if t == 'wmlabel':
            tif[tif > 1] = 2

        patch = tif[roi[idx][0], roi[idx][2]:roi[idx][4], roi[idx][1]:roi[idx][3]]
        patch = patch - patch.min()
        patch = patch / patch.max()
        to_print.append(patch)
    to_print = np.concatenate(to_print, 1)
    imagesc(to_print, show=False, save=root+'skel' + str(idx) + '.png')