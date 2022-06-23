import os, glob
from shutil import copyfile

root = '/media/ghc/GHc_data1/paired_images/pain/train/'
ys = sorted(glob.glob(root + 'beff1/*'))

for y in ys:
    name = y.split('/')[-1]
    copyfile(root+'aregis1eff/'+name, root+'aregis1eff1/'+name)
