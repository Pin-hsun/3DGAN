import os, glob
import tifffile as tiff
from utils.data_utils import imagesc
import torch


def to_cubes(images, destination, dx=256, criteria=0.0, save_names=None):
    file_names = list(images.keys())
    if save_names is None:
        save_names = file_names

    Z, H, W = images[list(images.keys())[0]].shape
    print((Z, H, W))

    # open top folder
    os.makedirs(destination, exist_ok=True)

    # open sub folder
    for name in save_names:
        os.makedirs(destination + name + '/', exist_ok=True)

    for s in range(Z // dx):
        for i in range(H // dx):
            for j in range(W // dx):
                name = file_names[0]
                patch = images[name][s * dx:(s + 1) * dx, i * dx:(i + 1) * dx, j * dx:(j + 1) * dx]
                avg = (patch > 0).mean()
                if avg >= criteria:
                    tiff.imsave(destination + name + '/' + str(s) + '_' + str(i) + '_' + str(j)
                                          + '_' + "{:.2f}".format(avg) + '.tif', patch)

                    for k in range(1, len(file_names)):
                        name = file_names[k]
                        patch = images[name][s * dx:(s + 1) * dx, i * dx:(i + 1) * dx, j * dx:(j + 1) * dx]
                        tiff.imsave(destination + name + '/' + str(s) + '_' + str(i) + '_' + str(j)
                                              + '_' + "{:.2f}".format(avg) + '.tif', patch)


def to_patches(images, destination, dx=256, criteria=0.0, save_names=None):
    file_names = list(images.keys())
    if save_names is None:
        save_names = file_names

    Z, H, W = images[list(images.keys())[0]].shape
    print((Z, H, W))

    # open top folder
    os.makedirs(destination, exist_ok=True)

    # open sub folder
    for name in save_names:
        os.makedirs(destination + name + '/', exist_ok=True)

    for s in range(Z):
        for i in range(H // dx):
            for j in range(W // dx):
                name = file_names[0]
                patch = images[name][s, i * dx:(i + 1) * dx, j * dx:(j + 1) * dx]
                avg = (patch > 0).mean()
                if avg >= criteria:
                    tiff.imsave(destination + name + '/' + str(i) + '_' + str(j) + '_' + str(s)
                                          + '_' + "{:.2f}".format(avg) + '.tif', patch)

                    for k in range(1, len(file_names)):
                        name = file_names[k]
                        patch = images[name][s, i * dx:(i + 1) * dx, j * dx:(j + 1) * dx]
                        tiff.imsave(destination + name + '/' + str(i) + '_' + str(j) + '_' + str(s)
                                              + '_' + "{:.2f}".format(avg) + '.tif', patch)



def crop_tiff_by_range(old, crop):
    new = dict()
    for k in list(old.keys()):
        new[k] = old[k][crop, :, :]
    return new


if __name__ == '__main__':
    if 1:
        """
        images: dictionary consisted of images that will be broken into patches in the same time.
        Criteria will be applied based on the first image
        """
        root = '/media/ExtHDD01/Dataset/paired_images/FlyZ/'
        #images = dict([(x, tiff.imread(root + x + '.tif')) for x in ['xyweak', 'xyori', 'xyorisb', 'xyori8bit']])
        images = dict([(x, tiff.imread(root + x + '.tif')) for x in ['zyweak', 'zyori', 'zyorisb']])
        Z = images[list(images.keys())[0]].shape[0]

        to_patches(images=crop_tiff_by_range(images, range(0, (Z // 4 * 3))),
                   destination=root + 'train/', dx=256, criteria=0.05)

        to_patches(images=crop_tiff_by_range(images, range((Z // 4 * 3), Z)),
                   destination=root + 'test/', dx=256, criteria=0.05)



        #images = dict([(x, tiff.imread(root + x + '.tif')) for x in ['zyweak', 'zyori']])
        #to_cubes(images=images, destination=root + 'train/', dx=256, criteria=0.00)

    if 0:
        root = '/media/ExtHDD01/BRC/64246_VMAT_complete_560/used_for_getnet/'
        scale = 10
        file_names = ['original', 'deconv/' + str(scale).zfill(2)]
        save_names = ['original/' + str(scale).zfill(2) + '_', 'deconv/' + str(scale).zfill(2) + '_']
        images = dict([(x, io.imread(root + x + '.tif')) for x in file_names])
        Z = images[list(images.keys())[0]].shape[0]
        #to_patches(images=images, zrange=range(0, Z // 4 * 3), destination=root + 'train/', dx=256, criteria=0)
        to_patches(images=images, zrange=range(Z // 4 * 3, Z), destination=root + 'test/',
                   dx=256, criteria=0, save_names=save_names)


self.seg = torch.load('submodels/model_seg_ZIB_res18_256.pth').cuda()
self.seg.eval()

train_set = Dataset(root=os.environ.get('DATASET') + opt.dataset + '/train/',
                    path='badKL3afterreg',
                    opt=opt, mode='train')

#images = dict([(x, io.imread(root + x + '.tif')) for x in ['weakzy', 'orizy']])
#to_patches(images, root, dx=256, criteria=0.2)