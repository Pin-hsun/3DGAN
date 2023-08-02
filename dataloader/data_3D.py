from os.path import join
import glob
import random
import torch
from torch import nn
import torch.utils.data as data
import torchvision.transforms as transforms
import tifffile as tiff
from PIL import Image
import numpy as np
import os
from skimage import io
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import pandas as pd
import scipy
import torchio as tio
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def to_8bit(x):
    if type(x) == torch.Tensor:
        x = (x / x.max() * 255).numpy().astype(np.uint8)
    else:
        x = (x / x.max() * 255).astype(np.uint8)

    if len(x.shape) == 2:
        x = np.concatenate([np.expand_dims(x, 2)]*3, 2)
    return x


def imagesc(x, show=True, save=None):
    # switch
    if (len(x.shape) == 3) & (x.shape[0] == 3):
        x = np.transpose(x, (1, 2, 0))

    if isinstance(x, list):
        x = [to_8bit(y) for y in x]
        x = np.concatenate(x, 1)
        x = Image.fromarray(x)
    else:
        x = x - x.min()
        x = Image.fromarray(to_8bit(x))
    if show:
        x.show()
    if save:
        x.save(save)

def normalize_image11(x):
    min_val = torch.min(x)
    max_val = torch.max(x)
    image_range = max_val - min_val

    normalized_image = (x - min_val) * 2 / image_range - 1

    return normalized_image

def crop(x, size=256, rand=True): #torch.Size([1, 1, 318, 444, 444])
    if rand:
        a = random.randint(0, x.shape[2] - size)
        b = random.randint(0, x.shape[3] - size)
        c = random.randint(0, x.shape[4] - size)
    else:
        a = int((x.shape[2] - size) / 2)
        b = int((x.shape[3] - size) / 2)
        c = int((x.shape[4] - size) / 2)
    out = x[0, :, a:a+size, b:b+size, c:c+size] #torch.Size([3, 256, 256, 256])
    return out

def interpolation(img_cube, N, mode, crop_size):
    m = nn.Upsample(scale_factor=(N, 1, 1), mode='trilinear')
    input = img_cube.unsqueeze(0).transpose(1,2) #torch.Size([1, 1, 37, 444, 444])
    out = m(input) #torch.Size([1, 1, 318, 444, 444])
    # transformation
    if mode == 'train':
        out = crop(out, size=crop_size, rand=True)
    if mode == 'test':
        out = crop(out, size=crop_size, rand=False)
    out = normalize_image11(out)

    return out

def slice_cube(tensor, N, size, start=5, zy=False):
    step_size = int(size / N)
    print(step_size)

    selected_slices = []
    # Iterate through the tensor and select slices
    if zy:
        for i in range(start, size-start, step_size):
            selected_slices.append(tensor[:, :, i, :])
    else:
        for i in range(start, size-start, step_size):
            selected_slices.append(tensor[:, i, :, :])

    return selected_slices

class MultiData(data.Dataset):
    """
    Multiple unpaired data ccombined
    """
    def __init__(self, root, path, opt, mode, paired=False, transforms=None, filenames=False, index=None):
        super(MultiData, self).__init__()
        self.opt = opt
        self.mode = mode
        self.filenames = filenames
        self.subset = []
        # split the data input subsets by _
        if paired:
            paired_path = path.split('%')
        else:
            paired_path = [path]
        for p in range(len(paired_path)):
            self.subset.append(OAI_pretrain(root=root, path=paired_path[p],opt=opt, mode=mode, transforms=transforms,
                                            permute=self.opt.permute, zy=False, filenames=filenames, index=index))
            self.subset.append(OAI_pretrain(root=root, path=paired_path[p], opt=opt, mode=mode, transforms=transforms,
                                            permute=self.opt.permute, zy=True, filenames=filenames, index=index))

    def __len__(self):
        return min([len(x) for x in self.subset])

    def __getitem__(self, index):
        outputs_all = []
        filenames_all = []
        if self.filenames:
            for i in range(len(self.subset)):
                outputs, _, filenames = self.subset[i].__getitem__(index)
                outputs_all = outputs_all + outputs
                filenames_all = filenames_all + filenames
            return outputs_all, filenames_all
        else:
            for i in range(len(self.subset)):
                outputs = self.subset[i].__getitem__(index)
                outputs_all.append(outputs)
            return outputs_all

class OAI_pretrain(data.Dataset):
    def __init__(self, root, path, opt, mode, permute=False, zy=False, transforms=None, filenames=False, index=None):
        self.opt = opt
        self.mode = mode
        self.zy = zy
        self.permute = permute
        self.filenames = filenames
        self.index = index
        self.all_path = os.path.join(root, path)
        # get name of images from the first folder
        self.images = sorted([x.split('/')[-1] for x in glob.glob(self.all_path + '/*')])
        # resize should be 0
        # cropsize be final size for crop
        self.resize = tiff.imread(join(self.all_path, self.images[0])).shape[-1]

        if self.opt.cropsize == 0:
            self.size = self.resize
        else:
            self.size = self.opt.cropsize

    def read_3D_img(self, names):
        img = self.load_img(names) #(37, 444, 444)
        img = np.expand_dims(img, axis=1)
        # img = np.concatenate([img, img, img,], 1)
        img = torch.from_numpy(img)
        return img

    def load_img(self, path,):
        x = tiff.imread(path)
        x = np.array(x).astype(np.float32)
        if self.opt.trd is not None:
            x[x >= self.opt.trd] = self.opt.trd

        if not self.opt.nm == '00':
            x = x - x.min()  # this is added for the neurons images, where min != 0

            if x.max() > 0:  # scale to 0-1
                x = x / x.max()

        if len(x.shape) == 2:  # if grayscale
            x = np.expand_dims(x, 2)

        if not self.opt.gray:
            if x.shape[2] == 1:
                x = np.concatenate([x]*3, 2)

        return x

    def __getitem__(self, idx):
        if self.index is not None:
            index = self.index[idx]
        else:
            index = idx

        filenames = join(self.all_path, self.images[index])
        outputs = self.read_3D_img(filenames) #(37, 1, 444, 444)

        if self.opt.load_3D:
            outputs = interpolation(outputs, 8, crop_size=256, mode=self.mode)
            outputs = slice_cube(tensor=outputs, N=outputs.shape[1], size=outputs.shape[-1], start=0, zy=self.zy)
            outputs = torch.cat(outputs, 0).unsqueeze(0)

        if self.permute:
            outputs = interpolation(outputs, 8.6, crop_size=256, mode=self.mode)
            outputs = slice_cube(tensor=outputs, N=8, size=outputs.shape[-1], start=5, zy=self.zy)

        # return only images or with filenames
        if self.filenames:
            return outputs, self.labels[index], filenames
        else:
            return outputs#, self.labels[index]

    def __len__(self):
        if self.index is not None:
            return len(self.index)
        else:
            return len(self.images)


if __name__ == '__main__':
    from dotenv import load_dotenv
    import argparse
    load_dotenv('env/.t09')

    parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
    # Data
    parser.add_argument('--dataset', type=str, default='womac4min0/raw3D/')
    parser.add_argument('--bysubject', action='store_true', dest='bysubject', default=False)
    parser.add_argument('--direction', type=str, default='SagIwTSE', help='a2b or b2a')
    parser.add_argument('--flip', action='store_true', dest='flip', default=False, help='image flip left right')
    parser.add_argument('--resize', type=int, default=0)
    parser.add_argument('--cropsize', type=int, default=0)
    parser.add_argument('--n01', action='store_true', dest='n01', default=False)
    parser.add_argument('--gray', action='store_true', dest='gray', default=False, help='only use 1 channel')
    parser.add_argument('--mode', type=str, default='dummy')
    parser.add_argument('--port', type=str, default='dummy')
    parser.add_argument('--trd', type=float, dest='trd', help='threshold of images')
    parser.add_argument('--nm', type=str, help='paired: a_b, unpaired a%b ex:(a_b%c_d)')
    parser.add_argument('--permute', action='store_true', dest='permute', default=False, help='do interpolation and permutation')
    parser.add_argument('--load_3D', action='store_true', dest='load_3D', default=False, help='load 3D cube')
    opt = parser.parse_args()

    root = os.environ.get('DATASET') + opt.dataset
    print(root)
    opt.cropsize = 384
    opt.n01 = True

    opt.bysubject = False
    paired = '%' in opt.direction
    dataset3d = MultiData(root=root, path=opt.direction, opt=opt, mode='train', filenames=False, paired=paired)
    x3d = dataset3d.__getitem__(55)
    print(len(x3d))
    print(x3d[0].shape)

    # cube0 = [i[0].unsqueeze(0) for i in x3d[0]]
    # cube0 = torch.cat(cube0)
    # cube1 = [i[0].unsqueeze(0) for i in x3d[1]]
    # cube1 = torch.cat(cube1)
    # print(cube0.shape)
    #
    if 1:
        # cube0 = cube0.numpy()
        # tiff.imsave('out/raw.tif', cube0)
        # cube1 = cube1.numpy()
        # tiff.imsave('out/zy.tif', cube1)
        cube0 = x3d[0].numpy()
        tiff.imsave('out/raw_cube.tif', cube0)
        cube1 = x3d[1].numpy()
        tiff.imsave('out/zy_cube.tif', cube1)
        print(np.max(cube0), np.min(cube0))

    if 1:
        def three2twoD(tensor):
            print('tensor',tensor.shape)
            # intput(1,B,x,y,z)
            out = tensor.transpose(0, 3)
            out = out.squeeze()
            out = out.unsqueeze(0)
            print('three2twoD',out.shape)
            return out

        # xy_slice = slice_cube(x3d[1], N=4, size=x3d[1].shape[-1], start=0, zy=False)
        # xy_slice = torch.cat(xy_slice, 0)
        # xy_slice =xy_slice.unsqueeze(0).numpy().astype(np.uint8)
        xy_2D = three2twoD(x3d[1].unsqueeze(0)).numpy()
        tiff.imsave('out/xy_4slice.tif', xy_slice)
        tiff.imsave('out/xy_2D.tif', xy_2D)

# CUDA_VISIBLE_DEVICES=3 python dataloader/data_3D.py --dataset womac4min0/raw3D --direction SAG_IW_TSE