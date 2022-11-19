from os.path import join
import glob
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
import numpy as np
import os
from skimage import io
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import tifffile as tiff
import pandas as pd
import random


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


def separate_subjects_n_slices(img_list):
    "for knee project"
    temp = [x.split('.')[0].split('_') for x in img_list]
    subject = dict()
    for y in temp:
        if int(y[0]) not in subject.keys():
            subject[int(y[0])] = []
        subject[int(y[0])] = subject[int(y[0])] + [int(y[1])]
    for k in list(subject.keys()):
        subject[k].sort()
    return subject


def get_transforms(crop_size, resize, additional_targets, need=('train', 'test')):
    transformations = {}
    if 'train' in need:
        transformations['train'] = A.Compose([
            A.Resize(resize, resize),
            #A.augmentations.geometric.rotate.Rotate(limit=45, p=0.5),
            A.RandomCrop(height=crop_size, width=crop_size, p=1.),
            #A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=400),
            ToTensorV2(p=1.0),
        ], p=1.0, additional_targets=additional_targets)
    if 'test' in need:
        transformations['test'] = A.Compose([
            A.Resize(resize, resize),
            #A.CenterCrop(height=crop_size, width=crop_size, p=1.),
            ToTensorV2(p=1.0),
        ], p=1.0, additional_targets=additional_targets)
    return transformations


class MultiData(data.Dataset):
    """
    Multiple unpaired data combined
    """
    def __init__(self, root, path, opt, mode, labels=None, transforms=None, filenames=False, index=None):
        super(MultiData, self).__init__()
        self.opt = opt
        self.mode = mode
        self.filenames = filenames
        # split the data input subsets by %
        paired_path = path.split('%')
        self.subset = []
        for p in range(len(paired_path)):
            if self.opt.load3d:
                self.subset.append(PairedData3D(root=root, path=paired_path[p],
                                                opt=opt, mode=mode, labels=labels, transforms=transforms, filenames=filenames, index=index))
            else:
                self.subset.append(PairedData(root=root, path=paired_path[p],
                                              opt=opt, mode=mode, labels=labels, transforms=transforms, filenames=filenames, index=index))

    def shuffle_images(self):
        for set in self.subset:
            random.shuffle(set.images)

    def __len__(self):
        return min([len(x) for x in self.subset])

    def __getitem__(self, index):
        outputs_all = []
        filenames_all = []
        if self.filenames:
            for i in range(len(self.subset)):
                outputs, labels, filenames = self.subset[i].__getitem__(index)
                outputs_all = outputs_all + outputs
                filenames_all = filenames_all + filenames
            return {'img': outputs_all, 'labels': labels, 'filenames': filenames_all}
        else:
            for i in range(len(self.subset)):
                outputs, labels = self.subset[i].__getitem__(index)
                outputs_all = outputs_all + outputs
            return {'img': outputs_all, 'labels': labels}


class PairedData(data.Dataset):
    """
    Paired images with the same file name from different folders
    """
    def __init__(self, root, path, opt, mode, labels=None, transforms=None, filenames=False, index=None):
        super(PairedData, self).__init__()
        self.opt = opt
        self.mode = mode
        self.filenames = filenames
        self.index = index

        self.all_path = list(os.path.join(root, x) for x in path.split('_'))

        # get name of images from the first folder
        self.images = sorted([x.split('/')[-1] for x in glob.glob(self.all_path[0] + '/*')])
        if self.opt.resize == 0:
            self.resize = np.array(Image.open(join(self.all_path[0], self.images[0]))).shape[1]
        else:
            self.resize = self.opt.resize

        if self.opt.cropsize == 0:
            self.cropsize = self.resize
        else:
            self.cropsize = self.opt.cropsize

        if transforms is None:
            additional_targets = dict()
            for i in range(1, 9999):#len(self.all_path)):
                additional_targets[str(i).zfill(4)] = 'image'
            self.transforms = get_transforms(crop_size=self.cropsize,
                                             resize=self.resize,
                                             additional_targets=additional_targets)[mode]
        else:
            self.transforms = transforms

        if labels is None:
            self.labels = [0] * len(self.images)  # WRONG, label is not added yet
        else:
            self.labels = labels

    def load_to_dict(self, names):
        out = dict()
        for i in range(len(names)):
            out[str(i).zfill(4)] = self.load_img(names[i])
        out['image'] = out.pop('0000')  # the first image in albumentation need to be named "image"
        return out

    def get_augumentation(self, inputs):
        outputs = []
        augmented = self.transforms(**inputs)
        augmented['0000'] = augmented.pop('image')  # 'change image back to 0'
        for k in sorted(list(augmented.keys())):
            if self.opt.n01:
                outputs = outputs + [augmented[k], ]
            else:
                if augmented[k].shape[0] == 3:
                    outputs = outputs + [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(augmented[k]), ]
                elif augmented[k].shape[0] == 1:
                    outputs = outputs + [transforms.Normalize(0.5, 0.5)(augmented[k]), ]
        return outputs

    def load_img(self, path):
        x = Image.open(path)
        x = np.array(x).astype(np.float32)

        if self.opt.trd > 0:
            x[x >= self.opt.trd] = self.opt.trd

        if x.max() > 0:  # scale to 0-1
            x = x / x.max()

        if len(x.shape) == 2:  # if grayscale
            x = np.expand_dims(x, 2)
        if not self.opt.gray:
            if x.shape[2] == 1:
                x = np.concatenate([x]*3, 2)
        return x

    def __len__(self):
        if self.index is not None:
            return len(self.index)
        else:
            return len(self.images)

    def __getitem__(self, idx):
        if self.index is not None:
            index = self.index[idx]
        else:
            index = idx

        filenames = [join(x, self.images[index]) for x in self.all_path]
        inputs = self.load_to_dict(filenames)

        # Do augmentation
        outputs = self.get_augumentation(inputs)

        # return only images or with filenames
        if self.filenames:
            return outputs, self.labels[index], filenames
        else:
            return outputs, self.labels[index]


class PairedData3D(PairedData):
    """
    Multiple unpaired data combined
    """
    def __init__(self, root, path, opt, mode, labels=None, transforms=None, filenames=False, index=None):
        super(PairedData3D, self).__init__(root, path, opt, mode, labels=labels, transforms=transforms, filenames=filenames, index=index)
        self.filenames = filenames
        self.index = index

        subjects = sorted(list(set([x.replace('_' + x.split('_')[-1], '') for x in self.images])))
        self.subjects = dict()
        for s in subjects:
            self.subjects[s] = sorted([x for x in self.images if x.replace('_' + x.split('_')[-1], '') == s])

    def __len__(self):
        if self.index is not None:
            return len(self.index)
        else:
            return len(self.subjects.keys())

    def __getitem__(self, idx):
        if self.index is not None:
            index = self.index[idx]
        else:
            index = idx

        # add all the slices into the dict
        a_subject = sorted(self.subjects.keys())[index]  # get the subject name
        filenames = []
        length_of_each_path = []
        for i in range(len(self.all_path)):  # loop over all the paths
            selected = sorted(self.subjects[a_subject])
            slices = [join(self.all_path[i], x) for x in selected]
            filenames = filenames + slices
            length_of_each_path.append(len(slices))
        inputs = self.load_to_dict(filenames)

        # Do augmentation
        outputs = self.get_augumentation(inputs)

        # split to different paths
        total = []
        for split in length_of_each_path:
            temp = []
            for i in range(split):
                temp.append(outputs.pop(0).unsqueeze(3))
            total.append(torch.cat(temp, 3))
        outputs = total

        # return only images or with filenames
        if self.filenames:
            return outputs, self.labels[index], filenames
        else:
            return outputs, self.labels[index]


class PairedDataTif(data.Dataset):
    def __init__(self, root, directions, permute=None, crop=None, trd=0):
        self.directions = directions.split('_')

        self.tif = []
        for d in self.directions:
            print('loading...')
            if crop is not None:
                tif = tiff.imread(os.path.join(root, d + '.tif'))[crop[0]:crop[1], crop[2]:crop[3], crop[4]:crop[5]]
            else:
                tif = tiff.imread(os.path.join(root, d + '.tif'))
            print('done...')
            tif = tif.astype(np.float32)

            if trd > 0:
                tif[tif >= trd] = trd

            if tif.max() > 0:  # scale to 0-1
                tif = tif / tif.max()

            tif = (tif * 2) - 1
            if permute is not None:
                tif = np.transpose(tif, permute)
            self.tif.append(tif)

    def __len__(self):
        return self.tif[0].shape[0]

    def __getitem__(self, idx):
        outputs = []
        for t in self.tif:
            slice = torch.from_numpy(t[:, :]).unsqueeze(0)
            #slice = torch.permute(slice, (0, 2, 3, 1))
            outputs.append(slice)
        return {'img': outputs}


if __name__ == '__main__':
    from dotenv import load_dotenv
    import argparse
    load_dotenv('env/.t09')

    parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
    # Data
    parser.add_argument('--dataset', type=str, default='womac3/full/')
    parser.add_argument('--load3d', action='store_true', dest='load3d', default=True)
    parser.add_argument('--direction', type=str, default='areg_b', help='a2b or b2a')
    parser.add_argument('--flip', action='store_true', dest='flip', default=False, help='image flip left right')
    parser.add_argument('--resize', type=int, default=0)
    parser.add_argument('--cropsize', type=int, default=0)
    parser.add_argument('--trd', type=float, default=0)
    parser.add_argument('--n01', action='store_true', dest='n01', default=False)
    parser.add_argument('--gray', action='store_true', dest='gray', default=False, help='only use 1 channel')
    parser.add_argument('--mode', type=str, default='dummy')
    parser.add_argument('--port', type=str, default='dummy')
    opt = parser.parse_args()

    if 0:
        root = os.environ.get('DATASET') + opt.dataset
        opt.cropsize = 256
        opt.n01 = True
        dataset = PairedData(root=root, path=opt.direction, opt=opt, mode='train', filenames=False)
        x = dataset.__getitem__(100)
        dataset3d = PairedData3D(root=root, path=opt.direction, opt=opt, mode='train', filenames=False)
        x3d = dataset3d.__getitem__(100)

        # womac3
        dataset = MultiData(root=root, path='areg_b_aregseg_bseg', opt=opt, mode='train', filenames=False)
        xm = dataset3d.__getitem__(100)

    if 1:
        # fly3d
        root = '/media/ExtHDD01/Dataset/paired_images/Fly0B/train/' # change to your data root
        opt.n01 = False
        opt.load3d = True
        dataset = MultiData(root=root, path='xyweak_xysb',
                            opt=opt, mode='train', filenames=True)

        xm = dataset.__getitem__(5)

    # fly3d tif
    if 0:
        datatif = PairedDataTif(root='/media/ExtHDD01/Dataset/paired_images/Fly0B/',
                                directions='xyzweak_xyzsb', permute=(0, 2, 1),
                                crop=[0, 1890, 1024+512, 1024+512+32, 0, 1024])
        x = datatif.__getitem__(0)

    if 0:
        root = os.environ.get('DATASET') + opt.dataset
        opt.cropsize = 256
        opt.n01 = True
        #dataset = PairedData(root=root, path=opt.direction, opt=opt, mode='train', filenames=False)
        #x = dataset.__getitem__(100)
        #dataset3d = PairedData3D(root=root, path=opt.direction, opt=opt, mode='train', filenames=False)
        #x3d = dataset3d.__getitem__(100)

        # womac3
        opt.load3d = True
        dataset3d = MultiData(root=root, path='a_effusion/aeffpain_b_effusion/beffpain', opt=opt, mode='train', filenames=False)
        xm = dataset3d.__getitem__(210)