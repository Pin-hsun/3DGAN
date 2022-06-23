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
            A.CenterCrop(height=crop_size, width=crop_size, p=1.),
            ToTensorV2(p=1.0),
        ], p=1.0, additional_targets=additional_targets)
    return transformations


def save_segmentation(dataset, names, destination, use_t2d):
    """
    turn images into segmentation and save it
    """
    os.makedirs(destination, exist_ok=True)
    seg = torch.load('submodels/model_seg_ZIB_res18_256.pth').cuda()
    t2d = torch.load('submodels/tse_dess_unet32.pth')
    seg.eval()
    t2d.eval()
    for i in range(len(dataset)):
        x = dataset.__getitem__(i)[0].unsqueeze(0).cuda()
        if use_t2d:
            x = t2d(x)[0]
        out = seg(x)
        out = torch.argmax(out, 1).squeeze().detach().cpu().numpy().astype(np.uint8)
        tiff.imsave(destination + names[i], out)


class MultiData(data.Dataset):
    """
    Multiple unpaired data ccombined
    """
    def __init__(self, root, path, opt, mode, transforms=None, filenames=False, index=None):
        super(MultiData, self).__init__()
        self.opt = opt
        self.mode = mode
        self.filenames = filenames
        # split the data input subsets by %
        paired_path = path.split('%')
        self.subset = []
        for p in range(len(paired_path)):
            if self.opt.bysubject:
                self.subset.append(PairedData3D(root=root, path=paired_path[p],
                                                opt=opt, mode=mode, transforms=transforms, filenames=filenames, index=index))
            else:
                self.subset.append(PairedData(root=root, path=paired_path[p],
                                              opt=opt, mode=mode, transforms=transforms, filenames=filenames, index=index))

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
                outputs_all = outputs_all + outputs
            return outputs_all


class PairedData(data.Dataset):
    """
    Paired images with the same file name from different folders
    """
    def __init__(self, root, path, opt, mode, transforms=None, filenames=False, index=None):
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
            for i in range(1, 1000):#len(self.all_path)):
                additional_targets[str(i).zfill(4)] = 'image'
            self.transforms = get_transforms(crop_size=self.cropsize,
                                             resize=self.resize,
                                             additional_targets=additional_targets)[mode]
        else:
            self.transforms = transforms

        if 0:
            df = pd.read_csv('notinuse/OAI00womac3.csv')
            self.labels = [(x, ) for x in df.loc[df['SIDE'] == 1, 'P01KPN#EV'].astype(np.int8)]
        else:
            self.labels = [0] * len(self.images)  ## WRONG

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
        #if x.max() > 0:  # scale to 0-1
        #    x = x / x.max()
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

        for i in range(len(outputs)):
            outputs[i] = outputs[i] / outputs[i].max()
            if not self.opt.n01:
                if self.opt.gray:
                    outputs[i] = transforms.Normalize(0.5, 0.5)(outputs[i])
                else:
                    outputs[i] = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0,5, 0.5))(outputs[i])

        # return only images or with filenames
        if self.filenames:
            return outputs, self.labels[index], filenames
        else:
            return outputs#, self.labels[index]


class PairedData3D(PairedData):
    """
    Multiple unpaired data ccombined
    """
    def __init__(self, root, path, opt, mode, transforms=None, filenames=False, index=None):
        super(PairedData3D, self).__init__(root, path, opt, mode, transforms=transforms, filenames=filenames, index=index)
        self.filenames = filenames
        self.index = index
        self.opt = opt

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
            a = torch.cat(temp, 3)

            a = a / a.max()

            if not self.opt.n01:
                (a0, a1, a2, a3) = a.shape
                a = a.view(a0, a1, a2 * a3)
                if self.opt.gray:
                    a = transforms.Normalize(0.5, 0.5)(a)
                else:
                    a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0, 5, 0.5))(a)
                a = a.view(a0, a1, a2, a3)

            total.append(a)
        outputs = total

        # return only images or with filenames
        if self.filenames:
            return outputs, self.labels[index], filenames
        else:
            return outputs#, self.labels[index]


if __name__ == '__main__':
    from dotenv import load_dotenv
    import argparse
    load_dotenv('.env')

    parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
    # Data
    parser.add_argument('--dataset', type=str, default='womac3')
    parser.add_argument('--bysubject', action='store_true', dest='bysubject', default=False)
    parser.add_argument('--direction', type=str, default='areg_b', help='a2b or b2a')
    parser.add_argument('--flip', action='store_true', dest='flip', default=False, help='image flip left right')
    parser.add_argument('--resize', type=int, default=0)
    parser.add_argument('--cropsize', type=int, default=0)
    parser.add_argument('--n01', action='store_true', dest='n01', default=False)
    parser.add_argument('--gray', action='store_true', dest='gray', default=False, help='only use 1 channel')
    parser.add_argument('--mode', type=str, default='dummy')
    parser.add_argument('--port', type=str, default='dummy')
    opt = parser.parse_args()

    root = os.environ.get('DATASET') + opt.dataset + '/train/'
    opt.cropsize = 256
    opt.n01 = True
    dataset = PairedData(root=root, path=opt.direction, opt=opt, mode='train', filenames=False)
    x = dataset.__getitem__(100)
    dataset3d = PairedData3D(root=root, path=opt.direction, opt=opt, mode='train', filenames=False)
    x3d = dataset3d.__getitem__(100)

    dataset3d = MultiData(root=root, path='areg_b_aregseg_bseg', opt=opt, mode='train', filenames=False)
    xm = dataset3d.__getitem__(100)

    #  Dataset
    if 0:
        #x = train_set.__getitem__(100)
        names = [x.split('/')[-1] for x in sorted(glob.glob(root + source + '/*'))]
        save_segmentation(dataset=dataset,
                          names=names,
                          destination=root + destination, use_t2d=True)

    if 0:
        root = os.environ.get('DATASET') + opt.dataset + '/test/'
        source = 'aregis1/'
        mask = 'aseg/'
        destination = 'amask/'
        images = [x.split('/')[-1] for x in sorted(glob.glob(root + source + '*'))]

        os.makedirs(root + destination, exist_ok=True)
        for im in images:
            x = np.array(Image.open(root + source + im))
            m = np.array(Image.open(root + mask + im))
            m = (m == 1) + (m == 3)
            masked = np.multiply(x, m)
            tiff.imsave(root + destination + im, masked)