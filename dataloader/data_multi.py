from os.path import join
import glob
import torch
from torch import nn
import torch.utils.data as data
import torchvision.transforms as transforms
import tifffile as tiff
from PIL import Image
import numpy as np
import os
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import random
from functools import reduce


def get_transforms(crop_size, resize, additional_targets, need=('train', 'test')):
    transformations = {}
    if 'train' in need:
        transformations['train'] = A.Compose([
            A.Resize(resize, resize),
            # A.augmentations.geometric.rotate.Rotate(limit=45, p=1.),
            A.RandomCrop(height=crop_size, width=crop_size, p=1.),
            A.RandomRotate90(p=1.),
            #A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=400),
            ToTensorV2(p=1.0),
        ], p=1.0, additional_targets=additional_targets)
    if 'test' in need:
        transformations['test'] = A.Compose([
            A.Resize(resize, resize),
            # may have problem here _----------------------------------
            # A.CenterCrop(height=crop_size, width=crop_size, p=1.),
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

        dataset_mode = globals()[opt.dataset_mode]
        #getattr(self.opt, 'dataset_mode')
        #print(dataset_mode)

        for p in range(len(paired_path)):
            self.subset.append(dataset_mode(root=root, path=paired_path[p],
                                            opt=opt, mode=mode, labels=labels, transforms=transforms,
                                            filenames=filenames, index=index))
            #if self.opt.load3d:
            #    self.subset.append(PairedData3D(root=root, path=paired_path[p],
            #                                    opt=opt, mode=mode, labels=labels, transforms=transforms, filenames=filenames, index=index))
            #else:
            #    self.subset.append(PairedData(root=root, path=paired_path[p],
            #                                  opt=opt, mode=mode, labels=labels, transforms=transforms, filenames=filenames, index=index))

    def shuffle_images(self):
        for set in self.subset:
            random.shuffle(set.subjects_keys)

    def __len__(self):
        return min([len(x) for x in self.subset])

    def __getitem__(self, index):
        imgs_all = []
        filenames_all = []
        for i in range(len(self.subset)):
            #{'img': outputs, 'labels': self.labels[index], 'filenames': filenames, 'inputs': inputs}
            out = self.subset[i].__getitem__(index)
            imgs_all = imgs_all + out['img']
            filenames_all = filenames_all + out['filenames']
        labels = out['labels']  # there is only one label for one set of images
        return {'img': imgs_all, 'labels': labels, 'filenames': filenames_all}


class PairedSlices(data.Dataset):
    """
    Paired images with the same file name from different folders
    """
    def __init__(self, root, path, opt, mode, labels=None, transforms=None, filenames=False, index=None):
        super(PairedSlices, self).__init__()
        self.opt = opt
        self.mode = mode
        self.filenames = filenames
        self.index = index
        self.all_path = list(os.path.join(root, x) for x in path.split('_'))

        # get name of images if exist in the all the paired folders
        image_list = [sorted([x.split('/')[-1] for x in glob.glob(self.all_path[i] + '/*')]) for i in range(len(self.all_path))]
        self.images = list(reduce(set.intersection, [set(item) for item in image_list]))

        #print('LL ' + str(len(self.images)))

        self.subjects = dict([(x, [x]) for x in self.images])
        self.subjects_keys = sorted(self.subjects.keys())

        # if no resize than resize = image size
        if self.opt.resize == 0:
            try:
                self.resize = np.array(Image.open(join(self.all_path[0], self.images[0]))).shape[1]
            except:
                self.resize = tiff.imread(join(self.all_path[0], self.images[0])).shape[1]
        else:
            self.resize = self.opt.resize

        # if no cropsize than cropsize = resize
        if self.opt.cropsize == 0:
            self.cropsize = self.resize
        else:
            self.cropsize = self.opt.cropsize

        # if no transform than transform = get_transforms
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
            outputs = outputs + [augmented[k], ]
        return outputs

    def __len__(self):
        if self.index is not None:
            return len(self.index)
        else:
            return len(self.subjects_keys)

    def load_img(self, path):
        # loading image
        try:
            x = tiff.imread(path)
        except:
            x = Image.open(path)
        x = np.array(x).astype(np.float32)

        # thresholding and normalization
        if len(self.opt.trd) == 1:
            trd = self.opt.trd[0]
        else: # if there are two thresholds, use the first one for the first image and the second one for the second image
            if 'CorIwTSE' in path:
                trd = self.opt.trd[0]
            else:
                trd = self.opt.trd[1]
        if trd > 0:
            x[x >= trd] = trd

        if not self.opt.nm == '00':  # if no normalization, otherwise normalize to 0-1
            x = x - x.min()
            if x.max() > 0:  # scale to 0-1
                x = x / x.max()

        return x

    def __getitem__(self, idx):
        if self.index is not None:
            index = self.index[idx]
        else:
            index = idx

        # add all the slices into the dict
        filenames = []
        slices = sorted(self.subjects[self.subjects_keys[index]])  # get all the slices of this subject
        for i in range(len(self.all_path)):  # loop over all the paths
            slices_per_path = [join(self.all_path[i], x) for x in slices]
            filenames = filenames + slices_per_path
        inputs = self.load_to_dict(filenames)

        # Do augmentation
        outputs = self.get_augumentation(inputs)

        # normalize tensor
        if self.opt.nm == '11':
            outputs = [transforms.Normalize((0.5, ) * x.shape[0], (0.5, ) * x.shape[0])(x) for x in outputs]

        return {'img': outputs, 'labels': self.labels[index], 'filenames': filenames} #, inputs


class PairedSlices3D(PairedSlices):
    # Paired images with the same file name from different folders
    def __init__(self, root, path, opt, mode, labels=None, transforms=None, filenames=False, index=None):
        super(PairedSlices3D, self).__init__(root, path, opt, mode, labels=labels, transforms=transforms, filenames=filenames, index=index)
        self.filenames = filenames
        self.index = index

        # split images to list by subjects
        subjects = sorted(list(set([x.replace('_' + x.split('_')[-1], '') for x in self.images])))
        self.subjects = dict()
        for s in subjects:
            self.subjects[s] = sorted([x for x in self.images if x.replace('_' + x.split('_')[-1], '') == s])
        self.subjects_keys = sorted(self.subjects.keys())

    def __len__(self):
        if self.index is not None:
            return len(self.index)
        else:
            return len(self.subjects_keys)

    def __getitem__(self, idx):
        if self.index is not None:
            index = self.index[idx]
        else:
            index = idx

        # add all the slices into the dict
        location_to_split = [0]
        filenames = []
        slices = sorted(self.subjects[self.subjects_keys[index]]) # get all the slices of this subject
        print(slices)
        for i in range(len(self.all_path)):  # loop over all the paths
            slices_per_path = [join(self.all_path[i], x) for x in slices]
            filenames = filenames + slices_per_path
            location_to_split.append(location_to_split[-1] + len(slices_per_path))
        inputs = self.load_to_dict(filenames)

        # Do augmentation
        outputs = self.get_augumentation(inputs)

        # split by different paths

        outputs = torch.stack(outputs, dim=3)
        outputs = torch.tensor_split(outputs, location_to_split[1:], dim=3)[:-1]
        outputs = [x for x in outputs]

        # normalize tensor
        if self.opt.nm == '11':
            outputs = [transforms.Normalize((0.5, ) * x.shape[0], (0.5, ) * x.shape[0])(x) for x in outputs]

        #return {'img': outputs, 'labels': self.labels[index], 'filenames': filenames, 'inputs': inputs}
        return {'img': outputs, 'labels': self.labels[index], 'filenames': filenames}  # , inputs


class RawTif(data.Dataset):
    def __init__(self, root, directions, permute=None, crop=None, trd=0):
        self.directions = directions.split('_')

        self.tif = []
        for i, d in enumerate(self.directions):
            print('loading...')
            if crop is not None:
                tif = tiff.imread(os.path.join(root, d + '.tif'))[crop[0]:crop[1], crop[2]:crop[3], crop[4]:crop[5]]
            else:
                tif = tiff.imread(os.path.join(root, d + '.tif'))
            print('done...')
            tif = tif.astype(np.float32)

            if trd is not None:
                tif[tif >= trd[i]] = trd[i]

            tif = tif - tif.min()  # this is added for the neurons images, where min != 0
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


class PairedCubes(PairedSlices):
    def __init__(self, root, path, opt, mode, labels=None, transforms=None, filenames=False, index=None):
        super(PairedCubes, self).__init__(root, path, opt, mode, labels=labels, transforms=transforms, filenames=filenames, index=index)

    def load_to_dict_3d(self, names):
        out = dict()
        location_to_split = [0]
        for i in range(len(names)):
            x3d = self.load_img(names[i])
            location_to_split.append(location_to_split[-1] + x3d.shape[0])
            for s in range(x3d.shape[0]):
                out[str(1000 * i + s).zfill(4)] = x3d[s, :, :]
        out['image'] = out.pop('0000')  # the first image in albumentation need to be named "image"
        return out, location_to_split

    def __len__(self):
        if self.index is not None:
            return len(self.index)
        else:
            return len(self.subjects_keys)

    def __getitem__(self, idx):
        if self.index is not None:
            index = self.index[idx]
        else:
            index = idx

        # add all the slices into the dict
        filenames = []
        slices = sorted(self.subjects[self.subjects_keys[index]])  # get all the slices of this subject
        for i in range(len(self.all_path)):  # loop over all the paths
            slices_per_path = [join(self.all_path[i], x) for x in slices]
            filenames = filenames + slices_per_path
        inputs, location_to_split = self.load_to_dict_3d(filenames)

        # Do augmentation
        outputs = self.get_augumentation(inputs)
        outputs = torch.stack(outputs, dim=3)
        outputs = torch.tensor_split(outputs, location_to_split[1:], dim=3)[:-1]

        # normalize tensor
        if self.opt.nm == '11':
            outputs = [transforms.Normalize((0.5, ) * x.shape[0], (0.5, ) * x.shape[0])(x) for x in outputs]
            # outputs = [(x - 0.5) * 2 for x in outputs]
            # print(outputs[0].max(), outputs[0].min())

        # croppiing the first dimension
        if self.opt.cropz > 0:
            try:
                min_z = min(outputs[0].shape[3], outputs[1].shape[3])
            except:
                min_z = outputs[0].shape[3]
            assert min_z >= self.opt.cropz
            cropz_range = np.random.randint(0, min_z - self.opt.cropz + 1)
            outputs = [x[:, :, :, cropz_range:cropz_range + self.opt.cropz] for x in outputs]

        return {'img': outputs, 'labels': self.labels[index], 'filenames': filenames, 'inputs': inputs}


if __name__ == '__main__':
    from dotenv import load_dotenv
    import argparse
    load_dotenv('env/.t09')

    parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
    # Data
    parser.add_argument('--dataset', type=str, default='womac4/full/')
    parser.add_argument('--load3d', action='store_true', dest='load3d', default=True)
    parser.add_argument('--direction', type=str, default='ap_bp', help='a2b or b2a')
    parser.add_argument('--flip', action='store_true', dest='flip', default=False, help='image flip left right')
    parser.add_argument('--resize', type=int, default=0)
    parser.add_argument('--cropsize', type=int, default=0)
    parser.add_argument('--cropz', type=int, default=0)
    parser.add_argument('--trd', type=float, default=None)
    parser.add_argument('--mode', type=str, default='dummy')
    parser.add_argument('--port', type=str, default='dummy')
    opt = parser.parse_args()

    root = os.environ.get('DATASET') + opt.dataset
    opt.cropsize = 256
    opt.nm = '11'
    d1 = PairedData(root=root, path=opt.direction, opt=opt, mode='train', filenames=False)
    x1 = d1.__getitem__(0)

    #d3 = PairedData3D(root=root, path='ap_bp_ap', opt=opt, mode='train', filenames=False)
    #x3 = d3.__getitem__(0)

    opt.load3d = True
    d3x = MultiData(root=root, path='ap_bp_ap', opt=opt, mode='train', filenames=False)
    x3x = d3x.__getitem__(0)

    #d3d = Paired3DTif(root='/media/ExtHDD01/Dataset/paired_images/KL2min0V10/', path='a_b', opt=opt, mode='train',
    #                  filenames=False)
    #opt.cropz = 16
    #x3d = d3d.__getitem__(0)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset=d3x, num_workers=4, batch_size=2, shuffle=True,
                              pin_memory=True)
    for i, x in enumerate(train_loader):
        print(i)