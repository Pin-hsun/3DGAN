import os
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import tifffile as tiff
import albumentations as A
import random


class CustomDataset(Dataset):
    def __init__(self, img_root, mode):
        self.root = img_root
        self.mode = mode

        if mode == 'train':
            self.data = os.listdir(self.root)[:400]
        if mode == 'eval':
            self.data = os.listdir(self.root)[400:]


    def transforms(self, mode):
        if mode == 'train':
            transform = A.Compose([
                A.RandomCrop(width=256, height=256),
                A.Rotate(limit=20),
                A.CoarseDropout(max_holes=10),
                # A.RandomBrightnessContrast(p=0.2),
                ])
        else:
            transform = A.Compose([
                A.CenterCrop(width=384, height=384),
                ])
        return transform


    def __getitem__(self, index):
        img = tiff.imread(os.path.join(self.root, self.data[index])) # 384, 384
        label = tiff.imread(os.path.join(self.root, self.data[index]).replace('images', 'labels')) # 384, 384

        transform = self.transforms(self.mode)
        transformed = transform(image=img, mask=label)
        img = transformed["image"]
        img = ((img - img.min()) / (img.max() - img.min()) - 0.5) / 0.5
        img = torch.from_numpy(img).unsqueeze(0)
        label = transformed["mask"]
        if label.max() > 2:
            img, label, name = self.__getitem__(random.randint(0, len(self.data)))
            return img.float(), label, name
        return img.float(), label, self.data[index]

    def __len__(self):
        return len(self.data)


class TSE_Inference_Dataset(Dataset):
    def __init__(self, img_root, mode):
        self.root = img_root
        self.mode = mode

        if mode == 'train':
            self.data = os.listdir(self.root)[:400]
        if mode == 'eval':
            self.data = os.listdir(self.root)[:10]

    def transforms(self, mode, additional_targets=None):
        if mode == 'train':
            transform = A.Compose([
                A.RandomCrop(width=256, height=256),
                A.Rotate(limit=20),
                A.CoarseDropout(max_holes=10),
                # A.RandomBrightnessContrast(p=0.2),
                ], additional_targets=additional_targets)
        else:
            transform = A.Compose([
                A.CenterCrop(width=384, height=384),
                ], additional_targets=additional_targets)
        return transform


    def __getitem__(self, index):
        img = tiff.imread(os.path.join(self.root, self.data[index])) # 384, 384

        # img = ((img -img.min()) / (img.max() - img.min()) - 0.5) / 0.5

        transform = self.transforms(self.mode)
        transformed = transform(image=img)
        img = transformed["image"]
        img = ((img - img.min()) / (img.max() - img.min()) - 0.5) / 0.5
        img = torch.from_numpy(img).unsqueeze(0)
        return img.float(), self.data[index]

    def __len__(self):
        return len(self.data)


class TSE_Inference_Dataset_3D(TSE_Inference_Dataset):
    def __init__(self, img_root, mode):
        self.root = img_root
        self.mode = mode
        self.data = os.listdir(self.root)[:1]

        additional_targets = dict()
        for i in range(1, 9999):  # len(self.all_path)):
            additional_targets[str(i).zfill(4)] = 'image'
        self.transforms = self.transforms(self.mode, additional_targets)

    def load_to_dict_3d(self, name):
        out = dict()
        x3d = tiff.imread(name).astype(float)
        x3d[x3d >= 800] = 800
        location_to_split = x3d.shape[0]
        for s in range(x3d.shape[0]):
            out[str(0000 + s).zfill(4)] = x3d[s, :, :]
        out['image'] = out.pop('0000')  # the first image in albumentation need to be named "image"
        return out, location_to_split

    def get_augumentation(self, inputs):
        outputs = []
        augmented = self.transforms(**inputs)
        augmented['0000'] = augmented.pop('image')  # 'change image back to 0'
        for k in sorted(list(augmented.keys())):
            outputs = outputs + [augmented[k], ]
        outputs = [torch.from_numpy(i).unsqueeze(0) for i in outputs]
        return outputs

    def __getitem__(self, index):
        # img = tiff.imread(os.path.join(self.root, self.data[index])) # 384, 384

        # add all the slices into the dict
        filenames = os.path.join(self.root, self.data[index])  # get all the slices of this subject
        inputs, location_to_split = self.load_to_dict_3d(filenames)

        # Do augmentation
        outputs = self.get_augumentation(inputs)
        outputs = torch.stack(outputs, dim=0)
        # outputs = torch.tensor_split(outputs, location_to_split, dim=3)[:-1]
        outputs = ((outputs - outputs.min()) / (outputs.max() - outputs.min()) - 0.5) / 0.5

        return outputs.float(), self.data[index]

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    # img_root = f'/media/ExtHDD01/Dataset/label_OAI/sample_10005/images'
    img_root = '/media/ExtHDD01/Dataset/OAI_DESS_segmentation/ZIB/original'
    train_set = ZIBDataset(img_root, 'train')
    train_loader = DataLoader(train_set, batch_size=10, shuffle=True, num_workers=10,
                              drop_last=True)

    for img_a, att_a, _ in train_loader:
        print(img_a.shape)
        print(att_a.shape)
