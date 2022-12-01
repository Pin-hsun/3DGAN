import torch, copy
import torch.nn as nn
import torchvision
import torch.optim as optim
from math import log10
import time, os
import pytorch_lightning as pl
from utils.metrics_segmentation import SegmentationCrossEntropyLoss
from utils.metrics_classification import CrossEntropyLoss, GetAUC
from utils.data_utils import *
from models.base import BaseModel, combine
import pandas as pd
from models.helper_oai import OaiSubjects, classify_easy_3d, swap_by_labels
from models.helper import reshape_3d, tile_like
from models.descar4 import GAN as GanBase


class GAN(GanBase):
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints)
        # First, modify the hyper-parameters if necessary
        # Initialize the networks
        self.net_g, self.net_d = self.set_networks()
        self.net_dX = copy.deepcopy(self.net_d)
        self.classifier = nn.Conv2d(256, 1, 1, stride=1, padding=0).cuda()

        # update names of the models for optimization
        self.netg_names = {'net_g': 'net_g'}
        self.netd_names = {'net_d': 'net_d', 'classifier': 'classifier', 'net_dX': 'net_dX'}#, 'net_class': 'netDC'}

        self.oai = OaiSubjects(self.hparams.dataset)

        # Finally, initialize the optimizers and scheduler
        self.init_optimizer_scheduler()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--lbx", dest='lbx', type=float, default=1)
        parser.add_argument("--dc0", dest='dc0', type=float, default=1)
        return parent_parser

    @staticmethod
    def test_method(net_g, img, a=None):
        oriX = img[0]

        imgXX, imgCX = net_g(oriX, a=torch.FloatTensor([0]))
        imgXX = nn.Sigmoid()(imgXX)  # mask

        imgXY, imgCY = net_g(oriX, a=torch.FloatTensor([a]))
        imgXY = nn.Sigmoid()(imgXY)  # mask
        imgCY = nn.Sigmoid()(imgCY)  # mask

        imgXX = combine(imgXX, oriX, method='mul')
        imgXY = combine(imgXY, oriX, method='mul') + combine(1 - imgXY, imgCY, method='mul')

        return imgXY

    def generation(self, batch):
        if self.hparams.load3d:  # if working on 3D input, bring the Z dimension to the first and combine with batch
            batch['img'] = reshape_3d(batch['img'])

        # pain label
        self.labels = self.oai.labels_unilateral(filenames=batch['filenames'])
        self.oriX = batch['img'][0]
        self.oriY = batch['img'][1]

        self.imgXY, self.imgCY = self.net_g(self.oriX, a=torch.abs(self.labels['paindiff']))
        self.imgXY = nn.Sigmoid()(self.imgXY)  # mask
        self.imgCY = nn.Sigmoid()(self.imgCY)  # mask
        self.imgXY = combine(self.imgXY, self.oriX, method='mul') + combine(1 - self.imgXY, self.imgCY, method='mul')

        self.imgXX, self.imgCX = self.net_g(self.oriX, a=0 * torch.abs(self.labels['paindiff']))
        self.imgXX = nn.Sigmoid()(self.imgXX)  # mask
        self.imgCX = nn.Sigmoid()(self.imgCX)  # mask
        self.imgXX = combine(self.imgXX, self.oriX, method='mul') + combine(1 - self.imgXX, self.imgCX, method='mul')
        #self.imgYY = combine(self.imgYY, self.oriY, method='mul')


# CUDA_VISIBLE_DEVICES=0 python train.py --jsn womac3 --prj 3D/test4att/  --models descar4att --netG dsmcrel0a --netD bpatch_16 --split moaks