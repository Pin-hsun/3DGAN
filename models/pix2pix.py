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
from models.base import BaseModel, combine, VGGLoss
import pandas as pd
from models.helper_oai import OaiSubjects, classify_easy_3d, swap_by_labels
from models.helper import reshape_3d, tile_like


class GAN(BaseModel):
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints)
        # First, modify the hyper-parameters if necessary
        # Initialize the networks
        if self.hparams.stack:
            self.net_g = self.set_networks(net='g')
            self.hparams.output_nc = self.hparams.output_nc * 2
            self.net_d = self.set_networks(net='d')
        else:
            self.net_g, self.net_d = self.set_networks()

        # update names of the models for optimization
        self.netg_names = {'net_g': 'net_g'}
        self.netd_names = {'net_d': 'net_d'}

        # Finally, initialize the optimizers and scheduler
        self.init_optimizer_scheduler()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--stack", dest='stack', action='store_true', default=False)
        return parent_parser

    @staticmethod
    def test_method(net_g, img, a=None):
        oriX = img[0]

        print(a)
        #imgXX, _ = net_g(oriX, a=torch.FloatTensor([0]))
        #imgXX = nn.Sigmoid()(imgXX)  # mask

        imgXY = net_g(oriX, a=torch.FloatTensor([a]))
        imgXY = nn.Sigmoid()(imgXY['out0'])  # mask

        #imgXX = combine(imgXX, oriX, method='mul')
        imgXY = combine(imgXY, oriX, method='mul')

        return imgXY

    def generation(self, batch):
        img = batch['img']
        self.oriX = img[0]
        self.oriY = img[1]
        self.imgX0 = self.net_g(self.oriX, a=None)['out0']

    def backward_g(self):
        # ADV(X0, Y)+
        loss_g = 0
        if self.hparams.stack:
            loss_g += self.add_loss_adv(a=torch.cat([self.oriX, self.imgX0], 1), net_d=self.net_d, truth=True) * 1
        else:
            loss_g += self.add_loss_adv(a=self.imgX0, net_d=self.net_d, truth=True) * 1

        # L1(X0, Y)
        loss_g += self.add_loss_l1(a=self.imgX0, b=self.oriY) * self.hparams.lamb

        return {'sum': loss_g, 'loss_g': loss_g}

    def backward_d(self):
        loss_d = 0
        # ADV(X0, Y)-
        if self.hparams.stack:
            loss_d += self.add_loss_adv(a=torch.cat([self.oriX, self.imgX0], 1), net_d=self.net_d, truth=False) * 0.5
        else:
            loss_d += self.add_loss_adv(a=self.imgX0, net_d=self.net_d, truth=False) * 0.5

        # ADV(X, Y)+
        if self.hparams.stack:
            loss_d += self.add_loss_adv(a=torch.cat([self.oriX, self.oriY], 1), net_d=self.net_d, truth=True) * 0.5
        else:
            loss_d += self.add_loss_adv(a=self.oriY, net_d=self.net_d, truth=True) * 0.5

        return {'sum': loss_d, 'loss_d': loss_d}


# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --jsn wnwp3d --prj NS/ori_ft0 --models NS -b 16 --direction xyori_xyft0 --trd 2000 --nm 11 --netG descarnoumc --mc --preload