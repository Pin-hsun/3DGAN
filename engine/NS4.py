import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from models.networks import define_G, define_D
from models.networks import get_scheduler
from models.loss import GANLoss
from math import log10
import time, os
import pytorch_lightning as pl
from utils.metrics_segmentation import SegmentationCrossEntropyLoss
from utils.metrics_classification import CrossEntropyLoss, GetAUC
from utils.data_utils import *
from engine.base import BaseModel, combine


class GAN(BaseModel):
    """
    There is a lot of patterned noise and other failures when using lightning
    """
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints)
        print('using pix2pix.py')

    @staticmethod
    def add_model_specific_args(parent_parser):
        return parent_parser

    def generation(self):
        self.oriX = self.batch[0]
        self.oriY = self.batch[1]
        self.imgX0 = self.net_g(self.oriX, a=torch.zeros(self.oriX.shape[0], self.net_g_inc).cuda())[0]
        self.imgX1 = self.net_g(self.oriX, a=torch.ones(self.oriX.shape[0], self.net_g_inc).cuda())[0]

        if self.hparams.cmb != 'none':
            self.imgX0 = combine(self.imgX0, self.oriX, method=self.hparams.cmb)
            self.imgX1 = combine(self.imgX1, self.oriX, method=self.hparams.cmb)

    def backward_g(self, inputs):
        # ADV(X0)+
        loss_g = 0
        loss_g += self.add_loss_adv(a=self.imgX0, net_d=self.net_d, coeff=1, truth=True, stacked=False)

        # L1(X0, Y)
        loss_g += self.add_loss_L1(a=self.imgX0, b=self.oriY, coeff=self.hparams.lamb)

        # L1(X1, X)
        loss_g += self.add_loss_L1(a=self.imgX1, b=self.oriX, coeff=self.hparams.lamb * 10)

        return {'sum': loss_g, 'loss_g': loss_g}

    def backward_d(self, inputs):
        loss_d = 0
        # ADV(X0)-
        loss_d += self.add_loss_adv(a=self.imgX0, net_d=self.net_d, coeff=0.5, truth=False, stacked=False)

        # ADV(Y)+
        loss_d += self.add_loss_adv(a=self.oriY, net_d=self.net_d, coeff=0.5, truth=True)

        return {'sum': loss_d, 'loss_d': loss_d}

# CUDA_VISIBLE_DEVICES=0 python train.py --dataset pain -b 16 --prj VryNS4 --direction aregis1_b --resize 286 --engine NS4 --netG attgan
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset womac3 -b 16 --prj NS4/r256GDatt --direction aregis1_b --resize 256 --engine NS4 --netG attgan --netD attgan

# CUDA_VISIBLE_DEVICES=1 python train.py --dataset kl3 -b 16 --prj NS/unet128 --direction badKL3afterreg_goodKL3reg --resize 384 --cropsize 256 --engine pix2pixNS --netG unet_128