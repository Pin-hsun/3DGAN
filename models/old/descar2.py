import torch, copy
import torch.nn as nn
import torchvision
import torch.optim as optim
from networks.loss import GANLoss
from math import log10
import time, os
import pytorch_lightning as pl
from utils.metrics_segmentation import SegmentationCrossEntropyLoss
from utils.metrics_classification import CrossEntropyLoss, GetAUC
from utils.data_utils import *
from models.base import BaseModel, combine


class GAN(BaseModel):
    """
    There is a lot of patterned noise and other failures when using lightning
    """
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints)
        self.net_dY = copy.deepcopy(self.net_d)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--lb1", dest='lb1', type=float, default=10)
        return parent_parser

    def test_method(self, net_g, img):
        self.oriX = img[0]

        self.imgX0, self.imgX1 = net_g(self.oriX, a=None)
        return self.imgX0

    def generation(self):
        img = self.batch['img']
        self.oriX = img[0]
        self.oriY = img[1]

        self.imgX0, self.imgX1 = self.net_g(self.oriX, a=None)

        if self.hparams.cmb is not None:
            self.imgX0 = combine(self.imgX0, self.oriX, method=self.hparams.cmb)
            self.imgX1 = combine(self.imgX1, self.oriX, method=self.hparams.cmb)

    def backward_g(self, inputs):
        # ADV(X0)+
        loss_g = 0
        loss_g += self.add_loss_adv(a=self.imgX0, net_d=self.net_d, coeff=1, truth=True, stacked=False)

        # L1(X0, Y)
        loss_g += self.add_loss_L1(a=self.imgX0, b=self.oriY, coeff=self.hparams.lamb)

        # L1(X1, X)
        loss_g += self.add_loss_L1(a=self.imgX1, b=self.oriX, coeff=self.hparams.lb1)

        # ADV(X1)+
        #loss_g = self.add_loss_adv(a=self.imgX1, net_d=self.net_dY, loss=loss_g, coeff=1, truth=True, stacked=False)

        # L1(X0, X1)
        #loss_g = self.add_loss_L1(a=self.imgX0, b=self.imgX1, loss=loss_g, coeff=self.hparams.lamb * 0.1)

        return {'sum': loss_g, 'loss_g': loss_g}

    def backward_d(self, inputs):
        loss_d = 0
        # ADV(X0)-
        loss_d += self.add_loss_adv(a=self.imgX0, net_d=self.net_d, coeff=0.5, truth=False, stacked=False)

        # ADV(Y)+
        loss_d += self.add_loss_adv(a=self.oriY, net_d=self.net_d, coeff=0.5, truth=True)

        # ADV(X1)-
        #loss_d = self.add_loss_adv(a=self.imgX1, net_d=self.net_dY, loss=loss_d, coeff=0.5, truth=False, stacked=False)

        # ADV(X)+
        #loss_d = self.add_loss_adv(a=self.oriX, net_d=self.net_dY, loss=loss_d, coeff=0.5, truth=True)

        return {'sum': loss_d, 'loss_d': loss_d}

# CUDA_VISIBLE_DEVICES=1 python train.py --jsn womac3 --prj mcfix/descar2/Gdescarsmc_index2_check --engine descar2 --netG descarsmc --mc --direction areg_b --index --gray