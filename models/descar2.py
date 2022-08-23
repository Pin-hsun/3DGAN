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

        self.imgXY, self.imgXX = self.net_g(self.oriX, a=None)

        if self.hparams.cmb is not None:
            self.imgXY = combine(self.imgXY, self.oriX, method=self.hparams.cmb)
            self.imgXX = combine(self.imgXX, self.oriX, method=self.hparams.cmb)

    def backward_g(self, inputs):
        # ADV(XY)+
        axy = self.add_loss_adv(a=self.imgXY, net_d=self.net_d, coeff=1, truth=True, stacked=False)

        # L1(XY, Y)
        loss_l1a = self.add_loss_L1(a=self.imgXY, b=self.oriY, coeff=1)

        # L1(XX, X)
        loss_l1b = self.add_loss_L1(a=self.imgXX, b=self.oriX, coeff=1)

        # ADV(X1)+
        #loss_g = self.add_loss_adv(a=self.imgX1, net_d=self.net_dY, loss=loss_g, coeff=1, truth=True, stacked=False)

        # L1(X0, X1)
        #loss_g = self.add_loss_L1(a=self.imgX0, b=self.imgX1, loss=loss_g, coeff=self.hparams.lamb * 0.1)

        self.log('gaxy', axy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('l1xy_y', loss_l1a, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('l1xx_y', loss_l1b, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        loss_g = axy + self.hparams.lamb * loss_l1a + self.hparams.lamb * loss_l1b
        return {'sum': loss_g, 'loss_g': loss_g}

    def backward_d(self, inputs):
        # ADV(X0)-
        axy = self.add_loss_adv(a=self.imgXY, net_d=self.net_d, coeff=1, truth=False, stacked=False)

        # ADV(Y)+
        ay = self.add_loss_adv(a=self.oriY, net_d=self.net_d, coeff=1, truth=True)

        # ADV(X1)-
        #loss_d = self.add_loss_adv(a=self.imgX1, net_d=self.net_dY, loss=loss_d, coeff=0.5, truth=False, stacked=False)

        # ADV(X)+
        #loss_d = self.add_loss_adv(a=self.oriX, net_d=self.net_dY, loss=loss_d, coeff=0.5, truth=True)

        self.log('daxy', axy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('day', ay, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        loss_d = 0.5 * axy + 0.5 * ay
        return {'sum': loss_d, 'loss_d': loss_d}

# CUDA_VISIBLE_DEVICES=0,1,2 python train.py --jsn womac3 --prj GDs/descar2/GdsmcCheck --engine descar2check --netG descarsmc --mc --direction areg_b --index --gray
#python testoai.py --jsn womac3 --direction a_b --prj GDs/descar2/GdsmcCheck --cropsize 384 --n01 --cmb mul --gray --nepochs 40 201 40 --nalpha 0 100 100 --engine descar2check