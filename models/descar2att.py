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
        print('using pix2pix.py')

        self.net_dY = copy.deepcopy(self.net_d)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--lb1", dest='lb1', type=float, default=10)
        parser.add_argument("--lb2", dest='lb2', type=float, default=10)
        return parent_parser

    def test_method(self, net_g, img):
        self.oriX = img[0]
        self.oriY = img[1]

        self.imgX0C, self.imgX0A = net_g(self.oriX, a=None)
        # self.imgY0, self.imgY1 = self.net_g(self.oriY, a=torch.zeros(self.oriX.shape[0], self.net_g_inc).cuda())

        self.imgX0A = nn.Sigmoid()(self.imgX0A)  # mask

        self.imgX0 = combine(self.imgX0C, self.imgX0A, method='mul') + combine(self.oriX, 1 - self.imgX0A, method='mul')

        self.imgX0[self.imgX0<=0] = 0
        self.imgX0[self.imgX0>=1] = 1

        if 0:
            self.imgX0 = self.imgX0 - self.imgX0.min()
            self.imgX0 = self.imgX0 / self.imgX0.max()
            self.imgX0 = self.imgX0 * 2 - 1
        #print([x.detach().cpu().numpy() for x in [self.oriX.min(), self.oriX.max(), self.imgX0.min(), self.imgX0.max()]])

        return self.imgX0

    def generation(self):
        img = self.batch['img']
        self.oriX = img[0]
        self.oriY = img[1]

        self.imgXYC, self.imgXYA = self.net_g(self.oriX)
        #self.imgY0, self.imgY1 = self.net_g(self.oriY, a=torch.zeros(self.oriX.shape[0], self.net_g_inc).cuda())

        self.imgXYA = nn.Sigmoid()(self.imgXYA)  # mask

        self.imgXY = combine(self.imgXYC, self.imgXYA, method='mul') + combine(self.oriX, 1 - self.imgXYA, method='mul')

        #self.imgY0 = nn.Sigmoid()(self.imgY0)  # mask

    def backward_g(self, inputs):
        # ADV(XY)+
        axy = self.add_loss_adv(a=self.imgXY, net_d=self.net_d, coeff=1, truth=True, stacked=False)

        # L1(X0, Y)
        loss_l1xy_y = self.add_loss_L1(a=self.imgXY, b=self.oriY, coeff=1)

        self.log('gaxy', axy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('l1xy_y', loss_l1xy_y, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        loss_g = axy + self.hparams.lamb * loss_l1xy_y
        return {'sum': loss_g, 'loss_g': loss_g}

    def backward_d(self, inputs):
        # ADV(XY)-
        axy = self.add_loss_adv(a=self.imgXY, net_d=self.net_d, coeff=1, truth=False, stacked=False)

        # ADV(Y)+
        ay = self.add_loss_adv(a=self.oriY, net_d=self.net_d, coeff=1, truth=True)

        self.log('daxy', axy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('day', ay, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        loss_d = 0.5 * axy + 0.5 * ay
        return {'sum': loss_d, 'loss_d': loss_d}

# CUDA_VISIBLE_DEVICES=0,1,2 python train.py --jsn womac3 --prj descar2att/descar2att/GdsattmcDugatit --mc --engine descar2att --netG dsattmc --netD ugatit  --direction areg_b --index --gray --final none --env a6k --n01