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
from engine.base import BaseModel


class GAN(BaseModel):
    """
    There is a lot of patterned noise and other failures when using lightning
    """
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints)
        print('using pix2pix.py')

    def generation(self):
        oriX = self.oriX
        self.imgX0 = self.net_g(oriX, a=torch.zeros(oriX.shape[0], self.net_g_inc).cuda())[0]
        self.imgX1 = self.net_g(oriX, a=torch.ones(oriX.shape[0], self.net_g_inc).cuda())[0]
        #self.imgX0 = self.net_g(oriX)[0]
        #self.imgX1 = self.net_g(oriX)[0]

    def backward_g(self, inputs):
        # ADV(X0)+
        loss_g = 0
        loss_g = self.add_loss_adv(a=self.imgX0, b=None, net_d=self.net_d, loss=loss_g, coeff=1, truth=True, stacked=False)

        # L1(X0, Y)
        loss_g = self.add_loss_L1(a=self.imgX0, b=self.oriY, loss=loss_g, coeff=self.hparams.lamb)

        # L1(X1, X)
        loss_g = self.add_loss_L1(a=self.imgX1, b=self.oriX, loss=loss_g, coeff=self.hparams.lamb * 0.1)

        # ADV(X1)+
        loss_g = self.add_loss_adv(a=self.imgX1, b=None, net_d=self.net_d, loss=loss_g, coeff=1, truth=True, stacked=False)
        return loss_g

    def backward_d(self, inputs):
        loss_d = 0
        # ADV(X0)-
        loss_d = self.add_loss_adv(a=self.imgX0, b=None, net_d=self.net_d, loss=loss_d, coeff=0.25, truth=False, stacked=False)

        # ADV(Y)+
        loss_d = self.add_loss_adv(a=self.oriY, b=None, net_d=self.net_d, loss=loss_d, coeff=0.5, truth=True)

        # ADV(X1)-
        loss_d = self.add_loss_adv(a=self.imgX1, b=None, net_d=self.net_d, loss=loss_d, coeff=0.25, truth=False, stacked=False)
        return loss_d

