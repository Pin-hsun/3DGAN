import torch, copy
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

        self.seg = torch.load('submodels/model_seg_ZIB_res18_256.pth').cuda()
        self.t2d = torch.load('submodels/tse_dess_unet32.pth').cuda()
        self.seg.eval()
        self.t2d.eval()
        from models.cyclegan.models import Discriminator
        self.net_dseg = Discriminator(input_shape=(4, 256, 256), patch=16)

        self.netd_names = {'net_d': 'netD', 'net_dseg': 'netDseg'}

    @staticmethod
    def add_model_specific_args(parent_parser):
        return parent_parser

    def generation(self):
        self.oriX = self.batch[0]
        self.oriY = self.batch[1]
        self.imgX0, self.imgX1 = self.net_g(self.oriX, a=torch.zeros(self.oriX.shape[0], self.net_g_inc).cuda())

        self.oriXseg = torch.argmax(self.seg(self.t2d(self.oriX)[0]), 1).unsqueeze(1)
        self.imgX0seg = torch.argmax(self.seg(self.t2d(self.imgX0)[0]), 1).unsqueeze(1)

    def backward_g(self, inputs):
        # ADV(X0)+
        loss_g = 0
        loss_g = self.add_loss_adv(a=self.imgX0, net_d=self.net_d, loss=loss_g, coeff=1, truth=True, stacked=False)

        # L1(X0, Y)
        loss_g = self.add_loss_L1(a=self.imgX0, b=self.oriY, loss=loss_g, coeff=self.hparams.lamb)

        # L1(X1, X)
        loss_g = self.add_loss_L1(a=self.imgX1, b=self.oriX, loss=loss_g, coeff=self.hparams.lamb * 0.1)

        # ADV(X0, X0seg)+
        loss_g = self.add_loss_adv(a=self.imgX0, b=self.imgX0seg, net_d=self.net_dseg, loss=loss_g, coeff=1, truth=True, stacked=True)

        # ADV(X1)+
        #loss_g = self.add_loss_adv(a=self.imgX1, net_d=self.net_dY, loss=loss_g, coeff=1, truth=True, stacked=False)

        # L1(X0, X1)
        #loss_g = self.add_loss_L1(a=self.imgX0, b=self.imgX1, loss=loss_g, coeff=self.hparams.lamb * 0.1)

        return loss_g

    def backward_d(self, inputs):
        loss_d = 0
        # ADV(X0)-
        loss_d = self.add_loss_adv(a=self.imgX0, net_d=self.net_d, loss=loss_d, coeff=0.5, truth=False, stacked=False)

        # ADV(Y)+
        loss_d = self.add_loss_adv(a=self.oriY, net_d=self.net_d, loss=loss_d, coeff=0.5, truth=True)

        # ADV(X0, X0seg)-
        loss_d = self.add_loss_adv(a=self.imgX0, b=self.imgX0seg, net_d=self.net_dseg, loss=loss_d, coeff=0.5, truth=False, stacked=True)

        # ADV(X, Xseg)+
        loss_d = self.add_loss_adv(a=self.oriX, b=self.oriXseg, net_d=self.net_dseg, loss=loss_d, coeff=0.5, truth=True, stacked=True)

        # ADV(X1)-
        #loss_d = self.add_loss_adv(a=self.imgX1, net_d=self.net_dY, loss=loss_d, coeff=0.5, truth=False, stacked=False)

        # ADV(X)+
        #loss_d = self.add_loss_adv(a=self.oriX, net_d=self.net_dY, loss=loss_d, coeff=0.5, truth=True)

        return loss_d

# CUDA_VISIBLE_DEVICES=0 python train.py --dataset pain -b 16 --prj VryNS4 --direction aregis1_b --resize 286 --engine NS4 --netG attgan
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset womac3 -b 16 --prj NS4/r256GDatt --direction aregis1_b --resize 256 --engine NS4 --netG attgan --netD attgan