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
from networks.losses_srgan import GANLoss, TVLoss, VGGLoss


class GAN(BaseModel):
    def __init__(self, hparams, train_loader, test_loader, checkpoints, resume_ep):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints, resume_ep)
        # First, modify the hyper-parameters if necessary
        # Initialize the networks

        from networks.networks_srgan import SRResNet, Discriminator
        self.net_g = SRResNet(self.hparams.scale_factor, 64, 16)
        self.net_d = Discriminator(65)

        # update names of the models for optimization
        self.netg_names = {'net_g': 'net_g'}
        self.netd_names = {'net_d': 'net_d'}

        # training criterions
        self.criterion_MSE = nn.MSELoss()
        self.criterion_VGG = VGGLoss(net_type='vgg19', layer='relu5_4')
        self.criterion_GAN = GANLoss(gan_mode='wgangp')
        self.criterion_TV = TVLoss()

        # Finally, initialize the optimizers and scheduler
        self.configure_optimizers()

        self.upsample = torch.nn.Upsample(size=(hparams.cropsize, hparams.cropsize, hparams.cropz * 8))
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--stack", dest='stack', action='store_true', default=False)
        parser.add_argument("--scale_factor", dest='scale_factor', type=int, help='scale factor', default=4)
        return parent_parser

    def generation(self, batch):
        img = batch['img']
        ori = img[0]
        self.oriX = nn.Upsample(size=(16, 16))(ori)
        self.oriY = nn.Upsample(size=(128, 128))(ori)

        self.imgX0 = self.net_g(self.oriX)

    def backward_g(self):
        # MSE Loss

        mse_loss = self.criterion_MSE(self.oriY,  # \in [-1, 1]
                                      self.imgX0)  # \in [-1, 1]
        # VGG Loss
        vgg_loss = self.criterion_VGG(self.oriY, self.imgX0)
        content_loss = (vgg_loss + mse_loss) / 2

        # tv loss
        tv_loss = self.criterion_TV(self.imgX0)

        # ADV(X0, Y)+
        adv_loss = self.add_loss_adv(a=self.imgX0, net_d=self.net_d, truth=True) * 1

        g_loss = 1e3 * content_loss + 1 * adv_loss + 2e-5 * tv_loss

        return {'sum': g_loss, 'adv_loss': adv_loss, 'mse': mse_loss, 'vgg': vgg_loss, 'tv': tv_loss}

    def backward_d(self):
        loss_d = 0
        # ADV(X0, Y)-
        loss_d += self.add_loss_adv(a=self.imgX0, net_d=self.net_d, truth=False) * 0.5

        # ADV(X, Y)+
        loss_d += self.add_loss_adv(a=self.oriY, net_d=self.net_d, truth=True) * 0.5

        return {'sum': loss_d, 'loss_d': loss_d}


# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --jsn wnwp3d --prj NS/ori_ft0 --models NS -b 16 --direction xyori_xyft0 --trd 2000 --nm 11 --netG descarnoumc --mc --preload