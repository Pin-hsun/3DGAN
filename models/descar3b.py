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

from models.descar3 import GAN as GANbase


class GAN(GANbase):
    """
    There is a lot of patterned noise and other failures when using lightning
    """
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        GANbase.__init__(self, hparams, train_loader, test_loader, checkpoints)
        #self.net_class = copy.deepcopy(self.net_d)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--lb1", dest='lb1', type=float, default=10)
        return parent_parser

    def test_method(self, net_g, img):
        self.oriX = img[0]
        self.imgXY, self.imgXX = net_g(self.oriX, a=None)
        self.imgXY = nn.Sigmoid()(self.imgXY)  # mask
        self.imgXY = combine(self.imgXY, self.oriX, method='mul')

        return self.imgXY

    def generation(self):
        img = self.batch['img']
        self.filenames = self.batch['filenames']
        self.oriX = img[0]
        self.oriY = img[1]

        self.imgXY, self.imgXX = self.net_g(self.oriX, a=None)
        #self.imgYY, self.imgYX = self.net_g(self.oriY, a=None)

        self.imgXY = nn.Sigmoid()(self.imgXY)  # mask
        #self.imgYY = nn.Sigmoid()(self.imgYY)  # mask
        self.imgXY = combine(self.imgXY, self.oriX, method='mul')
        #self.imgYY = combine(self.imgYY, self.oriY, method='mul')

    def backward_g(self, inputs):
        # ADV(XY)+ -
        axy, _ = self.add_loss_adv_classify3d(a=self.imgXY, net_d=self.net_d, truth_adv=True, truth_classify=False)

        # ADV(XX)+ +
        #axx, cxx = self.add_loss_adv_classify(a=self.imgXX, net_d=self.net_d, truth_adv=True, truth_classify=True)

        # ADV(YY)+ -
        #ayy, cyy = self.add_loss_adv_classify(a=self.imgYY, net_d=self.net_d, truth_adv=True, truth_classify=False)

        # ADV(YX)+ +
        #ayx, cyx = self.add_loss_adv_classify(a=self.imgYX, net_d=self.net_d, truth_adv=True, truth_classify=True)

        # L1(XY, Y)
        loss_l1 = self.add_loss_L1(a=self.imgXY, b=self.oriY, coeff=self.hparams.lamb)

        loss_ga = axy
        #loss_gc = cxy + cxx + cyy + cyx
        loss_g = loss_ga + loss_l1

        self.log('l1', loss_l1, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('ga', loss_ga, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #self.log('gc', loss_gc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return {'sum': loss_g, 'loss_g': loss_g}

    def backward_d(self, inputs):
        id = self.filenames[0][0].split('/')[-1].split('_')[0]
        side = self.df.loc[self.df['ID'] == int(id), ['SIDE']].values[0][0]
        # ADV(XY)- -
        axy, _ = self.add_loss_adv_classify3d(a=self.imgXY, net_d=self.net_d, truth_adv=False, truth_classify=False)
        # ADV(XX)- +
        #axx, cxx = self.add_loss_adv_classify(a=self.imgXX, net_d=self.net_d, truth_adv=False, truth_classify=True)
        # ADV(YY)- -
        #ayy, cyy = self.add_loss_adv_classify(a=self.imgYY, net_d=self.net_d, truth_adv=False, truth_classify=False)
        # ADV(YX)- +
        #ayx, cyx = self.add_loss_adv_classify(a=self.imgYX, net_d=self.net_d, truth_adv=False, truth_classify=True)

        # ADV(X)+ +
        #ax, cx = self.add_loss_adv_classify3d(a=self.oriX, net_d=self.net_d, truth_adv=True, truth_classify=True)
        # ADV(Y)+ -
        #ay, cy = self.add_loss_adv_classify3d(a=self.oriY, net_d=self.net_d, truth_adv=True, truth_classify=False)
        truth_classify = (side == 'RIGHT')
        _, ay, cxy, _ = self.add_loss_adv_classify3d_paired(a=self.oriX, b=self.oriY, net_d=self.net_d, classifier=self.classifier,
                                                     truth_adv=True, truth_classify=truth_classify)

        _, _, cxxy, _ = self.add_loss_adv_classify3d_paired(a=self.oriX, b=self.imgXY, net_d=self.net_d, classifier=self.classifier,
                                                     truth_adv=True, truth_classify=truth_classify)

        loss_da = axy * 0.5 + ay * 0.5
        #loss_dc = 0.5 * cx + 0.5 * cy # + (cxy + cxx + cyy + cyx) * 0.5
        loss_dc = cxy * 1 + cxxy * 1
        loss_d = loss_da + loss_dc

        self.log('da', loss_da, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('dc', loss_dc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return {'sum': loss_d, 'loss_d': loss_d}



# CUDA_VISIBLE_DEVICES=0,1,2 python train.py --env a6k --jsn womac3 --prj Gds/descar3b/GdsmcDboatch16 --mc --engine descar3b --netG dsmc --netD bpatch_16 --direction ap_bp --index --bysubject --final none

#CUDA_VISIBLE_DEVICES=1 python testoai.py --jsn womac3 --direction a_b --prj Gds/descar3b/GdsmcDboatch16 --cropsize 384 --n01 --cmb not --gray --nepochs 40 201 20 --nalpha 0 100 100 --env a6k --engine descar3b