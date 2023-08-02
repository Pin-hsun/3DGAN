from models.base import BaseModel, combine
import copy
import torch
import torch.nn as nn
import tifffile as tiff
import os
from dotenv import load_dotenv
import numpy as np
load_dotenv('.env')


class GAN(BaseModel):
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints)

        self.net_g, self.net_d = self.set_networks()
        self.net_gXY = self.net_g
        self.net_gYX = copy.deepcopy(self.net_g)

        self.net_dXo = copy.deepcopy(self.net_d)
        self.net_dXw = copy.deepcopy(self.net_d)
        self.net_dYo = copy.deepcopy(self.net_d)
        self.net_dYw = copy.deepcopy(self.net_d)

        # save model names
        self.netg_names = {'net_gXY': 'netGXY', 'net_gYX': 'netGYX'}
        self.netd_names = {'net_dXw': 'netDXw', 'net_dYw': 'netDYw', 'net_dXo': 'netDXo', 'net_dYo': 'netDYo'}

        if self.hparams.downZ > 0:
            self.upz = nn.Upsample(size=(self.hparams.cropsize, self.hparams.cropsize), mode='bilinear').cuda()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        # coefficient for the identify loss
        parser.add_argument("--lambI", type=float, default=0.5)
        parser.add_argument("--downZ", type=int, default=0)
        return parent_parser

    def test_method(self, net_g, x):
        output, output1 = net_g(torch.cat((x[0], x[1]), 1), a=None)
        return output1

    def generation(self, batch):  # 0
        # zyweak_zyorisb%xyweak_xyorisb
        img = batch['img']

        self.oriXw = img[0]
        self.oriXo = img[1]

        #print((self.oriXw.min(), self.oriXw.max()))
        #print((self.oriXo.min(), self.oriXo.max()))

        if self.hparams.downZ > 0:
            self.oriXw = self.oriXw[:, :, ::self.hparams.downZ, :]
            self.oriXo = self.oriXo[:, :, ::self.hparams.downZ, :]
            self.oriXw = self.upz(self.oriXw)
            self.oriXo = self.upz(self.oriXo)

        self.oriYw = img[2]
        self.oriYo = img[3]

        outXY = self.net_gXY(torch.cat([img[0], img[1]], 1), a=None)
        outYX = self.net_gYX(torch.cat([img[2], img[3]], 1), a=None)
        self.imgXYw, self.imgXYo = outXY['out0'], outXY['out1']
        self.imgYXw, self.imgYXo = outYX['out0'], outYX['out1']

        outXYX = self.net_gYX(torch.cat([self.imgXYw, self.imgXYo], 1), a=None)
        outYXY = self.net_gXY(torch.cat([self.imgYXw, self.imgYXo], 1), a=None)
        self.imgXYXw, self.imgXYXo = outXYX['out0'], outXYX['out1']
        self.imgYXYw, self.imgYXYo = outYXY['out0'], outYXY['out1']

        if self.hparams.lambI > 0:
            outidtX = self.net_gYX(torch.cat([img[0], img[1]], 1), a=None)
            outidtY = self.net_gXY(torch.cat([img[2], img[3]], 1), a=None)
            self.idt_Xw, self.idt_Xo = outidtX['out0'], outidtX['out1']
            self.idt_Yw, self.idt_Yo = outidtY['out0'], outidtY['out1']

    def backward_g(self):
        loss_g = 0
        # ADV(XYw)+
        loss_g += self.add_loss_adv(a=self.imgXYw, net_d=self.net_dYw, truth=True)
        # ADV(YXw)+
        loss_g += self.add_loss_adv(a=self.imgYXw, net_d=self.net_dXw, truth=True)
        # ADV(XYo)+
        loss_g += self.add_loss_adv(a=self.imgXYo, net_d=self.net_dYo, truth=True)
        # ADV(YXo)+
        loss_g += self.add_loss_adv(a=self.imgYXo, net_d=self.net_dXo, truth=True)

        # Cyclic(XYXw, Xw)
        loss_g += self.add_loss_l1(a=self.imgXYXw, b=self.oriXw) * self.hparams.lamb
        # Cyclic(YXYw, Yw)
        loss_g += self.add_loss_l1(a=self.imgYXYw, b=self.oriYw) * self.hparams.lamb
        # Cyclic(XYXo, Xo)
        loss_g += self.add_loss_l1(a=self.imgXYXo, b=self.oriXo) * self.hparams.lamb
        # Cyclic(YXYo, Yo)
        loss_g += self.add_loss_l1(a=self.imgYXYo, b=self.oriYo) * self.hparams.lamb

        # Identity(idt_X, X)
        if self.hparams.lambI > 0:
            # Identity(idt_Xw, Xw)
            loss_g += self.add_loss_l1(a=self.idt_Xw, b=self.oriXw) * self.hparams.lambI
            # Identity(idt_Yw, Yw)
            loss_g += self.add_loss_l1(a=self.idt_Yw, b=self.oriYw) * self.hparams.lambI
            # Identity(idt_Xo, Xo)
            loss_g += self.add_loss_l1(a=self.idt_Xo, b=self.oriXo) * self.hparams.lambI
            # Identity(idt_Yo, Yo)
            loss_g += self.add_loss_l1(a=self.idt_Yo, b=self.oriYo) * self.hparams.lambI

        return {'sum': loss_g, 'loss_g': loss_g}

    def backward_d(self):
        loss_d = 0
        # ADV(XY)-
        loss_d += self.add_loss_adv(a=self.imgXYw, net_d=self.net_dYw, truth=False)
        loss_d += self.add_loss_adv(a=self.imgXYo, net_d=self.net_dYo, truth=False)

        # ADV(YX)-
        loss_d += self.add_loss_adv(a=self.imgYXw, net_d=self.net_dXw, truth=False)
        loss_d += self.add_loss_adv(a=self.imgYXo, net_d=self.net_dXo, truth=False)

        # ADV(Y)+
        loss_d += self.add_loss_adv(a=self.oriYw, net_d=self.net_dYw, truth=True)
        loss_d += self.add_loss_adv(a=self.oriYo, net_d=self.net_dYo, truth=True)

        # ADV(X)+
        loss_d += self.add_loss_adv(a=self.oriXw, net_d=self.net_dXw, truth=True)
        loss_d += self.add_loss_adv(a=self.oriXo, net_d=self.net_dXo, truth=True)

        return {'sum': loss_d, 'loss_d': loss_d}


# USAGE
# CUDA_VISIBLE_DEVICES=0,1 python train.py --jsn wnwp3d --prj cyc4/Check2023c --models cyc4 -b 16 --direction zyft0_zyori%xyzft0_xyzori --trd 2000 --nm 11 --netG descarnoumc --mc --preload
