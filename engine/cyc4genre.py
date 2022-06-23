from engine.base import BaseModel, combine
import copy
import torch
import tifffile as tiff
import os
from dotenv import load_dotenv
import numpy as np
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

load_dotenv('.env')


class GAN(BaseModel):
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints)
        torch.autograd.set_detect_anomaly(True)
        from models.genre.generator.Unet_base import SPADEUNet, SPADEUNet2s

        opt = Namespace(input_size=256, parsing_nc=1, norm_G='spectralspadebatch3x3', spade_mode='res2',
                        use_en_feature=False)

        self.net_gXY = SPADEUNet2s(opt=opt, in_channels=1, out_channels=1)
        self.net_gYX = SPADEUNet2s(opt=opt, in_channels=1, out_channels=1)

        self.net_dXo = copy.deepcopy(self.net_d)
        self.net_dXw = copy.deepcopy(self.net_d)
        self.net_dYo = copy.deepcopy(self.net_d)
        self.net_dYw = copy.deepcopy(self.net_d)

        # save model names
        self.netg_names = {'net_gXY': 'netGXY', 'net_gYX': 'netGYX'}
        self.netd_names = {'net_dXw': 'netDXw', 'net_dYw': 'netDYw', 'net_dXo': 'netDXo', 'net_dYo': 'netDYo'}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        # coefficient for the identify loss
        parser.add_argument("--lambI", type=float, default=0.5)
        return parent_parser

    def test_method(self, net_g, x):
        #x[0] = ((x[0] > 0) / 1) * 2 - 1
        output, output1 = net_g(x[0], x[1])
        return output1

    def generation(self):  # 0
        # zyweak_zyorisb%xyweak_xyorisb
        img = self.batch['img']
        names = self.batch['filenames']

        self.oriXw = img[0]
        self.oriXo = img[1]
        self.oriYw = img[2]
        self.oriYo = img[3]

        # threshold XXX
        self.imgXYw, self.imgXYo = self.net_gXY(self.oriXo, self.oriXw)
        self.imgYXw, self.imgYXo = self.net_gYX(self.oriYo, self.oriYw)

        self.imgXYXw, self.imgXYXo = self.net_gYX(self.imgXYo, self.imgXYw)
        self.imgYXYw, self.imgYXYo = self.net_gXY(self.imgYXo, self.imgYXw)

        if self.hparams.lambI > 0:
            self.idt_Xw, self.idt_Xo = self.net_gYX(self.oriXo, self.oriXw)
            self.idt_Yw, self.idt_Yo, = self.net_gXY(self.oriYo, self.oriYw)

    def backward_g(self, inputs):
        loss_g = 0
        # ADV(XYw)+
        loss_g += self.add_loss_adv(a=self.imgXYw, b=None, net_d=self.net_dYw, coeff=1, truth=True, stacked=False)
        # ADV(YXw)+
        loss_g += self.add_loss_adv(a=self.imgYXw, b=None, net_d=self.net_dXw, coeff=1, truth=True, stacked=False)
        # ADV(XYo)+
        loss_g += self.add_loss_adv(a=self.imgXYo, b=None, net_d=self.net_dYo, coeff=1, truth=True, stacked=False)
        # ADV(YXo)+
        loss_g += self.add_loss_adv(a=self.imgYXo, b=None, net_d=self.net_dXo, coeff=1, truth=True, stacked=False)

        # Cyclic(XYXw, Xw)
        loss_g += self.add_loss_L1(a=self.imgXYXw, b=self.oriXw, coeff=self.hparams.lamb)
        # Cyclic(YXYw, Yw)
        loss_g += self.add_loss_L1(a=self.imgYXYw, b=self.oriYw, coeff=self.hparams.lamb)
        # Cyclic(XYXo, Xo)
        loss_g += self.add_loss_L1(a=self.imgXYXo, b=self.oriXo, coeff=self.hparams.lamb)
        # Cyclic(YXYo, Yo)
        loss_g += self.add_loss_L1(a=self.imgYXYo, b=self.oriYo, coeff=self.hparams.lamb)

        # Identity(idt_X, X)
        if self.hparams.lambI > 0:
            # Identity(idt_Xw, Xw)
            loss_g += self.add_loss_L1(a=self.idt_Xw, b=self.oriXw, coeff=self.hparams.lambI)
            # Identity(idt_Yw, Yw)
            loss_g += self.add_loss_L1(a=self.idt_Yw, b=self.oriYw, coeff=self.hparams.lambI)
            # Identity(idt_Xo, Xo)
            loss_g += self.add_loss_L1(a=self.idt_Xo, b=self.oriXo, coeff=self.hparams.lambI)
            # Identity(idt_Yo, Yo)
            loss_g += self.add_loss_L1(a=self.idt_Yo, b=self.oriYo, coeff=self.hparams.lambI)

        return {'sum': loss_g, 'loss_g': loss_g}

    def backward_d(self, inputs):
        loss_d = 0
        # ADV(XY)-
        loss_d += self.add_loss_adv(a=self.imgXYw, net_d=self.net_dYw, coeff=1, truth=False, stacked=False)
        loss_d += self.add_loss_adv(a=self.imgXYo, net_d=self.net_dYo, coeff=1, truth=False, stacked=False)

        # ADV(YX)-
        loss_d += self.add_loss_adv(a=self.imgYXw, net_d=self.net_dXw, coeff=1, truth=False, stacked=False)
        loss_d += self.add_loss_adv(a=self.imgYXo, net_d=self.net_dXo, coeff=1, truth=False, stacked=False)

        # ADV(Y)+
        loss_d += self.add_loss_adv(a=self.oriYw, net_d=self.net_dYw, coeff=1, truth=True, stacked=False)
        loss_d += self.add_loss_adv(a=self.oriYo, net_d=self.net_dYo, coeff=1, truth=True, stacked=False)

        # ADV(X)+
        loss_d += self.add_loss_adv(a=self.oriXw, net_d=self.net_dXw, coeff=1, truth=True, stacked=False)
        loss_d += self.add_loss_adv(a=self.oriXo, net_d=self.net_dXo, coeff=1, truth=True, stacked=False)

        return {'sum': loss_d, 'loss_d': loss_d}


# USAGE
# CUDA_VISIBLE_DEVICES=0 python train.py --jsn wnwp3d --prj wnwp3d/cyc4/GdenuWBmc --mc --engine cyc4 -b 16 --netG descarnoumc --direction zyweak_zysb%xyweak_xysb