import xml.sax
import tifffile
from models.base import BaseModel, combine
import copy
import torch
import matplotlib.pyplot as plt
from torch.nn.functional import avg_pool2d
import numpy as np

class GAN(BaseModel):
    """
    There is a lot of patterned noise and other failures when using lightning
    """
    def __init__(self, hparams, train_loader, test_loader, checkpoints, resume_ep):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints, resume_ep)

        self.net_g, self.net_d = self.set_networks()

        # save model names
        self.netg_names = {'net_g': 'net_g'}
        self.netd_names = {'net_d': 'net_d'}

        # interpolation network
        self.upsample = torch.nn.Upsample(size=(hparams.cropsize, hparams.cropsize, hparams.cropz * 8))
        self.depth = 4

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        # coefficient for the identify loss
        parser.add_argument("--lambI", type=int, default=0.5)
        return parent_parser

    def test_method(self, net_g, img):
        output = net_g(img[0])
        #output = combine(output, x[0], method='mul')
        return output[0]

    def generation(self, batch):

        self.ori = batch['img'][0]  # (B, C, X, Y, Z)

        ori = self.upsample(self.ori)  # (B, C, X, Y, Z)
        # tifffile.imsave('out/x.tif', ori.detach().cpu().numpy())

        xy = ori.permute(4, 1, 2, 3, 0)[::1, :, :, :, 0]  # (Z, C, X, Y)

        self.oriX = xy
        #self.oriY = ori

        #self.imgXY = self.net_gXY(self.oriX)['out0']
        self.imgYX = self.net_g(ori)['out0']  # (B, C, X, Y, Z)
        # tifffile.imsave('out/xy.tif', self.imgYX.detach().cpu().numpy())

        #if self.hparams.lamb > 0:
        #    self.imgXYX = self.net_gYX(self.imgXY)['out0']
        #    self.imgYXY = self.net_gXY(self.imgYX)['out0']

        #if self.hparams.lambI > 0:
        #    self.idt_X = self.net_gYX(self.oriX)['out0']
        #    self.idt_Y = self.net_gXY(self.oriY)['out0']

    def backward_g(self):
        # ADV(XY)+
        #loss_g += self.add_loss_adv(a=self.imgXY, net_d=self.net_d, truth=True)
        # ADV(YX)+

        loss_g_gan = 0
        loss_g_gan += self.add_loss_adv(a=self.imgYX.permute(2, 1, 4, 3, 0)[:, :, :, :, 0],  # (X, C, Z, Y)
                                       net_d=self.net_d, truth=True)
        loss_g_gan += self.add_loss_adv(a=self.imgYX.permute(2, 1, 3, 4, 0)[:, :, :, :, 0],  # (X, C, Y, Z)
                                       net_d=self.net_d, truth=True)
        loss_g_gan += self.add_loss_adv(a=self.imgYX.permute(3, 1, 4, 2, 0)[:, :, :, :, 0],  # (Y, C, Z, X)
                                       net_d=self.net_d, truth=True)
        loss_g_gan += self.add_loss_adv(a=self.imgYX.permute(3, 1, 2, 4, 0)[:, :, :, :, 0],  # (Y, C, X, Z)
                                       net_d=self.net_d, truth=True)
        loss_g_gan = loss_g_gan / 4

        loss_g_l1 = self.add_loss_l1(a=self.imgYX[:, :, :, :, ::8], b=self.ori[:, :, :, :, :]) * self.hparams.lamb


        # Cyclic(XYX, X)
        #if self.hparams.lamb > 0:
        #    loss_g += self.add_loss_l1(a=self.imgXYX, b=self.oriX) * self.hparams.lamb
            # Cyclic(YXY, Y)
        #    loss_g += self.add_loss_l1(a=self.imgYXY, b=self.oriY) * self.hparams.lamb

        # Identity(idt_X, X)
        #if self.hparams.lambI > 0:
        #    loss_g += self.add_loss_l1(a=self.idt_X, b=self.oriX) * self.hparams.lambI
            # Identity(idt_Y, Y)
        #    loss_g += self.add_loss_l1(a=self.idt_Y, b=self.oriY) * self.hparams.lambI

        loss_g = loss_g_gan + loss_g_l1

        return {'sum': loss_g, 'loss_g_gan': loss_g_gan, 'loss_g_l1': loss_g_l1}

    def backward_d(self):
        loss_d = 0
        # ADV(XY)-
        #loss_d += self.add_loss_adv(a=self.imgXY, net_d=self.net_d, truth=False)

        loss_d_gan = 0
        loss_d_gan += self.add_loss_adv(a=self.imgYX.permute(2, 1, 4, 3, 0)[:, :, :, :, 0],  # (X, C, Z, Y)
                                       net_d=self.net_d, truth=False)
        loss_d_gan += self.add_loss_adv(a=self.imgYX.permute(2, 1, 3, 4, 0)[:, :, :, :, 0],  # (X, C, Y, Z)
                                       net_d=self.net_d, truth=False)
        loss_d_gan += self.add_loss_adv(a=self.imgYX.permute(3, 1, 4, 2, 0)[:, :, :, :, 0],  # (Y, C, Z, X)
                                       net_d=self.net_d, truth=False)
        loss_d_gan += self.add_loss_adv(a=self.imgYX.permute(3, 1, 2, 4, 0)[:, :, :, :, 0],  # (Y, C, X, Z)
                                       net_d=self.net_d, truth=False)
        loss_d_gan = loss_d_gan / 4
        loss_d += loss_d_gan

        # ADV(Y)+
        #loss_d += self.add_loss_adv(a=self.oriY, net_d=self.net_d, truth=True)

        # ADV(X)+
        loss_d += self.add_loss_adv(a=self.oriX, net_d=self.net_d, truth=True)

        return {'sum': loss_d, 'loss_d': loss_d}


# USAGE
# CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --jsn cyc_imorphics --prj cyc_oai3d_1/23d/ --models cyc_oai3d_1 --cropz 16 --cropsize 128 --netG ed023d