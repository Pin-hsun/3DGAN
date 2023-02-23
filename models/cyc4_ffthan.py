'''
20221227 add fft function
'''

from models.base import BaseModel, combine
import copy
import torch
import torch.nn as nn
import tifffile as tiff
import os
from dotenv import load_dotenv
import numpy as np
load_dotenv('.env')

class Global_Filter_nope(nn.Module):
    def __init__(self):
        super().__init__()
        self.complex_weight = nn.Parameter((torch.tensor([[1.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]])))

    def forward(self, x):
        weight = torch.view_as_complex(self.complex_weight.permute(1,0).contiguous())
        x = x.permute(0,2,3,1).contiguous()  
        out = torch.zeros((x.shape[0],x.shape[1],x.shape[2])).to('cuda')     
        x = torch.fft.fft(x, dim=3, norm='ortho')

        for i in range(x.shape[-1]):
            temp = x[:,:,:,i].contiguous() * weight[i]
            out = torch.add(out, temp)
        out = torch.unsqueeze(out, -1).permute(0,3,1,2).contiguous()
        out = out.type(torch.cuda.FloatTensor)
        return out


def Global_Filter(x, weight):
    #weight = weight.repeat(x.shape[0], 1, x.shape[2], x.shape[3]).cuda()
    #x = torch.multiply(x, weight.cuda()).sum(1).unsqueeze(1)
    #x = x[: :1, :]
    #x = x.permute(0,2,3,1).contiguous()
    #x = torch.fft.fft(x, dim=3, norm='ortho')
    #out = torch.unsqueeze(x[:,:,:,0], -1).permute(0,3,1,2).contiguous()ss

    out = [weight[i] * x[:, i, :, :] for i in range(x.shape[1])]
    out = torch.stack(out, 1).sum(1).unsqueeze(1)
    return out


class GAN(BaseModel):
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints)

        self.weight = nn.Parameter(torch.FloatTensor([1, 0, 0, 0]))#.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(16, 1, 128, 128)

        self.net_g, self.net_d = self.set_networks()

        #self.net_fft = Global_Filter()
        self.net_gXY = self.net_g
        self.net_gYX = copy.deepcopy(self.net_g)

        self.net_dXo = copy.deepcopy(self.net_d)
        self.net_dXw = copy.deepcopy(self.net_d)
        self.net_dYo = copy.deepcopy(self.net_d)
        self.net_dYw = copy.deepcopy(self.net_d)

        # save model names
        self.netg_names = {'net_gXY': 'netGXY', 'net_gYX': 'netGYX'}#,'net_fft': 'netFFT'}
        self.netd_names = {'net_dXw': 'netDXw', 'net_dYw': 'netDYw', 'net_dXo': 'netDXo', 'net_dYo': 'netDYo'}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        # coefficient for the identify loss
        parser.add_argument("--lambI", type=float, default=0.5)
        return parent_parser

    def test_method(self, net_g, x):
        output, output1 = net_g(torch.cat((x[0], x[1]), 1), a=None)
        return output1

    def generation(self, batch):
        img = batch['img']

        self.oriXw = Global_Filter(torch.cat([img[0], img[1], img[2], img[3]], 1), weight=self.weight)
        self.oriXo = img[4]
        self.oriYw = Global_Filter(torch.cat([img[5], img[6], img[7], img[8]], 1), weight=self.weight)
        self.oriYo = img[9]
        # tiff.imwrite('/home/meng-yun/Projects/weightedFFT/Results/test_fixed_w_3000/yzfft-1.tif', self.oriXw.detach().cpu().numpy())
        # tiff.imwrite('/home/meng-yun/Projects/weightedFFT/Results/test_fixed_w_3000/xyfft-1.tif', self.oriYw.detach().cpu().numpy())
        # assert 0

        xy = self.net_gXY(torch.cat([self.oriXw, self.oriXo], 1))
        yx = self.net_gYX(torch.cat([self.oriYw, self.oriYo], 1))
        self.imgXYw, self.imgXYo = xy['out0'], xy['out1']
        self.imgYXw, self.imgYXo = yx['out0'], yx['out1']

        xyx = self.net_gYX(torch.cat([self.imgXYw, self.imgXYo], 1))
        yxy = self.net_gXY(torch.cat([self.imgYXw, self.imgYXo], 1))
        self.imgXYXw, self.imgXYXo = xyx['out0'], xyx['out1']
        self.imgYXYw, self.imgYXYo = yxy['out0'], yxy['out1']

        if self.hparams.lambI > 0:
            yxi = self.net_gYX(torch.cat([self.oriXw, self.oriXo], 1))
            xyi = self.net_gXY(torch.cat([self.oriYw, self.oriYo], 1))
            self.idt_Xw, self.idt_Xo = yxi['out0'], yxi['out1']
            self.idt_Yw, self.idt_Yo = xyi['out0'], xyi['out1']

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

    def validation_epoch_end(self, x):
        print(self.weight)


# USAGE

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --jsn wnwp3d --prj cyc4fft/0 --models cyc4_ffthan -b 16 --direction zyf0_zyf1_zyf2_zyf3_zyori%xyf0_xyf1_xyf2_xyf3_xyzori --nm 11 --netG descarnoumc --env a6k --trd 2000