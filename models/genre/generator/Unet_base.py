# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File  : networks.py
# @Author: Jeffrey, Jehovah
# @Date  : 19-9

import torch
import torch.nn as nn
#from config.SAND_pix_opt import TrainOptions
#from models.genre.base_network import BaseNetwork
from models.genre.blocks.unet_block import UNetDown, SPADEUp


class SPADEUNet(nn.Module):
    def __init__(self, opt, in_channels=3, out_channels=3):
        super(SPADEUNet, self).__init__()
        self.opt = opt
        self.down1 = nn.Conv2d(in_channels, 64, 4, 2, 1)
        self.down2 = UNetDown(64, 128, norm_fun=nn.InstanceNorm2d)
        self.down3 = UNetDown(128, 256, norm_fun=nn.InstanceNorm2d)
        self.down4 = UNetDown(256, 512, norm_fun=nn.InstanceNorm2d)
        self.down5 = UNetDown(512, 512, norm_fun=nn.InstanceNorm2d)
        self.down6 = UNetDown(512, 512, norm_fun=nn.InstanceNorm2d)
        self.down7 = UNetDown(512, 512, norm_fun=nn.InstanceNorm2d)
        if self.opt.input_size > 256:
            self.down8 = UNetDown(512, 512, norm_fun=nn.InstanceNorm2d)
            self.down9 = UNetDown(512, 512, normalize=False)
        else:
            self.down8 = UNetDown(512, 512, normalize=False)

        self.up0 = SPADEUp(self.opt, 512, 512, 0.5)
        self.up1 = SPADEUp(self.opt, 1024, 512, 0.5)
        self.up2 = SPADEUp(self.opt, 1024, 512, 0.5)
        self.up3 = SPADEUp(self.opt, 1024, 512)
        if self.opt.input_size > 256:
            self.up3_plus = SPADEUp(self.opt, 1024, 512)
        self.up4 = SPADEUp(self.opt, 1024, 256)
        self.up5 = SPADEUp(self.opt, 512, 128)
        self.up6 = SPADEUp(self.opt, 256, 64)

        self.final = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x, parsing):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        if self.opt.input_size <= 256:
            u0 = self.up0(d8, parsing)
            u1 = self.up1(u0, parsing, d7)
            u2 = self.up2(u1, parsing, d6)
            u3 = self.up3(u2, parsing, d5)
            u4 = self.up4(u3, parsing, d4)
            u5 = self.up5(u4, parsing, d3)
            u6 = self.up6(u5, parsing, d2)
        else:
            d9 = self.down9(d8)
            u0 = self.up0(d9, parsing)
            u1 = self.up1(u0, parsing, d8)
            u2 = self.up2(u1, parsing, d7)
            u3 = self.up3(u2, parsing, d6)
            u3_p = self.up3_plus(u3, parsing, d5)
            u4 = self.up4(u3_p, parsing, d4)
            u5 = self.up5(u4, parsing, d3)
            u6 = self.up6(u5, parsing, d2)

        u7 = torch.cat([u6, d1], dim=1)
        u8 = self.final(u7)

        return u8, u8


class SPADEUNet_YPar(nn.Module):
    def __init__(self, opt, img_channel, par_channel, out_channels=3):
        super(SPADEUNet_YPar, self).__init__()
        self.opt = opt
        self.down_rgb = nn.Conv2d(img_channel, 64, 4, 2, 1)
        self.down_par = nn.Conv2d(par_channel, 64, 4, 2, 1)
        self.down2 = UNetDown(128, 128, norm_fun=nn.InstanceNorm2d)
        self.down3 = UNetDown(128, 256, norm_fun=nn.InstanceNorm2d)
        self.down4 = UNetDown(256, 512, norm_fun=nn.InstanceNorm2d)
        self.down5 = UNetDown(512, 512, norm_fun=nn.InstanceNorm2d)
        self.down6 = UNetDown(512, 512, norm_fun=nn.InstanceNorm2d)
        self.down7 = UNetDown(512, 512, norm_fun=nn.InstanceNorm2d)
        self.down8 = UNetDown(512, 512, normalize=False)
        self.up0 = SPADEUp(self.opt, 512, 512, 0.5, first=True)
        self.up1 = SPADEUp(self.opt, 1024, 512, 0.5)
        self.up2 = SPADEUp(self.opt, 1024, 512, 0.5)
        self.up3 = SPADEUp(self.opt, 1024, 512)
        self.up4 = SPADEUp(self.opt, 1024, 256)
        self.up5 = SPADEUp(self.opt, 512, 128)
        self.up6 = SPADEUp(self.opt, 256, 64)

        self.final_rgb = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )
        self.final_par = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, par_channel, 4, 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x, x_parsing, y_parsing):
        d1_rgb = self.down_rgb(x)
        d1_par = self.down_par(x_parsing)
        d1 = torch.cat([d1_rgb, d1_par], dim=1)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u0 = self.up0(d8, y_parsing)
        u1 = self.up1(u0, y_parsing, d7)
        u2 = self.up2(u1, y_parsing, d6)
        u3 = self.up3(u2, y_parsing, d5)
        u4 = self.up4(u3, y_parsing, d4)
        u5 = self.up5(u4, y_parsing, d3)
        u6, gamma_beta = self.up6(u5, y_parsing, d2, gamma_mode="final")

        u7_rgb = torch.cat([u6, d1_rgb], dim=1)
        u7_par = torch.cat([u6, d1_par], dim=1)
        u8_rgb = self.final_rgb(u7_rgb)
        u8_par = self.final_par(u7_par)

        return u8_rgb, u8_par
        # return u8_rgb


class SPADEUNet2(nn.Module):
    def __init__(self, opt, in_channels=3, out_channels=3):
        super(SPADEUNet2, self).__init__()
        self.opt = opt
        self.down1 = nn.Conv2d(in_channels, 64, 4, 2, 1)
        self.down2 = UNetDown(64, 128, norm_fun=nn.InstanceNorm2d)
        self.down3 = UNetDown(128, 256, norm_fun=nn.InstanceNorm2d)
        self.down4 = UNetDown(256, 512, norm_fun=nn.InstanceNorm2d)
        self.down5 = UNetDown(512, 512, norm_fun=nn.InstanceNorm2d)
        self.down6 = UNetDown(512, 512, norm_fun=nn.InstanceNorm2d)
        self.down7 = UNetDown(512, 512, norm_fun=nn.InstanceNorm2d)
        if self.opt.input_size > 256:
            self.down8 = UNetDown(512, 512, norm_fun=nn.InstanceNorm2d)
            self.down9 = UNetDown(512, 512, normalize=False)
        else:
            self.down8 = UNetDown(512, 512, normalize=False)

        self.up0 = SPADEUp(self.opt, 512, 512, 0.5)
        self.up1 = SPADEUp(self.opt, 1024, 512, 0.5)
        self.up2 = SPADEUp(self.opt, 1024, 512, 0.5)
        self.up3 = SPADEUp(self.opt, 1024, 512)
        if self.opt.input_size > 256:
            self.up3_plus = SPADEUp(self.opt, 1024, 512)
        self.up4 = SPADEUp(self.opt, 1024, 256)
        self.up5 = SPADEUp(self.opt, 512, 128)
        self.up6 = SPADEUp(self.opt, 256, 64)

        self.final = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

        self.final2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x, parsing):
        print(x.shape)
        d1 = self.down1(x)
        print(d1.shape)
        d2 = self.down2(d1)
        print(d2.shape)
        d3 = self.down3(d2)
        print(d3.shape)
        d4 = self.down4(d3)
        print(d4.shape)
        d5 = self.down5(d4)
        print(d5.shape)
        d6 = self.down6(d5)
        print(d6.shape)
        d7 = self.down7(d6)
        print(d7.shape)
        d8 = self.down8(d7)
        if self.opt.input_size <= 256:
            u0 = self.up0(d8, parsing)
            u1 = self.up1(u0, parsing, d7)
            u2 = self.up2(u1, parsing, d6)
            u3 = self.up3(u2, parsing, d5)
            u4 = self.up4(u3, parsing, d4)
            u5 = self.up5(u4, parsing, d3)
            u6 = self.up6(u5, parsing, d2)
        else:
            d9 = self.down9(d8)
            u0 = self.up0(d9, parsing)
            u1 = self.up1(u0, parsing, d8)
            u2 = self.up2(u1, parsing, d7)
            u3 = self.up3(u2, parsing, d6)
            u3_p = self.up3_plus(u3, parsing, d5)
            u4 = self.up4(u3_p, parsing, d4)
            u5 = self.up5(u4, parsing, d3)
            u6 = self.up6(u5, parsing, d2)

        u7 = torch.cat([u6, d1], dim=1)
        u8 = self.final(u7)
        u82 = self.final2(u7)

        return u8, u82


class SPADEUNet2s(nn.Module):
    def __init__(self, opt, in_channels=3, out_channels=3):
        super(SPADEUNet2s, self).__init__()
        self.opt = opt
        self.down1 = nn.Conv2d(in_channels, 64, 4, 2, 1)
        self.down2 = UNetDown(64, 128, norm_fun=nn.InstanceNorm2d)
        self.down3 = UNetDown(128, 256, norm_fun=nn.InstanceNorm2d)
        self.down4 = UNetDown(256, 512, norm_fun=nn.InstanceNorm2d)
        self.down5 = UNetDown(512, 512, norm_fun=nn.InstanceNorm2d)
        self.down6 = UNetDown(512, 512, norm_fun=nn.InstanceNorm2d)
        self.down7 = UNetDown(512, 512, norm_fun=nn.InstanceNorm2d)
        if self.opt.input_size > 256:
            self.down8 = UNetDown(512, 512, norm_fun=nn.InstanceNorm2d)
            self.down9 = UNetDown(512, 512, normalize=False)
        else:
            self.down8 = UNetDown(512, 512, normalize=False)

        self.up0 = SPADEUp(self.opt, 512, 512, 0.5)
        self.up1 = SPADEUp(self.opt, 1024, 512, 0.5)
        self.up2 = SPADEUp(self.opt, 512, 512, 0.5)
        self.up3 = SPADEUp(self.opt, 1024, 512)
        if self.opt.input_size > 256:
            self.up3_plus = SPADEUp(self.opt, 1024, 512)
        self.up4 = SPADEUp(self.opt, 1024, 256)
        self.up5 = SPADEUp(self.opt, 512, 128)
        self.up6 = SPADEUp(self.opt, 256, 64)

        self.final = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

        self.final2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x, parsing):
        #print(x.shape)
        d1 = self.down1(x)
        #print(d1.shape)
        d2 = self.down2(d1)
        #print(d2.shape)
        d3 = self.down3(d2)
        #print(d3.shape)
        d4 = self.down4(d3)
        #print(d4.shape)
        d5 = self.down5(d4)
        #print(d5.shape)
        d6 = self.down6(d5)
        #print(d6.shape)
        #d7 = self.down7(d6)
        #print(d7.shape)

        #u1 = self.up1(d7, parsing)
        u2 = self.up2(d6, parsing)
        u3 = self.up3(u2, parsing, d5)
        u4 = self.up4(u3, parsing, d4)
        u5 = self.up5(u4, parsing, d3)
        u6 = self.up6(u5, parsing, d2)


        u7 = torch.cat([u6, d1], dim=1)
        u8 = self.final(u7)
        u82 = self.final2(u7)

        return u8, u82


if __name__ == '__main__':
    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    dx = 384
    opt = Namespace(input_size=dx, parsing_nc=1, norm_G='spectralspadebatch3x3', spade_mode='res2',
                    use_en_feature=False)
    m = SPADEUNet2s(opt=opt, in_channels=1, out_channels=1)

    x = torch.randn([2, 1, dx, dx])
    parsing = (torch.randn([2, 1, dx, dx]) > 0).type(torch.FloatTensor)

    output = m(x, parsing)

    # model = OutterUNet(opt, in_channels=3, out_channels=1)
    #y_identity, gamma_beta = model.forward(style)
    #hat_y, _ = model.forward(x, use_basic=False, gamma=gamma_beta[0], beta=gamma_beta[1])
    #print(hat_y.size())
