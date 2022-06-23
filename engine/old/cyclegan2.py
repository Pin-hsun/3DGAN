from engine.base import BaseModel, combine
import copy
import torch
import tifffile as tiff
import os
from dotenv import load_dotenv
import numpy as np
load_dotenv('.env')


class GAN(BaseModel):
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints)

        args = self.train_loader.dataset.opt
        folder = '/train/'
        train_index = None
        args.bysubject = True
        args.cropsize = 128
        args.direction = 'zyweak128_zyori128'
        from dataloader.data_multi import MultiData as Dataset
        self.set3d = Dataset(root=os.environ.get('DATASET') + args.dataset + folder,
                            path=args.direction,
                            opt=args, mode='train', index=train_index)

        self.net_gXY = self.net_g
        self.net_gYX = copy.deepcopy(self.net_g)

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
        parser.add_argument("--lambI", type=int, default=0.5)
        return parent_parser

    def test_method(self, net_g, x):
        output, output1 = net_g(torch.cat((x[0], x[1]), 1), a=None)
        #output = output - output.min()
        #output1 = output1 - output1.min()
        #output = x[1]
        #output = combine(output, x[0], method='mul')
        #output1 = torch.mul(output1, (output > -0.7))
        return output

    def generation(self):  # 0
        # zyweak_zyorisb%xyweak_xyorisb
        self.oriXw = self.batch[0]
        self.oriXo = self.batch[1]
        self.oriYw = self.batch[2]
        self.oriYo = self.batch[3]

        #print(self.batch[0].shape)
        #print(self.batch[1].shape)
        #print(self.batch[2].shape)
        #print(self.batch[3].shape)

        self.imgXYw, self.imgXYo = self.net_gXY(torch.cat([self.batch[0], self.batch[1]], 1), a=None)
        self.imgYXw, self.imgYXo = self.net_gYX(torch.cat([self.batch[2], self.batch[3]], 1), a=None)

        self.imgXYXw, self.imgXYXo = self.net_gYX(torch.cat([self.imgXYw, self.imgXYo], 1), a=None)
        self.imgYXYw, self.imgYXYo = self.net_gXY(torch.cat([self.imgYXw, self.imgYXo], 1), a=None)

        if self.hparams.lambI > 0:
            self.idt_Xw, self.idt_Xo = self.net_gYX(torch.cat([self.batch[0], self.batch[1]], 1), a=None)
            self.idt_Yw, self.idt_Yo, = self.net_gXY(torch.cat([self.batch[2], self.batch[3]], 1), a=None)

        if self.epoch % 10 == 0:
            if self.batch_idx == 0:
                x = self.set3d.__getitem__(0)
                w3d = x[0].permute(3, 0, 1, 2).cuda()
                o3d = x[1].permute(3, 0, 1, 2).cuda()
                w3dxy, o3dxy = self.net_gXY(torch.cat([w3d, o3d], 1), a=None)

                tiff.imsave('temp/wzy' + str(self.epoch).zfill(3) + '.tif', w3dxy.detach().cpu().numpy()[:, 0, :, :])
                w3dxy = w3dxy.permute(2, 1, 0, 3)
                tiff.imsave('temp/wxy' + str(self.epoch).zfill(3) + '.tif', w3dxy.detach().cpu().numpy()[:, 0, :, :])

                #o3dxy = o3dxy.permute(2, 1, 0, 3)
                #tiff.imsave('temp/o' + str(self.epoch).zfill(3) + '.tif', o3dxy.detach().cpu().numpy()[:, 0, :, :])


    def generation1(self):  # 1
        # zyweak_zyorisb%xyweak_xyorisb
        self.oriXw = self.batch[0]
        self.oriXo = self.batch[1]
        self.oriYw = self.batch[2]
        self.oriYo = self.batch[3]

        self.imgXYw, self.imgXYo = self.net_gXY(torch.cat([self.batch[0], self.batch[1]], 1), a=None)
        self.imgYXw, self.imgYXo = self.net_gYX(torch.cat([self.batch[2], self.batch[3]], 1), a=None)

        self.imgXYw = combine(self.imgXYw, self.batch[0], method='mul')  # XY
        self.imgYXw = combine(self.imgYXw, self.batch[2], method='mul')  # YX
        self.imgXYo = combine(self.imgXYo, self.batch[1], method='mul')  # XY
        self.imgYXo = combine(self.imgYXo, self.batch[3], method='mul')  # YX

        self.imgXYXw, self.imgXYXo = self.net_gYX(torch.cat([self.imgXYw, self.imgXYo], 1), a=None)
        self.imgYXYw, self.imgYXYo = self.net_gXY(torch.cat([self.imgYXw, self.imgYXo], 1), a=None)

        self.imgXYXw = combine(self.imgXYXw, self.imgXYw, method='mul')  # YX
        self.imgYXYw = combine(self.imgYXYw, self.imgYXw, method='mul')  # XY
        self.imgXYXo = combine(self.imgXYXo, self.imgXYo, method='mul')  # YX
        self.imgYXYo = combine(self.imgYXYo, self.imgYXo, method='mul')  # XY

        if self.hparams.lambI > 0:
            self.idt_Xw, self.idt_Xo = self.net_gYX(torch.cat([self.batch[0], self.batch[1]], 1), a=None)
            self.idt_Yw, self.idt_Yo, = self.net_gXY(torch.cat([self.batch[2], self.batch[3]], 1), a=None)

            self.idt_Xw = combine(self.idt_Xw, self.batch[0], method='mul')  # YX
            self.idt_Yw = combine(self.idt_Yw, self.batch[2], method='mul')  # XY
            self.idt_Xo = combine(self.idt_Xo, self.batch[1], method='mul')  # YX
            self.idt_Yo = combine(self.idt_Yo, self.batch[3], method='mul')  # XY

    def backward_g(self, inputs):
        loss_g = 0
        # ADV(XYw)+
        loss_g = self.add_loss_adv(a=self.imgXYw, b=None, net_d=self.net_dYw, loss=loss_g, coeff=1, truth=True, stacked=False)
        # ADV(YXw)+
        loss_g = self.add_loss_adv(a=self.imgYXw, b=None, net_d=self.net_dXw, loss=loss_g, coeff=1, truth=True, stacked=False)
        # ADV(XYo)+
        loss_g = self.add_loss_adv(a=self.imgXYo, b=None, net_d=self.net_dYo, loss=loss_g, coeff=1, truth=True, stacked=False)
        # ADV(YXo)+
        loss_g = self.add_loss_adv(a=self.imgYXo, b=None, net_d=self.net_dXo, loss=loss_g, coeff=1, truth=True, stacked=False)

        # Cyclic(XYXw, Xw)
        loss_g = self.add_loss_L1(a=self.imgXYXw, b=self.oriXw, loss=loss_g, coeff=self.hparams.lamb)
        # Cyclic(YXYw, Yw)
        loss_g = self.add_loss_L1(a=self.imgYXYw, b=self.oriYw, loss=loss_g, coeff=self.hparams.lamb)
        # Cyclic(XYXo, Xo)
        loss_g = self.add_loss_L1(a=self.imgXYXo, b=self.oriXo, loss=loss_g, coeff=self.hparams.lamb)
        # Cyclic(YXYo, Yo)
        loss_g = self.add_loss_L1(a=self.imgYXYo, b=self.oriYo, loss=loss_g, coeff=self.hparams.lamb)

        # Identity(idt_X, X)
        if self.hparams.lambI > 0:
            # Identity(idt_Xw, Xw)
            loss_g = self.add_loss_L1(a=self.idt_Xw, b=self.oriXw, loss=loss_g, coeff=self.hparams.lambI)
            # Identity(idt_Yw, Yw)
            loss_g = self.add_loss_L1(a=self.idt_Yw, b=self.oriYw, loss=loss_g, coeff=self.hparams.lambI)
            # Identity(idt_Xo, Xo)
            loss_g = self.add_loss_L1(a=self.idt_Xo, b=self.oriXo, loss=loss_g, coeff=self.hparams.lambI)
            # Identity(idt_Yo, Yo)
            loss_g = self.add_loss_L1(a=self.idt_Yo, b=self.oriYo, loss=loss_g, coeff=self.hparams.lambI)

        return {'sum': loss_g, 'loss_g': loss_g}

    def backward_d(self, inputs):
        loss_d = 0
        # ADV(XY)-
        loss_d = self.add_loss_adv(a=self.imgXYw, net_d=self.net_dYw, loss=loss_d, coeff=1, truth=False, stacked=False)
        loss_d = self.add_loss_adv(a=self.imgXYo, net_d=self.net_dYo, loss=loss_d, coeff=1, truth=False, stacked=False)

        # ADV(YX)-
        loss_d = self.add_loss_adv(a=self.imgYXw, net_d=self.net_dXw, loss=loss_d, coeff=1, truth=False, stacked=False)
        loss_d = self.add_loss_adv(a=self.imgYXo, net_d=self.net_dXo, loss=loss_d, coeff=1, truth=False, stacked=False)

        # ADV(Y)+
        loss_d = self.add_loss_adv(a=self.oriYw, net_d=self.net_dYw, loss=loss_d, coeff=1, truth=True, stacked=False)
        loss_d = self.add_loss_adv(a=self.oriYo, net_d=self.net_dYo, loss=loss_d, coeff=1, truth=True, stacked=False)

        # ADV(X)+
        loss_d = self.add_loss_adv(a=self.oriXw, net_d=self.net_dXw, loss=loss_d, coeff=1, truth=True, stacked=False)
        loss_d = self.add_loss_adv(a=self.oriXo, net_d=self.net_dXo, loss=loss_d, coeff=1, truth=True, stacked=False)

        return {'sum': loss_d, 'loss_d': loss_d}


# USAGE
# CUDA_VISIBLE_DEVICES=0 python train.py --jsn wnwp --prj wnwp/cyc2/000 --engine cyclegan2 --direction zyweak_zyorisb%xyweak_xyorisb --input_nc 2 --output_nc 1

# CUDA_VISIBLE_DEVICES=0 python train.py --jsn wnwp3d --prj wnwp3d/cyc2/0sb --engine cyclegan2 --input_nc 2 --output_nc 1

# CUDA_VISIBLE_DEVICES=1 python train.py --jsn wnwp --prj wnwp/cyc2/1sb --engine cyclegan2 --cropsize 128 --direction zyweak_zyorisb%xyweak_xyorisb --input_nc 2 --output_nc 1 --gray --netG uneta_128