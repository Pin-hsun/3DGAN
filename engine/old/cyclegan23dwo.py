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

        args = copy.deepcopy(self.train_loader.dataset.opt)
        folder = '/train/'
        train_index = None
        args.bysubject = True
        args.cropsize = 128
        args.direction = 'zyweak128_zyori128'
        from dataloader.data_multi import MultiData as Dataset
        if 0:
            self.set3d = Dataset(root=os.environ.get('DATASET') + args.dataset + folder,
                                path=args.direction,
                                opt=args, mode='train', index=train_index)

        self.net_gXY = self.net_g
        self.net_gYX = copy.deepcopy(self.net_g)
        if 0:
            from models.networks2 import define_G
            self.net_gYX = define_G(input_nc=self.hparams.input_nc, output_nc=self.hparams.output_nc,
                                  ngf=self.hparams.ngf, netG='uneta_32',
                                  norm=self.hparams.norm, use_dropout=self.hparams.mc, init_type='normal', init_gain=0.02,
                                  gpu_ids=[],
                                  final=self.hparams.final)

        self.net_dXo = copy.deepcopy(self.net_d)
        self.net_dXw = copy.deepcopy(self.net_d)
        self.net_dYo = copy.deepcopy(self.net_d)
        self.net_dYw = copy.deepcopy(self.net_d)

        # save model names
        self.netg_names = {'net_gXY': 'netGXY', 'net_gYX': 'netGYX'}
        self.netd_names = {'net_dXw': 'netDXw', 'net_dYw': 'netDYw', 'net_dXo': 'netDXo', 'net_dYo': 'netDYo'}

        self.w2sb = torch.load('submodels/w2sb.pth').cuda()#torch.load('/media/ExtHDD01/logs/Fly3D/wpopsb/Gunet128Crop/checkpoints/netG_model_epoch_90.pth').cuda()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        # coefficient for the identify loss
        parser.add_argument("--lambI", type=float, default=0.5)
        parser.add_argument("--lambZ", type=float, default=0)
        return parent_parser

    def test_method(self, net_g, x):
        output, output1 = net_g(torch.cat((x[0], x[1]), 1), a=None)
        #output = output[:,:1,:, :]

        if 0:
            w2sb = torch.load('/media/ExtHDD01/logs/Fly3D/wpopsb/Gunet128Crop/checkpoints/netG_model_epoch_90.pth').cuda()
            s = output.shape[0] // 4
            oa = w2sb(output.repeat(1, 3, 1, 1)[:s, ::])[0][:, :1, :, :]
            ob = w2sb(output.repeat(1, 3, 1, 1)[s:2*s, ::])[0][:, :1, :, :]
            oc = w2sb(output.repeat(1, 3, 1, 1)[2*s:3*s, ::])[0][:, :1, :, :]
            od = w2sb(output.repeat(1, 3, 1, 1)[3*s:, ::])[0][:, :1, :, :]
            output = torch.cat([oa, ob, oc, od], 0)
        #output = output - output.min()
        #output1 = output1 - output1.min()
        #output = x[1]
        #output = combine(output, x[0], method='mul')
        #output1 = torch.mul(output1, (output > -0.7))
        return output1

    def generation(self):  # 0
        # zyweak_zyorisb%xyweak_xyorisb
        self.oriXw = self.batch[0]
        self.oriXo = self.batch[1]
        self.oriYw = self.batch[2]
        self.oriYo = self.batch[3]

        self.imgXYw, self.imgXYo = self.net_gXY(torch.cat([self.batch[0], self.batch[1]], 1), a=None)
        self.imgYXw, self.imgYXo = self.net_gYX(torch.cat([self.batch[2], self.batch[3]], 1), a=None)

        self.imgXYXw, self.imgXYXo = self.net_gYX(torch.cat([self.imgXYw, self.imgXYo], 1), a=None)
        self.imgYXYw, self.imgYXYo = self.net_gXY(torch.cat([self.imgYXw, self.imgYXo], 1), a=None)

        if self.hparams.lambI > 0:
            self.idt_Xw, self.idt_Xo = self.net_gYX(torch.cat([self.batch[0], self.batch[1]], 1), a=None)
            self.idt_Yw, self.idt_Yo, = self.net_gXY(torch.cat([self.batch[2], self.batch[3]], 1), a=None)

        self.imgXYo_target = self.w2sb(self.imgXYw.repeat(1, 3, 1, 1))[0][:, :1, :, :]

        if self.hparams.lambZ > 0:
            x = self.set3d.__getitem__(np.random.randint(len(self.set3d)))
            w3d = x[0].permute(3, 0, 1, 2).cuda()
            o3d = x[1].permute(3, 0, 1, 2).cuda()
            w3dxy, o3dxy = self.net_gXY(torch.cat([w3d, o3d], 1), a=None)
            self.w3d = w3dxy.permute(2, 1, 0, 3)
            self.o3d = o3dxy.permute(2, 1, 0, 3)
            #self.w3d = self.w3d[::8, :, :, :]
            self.o3d = self.o3d[::8, :, :, :]

            self.w3d = self.w2sb(self.o3d.repeat(1, 3, 1, 1))[0][:, :1, :, :]

            self.woo = x[0].permute(1, 0, 3, 2)[::8, ::].cuda()
            self.ooo = x[1].permute(1, 0, 3, 2)[::8, ::].cuda()

        if 0:#self.epoch % 10 == 0:
            if self.batch_idx == 0:
                x = self.set3d.__getitem__(0)  # (C, Z, Y, X)
                wzy = x[0].permute(3, 0, 1, 2).cuda()  # (X, C, Z, Y)
                ozy = x[1].permute(3, 0, 1, 2).cuda()
                wzyxy, ozyxy = self.net_gXY(torch.cat([wzy, ozy], 1), a=None)  # (X, C, Z, Y)
                os.makedirs('outputs/' + self.hparams.prj + '/', exist_ok=True)

                tiff.imsave('outputs/' + self.hparams.prj + '/wzy' + str(self.epoch).zfill(3) + '.tif', wzyxy.detach().cpu().numpy()[:, 0, :, :])
                wzyxy = wzyxy[:, :, ::8, :]  # (X, C, Z+, Y)
                wzyxyxy = wzyxy.permute(2, 1, 0, 3)  # (Z+, C, X, Y)
                tiff.imsave('outputs/' + self.hparams.prj + '/wxy' + str(self.epoch).zfill(3) + '.tif', wzyxyxy.detach().cpu().numpy()[:, 0, :, :])

                wxyoriginal = x[0].permute(1, 0, 3, 2)  # (Z, C, X, Y)
                wxyoriginal = wxyoriginal[::8, :, ::]
                tiff.imsave('outputs/' + self.hparams.prj + '/wxyo' + str(self.epoch).zfill(3) + '.tif', wxyoriginal.numpy()[:, 0, :, :])

                #o3dxy = o3dxy.permute(2, 1, 0, 3)
                #tiff.imsave('temp/o' + str(self.epoch).zfill(3) + '.tif', o3dxy.detach().cpu().numpy()[:, 0, :, :])

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

        if 0:
            # ADV 3D
            loss_g = self.add_loss_adv(a=self.w3d, b=None, net_d=self.net_dYw, loss=loss_g, coeff=1, truth=True,
                                       stacked=False)
            loss_g = self.add_loss_adv(a=self.o3d, b=None, net_d=self.net_dYo, loss=loss_g, coeff=1, truth=True,
                                       stacked=False)

        # L1 3D
        if self.hparams.lambZ > 0:
            loss_g = self.add_loss_L1(a=self.w3d, b=self.woo, loss=loss_g, coeff=self.hparams.lambZ)
            loss_g = self.add_loss_L1(a=self.o3d, b=self.ooo, loss=loss_g, coeff=self.hparams.lambZ)

        # L! w-o
        #loss_g = self.add_loss_L1(a=self.imgXYo, b=self.imgXYo_target, loss=loss_g, coeff=self.hparams.lamb)

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

        if 0:
            # ADV 3D
            loss_d = self.add_loss_adv(a=self.w3d, b=None, net_d=self.net_dYw, loss=loss_d, coeff=1, truth=False,
                                       stacked=False)
            loss_d = self.add_loss_adv(a=self.o3d, b=None, net_d=self.net_dYo, loss=loss_d, coeff=1, truth=False,
                                       stacked=False)
            # ADV(Y)+
            loss_d = self.add_loss_adv(a=self.oriYw, net_d=self.net_dYw, loss=loss_d, coeff=1, truth=True, stacked=False)
            loss_d = self.add_loss_adv(a=self.oriYo, net_d=self.net_dYo, loss=loss_d, coeff=1, truth=True, stacked=False)

        return {'sum': loss_d, 'loss_d': loss_d}


# USAGE
# CUDA_VISIBLE_DEVICES=0 python train.py --jsn wnwp --prj wnwp/cyc2/000 --engine cyclegan2 --direction zyweak_zyorisb%xyweak_xyorisb --input_nc 2 --output_nc 1

# CUDA_VISIBLE_DEVICES=0 python train.py --jsn wnwp3d --prj wnwp3d/cyc23dwo/0sb --engine cyclegan23dwo --input_nc 2 --output_nc 1

# CUDA_VISIBLE_DEVICES=0 python train.py --jsn wnwp3d --prj wnwp3d/cyc3/GdenuWSmcZ1 --mc --engine cyclegan23dwo --netG descarnoumc --lambZ 0

# CUDA_VISIBLE_DEVICES=0 python train.py --jsn wnwp3d --prj wnwp3d/cyc3/GdenuWBmcZ1 --mc --engine cyclegan23dwo -b 16 --netG descarnoumc --lambZ 0 --direction zyweak_zysb%xyweak_xysb