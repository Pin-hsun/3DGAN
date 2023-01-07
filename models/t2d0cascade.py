from models.base import BaseModel, combine
import torch
import copy


class GAN(BaseModel):
    """
    There is a lot of patterned noise and other failures when using lightning
    """
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints)

        # First, modify the hyper-parameters if necessary
        # Initialize the networks
        self.net_gLR, self.net_dLR = self.set_networks()
        self.segd = torch.load('submodels/model_seg_ZIB_res18_256.pth').cuda()

        self.net_d = copy.deepcopy(self.net_dLR)

        self.hparams.input_nc = 2
        self.net_g, _ = self.set_networks()

        #self.segt2d = copy.deepcopy(self.segd)

        self.hparams.output_nc = 6
        _, self.net_dseg = self.set_networks()

        # update names of the models for optimization
        self.netg_names = {'net_g': 'net_g', 'net_gLR': 'net_gLR'}
        self.netd_names = {'net_d': 'net_d', 'net_dseg': 'net_dseg'}

        # Finally, initialize the optimizers and scheduler
        self.init_optimizer_scheduler()

    @staticmethod
    def add_model_specific_args(parent_parser):
        return parent_parser

    @staticmethod
    def test_method(net_g, net_gLR, img):
        t = img[0]
        tLR = torch.nn.Upsample(scale_factor=0.5, mode='bilinear')(t)
        t2dLR = net_gLR(tLR)[0]
        t2dLRHR = torch.nn.Upsample(scale_factor=2.0, mode='bilinear')(t2dLR)
        t2d = net_g(torch.cat([t, t2dLRHR], 1))[0]
        return t2d

    def generation(self, batch):
        img = batch['img']

        self.t = img[0]
        self.d = img[1]

        self.tLR = torch.nn.Upsample(scale_factor=0.5, mode='bilinear')(self.t)
        self.dLR = torch.nn.Upsample(scale_factor=0.5, mode='bilinear')(self.d)
        self.t2dLR = self.net_gLR(self.tLR)[0]

        self.t2dLRHR = torch.nn.Upsample(scale_factor=2.0, mode='bilinear')(self.t2dLR)

        self.t2d = self.net_g(torch.cat([self.t, self.t2dLRHR], 1))[0]

        self.dseg = self.segd(self.d.repeat(1, 3, 1, 1))
        self.t2dseg = self.segd(self.t2d.repeat(1, 3, 1, 1))

        self.t2dLRHRseg = self.segd(self.t2dLRHR.repeat(1, 3, 1, 1))

    def backward_g(self):
        # ADV(X0, Y)+
        loss_ga = self.add_loss_adv(a=self.t2d, net_d=self.net_d, truth=True)

        # L1(X0, Y)
        loss_l1 = self.add_loss_l1(a=self.t2d, b=self.d)

        # ADV(X0, Y)+
        loss_gaseg = self.add_loss_adv(a=torch.cat([self.t2d, self.t2dseg], 1), net_d=self.net_dseg, truth=True)

        # ADV(X0, Y)+
        loss_gaLR = self.add_loss_adv(a=self.t2dLR, net_d=self.net_dLR, truth=True)

        # L1(X0, Y)
        loss_l1LR = self.add_loss_l1(a=self.t2dLR, b=self.dLR)

        # ADV(X0, Y)+
        loss_gasegLRHR = self.add_loss_adv(a=torch.cat([self.t2dLRHR, self.t2dLRHRseg], 1), net_d=self.net_dseg, truth=True)

        #loss_l1seg = self.add_loss_l1(a=self.t2dseg, b=self.dseg)

        loss_g = loss_ga + loss_l1 * self.hparams.lamb + loss_gaseg + loss_gaLR + loss_l1LR * self.hparams.lamb + loss_gasegLRHR

        return {'sum': loss_g, 'l1': loss_l1, 'l1LR': loss_l1LR}

    def backward_d(self):
        # ADV(X0, Y)-
        a0 = self.add_loss_adv(a=self.t2d, net_d=self.net_d, truth=False)

        # ADV(X0, Y)+
        a1 = self.add_loss_adv(a=self.d, net_d=self.net_d, truth=True)

        loss_da = 0.5 * a0 + 0.5 * a1

        # ADV(X0, Y)-
        a0seg = self.add_loss_adv(a=torch.cat([self.t2d, self.t2dseg], 1), net_d=self.net_dseg, truth=False)

        # ADV(X0, Y)+
        a1seg = self.add_loss_adv(a=torch.cat([self.d, self.dseg], 1), net_d=self.net_dseg, truth=True)

        loss_daseg = 0.5 * a0seg + 0.5 * a1seg

        # ADV(X0, Y)-
        a0LR = self.add_loss_adv(a=self.t2dLR, net_d=self.net_dLR, truth=False)

        # ADV(X0, Y)+
        a1LR = self.add_loss_adv(a=self.dLR, net_d=self.net_dLR, truth=True)

        loss_daLR = 0.5 * a0LR + 0.5 * a1LR

        # ADV(X0, Y)-
        a0segLRHR = self.add_loss_adv(a=torch.cat([self.t2dLRHR, self.t2dLRHRseg], 1), net_d=self.net_dseg, truth=False)

        # ADV(X0, Y)+
        a1segLRHR = self.add_loss_adv(a=torch.cat([self.d, self.dseg], 1), net_d=self.net_dseg, truth=True)

        loss_dasegLRHR = 0.5 * a0segLRHR + 0.5 * a1segLRHR

        loss_d = loss_da + loss_daseg + loss_daLR + loss_dasegLRHR

        return {'sum': loss_d, 'da': loss_da, 'daseg': loss_daseg, 'daLR': loss_daLR, 'dasegLH': loss_dasegLRHR}

# CUDA_VISIBLE_DEVICES=0,1 python train.py --jsn t2d --prj cascade0/  --models t2d0cascade --netG dsmc --netD bpatch_16 --split a --dataset t2d --direction tres_d -b 16