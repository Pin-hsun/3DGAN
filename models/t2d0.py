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
        self.net_g, self.net_d = self.set_networks()
        self.segd = torch.load('submodels/model_seg_ZIB_res18_256.pth').cuda()
        #self.segt2d = copy.deepcopy(self.segd)

        self.hparams.output_nc = 6
        _, self.net_dseg = self.set_networks()

        # update names of the models for optimization
        self.netg_names = {'net_g': 'net_g'}
        self.netd_names = {'net_d': 'net_d', 'net_dseg': 'net_dseg'}

        # Finally, initialize the optimizers and scheduler
        self.init_optimizer_scheduler()

    @staticmethod
    def add_model_specific_args(parent_parser):
        return parent_parser

    def test_method(self, net_g, img):
        self.oriX = img[0]
        self.imgX0 = net_g(self.oriX)[0]
        return self.imgX0

    def generation(self, batch):
        img = batch['img']
        self.t = img[0]
        self.d = img[1]
        #self.dseg = img[2]

        #self.t = self.t / self.t.max()
        #self.d = self.d / self.d.max()

        self.t2d = self.net_g(self.t)[0]

        self.dseg = self.segd(self.d.repeat(1, 3, 1, 1))
        self.t2dseg = self.segd(self.t2d.repeat(1, 3, 1, 1))

    def backward_g(self):
        # ADV(X0, Y)+
        loss_ga = self.add_loss_adv(a=self.t2d, net_d=self.net_d, truth=True)

        # L1(X0, Y)
        loss_l1 = self.add_loss_l1(a=self.t2d, b=self.d)

        # ADV(X0, Y)+
        loss_gaseg = self.add_loss_adv(a=torch.cat([self.t2d, self.t2dseg], 1), net_d=self.net_dseg, truth=True)

        #loss_l1seg = self.add_loss_l1(a=self.t2dseg, b=self.dseg)

        loss_g = loss_ga + loss_l1 * self.hparams.lamb + loss_gaseg

        return {'sum': loss_g, 'l1': loss_l1, 'ga': loss_ga}

    def backward_d(self):
        # ADV(X0, Y)-
        a0 = self.add_loss_adv(a=self.t2d, net_d=self.net_d, truth=False)

        # ADV(X0, Y)-
        a1 = self.add_loss_adv(a=self.d, net_d=self.net_d, truth=True)

        loss_da = 0.5 * a0 + 0.5 * a1

        # ADV(X0, Y)-
        a0seg = self.add_loss_adv(a=torch.cat([self.t2d, self.t2dseg], 1), net_d=self.net_dseg, truth=False)

        # ADV(X0, Y)-
        a1seg = self.add_loss_adv(a=torch.cat([self.d, self.dseg], 1), net_d=self.net_dseg, truth=True)

        loss_daseg = 0.5 * a0seg + 0.5 * a1seg

        loss_d = loss_da + loss_daseg

        return {'sum': loss_d, 'loss_da': loss_da, 'loss_daseg': loss_daseg}

# CUDA_VISIBLE_DEVICES=0,1 python train.py --jsn t2d --prj 1/  --models t2d0 --netG dsmc --netD bpatch_16 --split a --dataset t2d --direction tres_d -b 16