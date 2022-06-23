from engine.base import BaseModel, combine
import torch


class GAN(BaseModel):
    """
    There is a lot of patterned noise and other failures when using lightning
    """
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints)

    @staticmethod
    def add_model_specific_args(parent_parser):
        return parent_parser

    def generation(self):
        self.oriX = self.batch[0]
        self.oriY = self.batch[1]
        self.oriXseg = self.batch[2]
        self.oriYseg = self.batch[3]
        try:
            self.imgX0 = self.net_g(self.oriX, a=torch.zeros(self.oriX.shape[0], self.net_g_inc).cuda())[0]
        except:
            self.imgX0 = self.net_g(self.oriX)[0]

        if self.hparams.cmb is not None:
            self.imgX0 = combine(self.imgX0, self.oriX, method=self.hparams.cmb)

    def backward_g(self, inputs):
        # ADV(X0, Y)+
        loss_g = 0
        loss_g = self.add_loss_adv(a=self.imgX0, b=self.oriXseg, net_d=self.net_d, loss=loss_g, coeff=1, truth=True, stacked=True)

        # L1(X0, Y)
        loss_g = self.add_loss_L1(a=self.imgX0, b=self.oriY, loss=loss_g, coeff=self.hparams.lamb)

        return {'sum': loss_g, 'loss_g': loss_g}

    def backward_d(self, inputs):
        loss_d = 0
        # ADV(X0, Y)-
        loss_d = self.add_loss_adv(a=self.imgX0, b=self.oriXseg, net_d=self.net_d, loss=loss_d, coeff=0.5, truth=False, stacked=True)

        # ADV(X, Y)+
        loss_d = self.add_loss_adv(a=self.oriY, b=self.oriYseg, net_d=self.net_d, loss=loss_d, coeff=0.5, truth=True, stacked=True)

        return {'sum': loss_d, 'loss_d': loss_d}



# CUDA_VISIBLE_DEVICES=1 python train.py --jsn womac3 --prj mcfix/NSseg/Gdescarsmc_index2 --engine NSseg --netG descarsmc --mc --direction areg_b_aregseg_bseg --gray --index