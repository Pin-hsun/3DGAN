from engine.base import BaseModel, combine
import copy
import torch


class GAN(BaseModel):
    """
    There is a lot of patterned noise and other failures when using lightning
    """
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints)

        self.net_gXY = self.net_g
        self.net_gYX = copy.deepcopy(self.net_g)

        self.net_dX = self.net_d
        self.net_dY = copy.deepcopy(self.net_d)

        # save model names
        self.netg_names = {'net_gXY': 'netGXY', 'net_gYX': 'netGYX'}
        self.netd_names = {'net_dX': 'netDX', 'net_dY': 'netDY'}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        # coefficient for the identify loss
        parser.add_argument("--lambI", type=int, default=0.5)
        return parent_parser

    def test_method(self, net_g, x):
        output, = net_g(torch.cat((x[0], x[1]), 1))
        output = output[:, 1:, :, :]
        return output

    def generation0(self):
        self.oriX = torch.cat([self.batch[0], self.batch[1]], 1)
        self.oriY = torch.cat([self.batch[2], self.batch[3]], 1)

        self.imgXY, = self.net_gXY(self.oriX)
        self.imgYX, = self.net_gYX(self.oriY)

        self.imgXYX, = self.net_gYX(self.imgXY)
        self.imgYXY, = self.net_gXY(self.imgYX)

        if self.hparams.lambI > 0:
            self.idt_X, = self.net_gYX(self.oriX)
            self.idt_Y, = self.net_gXY(self.oriY)

    def generation(self):
        # zyweak_zyorisb%xyweak_xyorisb
        self.oriX = torch.cat([self.batch[0], self.batch[1]], 1)
        self.oriY = torch.cat([self.batch[2], self.batch[3]], 1)
        tXY, = self.net_gXY(self.oriX)
        tYX, = self.net_gYX(self.oriY)

        self.imgXY = torch.cat([combine(tXY[:, :1, :, :], self.batch[0], method='mul'), tXY[:, 1:, :, :]], 1)
        self.imgYX = torch.cat([combine(tYX[:, :1, :, :], self.batch[2], method='mul'), tYX[:, 1:, :, :]], 1)

        tXYX, = self.net_gYX(self.imgXY)
        tYXY, = self.net_gXY(self.imgYX)

        self.imgXYX = torch.cat([combine(tXYX[:, :1, :, :], self.imgXY[:, :1, :, :], method='mul'), tXYX[:, 1:, :, :]], 1)
        self.imgYXY = torch.cat([combine(tYXY[:, :1, :, :], self.imgYX[:, :1, :, :], method='mul'), tYXY[:, 1:, :, :]], 1)

        if self.hparams.lambI > 0:
            tidt_X, = self.net_gYX(self.oriX)
            tidt_Y, = self.net_gXY(self.oriY)
            self.idt_X = torch.cat([combine(tidt_X[:, :1, :, :], self.batch[0], method='mul'), tidt_X[:, 1:, :, :]], 1)
            self.idt_Y = torch.cat([combine(tidt_Y[:, :1, :, :], self.batch[1], method='mul'), tidt_Y[:, 1:, :, :]], 1)

    def backward_g(self, inputs):
        loss_g = 0
        # ADV(XY)+
        loss_g = self.add_loss_adv(a=self.imgXY, b=None, net_d=self.net_dY, loss=loss_g, coeff=1, truth=True, stacked=False)
        # ADV(YX)+
        loss_g = self.add_loss_adv(a=self.imgYX, b=None, net_d=self.net_dX, loss=loss_g, coeff=1, truth=True, stacked=False)

        # Cyclic(XYX, X)
        loss_g = self.add_loss_L1(a=self.imgXYX, b=self.oriX, loss=loss_g, coeff=self.hparams.lamb)
        # Cyclic(YXY, Y)
        loss_g = self.add_loss_L1(a=self.imgYXY, b=self.oriY, loss=loss_g, coeff=self.hparams.lamb)

        # Identity(idt_X, X)
        if self.hparams.lambI > 0:
            loss_g = self.add_loss_L1(a=self.idt_X, b=self.oriX, loss=loss_g, coeff=self.hparams.lambI)
            # Identity(idt_Y, Y)
            loss_g = self.add_loss_L1(a=self.idt_Y, b=self.oriY, loss=loss_g, coeff=self.hparams.lambI)

        return {'sum': loss_g, 'loss_g': loss_g}

    def backward_d(self, inputs):
        loss_d = 0
        # ADV(XY)-
        loss_d = self.add_loss_adv(a=self.imgXY, net_d=self.net_dY, loss=loss_d, coeff=1, truth=False, stacked=False)

        # ADV(YX)-
        loss_d = self.add_loss_adv(a=self.imgYX, net_d=self.net_dX, loss=loss_d, coeff=1, truth=False, stacked=False)

        # ADV(Y)+
        loss_d = self.add_loss_adv(a=self.oriY, net_d=self.net_dY, loss=loss_d, coeff=1, truth=True, stacked=False)

        # ADV(X)+
        loss_d = self.add_loss_adv(a=self.oriX, net_d=self.net_dX, loss=loss_d, coeff=1, truth=True, stacked=False)

        return {'sum': loss_d, 'loss_d': loss_d}


# USAGE
# CUDA_VISIBLE_DEVICES=0 python train.py --jsn wnwp --prj wnwp/dual/cmb --engine cyclegandual --cropsize 128 --dataset Fly3D --direction zyweak_zyorisb%xyweak_xyorisb --input_nc 2 --output_nc 2 --gray

