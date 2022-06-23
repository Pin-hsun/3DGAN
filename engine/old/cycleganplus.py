from engine.base import BaseModel
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

        self.magic286 = torch.load('/media/ExtHDD01/checkpoints/FlyZ_WpOp/netG_model_epoch_170.pth').cuda()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        # coefficient for the identify loss
        parser.add_argument("--lambI", type=int, default=0.5)
        return parent_parser

    def generation(self):
        self.oriX = self.batch[0]
        self.oriY = self.batch[1]
        self.oriZ = self.batch[2]

        self.imgXY = self.net_gXY(self.oriX)[0]
        self.imgYX = self.net_gYX(self.oriY)[0]

        self.imgYXZ = self.magic286(self.imgYX)[0]

        self.imgXYX = self.net_gYX(self.imgXY)[0]
        self.imgYXY = self.net_gXY(self.imgYX)[0]

        if self.hparams.lambI > 0:
            self.idt_X = self.net_gYX(self.oriX)[0]
            self.idt_Y = self.net_gXY(self.oriY)[0]

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

        loss_g = self.add_loss_L1(a=self.imgYXZ, b=self.oriZ, loss=loss_g, coeff=self.hparams.lamb)

        return loss_g

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

        return loss_d


# USAGE
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset FlyZ -b 16 --prj WpWn256Z --direction xyweak%zyweak_zyori --resize 256 --engine cycleganplus --lamb 10

