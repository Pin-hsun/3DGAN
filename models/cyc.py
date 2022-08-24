from models.base import BaseModel, combine
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
        output = net_g(x[0])
        #output = combine(output, x[0], method='mul')
        return output[0]

    def generation(self):
        img = self.batch['img']

        self.oriX = img[0]
        self.oriY = img[1]

        self.imgXY = self.net_gXY(self.oriX)[0]
        self.imgYX = self.net_gYX(self.oriY)[0]

        self.imgXYX = self.net_gYX(self.imgXY)[0]
        self.imgYXY = self.net_gXY(self.imgYX)[0]

        if self.hparams.lambI > 0:
            self.idt_X = self.net_gYX(self.oriX)[0]
            self.idt_Y = self.net_gXY(self.oriY)[0]

    def backward_g(self, inputs):
        loss_g = 0
        # ADV(XY)+
        loss_g += self.add_loss_adv(a=self.imgXY, b=None, net_d=self.net_dY, coeff=1, truth=True, stacked=False)
        # ADV(YX)+
        loss_g += self.add_loss_adv(a=self.imgYX, b=None, net_d=self.net_dX, coeff=1, truth=True, stacked=False)

        # Cyclic(XYX, X)
        loss_g += self.add_loss_l1(a=self.imgXYX, b=self.oriX, coeff=self.hparams.lamb)
        # Cyclic(YXY, Y)
        loss_g += self.add_loss_l1(a=self.imgYXY, b=self.oriY, coeff=self.hparams.lamb)

        # Identity(idt_X, X)
        if self.hparams.lambI > 0:
            loss_g += self.add_loss_l1(a=self.idt_X, b=self.oriX, coeff=self.hparams.lambI)
            # Identity(idt_Y, Y)
            loss_g += self.add_loss_l1(a=self.idt_Y, b=self.oriY, coeff=self.hparams.lambI)

        return {'sum': loss_g, 'loss_g': loss_g}

    def backward_d(self, inputs):
        loss_d = 0
        # ADV(XY)-
        loss_d += self.add_loss_adv(a=self.imgXY, net_d=self.net_dY, coeff=1, truth=False, stacked=False)

        # ADV(YX)-
        loss_d += self.add_loss_adv(a=self.imgYX, net_d=self.net_dX, coeff=1, truth=False, stacked=False)

        # ADV(Y)+
        loss_d += self.add_loss_adv(a=self.oriY, net_d=self.net_dY, coeff=1, truth=True, stacked=False)

        # ADV(X)+
        loss_d += self.add_loss_adv(a=self.oriX, net_d=self.net_dX, coeff=1, truth=True, stacked=False)

        return {'sum': loss_d, 'loss_d': loss_d}


# USAGE
# CUDA_VISIBLE_DEVICES=1 python train.py --jsn wnwp3d --prj wnwp3d/cyc/GdenuOmc --mc --models cyc -b 16 --netG descarnoumc  --direction zyori%xyori --dataset Fly0B --input_nc 1

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --jsn 40x2fly10 --prj cyc/test1 --models cyc -b 16 --direction 40xdown2%xyori --dataset 40x --input_nc 1 --trd 500