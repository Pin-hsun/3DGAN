from models.base import BaseModel
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

        # segmentation models
        self.seg_cartilage = torch.load('submodels/oai_cartilage_384.pth')#model_seg_ZIB.pth')
        self.seg_cartilage.eval()
        self.net_dseg = copy.deepcopy(self.net_d)

        # save model names
        self.netg_names = {'net_gXY': 'netGXY', 'net_gYX': 'netGYX'}
        self.netd_names = {'net_dX': 'netDX', 'net_dY': 'netDY', 'net_dseg': 'netDseg'}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        # coefficient for the identify loss
        parser.add_argument("--lambI", type=int, default=0.5)
        return parent_parser

    def generation(self):
        self.oriX = self.batch[0]
        self.oriY = self.batch[1]

        self.imgXY = self.net_gXY(self.oriX)[0]
        self.imgYX = self.net_gYX(self.oriY)[0]

        self.imgXYX = self.net_gYX(self.imgXY)[0]
        self.imgYXY = self.net_gXY(self.imgYX)[0]

        if self.hparams.lambI > 0:
            self.idt_X = self.net_gYX(self.oriX)[0]
            self.idt_Y = self.net_gXY(self.oriY)[0]

        self.oriYseg = self.seg_cartilage(self.oriY)
        self.imgXYseg = self.seg_cartilage(self.imgXY)
        self.imgYXYseg = self.seg_cartilage(self.imgYXY)

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

        # Segmentation
        # L1 (oriYseg, imgYXYseg)
        loss_g = self.add_loss_L1(a=self.oriYseg, b=self.imgYXYseg, loss=loss_g, coeff=self.hparams.lamb)
        # ADV (imgXYseg)+
        loss_g = self.add_loss_adv(a=self.imgXYseg, net_d=self.net_dseg, loss=loss_g, coeff=1, truth=True, stacked=False)

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

        # ADV (imgXYseg)-
        loss_d = self.add_loss_adv(a=self.imgXYseg, net_d=self.net_dseg, loss=loss_d, coeff=1, truth=False, stacked=False)

        # ADV (oriYseg)+
        loss_d = self.add_loss_adv(a=self.oriYseg, net_d=self.net_dseg, loss=loss_d, coeff=1, truth=True, stacked=False)

        return loss_d


# USAGE
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset FlyZ -b 16 --prj WpWn256test --direction xyweak%zyweak --resize 256 --engine cyclegan --lamb 10
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset TSE_DESS -b 16 --prj Cyclesegadv --netG unet_32 --direction a_b --cropsize 256  --engine cycleganseg --lamb 10
