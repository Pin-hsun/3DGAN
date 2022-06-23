from engine.base import BaseModel
import torch
import copy


class GAN(BaseModel):
    """
    There is a lot of patterned noise and other failures when using lightning
    """
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints)
        # segmentation models
        self.seg_cartilage = torch.load('submodels/oai_cartilage_384.pth')#model_seg_ZIB.pth')
        self.seg_cartilage.eval()
        self.net_dseg = copy.deepcopy(self.net_d)
        self.netd_names = {'net_d': 'netD', 'net_dseg': 'netDseg'}

    @staticmethod
    def add_model_specific_args(parent_parser):
        return parent_parser

    def generation(self):
        self.oriX = self.batch[0]
        self.oriY = self.batch[1]
        self.imgX0 = self.net_g(self.oriX)[0]

        self.oriYseg = self.seg_cartilage(self.oriY)
        self.imgX0seg = self.seg_cartilage(self.imgX0)

    def backward_g(self, inputs):
        # ADV(X0)+
        loss_g = 0
        loss_g = self.add_loss_adv(a=self.imgX0, net_d=self.net_d, loss=loss_g, coeff=1, truth=True, stacked=False)

        # L1(X0, Y)
        loss_g = self.add_loss_L1(a=self.imgX0, b=self.oriY, loss=loss_g, coeff=self.hparams.lamb)

        # ADV(X0seg)+
        #loss_g = self.add_loss_adv(a=self.imgX0seg, net_d=self.net_dseg, loss=loss_g, coeff=1, truth=True, stacked=False)

        loss_g = self.add_loss_L1(a=self.imgX0seg, b=self.oriYseg, loss=loss_g, coeff=self.hparams.lamb)

        return loss_g

    def backward_d(self, inputs):
        loss_d = 0
        # ADV(X0, Y)-
        loss_d = self.add_loss_adv(a=self.imgX0, net_d=self.net_d, loss=loss_d, coeff=0.5, truth=False, stacked=False)

        # ADV(X, Y)+
        loss_d = self.add_loss_adv(a=self.oriY, net_d=self.net_d, loss=loss_d, coeff=0.5, truth=True, stacked=False)

        # ADV(X0seg)-
        #loss_d = self.add_loss_adv(a=self.imgX0seg, net_d=self.net_dseg, loss=loss_d, coeff=0.5, truth=False, stacked=False)

        # ADV(Yseg)+
        #loss_d = self.add_loss_adv(a=self.oriYseg, net_d=self.net_dseg, loss=loss_d, coeff=0.5, truth=True, stacked=False)

        return loss_d



# CUDA_VISIBLE_DEVICES=1 python train.py --dataset FlyZ -b 16 --prj WpOp/sb256 --direction xyweak_xyorisb --resize 256 --engine pix2pixNS


