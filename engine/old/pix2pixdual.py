from engine.base import BaseModel
import torch

class GAN(BaseModel):
    """
    There is a lot of patterned noise and other failures when using lightning
    """
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints)
        self.OpOn = torch.load('submodels/OpOn.pth').cuda()
        print('using pix2pix.py')

    def generation(self):
        OpOn, = self.OpOn(self.oriY)
        oriX = torch.cat([OpOn, self.oriX], 1)

        self.imgX0 = self.net_g(oriX)[0]

    def backward_g(self, inputs):
        # ADV(X0, Y)+
        loss_g = 0
        loss_g = self.add_loss_adv(a=self.imgX0, net_d=self.net_d, loss=loss_g, coeff=1, truth=True, stacked=False)

        # L1(X0, Y)
        loss_g = self.add_loss_L1(a=self.imgX0, b=self.oriY, loss=loss_g, coeff=self.hparams.lamb)

        return loss_g

    def backward_d(self, inputs):
        loss_d = 0
        # ADV(X0, Y)-
        loss_d = self.add_loss_adv(a=self.imgX0, net_d=self.net_d, loss=loss_d, coeff=0.5, truth=False, stacked=False)

        # ADV(X, Y)+
        loss_d = self.add_loss_adv(a=self.oriY, net_d=self.net_d, loss=loss_d, coeff=0.5, truth=True, stacked=False)

        return loss_d


# CUDA_VISIBLE_DEVICES=1 python train.py --dataset FlySide -b 16 --prj FlyDual --direction weakxy_orixy --resize 286 --engine pix2pixdual --input_nc 6