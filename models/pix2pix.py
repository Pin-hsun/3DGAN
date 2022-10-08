from models.base import BaseModel
import torch


class GAN(BaseModel):
    """
    There is a lot of patterned noise and other failures when using lightning
    """
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints)
        print('using pix2pix.py')

    @staticmethod
    def add_model_specific_args(parent_parser):
        return parent_parser

    def test_method(self, net_g, img):
        self.oriX = img[0]
        self.oriY = img[1]
        self.imgX0 = net_g(self.oriX)[0]

        return self.imgX0

    def generation(self):
        self.img = self.batch['img']
        self.oriX = self.img[0]
        self.oriY = self.img[1]
        self.imgX0 = self.net_g(self.oriX)[0]

    def backward_g(self, inputs):
        # ADV(X0, Y)+
        loss_g = 0
        loss_g += self.add_loss_adv(a=torch.cat([self.imgX0, self.oriY], 1), net_d=self.net_d, coeff=1, truth=True)

        # L1(X0, Y)
        loss_g += self.add_loss_l1(a=self.imgX0, b=self.oriY, coeff=self.hparams.lamb)

        return {'sum': loss_g, 'loss_g': loss_g}

    def backward_d(self, inputs):
        loss_d = 0
        # ADV(X0, Y)-
        loss_d += self.add_loss_adv(a=torch.cat([self.imgX0, self.oriY], 1), net_d=self.net_d, coeff=0.5, truth=False)

        # ADV(X, Y)+
        loss_d += self.add_loss_adv(a=torch.cat([self.oriX, self.oriY], 1), net_d=self.net_d, coeff=0.5, truth=True)

        return {'sum': loss_d, 'loss_d': loss_d}


#CUDA_VISIBLE_DEVICES=0,1 python train.py --jsn womac3 --prj compare/pix2pix/Gunet128 --models pix2pix --netG unet_128 --direction ap_bp --final sigmoid -b 1 --split moaks --cmb not
