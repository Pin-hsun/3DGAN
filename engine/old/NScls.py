from engine.base import BaseModel, combine
import torch
import torch.nn as nn
import numpy as np


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(256, 2),
        )

    def forward(self, x):
        x, _ = torch.max(x, 0)
        return self.classifier(x).unsqueeze(0)


class GAN(BaseModel):
    """
    There is a lot of patterned noise and other failures when using lightning
    """
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints)
        self.classifier = Classifier()

        from utils.metrics_classification import ClassificationLoss, GetAUC
        self.loss_function = ClassificationLoss()
        self.metrics = GetAUC()

        # save model names
        self.netd_names = {'net_d': 'netD', 'classifier': 'classifier'}
        self.loss_g_names = ['loss_g']
        self.loss_d_names = ['loss_d', 'loss_cls']

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--lambC", type=float, default=1)
        return parent_parser

    def generation(self):
        self.oriX = self.batch[0]
        self.oriY = self.batch[1]
        try:
            self.imgX0 = self.net_g(self.oriX, a=torch.zeros(self.oriX.shape[0], self.net_g_inc).cuda())[0]
        except:
            self.imgX0 = self.net_g(self.oriX)[0]

        if self.hparams.cmb is not None:
            self.imgX0 = combine(self.imgX0, self.oriX, method=self.hparams.cmb)

        # classification

    def backward_g(self, inputs):
        loss_g = 0
        # ADV(X0, Y)+
        loss_g = self.add_loss_adv(a=self.imgX0, net_d=self.net_d, loss=loss_g, coeff=1, truth=True, stacked=False)

        # L1(X0, Y)
        loss_g = self.add_loss_L1(a=self.imgX0, b=self.oriY, loss=loss_g, coeff=self.hparams.lamb)

        return {'sum': loss_g, 'loss_g': loss_g}

    def backward_d(self, inputs):
        loss_d = 0
        # ADV(X0, Y)-
        loss_d = self.add_loss_adv(a=self.imgX0, net_d=self.net_d, loss=loss_d, coeff=0.5, truth=False, stacked=False)

        # ADV(X, Y)+
        loss_d = self.add_loss_adv(a=self.oriY, net_d=self.net_d, loss=loss_d, coeff=0.5, truth=True, stacked=False)

        # Classification
        clsX = self.net_d(torch.cat((self.oriX, self.oriX), 1))[2]
        clsY = self.net_d(torch.cat((self.oriY, self.oriY), 1))[2]
        clsX = self.classifier(clsX[:, :, 0, 0])
        clsY = self.classifier(clsY[:, :, 0, 0])

        if (np.random.rand(1) > 0.5)[0]:
            cls = torch.cat([clsX, clsY], 0)
            labels = torch.cat([torch.ones(1), torch.zeros(1)], 0).type(torch.LongTensor).cuda()
        else:
            cls = torch.cat([clsY, clsX], 0)
            labels = torch.cat([torch.zeros(1), torch.ones(1)], 0).type(torch.LongTensor).cuda()
        loss_cls = self.loss_function(cls, labels)
        return {'sum': loss_d + loss_cls * self.hparams.lambC, 'loss_d': loss_d, 'loss_cls': loss_cls}



# CUDA_VISIBLE_DEVICES=1 python train.py --dataset FlyZ -b 16 --prj WpOp/sb256 --direction xyweak_xyorisb --resize 256 --engine pix2pixNS


# CUDA_VISIBLE_DEVICES=0 python train.py --dataset womac3 -b 16 --prj NS/GattNoL1 --direction aregis1_b --cropsize 256 --engine pix2pixNS --lamb 0 --netG attgan
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset womac3 -b 16 --prj seg/NS_Gatt --direction amask_bmask --cropsize 256 --engine pix2pixNS --netG attgan

# CUDA_VISIBLE_DEVICES=1 python train.py --dataset kl3 -b 16 --prj NS/0 --direction badKL3afterreg_gooodKL3reg --cropsize 256 --engine pix2pixNS

# CUDA_VISIBLE_DEVICES=0 python train.py --dataset kl3 -b 16 --prj seg/res6L10 --direction badseg_goodseg --resize 384 --cropsize 256 --engine pix2pixNS --netG resnet_6blocks --lamb 0

# Residual
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset womac3 -b 16 --prj N01/Unet32Res --direction aregis1_b --cropsize 256 --engine pix2pixNS --netG unet_32 --res --n01 --final sigmoid