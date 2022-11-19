import torch, copy
import torch.nn as nn
import torchvision
import torch.optim as optim
from math import log10
import time, os
import pytorch_lightning as pl
from utils.metrics_segmentation import SegmentationCrossEntropyLoss
from utils.metrics_classification import CrossEntropyLoss, GetAUC
from utils.data_utils import *
from models.base import BaseModel, combine
import pandas as pd


class GAN(BaseModel):
    """
    There is a lot of patterned noise and other failures when using lightning
    """
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints)
        #self.net_class = copy.deepcopy(self.net_d)

        self.classifier = nn.Conv2d(256, 1, 1, stride=1, padding=0).cuda()

        # update model names\
        self.netg_names = {'net_g': 'netG'}
        self.netd_names = {'net_d': 'netD', 'classifier': 'classifier'}#, 'net_class': 'netDC'}

        #self.df = pd.read_csv(os.getenv("HOME") + '/Dropbox/TheSource/scripts/OAI_pipelines/meta/subjects_unipain_womac3.csv')

        self.df = pd.read_csv('env/subjects_unipain_womac3.csv')

        self.init_networks_optimizer_scheduler()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--lbx", dest='lbx', type=float, default=1)
        parser.add_argument("--dc0", dest='dc0', type=float, default=1)
        return parent_parser

    @staticmethod
    def test_method(net_g, img, a=None):
        oriX = img[0]
        imgXY, = net_g(oriX, a=a)
        imgXY = nn.Sigmoid()(imgXY)  # mask

        return imgXY

    def generation(self):
        img = self.batch['img']
        id = self.batch['filenames'][0][0].split('/')[-1].split('_')[0]
        paindiff = np.abs(self.df.loc[self.df['ID'] == int(id), ['V00WOMKPR']].values[0][0]\
                   - self.df.loc[self.df['ID'] == int(id), ['V00WOMKPL']].values[0][0])
        paindiff = 1#(paindiff / 10)

        self.oriX = img[0]
        self.oriY = img[1]

        self.imgXY, = self.net_g(self.oriX, a=paindiff)
        self.imgXY = nn.Sigmoid()(self.imgXY)  # mask
        self.imgXY = combine(self.imgXY, self.oriX, method='mul')

        self.imgXX, = self.net_g(self.oriX, a=0)
        self.imgXX = nn.Sigmoid()(self.imgXX)  # mask
        self.imgXX = combine(self.imgXX, self.oriX, method='mul')

        #self.imgYY = combine(self.imgYY, self.oriY, method='mul')

    def backward_g(self):
        # ADV(XY)+ -
        axy, _ = self.add_loss_adv_classify3d(a=self.imgXY, net_d=self.net_d, truth_adv=True, truth_classify=False)

        # L1(XY, Y)
        loss_l1 = self.add_loss_l1(a=self.imgXY, b=self.oriY, coeff=self.hparams.lamb)

        # L1(XX, X)
        loss_l1x = self.add_loss_l1(a=self.imgXX, b=self.oriX, coeff=self.hparams.lbx)

        loss_ga = axy
        #loss_gc = cxy + cxx + cyy + cyx
        loss_g = loss_ga + loss_l1 + loss_l1x

        #self.log('l1', loss_l1, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #self.log('ga', loss_ga, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #self.log('gc', loss_gc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return {'sum': loss_g, 'l1': loss_l1, 'ga': loss_ga}

    def backward_d(self):
        id = self.batch['filenames'][0][0].split('/')[-1].split('_')[0]
        side = self.df.loc[self.df['ID'] == int(id), ['SIDE']].values[0][0]
        # ADV(XY)- -
        # aversarial of xy
        axy, _ = self.add_loss_adv_classify3d(a=self.imgXY, net_d=self.net_d, truth_adv=False, truth_classify=False)

        truth_classify = (side == 'RIGHT')
        # ax: adversarial of x, ay: adversarial of y
        ax, ay, cxy, _ = self.add_loss_adv_classify3d_paired(a=self.oriX, b=self.oriY, net_d=self.net_d, classifier=self.classifier,
                                                             truth_adv=True, truth_classify=truth_classify)
        # adversarial of xy (-) and y (+)
        loss_da = axy * 0.5 + ay * 0.5
        # classify x (+) vs y (-)
        loss_dc = cxy
        loss_d = loss_da + loss_dc * self.hparams.dc0

        #self.log('da', loss_da, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #self.log('dc', loss_dc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return {'sum': loss_d, 'da': loss_da, 'dc': loss_dc}

    def add_loss_adv_classify3d(self, a, net_d, truth_adv, truth_classify, log=None):
        adv_logits, classify_logits = net_d(a)

        # 3D classification
        classify_logits = nn.AdaptiveAvgPool2d(1)(classify_logits)
        classify_logits = classify_logits.sum(0).unsqueeze(0)

        if truth_adv:
            adv = self.criterionGAN(adv_logits, torch.ones_like(adv_logits))
        else:
            adv = self.criterionGAN(adv_logits, torch.zeros_like(adv_logits))

        if truth_classify:
            classify = self.criterionGAN(classify_logits, torch.ones_like(classify_logits))
        else:
            classify = self.criterionGAN(classify_logits, torch.zeros_like(classify_logits))

        return adv, classify

    def add_loss_adv_classify3d_paired(self, a, b, net_d, classifier, truth_adv, truth_classify, log=None):
        adv_a, classify_a = net_d(a)
        adv_b, classify_b = net_d(b)

        if truth_adv:
            adv_a = self.criterionGAN(adv_a, torch.ones_like(adv_a))
            adv_b = self.criterionGAN(adv_b, torch.ones_like(adv_b))
        else:
            adv_a = self.criterionGAN(adv_a, torch.zeros_like(adv_a))
            adv_b = self.criterionGAN(adv_b, torch.zeros_like(adv_b))

        if truth_classify:  # if right knee pain
            classify_logits = nn.AdaptiveAvgPool2d(1)(classify_a - classify_b)  # (right knee - left knee)
        else:  # if left knee pain
            classify_logits = nn.AdaptiveAvgPool2d(1)(classify_b - classify_a)  # (right knee - left knee)

        classify_logits, _ = torch.max(classify_logits, 0)
        classify_logits = classify_logits.unsqueeze(0)
        classify_logits = classifier(classify_logits)

        if truth_classify:
            classify = self.criterionGAN(classify_logits, torch.ones_like(classify_logits))
        else:
            classify = self.criterionGAN(classify_logits, torch.zeros_like(classify_logits))

        return adv_a, adv_b, classify, classify_logits





# CUDA_VISIBLE_DEVICES=0,1,2 python train.py --jsn womac3 --prj Gds/descar3/Gdsmc3DB --mc --engine descar3 --netG dsmc --netD descar --direction areg_b --index --gray --load3d --final none
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --jsn womac3 --prj 3D/descar3/GdsmcDbpatch16Trd800 --mc --engine descar3 --netG dsmc --netD bpatch_16 --direction ap_bp --split moaks --load3d --final none --n_epochs 400 --trd 800
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --jsn womac3 --prj 3D/descar3/GdsmcDbpatch16Trd800  --models descar3 --netG dsmc --netD bpatch_16 --direction ap_bp --final none -b 1 --split moaks --final none --n_epochs 400 --trd 800