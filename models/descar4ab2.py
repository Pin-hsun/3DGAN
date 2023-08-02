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
from models.base import BaseModel, combine, VGGLoss
import pandas as pd
from models.helper_oai import OaiSubjects, classify_easy_3d, swap_by_labels
from models.helper import reshape_3d, tile_like


class GAN(BaseModel):
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints)
        # First, modify the hyper-parameters if necessary
        # Initialize the networks
        self.net_g, self.net_d = self.set_networks()
        #self.net_dX = copy.deepcopy(self.net_d)
        self.classifier = nn.Conv2d(256, 2, 1, stride=1, padding=0).cuda()

        # update names of the models for optimization
        self.netg_names = {'net_g': 'net_g'}
        self.netd_names = {'net_d': 'net_d', 'classifier': 'classifier'}

        self.oai = OaiSubjects(self.hparams.dataset)

        if hparams.lbvgg > 0:
            self.VGGloss = VGGLoss().cuda()

        # Finally, initialize the optimizers and scheduler
        self.init_optimizer_scheduler()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--lbx", dest='lbx', type=float, default=0)
        parser.add_argument("--dc0", dest='dc0', type=float, default=0)
        parser.add_argument("--lbvgg", dest='lbvgg', type=float, default=0)
        return parent_parser

    @staticmethod
    def test_method(net_g, img, args, a=None):
        oriX = img[0]

        #imgXX, _ = net_g(oriX, a=torch.FloatTensor([0]))
        #imgXX = nn.Sigmoid()(imgXX)  # mask

        out = net_g(oriX, a=None)
        imgXY = out['out0']
        imgXY = nn.Sigmoid()(imgXY)  # mask
        imgXX = out['out1']
        imgXX = nn.Sigmoid()(imgXX)  # mask

        combinedXY = combine(imgXY, oriX, method='mul')
        combinedXX = combine(imgXX, oriX, method='mul')

        return {'imgXY': imgXY[0, ::].detach().cpu(),
                'imgXX': imgXX[0, ::].detach().cpu(),
                'combinedXY': combinedXY[0, ::].detach().cpu(),
                'combinedXX': combinedXX[0, ::].detach().cpu()}

    def generation(self, batch):
        if self.hparams.load3d:  # if working on 3D input, bring the Z dimension to the first and combine with batch
            batch['img'] = reshape_3d(batch['img'])

        # pain label
        self.labels = self.oai.labels_unilateral(filenames=batch['filenames'])
        self.oriX = batch['img'][0]
        self.oriY = batch['img'][1]
        self.ori0 = batch['img'][2]

        self.imgXY = nn.Sigmoid()(self.net_g(self.oriX, a=None)['out0'])  # mask
        self.imgXX = nn.Sigmoid()(self.net_g(self.ori0, a=None)['out1'])  # mask

        self.imgXY = combine(self.imgXY, self.oriX, method='mul')
        self.imgXX = combine(self.imgXX, self.oriX, method='mul')

    def backward_g(self):
        # ADV(XY)+
        axy = self.add_loss_adv(a=self.imgXY, net_d=self.net_d, truth=True)

        # ADV(XX)+
        #axx = self.add_loss_adv(a=self.imgXX, net_d=self.net_dX, truth=True)

        # L1(XY, Y)
        loss_l1 = self.add_loss_l1(a=self.imgXY, b=self.oriY)

        loss_ga = axy * 1.0 #+ axx * 0.5

        loss_g = loss_ga + loss_l1 * self.hparams.lamb

        if self.hparams.lbx > 0:
            loss_l1x = self.add_loss_l1(a=self.imgXX, b=self.oriX)
            loss_g += loss_l1x * self.hparams.lbx

        if self.hparams.lbvgg > 0:
            loss_gvgg = self.VGGloss(torch.cat([self.imgXY] * 3, 1), torch.cat([self.oriY] * 3, 1))
            loss_g += loss_gvgg * self.hparams.lbvgg

        return {'sum': loss_g, 'l1': loss_l1, 'ga': loss_ga}#, 'gvgg': loss_gvgg}

    def backward_d(self):
        # ADV(XY)-
        axy = self.add_loss_adv(a=self.imgXY, net_d=self.net_d, truth=False)

        # ADV(XX)-
        #axx = self.add_loss_adv(a=self.imgXX, net_d=self.net_dX, truth=False)

        # ADV(Y)+
        #ay = self.add_loss_adv(a=self.oriY, net_d=self.net_d, truth=True)

        # ADV(X)+
        #ax = self.add_loss_adv(a=self.oriX, net_d=self.net_dX, truth=True)

        #axx = self.add_loss_adv(a=self.imgXX, net_d=self.net_dX, truth=False)
        #axx, _ = self.add_loss_adv_classify3d(a=self.imgXX, net_d=self.net_dX, truth_adv=False, truth_classify=False)

        # ax: adversarial of x, ay: adversarial of y
        _, ay, _, _ = self.add_loss_adv_classify3d_paired(a=self.oriX, b=self.oriY, net_d=self.net_d,
                                                          classifier=self.classifier,
                                                          truth_adv=True, truth_classify=self.labels['painbinary'])
        # adversarial of xy (-) and y (+)
        loss_da = axy * 0.5 + ay * 0.5 #+ axx * 0.5 + ax * 0.5
        # classify x (+) vs y (-)
        #loss_dc = cxy
        loss_d = loss_da

        return {'sum': loss_d, 'da': loss_da}

    def add_loss_adv_classify3d_paired(self, a, b, net_d, classifier, truth_adv, truth_classify):
        adv_a, classify_a = net_d(a)  # (B*Z, 1, dH, dW), (B*Z, C, dH, dW)
        adv_b, classify_b = net_d(b)

        if truth_adv:
            adv_a = self.criterionGAN(adv_a, torch.ones_like(adv_a))
            adv_b = self.criterionGAN(adv_b, torch.ones_like(adv_b))
        else:
            adv_a = self.criterionGAN(adv_a, torch.zeros_like(adv_a))
            adv_b = self.criterionGAN(adv_b, torch.zeros_like(adv_b))

        classify_logits = swap_by_labels(sign_swap=(truth_classify * 2) - 1, classify_logits=(classify_b - classify_a))

        classify, classify_logits = classify_easy_3d(classify_logits, truth_classify, classifier, nn.BCEWithLogitsLoss())

        return adv_a, adv_b, classify, classify_logits


# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --jsn womac3 --prj descar4ab2/L1_100_Vgg10_att/ --dataset womac4 --models descar4ab2  --netD bpatch_16 --split a  --lbvgg 10 --mc --direction ap_bp_a --lbx 1 --lamb 100 --env a6k --netG dsmcatt