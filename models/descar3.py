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

        # save model names
        self.netg_names = {'net_g': 'netG'}
        self.netd_names = {'net_d': 'netD', 'classifier': 'classifier'}#, 'net_class': 'netDC'}

        #self.df = pd.read_csv(os.getenv("HOME") + '/Dropbox/TheSource/scripts/OAI_pipelines/meta/subjects_unipain_womac3.csv')

        self.df = pd.read_csv('env/subjects_unipain_womac3.csv')

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--lb1", dest='lb1', type=float, default=10)
        return parent_parser

    def test_method(self, net_g, img):
        self.oriX = img[0]
        self.imgXY, self.imgXX = net_g(self.oriX, a=None)
        self.imgXY = nn.Sigmoid()(self.imgXY)  # mask
        self.imgXY = combine(self.imgXY, self.oriX, method='mul')

        return self.imgXY

    def generation(self):
        img = self.batch['img']
        self.filenames = self.batch['filenames']
        self.oriX = img[0]
        self.oriY = img[1]

        self.imgXY, self.imgXX = self.net_g(self.oriX, a=None)
        #self.imgYY, self.imgYX = self.net_g(self.oriY, a=None)

        self.imgXY = nn.Sigmoid()(self.imgXY)  # mask
        #self.imgYY = nn.Sigmoid()(self.imgYY)  # mask
        self.imgXY = combine(self.imgXY, self.oriX, method='mul')
        #self.imgYY = combine(self.imgYY, self.oriY, method='mul')

    def backward_g(self, inputs):
        # ADV(XY)+ -
        axy, _ = self.add_loss_adv_classify3d(a=self.imgXY, net_d=self.net_d, truth_adv=True, truth_classify=False)

        # ADV(XX)+ +
        #axx, cxx = self.add_loss_adv_classify(a=self.imgXX, net_d=self.net_d, truth_adv=True, truth_classify=True)

        # ADV(YY)+ -
        #ayy, cyy = self.add_loss_adv_classify(a=self.imgYY, net_d=self.net_d, truth_adv=True, truth_classify=False)

        # ADV(YX)+ +
        #ayx, cyx = self.add_loss_adv_classify(a=self.imgYX, net_d=self.net_d, truth_adv=True, truth_classify=True)

        # L1(XY, Y)
        loss_l1 = self.add_loss_l1(a=self.imgXY, b=self.oriY, coeff=self.hparams.lamb)

        loss_ga = axy
        #loss_gc = cxy + cxx + cyy + cyx
        loss_g = loss_ga + loss_l1

        self.log('l1', loss_l1, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('ga', loss_ga, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #self.log('gc', loss_gc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return {'sum': loss_g, 'loss_g': loss_g}

    def backward_d(self, inputs):
        id = self.filenames[0][0].split('/')[-1].split('_')[0]
        side = self.df.loc[self.df['ID'] == int(id), ['SIDE']].values[0][0]
        # ADV(XY)- -
        axy, _ = self.add_loss_adv_classify3d(a=self.imgXY, net_d=self.net_d, truth_adv=False, truth_classify=False)
        # ADV(XX)- +
        #axx, cxx = self.add_loss_adv_classify(a=self.imgXX, net_d=self.net_d, truth_adv=False, truth_classify=True)
        # ADV(YY)- -
        #ayy, cyy = self.add_loss_adv_classify(a=self.imgYY, net_d=self.net_d, truth_adv=False, truth_classify=False)
        # ADV(YX)- +
        #ayx, cyx = self.add_loss_adv_classify(a=self.imgYX, net_d=self.net_d, truth_adv=False, truth_classify=True)

        # ADV(X)+ +
        #ax, cx = self.add_loss_adv_classify3d(a=self.oriX, net_d=self.net_d, truth_adv=True, truth_classify=True)
        # ADV(Y)+ -
        #ay, cy = self.add_loss_adv_classify3d(a=self.oriY, net_d=self.net_d, truth_adv=True, truth_classify=False)
        truth_classify = (side == 'RIGHT')
        ax, ay, cxy, _ = self.add_loss_adv_classify3d_paired(a=self.oriX, b=self.oriY, net_d=self.net_d, classifier=self.classifier,
                                                             truth_adv=True, truth_classify=truth_classify)

        loss_da = axy * 0.5 + ay * 0.5
        #loss_dc = 0.5 * cx + 0.5 * cy # + (cxy + cxx + cyy + cyx) * 0.5
        loss_dc = cxy
        loss_d = loss_da + loss_dc

        self.log('da', loss_da, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('dc', loss_dc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return {'sum': loss_d, 'loss_d': loss_d}

    def validation_step(self, batch, batch_idx):
        self.batch = batch
        if self.hparams.load3d:  # if working on 3D input
            self.batch = batch
            if self.hparams.load3d:  # if working on 3D input, bring the Z dimension to the first and combine with batch
                self.batch['img'] = self.reshape_3d(self.batch['img'])

        img = self.batch['img']
        self.filenames = self.batch['filenames']
        self.oriX = img[0]
        self.oriY = img[1]

        #
        id = self.filenames[0][0].split('/')[-1].split('_')[0]
        side = self.df.loc[self.df['ID'] == int(id), ['SIDE']].values[0][0]

        truth_classify = (side == 'RIGHT')
        ax, ay, cxy, lxy = self.add_loss_adv_classify3d_paired(a=self.oriX, b=self.oriY, net_d=self.net_d, classifier=self.classifier,
                                                     truth_adv=True, truth_classify=truth_classify)
        loss_dc = cxy
        self.log('valdc', loss_dc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # AUC metrics
        if truth_classify:
            label = torch.zeros(1).type(torch.LongTensor)
        else:
            label = torch.ones(1).type(torch.LongTensor)
        out = lxy[:, :, 0, 0]
        self.all_label.append(label)
        self.all_out.append(out.cpu().detach())

        return loss_dc

    def validation_epoch_end(self, x):
        all_out = torch.cat(self.all_out, 0)
        all_label = torch.cat(self.all_label, 0)
        metrics = GetAUC()(all_label, all_out)

        auc = torch.from_numpy(np.array(metrics)).cuda()
        for i in range(len(auc)):
            self.log('auc' + str(i), auc[i], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return metrics

    def add_loss_adv_classify3d(self, a, net_d, truth_adv, truth_classify, log=None):
        fake_in = torch.cat((a, a), 1)
        adv_logits, classify_logits = net_d(fake_in)

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

        if log is not None:
            self.log(log, adv, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return adv, classify

    def add_loss_adv_classify3d_paired(self, a, b, net_d, classifier, truth_adv, truth_classify, log=None):
        a_in = torch.cat((a, a), 1)
        adv_a, classify_a = net_d(a_in)
        b_in = torch.cat((b, b), 1)
        adv_b, classify_b = net_d(b_in)

        if truth_adv:
            adv_a = self.criterionGAN(adv_a, torch.ones_like(adv_a))
            adv_b = self.criterionGAN(adv_b, torch.ones_like(adv_b))
        else:
            adv_a = self.criterionGAN(adv_a, torch.zeros_like(adv_a))
            adv_b = self.criterionGAN(adv_b, torch.zeros_like(adv_b))

        if truth_classify:
            classify_logits = nn.AdaptiveAvgPool2d(1)(classify_a - classify_b)
        else:
            classify_logits = nn.AdaptiveAvgPool2d(1)(classify_b - classify_a)

        classify_logits, _ = torch.max(classify_logits, 0)
        classify_logits = classify_logits.unsqueeze(0)
        classify_logits = classifier(classify_logits)

        if truth_classify:
            classify = self.criterionGAN(classify_logits, torch.ones_like(classify_logits))
        else:
            classify = self.criterionGAN(classify_logits, torch.zeros_like(classify_logits))

        if log is not None:
            self.log(log, adv, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return adv_a, adv_b, classify, classify_logits

    def add_loss_adv_classify3d_paired2(self, a, b, a2, b2, net_d, classifier, truth_adv, truth_classify, log=None):
        a_in = torch.cat((a, a), 1)
        adv_a, classify_a = net_d(a_in)
        b_in = torch.cat((b, b), 1)
        adv_b, classify_b = net_d(b_in)

        a2_in = torch.cat((a2, a2), 1)
        adv_a2, classify_a2 = net_d(a2_in)
        b2_in = torch.cat((b2, b2), 1)
        adv_b2, classify_b2 = net_d(b2_in)

        if truth_adv:
            adv_a = self.criterionGAN(adv_a, torch.ones_like(adv_a))
            adv_b = self.criterionGAN(adv_b, torch.ones_like(adv_b))
        else:
            adv_a = self.criterionGAN(adv_a, torch.zeros_like(adv_a))
            adv_b = self.criterionGAN(adv_b, torch.zeros_like(adv_b))

        if truth_classify:
            classify_logits = nn.AdaptiveAvgPool2d(1)(classify_a - classify_b2) - nn.AdaptiveAvgPool2d(1)(classify_b - classify_a2)
        else:
            classify_logits = nn.AdaptiveAvgPool2d(1)(classify_b - classify_a2) - nn.AdaptiveAvgPool2d(1)(classify_a - classify_b2)

        classify_logits, _ = torch.max(classify_logits, 0)
        classify_logits = classify_logits.unsqueeze(0)
        classify_logits = classifier(classify_logits)

        if truth_classify:
            classify = self.criterionGAN(classify_logits, torch.ones_like(classify_logits))
        else:
            classify = self.criterionGAN(classify_logits, torch.zeros_like(classify_logits))

        if log is not None:
            self.log(log, adv, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return adv_a, adv_b, classify, classify_logits


# CUDA_VISIBLE_DEVICES=0,1,2 python train.py --jsn womac3 --prj Gds/descar3/Gdsmc3DB --mc --engine descar3 --netG dsmc --netD descar --direction areg_b --index --gray --load3d --final none
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --env  a6k --jsn womac3 --prj Gds/descar3b/GdsmcDboatch16 --mc --engine descar3b --netG dsmc --netD bpatch_16 --direction ap_bp --index --load3d --final none
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --env  a6k --jsn womac3 --prj Gds/descar3/check  --models descar3 --netG descarganshallow --netD bpatch_16 --direction ap_bp --final none -b 1 --split moaks