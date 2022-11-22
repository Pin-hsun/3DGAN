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


def tile_like(x, target):  # tile_size = 256 or 4
    x = x.view(x.size(0), x.size(1), 1, 1)
    x = x.repeat(1, 1, target.size(2), target.size(3))
    return x


class GAN(BaseModel):
    """
    There is a lot of patterned noise and other failures when using lightning
    """
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints)
        self.net_g, self.net_d = self.set_networks()
        self.net_dX = copy.deepcopy(self.net_d)
        self.classifier = nn.Conv2d(256, 1, 1, stride=1, padding=0).cuda()

        # update model names\
        self.netg_names = {'net_g': 'netG'}
        self.netd_names = {'net_d': 'netD', 'classifier': 'classifier', 'net_dX': 'netDX'}#, 'net_class': 'netDC'}

        self.df = pd.read_csv('env/subjects_unipain_womac3.csv')

        self.init_optimizer_scheduler()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--lbx", dest='lbx', type=float, default=0)
        parser.add_argument("--dc0", dest='dc0', type=float, default=1)
        return parent_parser

    @staticmethod
    def test_method(net_g, img, a=None):
        oriX = img[0]

        imgXX, = net_g(oriX, a=0)
        imgXX = nn.Sigmoid()(imgXX)  # mask

        imgXY, = net_g(oriX, a=a)
        imgXY = nn.Sigmoid()(imgXY)  # mask

        imgXX = combine(imgXX, oriX, method='mul')
        imgXY = combine(imgXY, imgXX, method='mul')

        return imgXY

    def generation(self):
        # pain label
        ids = [x.split('/')[-1].split('_')[0] for x in self.batch['filenames'][0]]
        paindiff = [(self.df.loc[self.df['ID'] == int(i), ['V00WOMKPR']].values[0][0]\
                   - self.df.loc[self.df['ID'] == int(i), ['V00WOMKPL']].values[0][0]) for i in ids] # Right - Left
        self.paindiff = torch.FloatTensor([x / 10 for x in paindiff])

        img = self.batch['img']
        self.oriX = img[0]
        self.oriY = img[1]

        self.imgXY, = self.net_g(self.oriX, a=torch.abs(self.paindiff))
        self.imgXY = nn.Sigmoid()(self.imgXY)  # mask
        self.imgXY = combine(self.imgXY, self.oriX, method='mul')

        #self.imgXX, = self.net_g(self.oriX, a=0 * torch.abs(self.paindiff))
        #self.imgXX = nn.Sigmoid()(self.imgXX)  # mask
        #self.imgXX = combine(self.imgXX, self.oriX, method='mul')

        #self.imgYY = combine(self.imgYY, self.oriY, method='mul')

    def backward_g(self):
        # ADV(XY)+ -
        axy, _ = self.add_loss_adv_classify3d(a=self.imgXY, net_d=self.net_d, truth_adv=True, truth_classify=False)

        # ADV(XY)+ -
        #axx, _ = self.add_loss_adv_classify3d(a=self.imgXX, net_d=self.net_dX, truth_adv=True, truth_classify=False)

        # L1(XY, Y)
        loss_l1 = self.add_loss_l1(a=self.imgXY, b=self.oriY, coeff=self.hparams.lamb)

        # L1(XX, X)
        #loss_l1x = self.add_loss_l1(a=self.imgXX, b=self.oriX, coeff=self.hparams.lbx)

        loss_ga = axy# * 0.5 + axx * 0.5
        loss_g = loss_ga + loss_l1# + loss_l1x

        return {'sum': loss_g, 'l1': loss_l1, 'ga': loss_ga}

    def backward_d(self):
        id = self.batch['filenames'][0][0].split('/')[-1].split('_')[0]
        side = self.df.loc[self.df['ID'] == int(id), ['SIDE']].values[0][0]
        # ADV(XY)- -
        # aversarial of xy
        axy, _ = self.add_loss_adv_classify3d(a=self.imgXY, net_d=self.net_d, truth_adv=False, truth_classify=False)

        # aversarial of xx
        #axx, _ = self.add_loss_adv_classify3d(a=self.imgXX, net_d=self.net_dX, truth_adv=False, truth_classify=False)

        # ax: adversarial of x, ay: adversarial of y

        side_classify = (torch.sign(self.paindiff) + 1) / 2
        ax, ay, cxy, _ = self.add_loss_adv_classify3d_paired(a=self.oriX, b=self.oriY, net_d=self.net_d,
                                                             classifier=self.classifier,
                                                             truth_adv=True, truth_classify=side_classify)
        # adversarial of xy (-) and y (+)
        loss_da = axy * 0.5 + ay * 0.5#axy * 0.25 + axx * 0.25 + ax * 0.25 + ay * 0.25
        # classify x (+) vs y (-)
        loss_dc = cxy
        loss_d = loss_da + loss_dc * self.hparams.dc0

        return {'sum': loss_d, 'da': loss_da, 'dc': loss_dc}

    def add_loss_adv_classify3d(self, a, net_d, truth_adv, truth_classify):
        adv_logits, classify_logits = net_d(a)

        if truth_adv:
            adv = self.criterionGAN(adv_logits, torch.ones_like(adv_logits))
        else:
            adv = self.criterionGAN(adv_logits, torch.zeros_like(adv_logits))

        if 0: # No Single Knee Classification
            # 3D classification
            classify_logits = nn.AdaptiveAvgPool2d(1)(classify_logits)
            classify_logits = classify_logits.sum(0).unsqueeze(0)
            if truth_classify:
                classify = self.criterionGAN(classify_logits, torch.ones_like(classify_logits))
            else:
                classify = self.criterionGAN(classify_logits, torch.zeros_like(classify_logits))

        return adv, None#classify

    def add_loss_adv_classify3d_pairedOLD(self, a, b, net_d, classifier, truth_adv, truth_classify):

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

    def add_loss_adv_classify3d_paired(self, a, b, net_d, classifier, truth_adv, truth_classify):
        adv_a, classify_a = net_d(a)  # (B*Z, 1, dH, dW)
        adv_b, classify_b = net_d(b)  # (B*Z, C, dH, dW)

        if truth_adv:
            adv_a = self.criterionGAN(adv_a, torch.ones_like(adv_a))
            adv_b = self.criterionGAN(adv_b, torch.ones_like(adv_b))
        else:
            adv_a = self.criterionGAN(adv_a, torch.zeros_like(adv_a))
            adv_b = self.criterionGAN(adv_b, torch.zeros_like(adv_b))


        #print(classify_a.shape)  #(46, 256, 16, 16)
        #print(truth_classify.shape)  # (2) (0~1)
        # STUPID MESS UP PART WHERE I MULTIPLE FEATURES DIFF WITH LABEL SIGN
        if 1:
            B = truth_classify.shape[0]
            Z = classify_a.shape[0] // B

            classify_diff = (classify_a - classify_b)
            classify_diff = classify_diff.view((B, Z) + classify_diff.shape[1:4])

            sign_diff = (truth_classify * 2) - 1  # (-1~1)
            sign_diff = sign_diff.view(-1, 1, 1, 1, 1)\
                .repeat((1,) + classify_diff.shape[1:5]).type_as(classify_diff)

            classify_logits = nn.AdaptiveAvgPool2d(1)(torch.mul(classify_diff, sign_diff))  #(B, Z, 256, 1, 1)

            classify_logits, _ = torch.max(classify_logits, 1)
            classify_logits = classifier(classify_logits)

        elif 0:
            if truth_classify[0]:  # if right knee pain
                classify_logits = nn.AdaptiveAvgPool2d(1)(classify_a - classify_b)  # (right knee - left knee)
            else:  # if left knee pain
                classify_logits = nn.AdaptiveAvgPool2d(1)(classify_b - classify_a)  # (right knee - left knee)

            classify_logits, _ = torch.max(classify_logits, 0)
            classify_logits = classify_logits.unsqueeze(0)
            classify_logits = classifier(classify_logits)

        classify = self.criterionGAN(classify_logits, truth_classify.view(-1, 1, 1, 1).type_as(classify_logits))

        return adv_a, adv_b, classify, classify_logits

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
        ids = [x.split('/')[-1].split('_')[0] for x in self.batch['filenames'][0]]
        paindiff = [(self.df.loc[self.df['ID'] == int(i), ['V00WOMKPR']].values[0][0]\
                   - self.df.loc[self.df['ID'] == int(i), ['V00WOMKPL']].values[0][0]) for i in ids]  # Right - Left
        self.paindiff = torch.FloatTensor([x / 10 for x in paindiff])

        side_classify = (torch.sign(self.paindiff) + 1) / 2

        ax, ay, cxy, lxy = self.add_loss_adv_classify3d_paired(a=self.oriX, b=self.oriY, net_d=self.net_d,
                                                               classifier=self.classifier,
                                                               truth_adv=True, truth_classify=side_classify)
        loss_dc = cxy
        self.log('valdc', loss_dc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # STUPID WHY INVERSE?
        label = 1 - side_classify.type(torch.LongTensor)

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

    def validation_loggingXXX(self, batch, batch_idx):
        print('val step')
        self.batch_idx = batch_idx
        self.batch = batch
        if self.hparams.load3d:  # if working on 3D input, bring the Z dimension to the first and combine with batch
            self.batch['img'] = self.reshape_3d(self.batch['img'])

        self.generation()

        ### STUPID
        id = self.batch['filenames'][0][0].split('/')[-1].split('_')[0]
        #if id in ['9026695', '9039627']:
        if self.batch_idx in [5, 6, 7]:
            self.log_helper.append(self.oriX)
            self.log_helper.append(self.imgXY)
        ### STUPID


# CUDA_VISIBLE_DEVICES=0,1,2 python train.py --jsn womac3 --prj Gds/descar3/Gdsmc3DB --mc --engine descar3 --netG dsmc --netD descar --direction areg_b --index --gray --load3d --final none
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --jsn womac3 --prj 3D/descar3/GdsmcDbpatch16Trd800 --mc --engine descar3 --netG dsmc --netD bpatch_16 --direction ap_bp --split moaks --load3d --final none --n_epochs 400 --trd 800
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --jsn womac3 --prj 3D/descar3/GdsmcDbpatch16Trd800  --models descar3 --netG dsmc --netD bpatch_16 --direction ap_bp --final none -b 1 --split moaks --final none --n_epochs 400 --trd 800