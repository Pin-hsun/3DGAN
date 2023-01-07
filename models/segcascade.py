from models.base import BaseModel
import torch
import torch.optim as optim
from utils.metrics_segmentation import SegmentationCrossEntropyLoss, SegmentationDiceCoefficient
import torch.nn as nn
from networks.networks import get_scheduler
import numpy as np
from models.helper import reshape_3d
import copy


class GAN(BaseModel):
    """
    There is a lot of patterned noise and other failures when using lightning
    """
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints)

        # set networks
        self.hparams.input_nc = 2
        self.hparams.output_nc = 7
        self.segnet, _ = self.set_networks()
        self.hparams.input_nc = 1
        self.hparams.output_nc = 7
        self.segnetA, _ = self.set_networks()
        # update model names
        self.model_names = {'segnet': 'segnet', 'segnetA': 'segnetA'}

        self.init_optimizer_scheduler()

        self.segloss = SegmentationCrossEntropyLoss()
        self.segdice = SegmentationDiceCoefficient()

        self.all_label = []
        self.all_out = []

    @staticmethod
    def add_model_specific_args(parent_parser):
        return parent_parser

    def configure_optimizers(self):
        seg_parameters = []
        for g in self.model_names.keys():
            seg_parameters = seg_parameters + list(getattr(self, g).parameters())

        self.optimizer = optim.Adam(seg_parameters, lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        # not using pl scheduler for now....
        return self.optimizer

    def init_optimizer_scheduler(self):
        # Optimizer and scheduler
        self.optimizer = self.configure_optimizers()
        self.scheduler = get_scheduler(self.optimizer, self.hparams)

    def generation(self, batch):
        if self.hparams.load3d:  # if working on 3D input, bring the Z dimension to the first and combine with batch
            batch['img'] = reshape_3d(batch['img'])

        img = batch['img']
        self.ori = img[1]
        self.ori = self.ori / self.ori.max()

        self.mask = img[0].type(torch.ByteTensor)
        self.mask[self.mask == 3] = 2

        self.oriLowRes = torch.nn.Upsample(scale_factor=0.5, mode='bilinear')(self.ori)
        self.maskLowRes = torch.nn.Upsample(scale_factor=0.5, mode='nearest')(self.mask)

        self.maskLowRes = self.maskLowRes.type(torch.LongTensor).to(self.ori.device)
        self.mask = self.mask.type(torch.LongTensor).to(self.ori.device)

        self.orisegLowRes = self.segnetA(self.oriLowRes)[0]
        self.orisegLowRes = torch.argmax(self.orisegLowRes, 1).unsqueeze(1).type(torch.ByteTensor)

        self.orisegLowResHigh = torch.nn.Upsample(scale_factor=2, mode='nearest')(self.orisegLowRes)
        #self.orisegLowResHigh = torch.argmax(self.orisegLowResHigh, 1).unsqueeze(1)

        self.orisegLowResHigh = self.orisegLowResHigh.type(torch.FloatTensor).cuda()

        self.oriseg = self.segnet(torch.cat([self.ori, self.orisegLowResHigh], 1))[0]

    def training_step(self, batch, batch_idx):
        self.batch_idx = batch_idx

        self.generation(batch)
        seg_loss, seg_prob = self.segloss(self.oriseg, self.mask)
        self.log('seglosst', seg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return seg_loss

    def validation_step(self, batch, batch_idx):
        self.segnet.train()

        self.generation(batch)
        seg_loss, seg_prob = self.segloss(self.oriseg, self.mask)
        self.log('seglossv', seg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # metrics
        self.all_label.append(self.mask.cpu())
        self.all_out.append(self.oriseg.cpu().detach())

        return seg_loss

    def training_epoch_end(self, outputs):
        self.train_loader.dataset.shuffle_images()

        # checkpoint
        if self.epoch % 20 == 0:
            for name in self.model_names.keys():
                path_g = self.dir_checkpoints + ('/' + self.model_names[name] + '_model_epoch_{}.pth').format(self.epoch)
                torch.save(getattr(self, name), path_g)
                print("Checkpoint saved to {}".format(path_g))

        self.epoch += 1
        self.scheduler.step()

        self.all_label = []
        self.all_out = []

    def validation_epoch_end(self, x):
        seg = torch.cat(self.all_out, 0)
        mask = torch.cat(self.all_label, 0)

        del self.all_out
        del self.all_label

        seg = torch.argmax(seg, 1).view(-1)
        mask = mask.view(-1)
        dice = []
        for i in range(7):
            tp = ((mask == i) & (seg == i)).sum().item()
            uni = (mask == i).sum().item() + (seg == i).sum().item()
            dice.append(2 * tp / (uni + 0.001))
        print(dice)

        #for i in range(len(auc)):
        #    self.log('auc' + str(i), auc[i], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.all_label = []
        self.all_out = []

        return 0#metrics




#CUDA_VISIBLE_DEVICES=0 python train.py --jsn seg --prj segmentation --models segcascade --split a --norm instance --load3d -b 1
