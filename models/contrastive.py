from models.base import BaseModel
import torch
import torch.optim as optim
from utils.metrics_segmentation import SegmentationCrossEntropyLoss, SegmentationDiceCoefficient
import torch.nn as nn
from networks.networks import get_scheduler
import numpy as np
from models.helper import reshape_3d


class GAN(BaseModel):
    """
    There is a lot of patterned noise and other failures when using lightning
    """
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints)

        # set networks
        self.hparams.output_nc = 2
        self.encoder, _ = self.set_networks()
        # update model names
        self.model_names = {'encoder': 'encoder'}

        self.init_optimizer_scheduler()

        self.segloss = SegmentationCrossEntropyLoss()
        self.segdice = SegmentationDiceCoefficient()

        self.all_label = []
        self.all_out = []

    @staticmethod
    def add_model_specific_args(parent_parser):
        return parent_parser

    def configure_optimizers(self):
        parameters = []
        for g in self.model_names.keys():
            parameters = parameters + list(getattr(self, g).parameters())

        self.optimizer = optim.Adam(parameters, lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
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

        self.imgX = img[0]
        self.imgY = img[1]

        self.imgXz = self.eocoder(self.imgX, a=None)['z']
        self.imgYz = self.eocoder(self.imgY, a=None)['z']

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





#CUDA_VISIBLE_DEVICES=0 python train.py --jsn seg --prj segmentation --models segmentation --split a --norm instance
