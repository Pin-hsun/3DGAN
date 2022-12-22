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

        # update model names
        self.seg_names = {'segnet': 'segnet'}

        self.init_networks_optimizer_scheduler()

        self.segloss = SegmentationCrossEntropyLoss()
        self.segdice = SegmentationDiceCoefficient()

        self.all_label = []
        self.all_out = []

    @staticmethod
    def add_model_specific_args(parent_parser):
        return parent_parser

    def configure_optimizers(self):
        seg_parameters = []
        for g in self.seg_names.keys():
            seg_parameters = seg_parameters + list(getattr(self, g).parameters())

        self.optimizer = optim.Adam(seg_parameters, lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        # not using pl scheduler for now....
        return self.optimizer

    def init_networks_optimizer_scheduler(self):
        # set networks
        self.hparams.output_nc = 7
        self.segnet, _ = self.set_networks()
        # Optimizer and scheduler
        self.optimizer = self.configure_optimizers()
        self.scheduler = get_scheduler(self.optimizer, self.hparams)

    def generation(self):
        self.img = self.batch['img']
        self.ori = self.img[1]

        self.ori = self.ori / self.ori.max()

        self.mask = self.img[0].type(torch.LongTensor).to(self.ori.device)
        self.oriseg = self.segnet(self.ori)[0]

    def training_step(self, batch, batch_idx, ):
        self.batch_idx = batch_idx
        self.batch = batch
        if self.hparams.load3d:  # if working on 3D input, bring the Z dimension to the first and combine with batch
            self.batch['img'] = self.reshape_3d(self.batch['img'])

        self.generation()
        seg_loss, seg_prob = self.segloss(self.oriseg, self.mask)
        self.log('seglosst', seg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return seg_loss

    def validation_step(self, batch, batch_idx):
        #self.segnet.train()
        self.batch_idx = batch_idx
        self.batch = batch
        if self.hparams.load3d:  # if working on 3D input, bring the Z dimension to the first and combine with batch
            self.batch['img'] = self.reshape_3d(self.batch['img'])

        self.generation()
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
            for name in self.seg_names.keys():
                path_g = self.dir_checkpoints + ('/' + self.seg_names[name] + '_model_epoch_{}.pth').format(self.epoch)
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

        #print(all_out.shape)
        #print(all_label.shape)
        #metrics = self.segdice(all_label, all_out)
        #print(metrics)
        #auc = torch.from_numpy(np.array(metrics)).cuda()
        #print(self.all_out.shape)
        seg = torch.argmax(seg, 1).view(-1)
        mask = mask.view(-1)
        dice = []
        for i in range(7):
            tp = ((mask == i) & (seg == i)).sum().item()
            uni = (mask == i).sum().item() + (seg == i).sum().item()
            dice.append(2 * tp / uni)
        print(dice)

        #for i in range(len(auc)):
        #    self.log('auc' + str(i), auc[i], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.all_label = []
        self.all_out = []

        return 0#metrics




#CUDA_VISIBLE_DEVICES=0 python train.py --jsn seg --prj segmentation --models segmentation --split a
