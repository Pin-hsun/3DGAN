import os, time
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
import numpy as np
from PIL import Image
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer

from models.cyclegan.models import GeneratorResNet, Discriminator
from models.cyclegan.utils import ReplayBuffer, LambdaLR

from models.networks import define_G, define_D
from models.loss import GANLoss

cuda = True if torch.cuda.is_available() else False
# Tensor type
Tensor = torch.cuda.HalfTensor if cuda else torch.FloatTensor


class CycleGanModel(LightningModule):
    def __init__(self,
                 hparams,
                 dir_checkpoints):
        super().__init__()
        self.hparams = hparams
        self.dir_checkpoints = dir_checkpoints
        self.epoch = 0
        self.tini = time.time()
        self.lr = hparams.lr
        self.b1 = hparams.beta1
        self.b2 = 0.999
        input_shape = (3, 384, 384)
        self.input_shape = input_shape
        self.lambda_cyc = hparams.lambda_cyc
        #self.lambda_id = lambda_id

        # Image transformations
        self.transforms_ = None

        # Losses
        self.criterion_GAN = GANLoss(hparams.gan_mode)#nn.BCEWithLogitsLoss()#torch.nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

        self.G_AB = define_G(input_nc=hparams.input_nc, output_nc=hparams.output_nc, ngf=64, netG=hparams.netG,
                             norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[])
        self.G_BA = define_G(input_nc=hparams.input_nc, output_nc=hparams.output_nc, ngf=64, netG=hparams.netG,
                             norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[])

        if hparams.netD == 'cycle':
            self.D_A = Discriminator(input_shape)
            self.D_B = Discriminator(input_shape)
            self.dPatch = 16
        else:
            self.D_A = define_D(input_nc=hparams.output_nc, ndf=64, netD=hparams.netD)
            self.D_B = define_D(input_nc=hparams.output_nc, ndf=64, netD=hparams.netD)
            if hparams.netD == 'pixel':
                self.dPatch = 1

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        # Losses
        #self.criterion_GAN = GANLoss(hparams.gan_mode)#nn.BCEWithLogitsLoss()#
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

    def forward(self, z):
        return self.G_AB(z)

    def adversarial_loss(self, y_hat, y):
        raise NotImplementedError

    def training_step(self, batch, batch_idx, optimizer_idx):
        # Set model input
        real_A, real_B = batch
        #real_A = Variable(batch["A"].type(Tensor))
        #eal_B = Variable(batch["B"].type(Tensor))

        # Adversarial ground truths
        #valid = Variable(
        #    Tensor(np.ones((real_A.size(0), *self.D_A.output_shape))), requires_grad=False)
        #fake = Variable(
        #    Tensor(np.zeros((real_A.size(0), *self.D_A.output_shape))), requires_grad=False)

        valid = torch.ones(real_A.shape[0], 1, self.dPatch, self.dPatch).cuda()
        fake = torch.zeros(real_A.shape[0], 1, self.dPatch, self.dPatch).cuda()

        fake_B = self.G_AB(real_A)
        fake_A = self.G_BA(real_B)

        # ------------------
        #  Train Generators
        # ------------------
        if optimizer_idx == 0:

            self.G_AB.train()
            self.G_BA.train()

            # Identity loss
            loss_id_A = self.criterion_identity(self.G_BA(real_A), real_A)
            loss_id_B = self.criterion_identity(self.G_AB(real_B), real_B)

            loss_identity = (loss_id_A * 1 + loss_id_B * 1) / 2

            # GAN loss
            loss_GAN_AB = self.criterion_GAN(self.D_B(fake_B), valid)
            loss_GAN_BA = self.criterion_GAN(self.D_A(fake_A), valid)
            loss_GAN = (loss_GAN_AB * 1 + loss_GAN_BA * 1) / 2

            # Cycle loss
            recov_A = self.G_BA(fake_B)
            loss_cycle_A = self.criterion_cycle(recov_A, real_A)
            recov_B = self.G_AB(fake_A)
            loss_cycle_B = self.criterion_cycle(recov_B, real_B)

            loss_cycle = (loss_cycle_A * 1 + loss_cycle_B * 1) / 2

            # Total loss
            loss_G = loss_GAN + self.lambda_cyc * loss_cycle + self.hparams.lambda_id * loss_identity
            tqdm_dict = {'loss_G': loss_G}

            output = OrderedDict({
                'loss': loss_G,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            self.log('loss_G', loss_G, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)
            return output

        if optimizer_idx > 0:
            # Real loss
            loss_real_A = self.criterion_GAN(self.D_A(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A_ = self.fake_A_buffer.push_and_pop(fake_A)
            loss_fake_A = self.criterion_GAN(
                self.D_A(fake_A_.detach()), fake)
            loss_D_A = (loss_real_A + loss_fake_A) / 2

            # -----------------------
            #  Train Discriminator A
            # -----------------------
            if optimizer_idx == 1:
                # Real loss
                loss_real = self.criterion_GAN(self.D_A(real_A), valid)
                # Fake loss (on batch of previously generated samples)
                fake_A_ = self.fake_A_buffer.push_and_pop(fake_A)
                loss_fake = self.criterion_GAN(self.D_A(fake_A_.detach()), fake)
                # Total loss
                loss_D_A = (loss_real + loss_fake) / 2

                tqdm_dict = {'loss_D_A': loss_D_A}

                output = OrderedDict({
                    'loss': loss_D_A,
                    'progress_bar': tqdm_dict,
                    'log': tqdm_dict
                })
                self.log('loss_D_A', loss_D_A, on_step=False, on_epoch=True,
                         prog_bar=True, logger=True, sync_dist=True)
                return output

            # -----------------------
            #  Train Discriminator B
            # -----------------------
            if optimizer_idx == 2:
                # Real loss
                loss_real_B = self.criterion_GAN(self.D_B(real_B), valid)
                # Fake loss (on batch of previously generated samples)
                fake_B_ = self.fake_B_buffer.push_and_pop(fake_B)
                loss_fake_B = self.criterion_GAN(
                    self.D_B(fake_B_.detach()), fake)
                # Total loss
                loss_D_B = (loss_real_B + loss_fake_B) / 2

                loss_D = (loss_D_A + loss_D_B) / 2

                tqdm_dict = {'loss_D': loss_D}

                output = OrderedDict({
                    'loss': loss_D,
                    'progress_bar': tqdm_dict,
                    'log': tqdm_dict
                })
                self.log('loss_D_B', loss_D_B, on_step=False, on_epoch=True,
                         prog_bar=True, logger=True, sync_dist=True)
                return output

    def configure_optimizers(self):
        lr = self.lr
        b1 = self.b1
        b2 = self.b2

        # Optimizers
        optimizer_G = torch.optim.Adam(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()), lr=lr, betas=(b1, b2)
        )
        optimizer_D_A = torch.optim.Adam(
            self.D_A.parameters(), lr=lr, betas=(b1, b2))
        optimizer_D_B = torch.optim.Adam(
            self.D_B.parameters(), lr=lr, betas=(b1, b2))

        # Learning rate update schedulers
        #lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        #    optimizer_G, lr_lambda=LambdaLR(
        #        self.n_epochs, self.epoch, self.decay_epoch).step
        #)
        #lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        #    optimizer_D_A, lr_lambda=LambdaLR(
        #        self.n_epochs, self.epoch, self.decay_epoch).step
        #)
        #lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        #    optimizer_D_B, lr_lambda=LambdaLR(
        #        self.n_epochs, self.epoch, self.decay_epoch).step
        #)

        return [optimizer_G, optimizer_D_A, optimizer_D_B], []

    def training_epoch_end(self, outputs):
        hparams = self.hparams
        # checkpoint
        dir_checkpoints = self.dir_checkpoints
        if self.epoch % 20 == 0:
            if not os.path.exists(dir_checkpoints):
                os.mkdir(dir_checkpoints)
            if not os.path.exists(os.path.join(dir_checkpoints, hparams.prj)):
                os.mkdir(os.path.join(dir_checkpoints, hparams.prj))
            net_g_ab_model_out_path = dir_checkpoints + "/{}/netG_ab_model_epoch_{}.pth".format(hparams.prj, self.epoch)
            net_g_ba_model_out_path = dir_checkpoints + "/{}/netG_ba_model_epoch_{}.pth".format(hparams.prj, self.epoch)
            torch.save(self.G_AB, net_g_ab_model_out_path)
            torch.save(self.G_BA, net_g_ba_model_out_path)
            net_d_model_out_path = dir_checkpoints + "/{}/netD_model_epoch_{}.pth".format(hparams.prj, self.epoch)
            torch.save(self.net_d, net_d_model_out_path)
            print("Checkpoint saved to {}".format(dir_checkpoints + '/' + hparams.prj))

        self.epoch += 1
        self.tini = time.time()
        self.avg_psnr = 0

        # shuffle index of image_b for cyclegan
        self.train_loader.dataset.reshuffle_b()

