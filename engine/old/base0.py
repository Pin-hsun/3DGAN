import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from models.networks import get_scheduler
from models.loss import GANLoss
from math import log10
import time, os
import pytorch_lightning as pl
from utils.metrics_segmentation import SegmentationCrossEntropyLoss
from utils.metrics_classification import CrossEntropyLoss, GetAUC
from utils.data_utils import *


def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def combine(x, y, method):
    if method == 'res':
        return x + y
    elif method == 'mul':
        return torch.mul(x, y)
    elif method == 'multanh':
        return torch.mul((x + 1) / 2, y)


class BaseModel(pl.LightningModule):
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        super(BaseModel, self).__init__()
        self.train_loader = train_loader

        # initialize
        self.tini = time.time()
        self.epoch = 0
        self.dir_checkpoints = checkpoints

        # save model names
        self.netg_names = {'net_g': 'netG'}
        self.netd_names = {'net_d': 'netD'}
        self.loss_g_names = ['loss_g']
        self.loss_d_names = ['loss_d']

        # hyperparameters
        hparams = {x: vars(hparams)[x] for x in vars(hparams).keys() if x not in hparams.not_tracking_hparams}
        hparams.pop('not_tracking_hparams', None)
        self.hparams.update(hparams)
        self.save_hyperparameters(self.hparams)

        # set networks
        self.set_networks()

        # Optimizer and scheduler
        [self.optimizer_d, self.optimizer_g], [] = self.configure_optimizers()
        self.net_g_scheduler = get_scheduler(self.optimizer_g, self.hparams)
        self.net_d_scheduler = get_scheduler(self.optimizer_d, self.hparams)

        # Define Loss Functions
        self.CELoss = CrossEntropyLoss()
        self.criterionL1 = nn.L1Loss().cuda()
        if self.hparams.gan_mode == 'vanilla':
            self.criterionGAN = nn.BCEWithLogitsLoss()
        else:
            self.criterionGAN = GANLoss(self.hparams.gan_mode).cuda()
        self.segLoss = SegmentationCrossEntropyLoss()

        # Final
        self.hparams.update(vars(self.hparams))   # updated hparams to be logged in tensorboard

    def test_method(self, net_g, x):
        output, = net_g(x[0])
        return output

    def set_networks(self):
        # GENERATOR
        if self.hparams.netG == 'attgan':
            from models.AttGAN.attgan import Generator
            print('use attgan generator')
            self.net_g = Generator(n_in=self.hparams.input_nc, enc_dim=self.hparams.ngf, dec_dim=self.hparams.ngf,
                                   n_attrs=self.hparams.n_attrs, img_size=256,
                                   enc_norm_fn=self.hparams.norm, dec_norm_fn=self.hparams.norm,
                                   final=self.hparams.final)
            self.net_g_inc = 1

        elif (self.hparams.netG).startswith('descar'):
            print('descar generator: ' + self.hparams.netG)
            Generator = getattr(getattr(__import__('models.DeScarGan.' + self.hparams.netG), 'DeScarGan'),
                                self.hparams.netG).Generator
            # descargan only has options for batchnorm or none
            if self.hparams.norm == 'batch':
                usebatch = True
            elif self.hparams.norm == 'none':
                usebatch = False
            self.net_g = Generator(n_channels=self.hparams.input_nc, out_channels=self.hparams.output_nc, batch_norm=usebatch, final=self.hparams.final)
            self.net_g_inc = 2
        elif (self.hparams.netG).startswith('uneta'):
            from models.networks2 import define_G
            self.net_g_inc = 1
            self.net_g = define_G(input_nc=self.hparams.input_nc, output_nc=self.hparams.output_nc,
                                  ngf=self.hparams.ngf, netG=self.hparams.netG,
                                  norm=self.hparams.norm, use_dropout=self.hparams.mc, init_type='normal', init_gain=0.02, gpu_ids=[],
                                  final=self.hparams.final)
        else:
            from models.networks import define_G
            self.net_g = define_G(input_nc=self.hparams.input_nc, output_nc=self.hparams.output_nc,
                                  ngf=self.hparams.ngf, netG=self.hparams.netG,
                                  norm=self.hparams.norm, use_dropout=self.hparams.mc, init_type='normal', init_gain=0.02, gpu_ids=[],
                                  final=self.hparams.final)
            self.net_g_inc = 0

        # DISCRIMINATOR
        if (self.hparams.netD).startswith('patch'):  # Patchgan from cyclegan (the pix2pix one is strange)
            from models.cyclegan.models import Discriminator
            self.net_d = Discriminator(input_shape=(self.hparams.output_nc * 2, 256, 256), patch=int((self.hparams.netD).split('_')[-1]))
        elif self.hparams.netD == 'sagan':
            from models.sagan.sagan import Discriminator
            print('use sagan discriminator')
            self.net_d = Discriminator(image_size=64)
        elif self.hparams.netD == 'acgan':
            from models.acgan import Discriminator
            print('use acgan discriminator')
            self.net_d = Discriminator(img_shape=(self.hparams.input_nc_nc * 2, 256, 256), n_classes=2)
        elif self.hparams.netD == 'attgan':
            from models.AttGAN.attgan import Discriminators
            print('use attgan discriminator')
            self.net_d = Discriminators(img_size=256, cls=2)
        elif self.hparams.netD == 'descar':
            from models.DeScarGan.descargan import Discriminator
            print('use descargan discriminator')
            self.net_d = Discriminator(n_channels=self.hparams.input_nc * 2)
        elif self.hparams.netD == 'descars':
            from models.DeScarGan.descarganshallow import Discriminator
            print('use descargan shallow discriminator')
            self.net_d = Discriminator(n_channels=self.hparams.input_nc * 2)
        # original pix2pix, the size of patchgan is strange, just use for pixel-D
        else:
            from models.networks import define_D
            self.net_d = define_D(input_nc=self.hparams.output_nc * 2, ndf=64, netD=self.hparams.netD)

        # Init. Network Parameters
        self.net_g = self.net_g.apply(_weights_init)
        if self.hparams.netD == 'sagan':
            print('not init for netD of sagan')
        else:
            self.net_d = self.net_d.apply(_weights_init)

    def configure_optimizers(self):
        netg_parameters = []
        for g in self.netg_names.keys():
            netg_parameters = netg_parameters + list(getattr(self, g).parameters())

        netd_parameters = []
        for d in self.netd_names.keys():
            netd_parameters = netd_parameters + list(getattr(self, d).parameters())

        self.optimizer_g = optim.Adam(netg_parameters, lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        self.optimizer_d = optim.Adam(netd_parameters, lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        return [self.optimizer_d, self.optimizer_g], []

    def add_loss_adv(self, a, net_d, loss, coeff, truth, b=None, log=None, stacked=False):
        if stacked:
            fake_in = torch.cat((a, b), 1)
        else:
            fake_in = torch.cat((a, a), 1)
        disc_logits = net_d(fake_in)[0]
        if truth:
            adv = self.criterionGAN(disc_logits, torch.ones_like(disc_logits))
        else:
            adv = self.criterionGAN(disc_logits, torch.zeros_like(disc_logits))
        loss = loss + coeff * adv
        if log is not None:
            self.log(log, coeff * adv, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def add_loss_L1(self, a, b, loss, coeff, log=None):
        l1 = self.criterionL1(a, b)
        loss = loss + coeff * l1
        if log is not None:
            self.log(log, coeff * l1, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.batch_idx = batch_idx
        self.batch = batch
        if self.hparams.bysubject:  # if working on 3D input
            if len(self.batch[0].shape) == 5:
                for i in range(len(self.batch)):
                    (B, C, H, W, Z) = self.batch[i].shape
                    self.batch[i] = self.batch[i].permute(0, 4, 1, 2, 3)
                    self.batch[i] = self.batch[i].reshape(B * Z, C, H, W)

        if optimizer_idx == 0:
            imgs = self.generation()
            loss_d = self.backward_d(imgs)
            for name in self.loss_d_names:
                self.log(name, loss_d[name], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            return loss_d['sum']

        if optimizer_idx == 1:
            imgs = self.generation()
            loss_g = self.backward_g(imgs)
            for name in self.loss_g_names:
                self.log(name, loss_g[name], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            return loss_g['sum']

    def training_epoch_end(self, outputs):
        # checkpoint
        if self.epoch % 10 == 0:
            for name in self.netg_names.keys():
                path = self.dir_checkpoints + ('/' + self.netg_names[name] + '_model_epoch_{}.pth').format(self.epoch)
                torch.save(getattr(self, name), path)
                print("Checkpoint saved to {}".format(path))

        self.epoch += 1
        self.tini = time.time()
        self.net_g_scheduler.step()
        self.net_d_scheduler.step()

    #def validation_step(self, batch, batch_idx):
    #    self.batch = batch
    #    print('v')

    def generation(self):
        return 0

    def backward_g(self, inputs):
        return 0

    def backward_d(self, inputs):
        return 0

# CUDA_VISIBLE_DEVICES=2 python train.py --dataset womac3 -b 16 --prj NS/unet128 --direction aregis1_b --cropsize 256 --engine pix2pixNS --netG unet_128
