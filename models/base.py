import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from networks.networks import get_scheduler
from networks.loss import GANLoss

import time, os
import pytorch_lightning as pl
from utils.metrics_segmentation import SegmentationCrossEntropyLoss
from utils.metrics_classification import CrossEntropyLoss, GetAUC
from utils.data_utils import *
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


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
    elif method == 'not':
        return x


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
        self.net_g, self.net_d = self.set_networks()


        # Optimizer and scheduler

        [self.optimizer_d, self.optimizer_g], _ = self.configure_optimizers()
        self.net_g_scheduler = get_scheduler(self.optimizer_g, self.hparams)
        self.net_d_scheduler = get_scheduler(self.optimizer_d, self.hparams)

        # Define Loss Functions
        self.CELoss = CrossEntropyLoss()
        self.criterionL1 = nn.L1Loss()
        if self.hparams.gan_mode == 'vanilla':
            self.criterionGAN = nn.BCEWithLogitsLoss()
        else:
            self.criterionGAN = GANLoss(self.hparams.gan_mode)
        self.segLoss = SegmentationCrossEntropyLoss()

        # Final
        self.hparams.update(vars(self.hparams))   # updated hparams to be logged in tensorboard
        self.train_loader.dataset.shuffle_images()

        self.all_label = []
        self.all_out = []

    def configure_optimizers(self):
        netg_parameters = []
        for g in self.netg_names.keys():
            netg_parameters = netg_parameters + list(getattr(self, g).parameters())

        netd_parameters = []
        for d in self.netd_names.keys():
            netd_parameters = netd_parameters + list(getattr(self, d).parameters())

        self.optimizer_g = optim.Adam(netg_parameters, lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        self.optimizer_d = optim.Adam(netd_parameters, lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        #self.net_g_scheduler = get_scheduler(self.optimizer_g, self.hparams)
        #self.net_d_scheduler = get_scheduler(self.optimizer_d, self.hparams)

        return [self.optimizer_d, self.optimizer_g], []#[self.net_d_scheduler, self.net_g_scheduler]

    def add_loss_adv(self, a, net_d, coeff, truth):
        disc_logits = net_d(a)[0]
        if truth:
            adv = self.criterionGAN(disc_logits, torch.ones_like(disc_logits))
        else:
            adv = self.criterionGAN(disc_logits, torch.zeros_like(disc_logits))
        return coeff * adv

    def add_loss_l1(self, a, b, coeff):
        l1 = self.criterionL1(a, b)
        return coeff * l1

    @staticmethod
    def reshape_3d(img3d):
        if len(img3d[0].shape) == 5:
            for i in range(len(img3d)):
                (B, C, H, W, Z) = img3d[i].shape
                img3d[i] = img3d[i].permute(0, 4, 1, 2, 3)
                img3d[i] = img3d[i].reshape(B * Z, C, H, W)
        return img3d

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.batch_idx = batch_idx
        self.batch = batch
        if self.hparams.load3d:  # if working on 3D input, bring the Z dimension to the first and combine with batch
            self.batch['img'] = self.reshape_3d(self.batch['img'])

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
        self.train_loader.dataset.shuffle_images()

        # checkpoint
        if self.epoch % 20 == 0:
            for name in self.netg_names.keys():
                path_g = self.dir_checkpoints + ('/' + self.netg_names[name] + '_model_epoch_{}.pth').format(self.epoch)
                torch.save(getattr(self, name), path_g)
                print("Checkpoint saved to {}".format(path_g))

            if self.hparams.save_d:
                for name in self.netd_names.keys():
                    path_d = self.dir_checkpoints + ('/' + self.netd_names[name] + '_model_epoch_{}.pth').format(self.epoch)
                    torch.save(getattr(self, name), path_d)
                    print("Checkpoint saved to {}".format(path_d))

        self.epoch += 1
        self.tini = time.time()
        self.net_g_scheduler.step()
        self.net_d_scheduler.step()

        self.all_label = []
        self.all_out = []

    def generation(self):
        pass

    def backward_g(self, inputs):
        pass

    def backward_d(self, inputs):
        pass

    def set_networks(self):
        # GENERATOR
        if self.hparams.netG == 'attgan':
            from networks.AttGAN.attgan import Generator
            print('use attgan generator')
            net_g = Generator(n_in=self.hparams.input_nc, enc_dim=self.hparams.ngf, dec_dim=self.hparams.ngf,
                              n_attrs=self.hparams.n_attrs, img_size=256,
                              enc_norm_fn=self.hparams.norm, dec_norm_fn=self.hparams.norm,
                              final=self.hparams.final)
        elif (self.hparams.netG).startswith('de'):
            print('descar generator: ' + self.hparams.netG)
            Generator = getattr(getattr(__import__('networks.DeScarGan.' + self.hparams.netG), 'DeScarGan'),
                                self.hparams.netG).Generator
            # descargan only has options for batchnorm or none
            if self.hparams.norm == 'batch':
                usebatch = True
            elif self.hparams.norm == 'none':
                usebatch = False
            net_g = Generator(n_channels=self.hparams.input_nc, out_channels=self.hparams.output_nc,
                              batch_norm=usebatch, final=self.hparams.final,
                              mc=self.hparams.mc)
        elif (self.hparams.netG).startswith('ds'):
            print('descar generator: ' + self.hparams.netG)
            Generator = getattr(getattr(__import__('networks.DSGan.' + self.hparams.netG), 'DSGan'),
                                self.hparams.netG).Generator
            # descargan only has options for batchnorm or none
            if self.hparams.norm == 'batch':
                usebatch = True
            elif self.hparams.norm == 'none':
                usebatch = False
            net_g = Generator(n_channels=self.hparams.input_nc, out_channels=self.hparams.output_nc,
                              batch_norm=usebatch, final=self.hparams.final, mc=self.hparams.mc)
        elif self.hparams.netG == 'ugatit':
            from networks.ugatit.networks import ResnetGenerator
            print('use ugatit generator')
            net_g = ResnetGenerator(input_nc=self.hparams.input_nc,
                                    output_nc=self.hparams.output_nc, ngf=self.hparams.ngf,
                                    n_blocks=4, img_size=128, light=True)
        elif self.hparams.netG == 'genre':
            from networks.genre.generator.Unet_base import SPADEUNet2s
            opt = Namespace(input_size=128, parsing_nc=1, norm_G='spectralspadebatch3x3', spade_mode='res2',
                            use_en_feature=False)
            net_g = SPADEUNet2s(opt=opt, in_channels=1, out_channels=1)
        else:
            from networks.networks import define_G
            net_g = define_G(input_nc=self.hparams.input_nc, output_nc=self.hparams.output_nc,
                             ngf=self.hparams.ngf, netG=self.hparams.netG,
                             norm=self.hparams.norm, use_dropout=self.hparams.mc, init_type='normal', init_gain=0.02, gpu_ids=[],
                             final=self.hparams.final)

        # DISCRIMINATOR
        if (self.hparams.netD).startswith('patch'):  # Patchgan from cyclegan (the pix2pix one is strange)
            from networks.cyclegan.models import Discriminator
            net_d = Discriminator(input_shape=(self.hparams.output_nc * 1, 256, 256), patch=int((self.hparams.netD).split('_')[-1]))
        elif (self.hparams.netD).startswith('bpatch'):  # Patchgan from cyclegan (the pix2pix one is strange)
            from networks.cyclegan.modelsb import Discriminator
            net_d = Discriminator(input_shape=(self.hparams.output_nc * 2, 256, 256), patch=int((self.hparams.netD).split('_')[-1]))
        elif (self.hparams.netD).startswith('cpatch'):  # Patchgan from cyclegan (the pix2pix one is strange)
            from networks.cyclegan.modelsc import Discriminator
            net_d = Discriminator(input_shape=(self.hparams.output_nc * 2, 256, 256), patch=int((self.hparams.netD).split('_')[-1]))
        elif self.hparams.netD == 'sagan':
            from networks.sagan.sagan import Discriminator
            print('use sagan discriminator')
            net_d = Discriminator(image_size=64)
        elif self.hparams.netD == 'acgan':
            from networks.acgan import Discriminator
            print('use acgan discriminator')
            net_d = Discriminator(img_shape=(self.hparams.input_nc_nc * 2, 256, 256), n_classes=2)
        elif self.hparams.netD == 'attgan':
            from networks.AttGAN.attgan import Discriminators
            print('use attgan discriminator')
            net_d = Discriminators(img_size=256, cls=2)
        elif self.hparams.netD == 'descar':
            from networks.DeScarGan.descargan import Discriminator
            print('use descargan discriminator')
            net_d = Discriminator(n_channels=self.hparams.input_nc * 2)
        elif self.hparams.netD == 'ugatit':
            from networks.ugatit.networks import Discriminator
            print('use ugatit discriminator')
            net_d = Discriminator(self.hparams.input_nc * 2, ndf=64, n_layers=5)
        elif self.hparams.netD == 'ugatitb':
            from networks.ugatit.networksb import Discriminator
            print('use ugatitb discriminator')
            net_d = Discriminator(self.hparams.input_nc * 2, ndf=64, n_layers=5)
        # original pix2pix, the size of patchgan is strange, just use for pixel-D
        else:
            from networks.networks import define_D
            net_d = define_D(input_nc=self.hparams.output_nc * 2, ndf=64, netD=self.hparams.netD)

        # Init. Network Parameters
        if self.hparams.netG == 'genre':
            print('no init netG of genre')
        else:
            net_g = net_g.apply(_weights_init)
        if self.hparams.netD == 'sagan':
            print('no init netD of sagan')
        else:
            net_d = net_d.apply(_weights_init)
        return net_g, net_d