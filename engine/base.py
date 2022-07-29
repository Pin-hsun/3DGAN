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
        self.set_networks()

        # Optimizer and scheduler
        [self.optimizer_d, self.optimizer_g], [] = self.configure_optimizers()
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

        elif (self.hparams.netG).startswith('de'):
            print('descar generator: ' + self.hparams.netG)
            Generator = getattr(getattr(__import__('models.DeScarGan.' + self.hparams.netG), 'DeScarGan'),
                                self.hparams.netG).Generator
            # descargan only has options for batchnorm or none
            if self.hparams.norm == 'batch':
                usebatch = True
            elif self.hparams.norm == 'none':
                usebatch = False
            self.net_g = Generator(n_channels=self.hparams.input_nc, out_channels=self.hparams.output_nc, batch_norm=usebatch, final=self.hparams.final,
                                   mc=self.hparams.mc)
            self.net_g_inc = 2
        elif (self.hparams.netG).startswith('ds'):
            print('descar generator: ' + self.hparams.netG)
            Generator = getattr(getattr(__import__('models.DSGan.' + self.hparams.netG), 'DSGan'),
                                self.hparams.netG).Generator
            # descargan only has options for batchnorm or none
            if self.hparams.norm == 'batch':
                usebatch = True
            elif self.hparams.norm == 'none':
                usebatch = False
            self.net_g = Generator(n_channels=self.hparams.input_nc, out_channels=self.hparams.output_nc, batch_norm=usebatch, final=self.hparams.final,
                                   mc=self.hparams.mc)
            self.net_g_inc = 2
        elif self.hparams.netG == 'ugatit':
            from models.ugatit.networks import ResnetGenerator
            print('use ugatit generator')
            self.net_g = ResnetGenerator(input_nc=self.hparams.input_nc,
                                         output_nc=self.hparams.output_nc, ngf=self.hparams.ngf,
                                         n_blocks=4, img_size=128, light=True)
            self.net_g_inc = 0
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
        elif self.hparams.netD == 'ugatit':
            from models.ugatit.networks import Discriminator
            print('use ugatit discriminator')
            self.net_d = Discriminator(self.hparams.input_nc * 2, ndf=64, n_layers=5)
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

    def add_loss_adv_classify3d_paired(self, a, b, net_d, truth_adv, truth_classify, log=None):
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
            classify_logits = classify_a - classify_b
        else:
            classify_logits = classify_b - classify_a

        # 3D classification
        classify_logits = nn.AdaptiveAvgPool2d(1)(classify_logits)
        classify_logits = classify_logits.sum(0).unsqueeze(0)

        if truth_classify:
            classify = self.criterionGAN(classify_logits, torch.ones_like(classify_logits))
        else:
            classify = self.criterionGAN(classify_logits, torch.zeros_like(classify_logits))

        if log is not None:
            self.log(log, adv, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return adv_a, adv_b, classify


    def add_loss_adv(self, a, net_d, coeff, truth, b=None, log=None, stacked=False):
        if stacked:
            fake_in = torch.cat((a, b), 1)
        else:
            fake_in = torch.cat((a, a), 1)
        disc_logits = net_d(fake_in)[0]
        if truth:
            adv = self.criterionGAN(disc_logits, torch.ones_like(disc_logits))
        else:
            adv = self.criterionGAN(disc_logits, torch.zeros_like(disc_logits))
        if log is not None:
            self.log(log, adv, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return coeff * adv

    def add_loss_L1(self, a, b, coeff, log=None):
        l1 = self.criterionL1(a, b)
        if log is not None:
            self.log(log, l1, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return coeff * l1

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.batch_idx = batch_idx
        self.batch = batch
        if self.hparams.bysubject:  # if working on 3D input
            if len(self.batch['img'][0].shape) == 5:
                for i in range(len(self.batch['img'])):
                    (B, C, H, W, Z) = self.batch['img'][i].shape
                    self.batch['img'][i] = self.batch['img'][i].permute(0, 4, 1, 2, 3)
                    self.batch['img'][i] = self.batch['img'][i].reshape(B * Z, C, H, W)

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
        if self.epoch % 10 == 0:
            for name in self.netg_names.keys():
                path = self.dir_checkpoints + ('/' + self.netg_names[name] + '_model_epoch_{}.pth').format(self.epoch)
                torch.save(getattr(self, name), path)
                print("Checkpoint saved to {}".format(path))

        self.epoch += 1
        self.tini = time.time()
        self.net_g_scheduler.step()
        self.net_d_scheduler.step()

        self.all_label = []
        self.all_out = []

    def validation_step(self, batch, batch_idx):
        self.batch = batch
        if self.hparams.bysubject:  # if working on 3D input
            if len(self.batch['img'][0].shape) == 5:
                for i in range(len(self.batch['img'])):
                    (B, C, H, W, Z) = self.batch['img'][i].shape
                    self.batch['img'][i] = self.batch['img'][i].permute(0, 4, 1, 2, 3)
                    self.batch['img'][i] = self.batch['img'][i].reshape(B * Z, C, H, W)
        img = self.batch['img']
        self.oriX = img[0]
        self.oriY = img[1]

        net_d = self.net_d
        # cx
        a = self.oriX
        fake_in = torch.cat((a, a), 1)
        _, classify_logits = net_d(fake_in)
        classify_logits = nn.AdaptiveAvgPool2d(1)(classify_logits)
        classify_logits = classify_logits.sum(0).unsqueeze(0)
        cx = self.criterionGAN(classify_logits, torch.ones_like(classify_logits))
        lx = classify_logits[:, :, 0, 0]
        # cy
        a = self.oriY
        fake_in = torch.cat((a, a), 1)
        _, classify_logits = net_d(fake_in)
        classify_logits = nn.AdaptiveAvgPool2d(1)(classify_logits)
        classify_logits = classify_logits.sum(0).unsqueeze(0)
        cy = self.criterionGAN(classify_logits, torch.zeros_like(classify_logits))
        ly = classify_logits[:, :, 0, 0]

        loss_dc = 0.5 * cx + 0.5 * cy # + (cxy + cxx + cyy + cyx) * 0.5
        self.log('valdc', loss_dc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # metrics
        flip = np.random.randint(2)
        if flip:
            label = torch.ones(1).type(torch.LongTensor)
            out = torch.cat([ly, lx], 1)
        else:
            label = torch.zeros(1).type(torch.LongTensor)
            out = torch.cat([lx, ly], 1)

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

    def generation(self):
        return 0

    def backward_g(self, inputs):
        return 0

    def backward_d(self, inputs):
        return 0

# CUDA_VISIBLE_DEVICES=2 python train.py --dataset womac3 -b 16 --prj NS/unet128 --direction aregis1_b --cropsize 256 --engine pix2pixNS --netG unet_128


