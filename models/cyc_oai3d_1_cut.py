import tifffile
from models.base import BaseModel, combine
import copy
import torch
import torch.nn as nn
from networks.networks_cut import Normalize, init_net
import numpy as np
from networks.networks_cut import PatchNCELoss

class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids
        self.max3d = nn.MaxPool3d(4)
        self.max2d = nn.MaxPool2d(2)

    def create_mlp(self, feature_shapes):
        for mlp_id, feat in enumerate(feature_shapes):
            input_nc = feat
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            #if len(self.gpu_ids) > 0:
            #mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            # B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            # print(feat.shape) torch.Size([1, 32, 128, 128, 128])
            # use 2d max pooling
            # feat = feat.reshape(feat.shape[0]*feat.shape[2], feat.shape[1], feat.shape[3], feat.shape[4]) #torch.Size([128, 32, 128, 128])
            # feat = self.max2d(feat)
            # feat = feat.permute(0, 2, 3, 1) # (B, H, W, C)
            # feat_reshape = feat.reshape(feat.shape[0], feat.shape[1] * feat.shape[2], feat.shape[3])
            # use 3d max pooling
            # feat = self.max3d(feat)
            feat = feat.permute(0, 2, 3, 4, 1)  # (B, H*W, C)
            feat_reshape = feat.reshape(feat.shape[0], feat.shape[1]*feat.shape[2]*feat.shape[3], feat.shape[4])

            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    # torch.randperm produces cudaErrorIllegalAddress for newer versions of PyTorch. https://github.com/taesungp/contrastive-unpaired-translation/issues/83
                    #patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = np.random.permutation(feat_reshape.shape[1])  # (random order of range(H*W))
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device) # first N patches
                patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)  # Channel (1, 128, 256, 256, 256) > (256, 256, 256, 256, 256)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        #print([x.shape for x in return_feats]) # (B * num_patches, 256) * level of features
        return return_feats, return_ids


class GAN(BaseModel):
    """
    There is a lot of patterned noise and other failures when using lightning
    """
    def __init__(self, hparams, train_loader, test_loader, checkpoints, resume_ep):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints, resume_ep)

        self.net_g, self.net_d = self.set_networks()

        netF = PatchSampleF(use_mlp=True, init_type='normal', init_gain=0.02, gpu_ids=[], nc=256)
        self.netF = init_net(netF, init_type='normal', init_gain=0.02, gpu_ids=[])
        # feature_shapes = [32, 64, 128, 256]
        feature_shapes = [64, 128, 128, 128]
        self.netF.create_mlp(feature_shapes)

        self.criterionNCE = []
        for nce_layer in range(4):#self.nce_layers:
            self.criterionNCE.append(PatchNCELoss(opt=hparams))  # .to(self.device)))

            # save model names
        self.netg_names = {'net_g': 'net_g', 'netF': 'netF'}
        self.netd_names = {'net_d': 'net_d'}#, 'net_dY': 'netDY'}

        # interpolation network
        self.upsample = torch.nn.Upsample(size=(hparams.cropsize, hparams.cropsize, hparams.cropz * 8))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        # coefficient for the identify loss
        parser.add_argument("--lambI", type=int, default=0.5)
        # parser.add_argument("--enc_direction", type=str, default='xy')
        return parent_parser

    def test_method(self, net_g, img):
        output = net_g(img[0])
        #output = combine(output, x[0], method='mul')
        return output[0]

    def generation(self, batch):
        self.ori = batch['img'][0]  # (B, C, X, Y, Z)

        ori = self.upsample(self.ori)  # (B, C, X, Y, Z)
        self.oriup = ori  # for CUT loss

        xy = ori.permute(4, 1, 2, 3, 0)[::1, :, :, :, 0]  # (Z, C, X, Y)
        self.oriX = xy

        #self.oriY = ori

        #self.imgXY = self.net_gXY(self.oriX)['out0']
        self.imgYX = self.net_g(ori)['out0']  # (B, C, X, Y, Z)

        #if self.hparams.lamb > 0:
        #    self.imgXYX = self.net_gYX(self.imgXY)['out0']
        #    self.imgYXY = self.net_gXY(self.imgYX)['out0']

        # if self.hparams.lambI > 0:
        #    self.idt_X = self.net_gYX(self.oriX)['out0']
        #    self.idt_Y = self.net_g(self.oriY)['out0']

    def backward_g(self):
        # ADV(XY)+
        #loss_g += self.add_loss_adv(a=self.imgXY, net_d=self.net_dY, truth=True)
        # ADV(YX)+

        loss_g_gan = 0
        loss_g_gan += self.add_loss_adv(a=self.imgYX.permute(2, 1, 4, 3, 0)[:, :, :, :, 0],  # (X, C, Z, Y)
                                       net_d=self.net_d, truth=True)
        loss_g_gan += self.add_loss_adv(a=self.imgYX.permute(2, 1, 3, 4, 0)[:, :, :, :, 0],  # (X, C, Y, Z)
                                       net_d=self.net_d, truth=True)
        loss_g_gan += self.add_loss_adv(a=self.imgYX.permute(3, 1, 4, 2, 0)[:, :, :, :, 0],  # (Y, C, Z, X)
                                       net_d=self.net_d, truth=True)
        loss_g_gan += self.add_loss_adv(a=self.imgYX.permute(3, 1, 2, 4, 0)[:, :, :, :, 0],  # (Y, C, X, Z)
                                       net_d=self.net_d, truth=True)
        loss_g_gan = loss_g_gan / 4

        loss_g_l1 = self.add_loss_l1(a=self.imgYX[:, :, :, :, ::8], b=self.ori[:, :, :, :, :]) * self.hparams.lamb

        # Cyclic(XYX, X)
        #if self.hparams.lamb > 0:
        #    loss_g += self.add_loss_l1(a=self.imgXYX, b=self.oriX) * self.hparams.lamb
            # Cyclic(YXY, Y)
        #    loss_g += self.add_loss_l1(a=self.imgYXY, b=self.oriY) * self.hparams.lamb

        # Identity(idt_X, X)
        #if self.hparams.lambI > 0:
        #    loss_g += self.add_loss_l1(a=self.idt_X, b=self.oriX) * self.hparams.lambI
            # Identity(idt_Y, Y)
        #    loss_g += self.add_loss_l1(a=self.idt_Y, b=self.oriY) * self.hparams.lambI

        # CUT NCE_loss
        feat_q = self.net_g(self.oriup, method='encode')
        feat_k = self.net_g(self.imgYX, method='encode')

        feat_k_pool, sample_ids = self.netF(feat_k, self.hparams.num_patches, None)  # get source patches by random id
        feat_q_pool, _ = self.netF(feat_q, self.hparams.num_patches, sample_ids)  # use the ids for query target

        total_nce_loss = 0.0
        # for f_q, f_k, crit in zip(feat_q_pool[2:], feat_k_pool[2:], self.criterionNCE):
        for f_q, f_k, crit in zip(feat_q_pool, feat_k_pool, self.criterionNCE):
            loss = crit(f_q, f_k) * self.hparams.lambda_NCE
            total_nce_loss += loss.mean()
        loss_nce = total_nce_loss / 4

        loss_g = loss_g_gan + loss_g_l1 + loss_nce

        return {'sum': loss_g, 'loss_g_gan': loss_g_gan, 'loss_g_l1': loss_g_l1, 'loss_nce': loss_nce}

    def backward_d(self):
        loss_d = 0
        # ADV(XY)-
        #loss_d += self.add_loss_adv(a=self.imgXY, net_d=self.net_dY, truth=False)

        loss_d_gan = 0
        loss_d_gan += self.add_loss_adv(a=self.imgYX.permute(2, 1, 4, 3, 0)[:, :, :, :, 0],  # (X, C, Z, Y)
                                       net_d=self.net_d, truth=False)
        loss_d_gan += self.add_loss_adv(a=self.imgYX.permute(2, 1, 3, 4, 0)[:, :, :, :, 0],  # (X, C, Y, Z)
                                       net_d=self.net_d, truth=False)
        loss_d_gan += self.add_loss_adv(a=self.imgYX.permute(3, 1, 4, 2, 0)[:, :, :, :, 0],  # (Y, C, Z, X)
                                       net_d=self.net_d, truth=False)
        loss_d_gan += self.add_loss_adv(a=self.imgYX.permute(3, 1, 2, 4, 0)[:, :, :, :, 0],  # (Y, C, X, Z)
                                       net_d=self.net_d, truth=False)
        loss_d_gan = loss_d_gan / 4
        loss_d += loss_d_gan

        # ADV(Y)+
        #loss_d += self.add_loss_adv(a=self.oriY, net_d=self.net_dY, truth=True)

        # ADV(X)+
        loss_d += self.add_loss_adv(a=self.oriX, net_d=self.net_d, truth=True)

        return {'sum': loss_d, 'loss_d': loss_d}


# USAGE
# CUDA_VISIBLE_DEVICES=2,3 python train.py --jsn cyc_imorphics --prj 0713_cut_max2d4 --models cyc_oai3d_1_cut --cropz 16 --cropsize 128 --netG ed023d --trd 800 --direction SagIwTSE