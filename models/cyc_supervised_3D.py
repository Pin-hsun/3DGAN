from models.base import BaseModel, combine
import copy
import torch
from utils.data_utils import three2twoD, save_tif, interpolation, slice_cube
import tifffile as tiff

class GAN(BaseModel):
    """
    There is a lot of patterned noise and other failures when using lightning
    """
    def __init__(self, hparams, train_loader, test_loader, checkpoints, resume_ep):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints, resume_ep)

        self.net_g, self.net_d = self.set_networks()

        self.net_gXY = self.net_g
        self.net_gXZ = copy.deepcopy(self.net_g)
        self.net_gZX = copy.deepcopy(self.net_g)

        self.net_dX = self.net_d
        self.net_dZ = copy.deepcopy(self.net_d)

        # save model names
        self.netg_names = {'net_gXY': 'net_gXY', 'net_gXZ': 'net_gXZ', 'net_gZX': 'net_gZX'}
        self.netd_names = {'net_dX': 'netDX', 'net_dZ': 'netDZ'}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        # coefficient for the identify loss
        parser.add_argument("--lambI", type=int, default=0.5)
        return parent_parser

    def test_method(self, net_g, img):
        output = net_g(img[0])
        #output = combine(output, x[0], method='mul')
        return output[0]

    def generation(self, batch):
        img = batch

        # interpolate and flip xy to yz
        x = img['imgs'][0]
        z = img['imgs'][1]
        x = interpolation(x, 8)
        z = interpolation(z, 8)
        self.x_SAG = x
        x = x.permute(0, 1, 4, 3, 2)
        self.oriX = x #torch.Size([1, 1, 32, 32, 32])
        self.oriZ = torch.flip(z, [4])
        # tiff.imsave('out/b.tif', self.oriY.cpu().numpy())
        # tiff.imsave('out/a.tif', self.oriX.cpu().numpy())

        self.imgXY = self.net_gXY(self.oriX)['out0']
        self.imgXZ = self.net_gXZ(self.x_SAG)['out0']
        self.imgXYZ = self.net_gXZ(self.imgXY)['out0']
        self.imgZX = self.net_gZX(self.oriZ)['out0']

        # get slices of xy: torch.Size([1, 1, 4, 32, 32])
        self.oriX_slice = slice_cube(tensor=self.oriX, N=4, size=self.oriX.shape[-1], start=0)
        self.imgXY_slice = slice_cube(tensor=self.imgXY, N=4, size=self.imgXY.shape[-1], start=0)
        # tiff.imsave('out/a_slice.tif', self.oriX_slice.cpu().numpy())
        # tiff.imsave('out/b_slice.tif', self.imgXY_slice.detach().cpu().numpy())

        # get 2D for discriminater input : torch.Size([32, 1, 32, 32])
        self.x_SAG_2D = three2twoD(self.x_SAG)
        self.oriZ_2D = three2twoD(self.oriZ)
        self.imgXY_2D = three2twoD(self.imgXY)
        self.imgXZ_2D = three2twoD(self.imgXZ)
        self.imgXYZ_2D = three2twoD(self.imgXYZ)
        self.imgZX_2D = three2twoD(self.imgZX)

        if self.hparams.lamb > 0:
            self.imgXZX = self.net_gZX(self.imgXZ)['out0']
            self.imgZXZ = self.net_gXZ(self.imgZX)['out0']

        if self.hparams.lambI > 0:
            self.idt_X = self.net_gZX(self.x_SAG)['out0']
            self.idt_Z = self.net_gXZ(self.oriZ)['out0']

    def backward_g(self):
        loss_g = 0
        # ADV(XY_xy)+
        loss_g += self.add_loss_adv(a=self.imgXY_2D, net_d=self.net_dX, truth=True) / 2
        # ADV(ZX)+
        loss_g += self.add_loss_adv(a=self.imgZX_2D, net_d=self.net_dX, truth=True) / 2
        # ADV(XZ)+
        loss_g += self.add_loss_adv(a=self.imgXZ_2D, net_d=self.net_dZ, truth=True) / 2
        # ADV(XYZ)+
        loss_g += self.add_loss_adv(a=self.imgXYZ_2D, net_d=self.net_dZ, truth=True) / 2

        # zy L1 loss (oriX_xy, imgXY_xy)
        loss_g += self.add_loss_l1(a=self.oriX_slice, b=self.imgXY_slice) * self.hparams.lamb

        # Cyclic(XYX, X)
        if self.hparams.lamb > 0:
            loss_g += self.add_loss_l1(a=self.imgXZX, b=self.x_SAG) * self.hparams.lamb
            # Cyclic(YXY, Y)
            loss_g += self.add_loss_l1(a=self.imgZXZ, b=self.oriZ) * self.hparams.lamb

        # Identity(idt_X, X)
        if self.hparams.lambI > 0:
            loss_g += self.add_loss_l1(a=self.idt_X, b=self.x_SAG) * self.hparams.lambI
            # Identity(idt_Y, Y)
            loss_g += self.add_loss_l1(a=self.idt_Z, b=self.oriZ) * self.hparams.lambI

        return {'sum': loss_g, 'loss_g': loss_g}

    def backward_d(self):
        loss_d = 0
        # ADV(XY_xy)-
        loss_d += self.add_loss_adv(a=self.imgXY_2D, net_d=self.net_dX, truth=False) / 2

        # ADV(ZX)-
        loss_d += self.add_loss_adv(a=self.imgZX_2D, net_d=self.net_dX, truth=False) / 2

        # ADV(X_xy)+
        loss_d += self.add_loss_adv(a=self.x_SAG_2D, net_d=self.net_dX, truth=True)

        # ADV(XZ)-
        loss_d += self.add_loss_adv(a=self.imgXZ_2D, net_d=self.net_dZ, truth=False) / 2

        # ADV(XYZ)-
        loss_d += self.add_loss_adv(a=self.imgXYZ_2D, net_d=self.net_dZ, truth=False) / 2

        # ADV(Y)+
        loss_d += self.add_loss_adv(a=self.oriZ_2D, net_d=self.net_dZ, truth=True)

        return {'sum': loss_d, 'loss_d': loss_d}


# USAGE
# CUDA_VISIBLE_DEVICES=1 python train.py --jsn wnwp3d --prj wnwp3d/cyc/GdenuOmc --mc --models cyc -b 16 --netG descarnoumc  --direction zyori%xyori --dataset Fly0B --input_nc 1

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --jsn 40x2fly10 --prj cyc/test1 --models cyc -b 16 --direction 40xdown2%xyori --dataset 40x --input_nc 1 --trd 500

# CUDA_VISIBLE_DEVICES=0,1 python train.py --jsn 40x2fly10_2 --prj csb/0 --models cyc -b 16 --direction xysb_xyweak --dataset Fly0B --input_nc 1 --trd 0 --netG dsmc --nm 11 --output_nc 1 --lamb 0 --lambI 0