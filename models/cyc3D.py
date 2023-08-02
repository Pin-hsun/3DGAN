from models.base import BaseModel, combine
import copy
import torch
import tifffile as tiff
from utils.data_utils import three2twoD, interpolation

class GAN(BaseModel):
    """
    There is a lot of patterned noise and other failures when using lightning
    """
    def __init__(self, hparams, train_loader, test_loader, checkpoints, resume_ep):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints, resume_ep)

        self.net_g, self.net_d = self.set_networks()

        self.net_gXY = self.net_g
        self.net_gYX = copy.deepcopy(self.net_g)

        self.net_dX = self.net_d
        self.net_dY = copy.deepcopy(self.net_d)

        # save model names
        self.netg_names = {'net_gXY': 'net_gXY', 'net_gYX': 'net_gYX'}
        self.netd_names = {'net_dX': 'netDX', 'net_dY': 'netDY'}

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
        # make img(B,1,256,256)
        x = img['imgs'][0]
        y = img['imgs'][1]
        x = interpolation(x, 8)
        y = interpolation(y, 8)
        # tiff.imsave('out/x.tif', x.cpu().numpy())
        # tiff.imsave('out/y.tif', y.cpu().numpy())
        # print(x.max(), x.min())
        # print(y.max(), y.min())
        # assert 0
        self.oriX = x  # torch.Size([1, 1, 32, 32, 32])
        self.oriY = y

        self.imgXY = self.net_gXY(self.oriX)['out0']
        tiff.imsave('out/xy.tif', self.imgXY.detach().cpu().numpy())
        self.imgYX = self.net_gYX(self.oriY)['out0']

        if self.hparams.lamb > 0:
            self.imgXYX = self.net_gYX(self.imgXY)['out0']
            self.imgYXY = self.net_gXY(self.imgYX)['out0']

        if self.hparams.lambI > 0:
            self.idt_X = self.net_gYX(self.oriX)['out0']
            self.idt_Y = self.net_gXY(self.oriY)['out0']

        self.oriX_2D = three2twoD(self.oriX)
        self.oriY_2D = three2twoD(self.oriY)
        self.imgXY_2D = three2twoD(self.imgXY)
        self.imgYX_2D = three2twoD(self.imgYX)

    def backward_g(self):
        loss_g = 0
        # ADV(XY)+
        loss_g += self.add_loss_adv(a=self.imgXY_2D, net_d=self.net_dY, truth=True)
        # ADV(YX)+
        loss_g += self.add_loss_adv(a=self.imgYX_2D, net_d=self.net_dX, truth=True)

        # Cyclic(XYX, X)
        if self.hparams.lamb > 0:
            loss_g += self.add_loss_l1(a=self.imgXYX, b=self.oriX) * self.hparams.lamb
            # Cyclic(YXY, Y)
            loss_g += self.add_loss_l1(a=self.imgYXY, b=self.oriY) * self.hparams.lamb

        # Identity(idt_X, X)
        if self.hparams.lambI > 0:
            loss_g += self.add_loss_l1(a=self.idt_X, b=self.oriX) * self.hparams.lambI
            # Identity(idt_Y, Y)
            loss_g += self.add_loss_l1(a=self.idt_Y, b=self.oriY) * self.hparams.lambI

        return {'sum': loss_g, 'loss_g': loss_g}

    def backward_d(self):
        loss_d = 0
        loss_x = 0
        loss_y = 0

        # ADV(XY)-
        loss_y += self.add_loss_adv(a=self.imgXY_2D, net_d=self.net_dY, truth=False)

        # ADV(YX)-
        loss_x += self.add_loss_adv(a=self.imgYX_2D, net_d=self.net_dX, truth=False)

        # ADV(Y)+
        loss_y += self.add_loss_adv(a=self.oriY_2D, net_d=self.net_dY, truth=True)

        # ADV(X)+
        loss_x += self.add_loss_adv(a=self.oriX_2D, net_d=self.net_dX, truth=True)

        loss_d += loss_x + loss_y

        return {'sum': loss_d, 'loss_x': loss_x, 'loss_y': loss_y}


# USAGE
# CUDA_VISIBLE_DEVICES=1 python train.py --jsn wnwp3d --prj wnwp3d/cyc/GdenuOmc --mc --models cyc -b 16 --netG descarnoumc  --direction zyori%xyori --dataset Fly0B --input_nc 1

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --jsn 40x2fly10 --prj cyc/test1 --models cyc -b 16 --direction 40xdown2%xyori --dataset 40x --input_nc 1 --trd 500

# CUDA_VISIBLE_DEVICES=0,1 python train.py --jsn 40x2fly10_2 --prj csb/0 --models cyc -b 16 --direction xysb_xyweak --dataset Fly0B --input_nc 1 --trd 0 --netG dsmc --nm 11 --output_nc 1 --lamb 0 --lambI 0