""" Dcgan model """
import torch
import torch.nn as nn

class Decriminator(nn.Module):
    """ dcgan Decriminator """
    def __init__(self, isize, nz, nc, ndf, n_extra_layers=0):
        super(Decriminator, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        main.append(nn.Conv2d(nc, ndf, 4, 2, 'pad', 1, bias=False))
        main.append(nn.LeakyReLU(0.2))

        csize, cndf = isize / 2, ndf

        # Extra layers
        for _ in range(n_extra_layers):
            main.append(nn.Conv2d(cndf, cndf, 3, 1, 'pad', 1, bias=False))
            main.append(nn.BatchNorm2d(cndf))
            main.append(nn.LeakyReLU(0.2))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2

            main.append(nn.Conv2d(in_feat, out_feat, 4, 2, 'pad', 1, bias=False))
            main.append(nn.BatchNorm2d(out_feat))
            main.append(nn.LeakyReLU(0.2))

            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.append(nn.Conv2d(cndf, 1, 4, 1, 'pad', 0, bias=False))
        self.main = main

    def foward(self, input1):
        """construct"""
        output = self.main(input1)
        output = output.mean(0)
        return output.view(1)


class Generator(nn.Module):
    """ dcgan generator """
    def __init__(self, isize, nz, nc, ngf, n_extra_layers=0):
        super(Generator, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.append(nn.ConvTranspose2d(nz, cngf, 4, 1, 'pad', 0, bias=False))
        main.append(nn.BatchNorm2d(cngf))
        main.append(nn.ReLU())

        csize = 4
        while csize < isize // 2:
            main.append(nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 'pad', 1, bias=False))
            main.append(nn.BatchNorm2d(cngf // 2))
            main.append(nn.ReLU())

            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for _ in range(n_extra_layers):
            main.append(nn.Conv2d(cngf, cngf, 3, 1, 'pad', 1, bias=False))
            main.append(nn.BatchNorm2d(cngf))
            main.append(nn.ReLU())

        main.append(nn.ConvTranspose2d(cngf, nc, 4, 2, 'pad', 1, bias=False))
        main.append(nn.Tanh())
        self.main = main

    def foward(self, input1):
        """construct"""
        output = self.main(input1)
        return output

if __name__ == '__main__':
    from torchsummary import summary
    model = Generator(384, 100, 1, 64, 1)
    summary(model, (100, 1, 1))
    model = Decriminator(384, 100, 1, 64, 1)
    summary(model, (1, 384, 384))