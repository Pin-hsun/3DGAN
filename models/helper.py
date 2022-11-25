import torchmetrics
from neptune.new.types import File
import torch, torchvision
from utils.data_utils import imagesc
from pytorch_lightning.utilities import rank_zero_only


class NeptuneHelper():
    def __init__(self):
        self.to_print = []

    def clear(self):
        self.to_print = []

    def append(self, x):
        self.to_print.append(x)

    @rank_zero_only
    def print(self, logger, epoch, destination="train/misclassified_images"):
        if len(self.to_print) > 0:  # if there is something to print
            to_print = [[self.to_print[j][i, 0, :, :].detach().cpu() for i in range(5, 12)] for j in range(len(self.to_print))]
            to_print = [torch.cat(x, 1) for x in to_print]
            to_print = torch.cat(to_print, 0)
            imagesc(to_print, show=False, save='temp/v' + str(epoch).zfill(3) + '.png')

            grid = torchvision.utils.make_grid(to_print)[0,::]
            logger.experiment[destination].log(File.as_image(grid))


def reshape_3d(img3d):
    if len(img3d[0].shape) == 5:
        for i in range(len(img3d)):
            (B, C, H, W, Z) = img3d[i].shape
            img3d[i] = img3d[i].permute(0, 4, 1, 2, 3)
            img3d[i] = img3d[i].reshape(B * Z, C, H, W)
    return img3d


def tile_like(x, target):  # tile_size = 256 or 4
    x = x.view(x.size(0), x.size(1), 1, 1)
    x = x.repeat(1, 1, target.size(2), target.size(3))
    return x