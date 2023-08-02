import torchmetrics
#from neptune.new.types import File
import torch, torchvision
from utils.data_utils import imagesc
from pytorch_lightning.utilities import rank_zero_only


class NeptuneHelper():
    def __init__(self):

        self.log = dict()

        self.log['to_print'] = []
        self.log['label'] = []
        self.log['out'] = []

    def get(self, key):
        return self.log[key]

    def clear(self, key):
        self.log[key] = []

    def append(self, key, x):
        self.log[key].append(x)

    @rank_zero_only
    def print(self, logger, epoch, destination="train/misclassified_images"):
        if len(self.to_print) > 0:  # if there is something to print
            to_print = [[self.to_print[j][i, 0, :, :].detach().cpu() for i in range(5, 12)] for j in range(len(self.to_print))]
            to_print = [torch.cat(x, 1) for x in to_print]
            to_print = torch.cat(to_print, 0)
            imagesc(to_print, show=False, save='temp/v' + str(epoch).zfill(3) + '.png')

            grid = torchvision.utils.make_grid(to_print)[0,::]
            logger.experiment[destination].log(File.as_image(grid))


class MyAccuracy():
    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # update metric states
        preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        # compute final result
        return self.correct.float() / self.total


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