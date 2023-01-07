import torch
import torch.nn as nn
import numpy as np


class SegmentationCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SegmentationCrossEntropyLoss, self).__init__()

    def __len__(self):
        """ length of the components of loss to display """
        return 1

    def forward(self, probs, true_masks):
        #probs = out.view(out.shape[0] * out.shape[1], out.shape[2], out.shape[3],`
        #                       out.shape[4])  # (B * T, C, H, W)
        if len(probs.shape) == 4:
            probs = probs.permute(0, 2, 3, 1)  # (B, H, W, C)
            probs = probs.reshape(probs.shape[0] * probs.shape[1] * probs.shape[2], probs.shape[3])  # (B * H * W, C)
        elif len(probs.shape) == 5:
            probs = probs.permute(0, 2, 3, 4, 1)  # (B, H, W, Z, C)
            probs = probs.reshape(probs.shape[0] * probs.shape[1] * probs.shape[2] * probs.shape[3], probs.shape[4])  # (B * H * W * Z, C)

        true_masks = true_masks.reshape(-1)  # (B * T * H * W)
        # probs = nn.Softmax()(probs)

        loss_s = nn.CrossEntropyLoss(reduction='none')(probs, true_masks)
        loss_s = torch.mean(loss_s)
        return loss_s, probs


class SegmentationCrossEntropyLossDual(nn.Module):
    def __init__(self):
        super(SegmentationCrossEntropyLossDual, self).__init__()
        self.SegmentationCrossEntropyLoss = SegmentationCrossEntropyLoss()

    def forward(self, output, labels):
        output = torch.cat([output[0][:, 0, ::], output[0][:, 1, ::]], 0)
        true_masks = torch.cat([labels[0][:, 0, ::], labels[0][:, 1, ::]], 0)
        loss_s, probs = self.SegmentationCrossEntropyLoss(output=(output, ), labels=(true_masks, ))
        return loss_s, probs


class SegmentationDiceCoefficient(nn.Module):
    def __init__(self):
        super(SegmentationDiceCoefficient, self).__init__()

    def forward(self, true_masks, out):
        n_classes = out.shape[1]
        probs = out.permute(0, 2, 3, 1)  # (B, H, W, C)
        probs = probs.reshape(probs.shape[0] * probs.shape[1] * probs.shape[2],
                                          probs.shape[3])  # (B * H * W, C)
        _, masks_pred = torch.max(probs, 1)

        dice = np.zeros(n_classes)
        dice_tp = np.zeros(n_classes)
        dice_div = np.zeros(n_classes)
        for c in range(n_classes):
            dice_tp[c] += ((masks_pred == c) & (true_masks.view(-1) == c)).sum().item()
            dice_div[c] += ((masks_pred == c).sum().item() + (true_masks.view(-1) == c).sum().item())
            dice[c] = 2 * dice_tp[c] / dice_div[c]

        return dice[:]


class SegmentationDiceCoefficientDual(nn.Module):
    """
    what is this for ???
    """
    def __init__(self):
        super(SegmentationDiceCoefficientDual, self).__init__()
        self.SegmentationDiceCoefficient = SegmentationDiceCoefficient()

    def forward(self, true_masks, out):
        true_masks = torch.cat([true_masks[:, 0, ::], true_masks[:, 1, ::]], 0)
        out = torch.cat([out[:, 0, ::], out[:, 1, ::]], 0)
        dice = self.SegmentationDiceCoefficient( true_masks, out)
        return dice


if __name__ == '__main__':
    loss = SegmentationCrossEntropyLoss()
    out = loss(torch.rand(7, 2, 50, 50), torch.rand(7, 1, 50, 50).type(torch.LongTensor))