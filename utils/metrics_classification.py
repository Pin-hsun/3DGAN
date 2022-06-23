import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from torch.nn import functional as F


class CrossEntropyLoss(nn.Module):
    """Dice loss of binary class
    Args:
    Returns:
        Loss tensor
    """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, x, y):
        loss_c = nn.CrossEntropyLoss()(x, y)
        _, classification_pred = torch.max(x, 1)
        #acc = (classification_pred == target).sum().type(torch.FloatTensor)

        return loss_c,


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output, target, size_average=True):
        (output1, output2) = output
        target = (1 - 1 * target)

        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))

        output = losses.mean() if size_average else losses.sum()
        return output,


class ClassificationLoss(nn.Module):
    def __init__(self):
        super(ClassificationLoss, self).__init__()
        self.cross_entropy_loss = CrossEntropyLoss()

    def __len__(self):
        """ length of the components of loss to display """
        return 1

    def forward(self, output, labels):
        x = output
        y = labels
        loss_val, = self.cross_entropy_loss(x, y)
        return loss_val


class ClassifyAndContrastiveLoss(nn.Module):
    """Dice loss of binary class
    Args:
    Returns:
        Loss tensor
    """
    def __init__(self):
        super(ClassifyAndContrastiveLoss, self).__init__()
        self.classification_loss = CrossEntropyLoss()
        self.contrastive_loss = ContrastiveLoss(margin=0.1)

    def __len__(self):
        """ length of the components of loss to display """
        return 2

    def forward(self, output, labels):
        x = output[0]
        x0 = x[:1, :]
        x1 = x[1:, :]

        features = output[1]

        y = labels[0]
        y0 = y[0, :y.shape[1] // 2]
        y1 = y[0, y.shape[1] // 2:]

        loss0, = self.classification_loss(x0, y0)
        loss1, = self.classification_loss(x1, y1)
        loss_classify = (loss0 + loss1) / 2

        loss_contrastive, = self.contrastive_loss((features[0, :, :, :], features[1, :, :, :]), labels[1])

        loss_all = (loss_classify, loss_contrastive)
        loss_val = loss_all[0] + loss_all[1] * 0

        return loss_val, loss_all


class GetAUC(nn.Module):
    def __init__(self):
        super(GetAUC, self).__init__()

    def forward(self, all_label, all_out):
        auc = []
        for n in range(1, all_out.shape[1]):
            fpr, tpr, thresholds = metrics.roc_curve(all_label, all_out[:, n], pos_label=n)
            auc.append(metrics.auc(fpr, tpr))

        return auc

