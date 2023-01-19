import torch
import torch.nn as nn
import pandas as pd


class OaiSubjects():
    def __init__(self, dataset):
        self.df = pd.read_csv('env/csv/' + dataset + '.csv')
        self.df['ID'] = [str(x) for x in self.df['ID']]

    def labels_unilateral(self, filenames):
        df = self.df
        #ids = [x.split('/')[-1].split('_')[0] for x in filenames[0]]
        ids = [x.split('/')[-1].replace('_' + x.split('/')[-1].split('_')[-1], '') for x in filenames[0]]

        paindiff = [(df.loc[df['ID'] == i, ['V$$WOMKPR']].values[0][0] \
                     - df.loc[df['ID'] == i, ['V$$WOMKPL']].values[0][0]) for i in ids] # Right - Left
        paindiff = torch.FloatTensor([x / 10 for x in paindiff])

        labels = {'paindiff': paindiff,  'painbinary': (torch.sign(paindiff) + 1) / 2}
        return labels


oai = OaiSubjects('womac3')


def classify_easy_3d(classify_logits, truth_classify, classifier, criterion):
    classify_logits = nn.AdaptiveAvgPool2d(1)(classify_logits)  # (B, Z, 256, 1, 1)
    classify_logits, _ = torch.max(classify_logits, 1)
    classify_logits = classifier(classify_logits)
    classify = criterion(classify_logits, truth_classify.view(-1, 1, 1, 1).type_as(classify_logits))
    return classify, classify_logits


def swap_by_labels(sign_swap, classify_logits):
    # print(classify_a.shape)  #(46, 256, 16, 16)
    # print(truth_classify.shape)  # (2) (0~1)
    # STUPID MESS UP PART WHERE I MULTIPLE FEATURES DIFF WITH LABEL SIGN
    B = sign_swap.shape[0]
    Z = classify_logits.shape[0] // B

    classify_logits = classify_logits.view((B, Z) + classify_logits.shape[1:4])

    sign_swap = sign_swap.view(-1, 1, 1, 1, 1) \
        .repeat((1,) + classify_logits.shape[1:5]).type_as(classify_logits)
    classify_logits = torch.mul(classify_logits, sign_swap)
    return classify_logits