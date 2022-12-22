from __future__ import print_function
import argparse, json
import os, glob, sys
from utils.data_utils import imagesc
import torch
from torchvision.utils import make_grid
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from utils.data_utils import norm_01
from skimage import io
import torchvision.transforms as transforms
import torch.nn as nn
import tifffile as tiff
from dataloader.data_multi import MultiData as Dataset
import os, glob
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--jsn', default='seg', type=str)
parser.add_argument('--env', type=str)
parser.add_argument('--mode', type=str)
parser.add_argument('--port', type=str)

# Read json file and update it
with open('env/jsn/' + parser.parse_args().jsn + '.json', 'rt') as f:
    t_args = argparse.Namespace()
    t_args.__dict__.update(json.load(f)['test'])
    args = parser.parse_args(namespace=t_args)

# environment file
if args.env is not None:
    load_dotenv('env/.' + args.env)
else:
    load_dotenv('env/.t09')

# Data
from env.custom_data_utils import customize_data_split
folder, train_index, test_index = customize_data_split(dataset=args.dataset, split=args.split)
test_set = Dataset(root=os.environ.get('DATASET') + args.dataset + folder,
                    path=args.direction,
                    opt=args, mode='train', index=test_index, filenames=True)

# Model
epoch = 180
path_g = os.path.join(os.environ.get('LOGS'), args.dataset, args.prj,
                      'checkpoints', args.netg + '_model_epoch_' + str(epoch) + '.pth')

net = torch.load(path_g)
#net = net.eval()

# Forward

tp_all = np.zeros(7)
uni_all = np.zeros(7)

for idx in range(len(test_set)):
    a = test_set.__getitem__(idx)
    ori = a['img'][1]
    mask = a['img'][0]

    ori = ori / ori.max()

    out, = net(ori.unsqueeze(0).type(torch.FloatTensor).cuda())
    seg = torch.argmax(out, 1).detach().cpu()

    for i in range(7):
        tp = ((mask == i) & (seg == i)).sum()
        uni = (mask == i).sum() + (seg == i).sum()
        tp_all[i] += tp
        uni_all[i] += uni

print(np.divide(2*tp_all, uni_all))