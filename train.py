from __future__ import print_function
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
import os, shutil, time
from tqdm import tqdm
from dotenv import load_dotenv

from utils.make_config import load_json, save_json
import json
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from dataloader.data_multi import MultiData as Dataset


def prepare_log(args):
    """
    finalize arguments, creat a folder for logging, save argument in json
    """
    args.not_tracking_hparams = ['mode', 'port', 'epoch_load', 'legacy', 'threads', 'test_batch_size']
    os.makedirs(os.environ.get('LOGS') + args.dataset + '/', exist_ok=True)
    os.makedirs(os.environ.get('LOGS') + args.dataset + '/' + args.prj + '/', exist_ok=True)
    save_json(args, os.environ.get('LOGS') + args.dataset + '/' + args.prj + '/' + '0.json')
    shutil.copy('models/' + args.models + '.py', os.environ.get('LOGS') + args.dataset + '/' + args.prj + '/' + args.models + '.py')
    return args


# Arguments
parser = argparse.ArgumentParser()#add_help=False)
# Env
parser.add_argument('--jsn', type=str, default='default', help='name of ini file')
parser.add_argument('--env', type=str, default=None, help='environment_to_use')
# Project name
parser.add_argument('--prj', type=str, help='name of the project')
parser.add_argument('--models', dest='models', type=str, help='use which models')
# Data
parser.add_argument('--dataset', type=str)
parser.add_argument('--preload', action='store_true')
parser.add_argument('--split', type=str, help='split of data')
parser.add_argument('--load3d', action='store_true', dest='load3d', default=False, help='do 3D')

parser.add_argument('--direction', type=str, help='paired: a_b, unpaired a%b ex:(a_b%c_d)')
parser.add_argument('--flip', action='store_true', dest='flip', help='image flip left right')
parser.add_argument('--resize', type=int, help='size for resizing before cropping, 0 for no resizing')
parser.add_argument('--cropsize', type=int, help='size for cropping, 0 for no crop')
parser.add_argument('--n01', dest='n01', action='store_true', help='normalize the image to 0~1')
parser.add_argument('--n11', dest='n01', action='store_false', help='otherwise normalize the image to -1~1')
parser.set_defaults(n01=False)
parser.add_argument('--gray', action='store_true', dest='gray', default=False, help='dont copy img to 3 channel')
parser.add_argument('--spd', action='store_true', dest='spd', default=False, help='USE SPADE?')
# Model
parser.add_argument('--gan_mode', type=str, help='gan mode')
parser.add_argument('--netG', type=str, help='netG model')
parser.add_argument('--norm', type=str, help='normalization in generator')
parser.add_argument('--mc', action='store_true', dest='mc', default=False, help='monte carlo dropout for some of the generators')
parser.add_argument('--netD', type=str, help='netD model')
parser.add_argument('--input_nc', type=int, help='input image channels')
parser.add_argument('--output_nc', type=int, help='output image channels')
parser.add_argument('--ngf', type=int, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, help='discriminator filters in first conv layer')
# parser.add_argument("--n_attrs", type=int, default=1)   # NOT IN USE?
parser.add_argument('--final', type=str, dest='final', help='activation of final layer')
parser.add_argument('--cmb', dest='cmb', help='method to combine the outputs to the original')
parser.add_argument('--trd', type=float, dest='trd', help='threshold of images')
# Training
parser.add_argument('-b', dest='batch_size', type=int, help='training batch size')
# parser.add_argument('--test_batch_size', type=int, help='testing batch size') # NOT IN USE?
parser.add_argument('--n_epochs', type=int, help='# of iter at starting learning rate')
parser.add_argument('--lr', type=float, help='initial learning rate f -or adam')
parser.add_argument('--beta1', type=float, help='beta1 for adam. default=0.5')
parser.add_argument('--threads', type=int,  help='number of threads for data loader to use')
parser.add_argument('--epoch_count', type=int, help='the starting epoch count')
parser.add_argument('--epoch_load', type=int, help='to load checkpoint form the epoch count')
parser.add_argument('--n_epochs_decay', type=int, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr_policy', type=str, help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--save_d', action='store_true', dest='save_d', default=False, help='save checkpoints of discriminators')
# Loss
parser.add_argument('--lamb', type=int, help='weight on L1 term in objective')
# Misc
parser.add_argument('--seed', type=int, help='random seed to use. Default=123')
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--port', type=str, default='dummy')

# Model-specific Arguments
models = parser.parse_known_args()[0].models
GAN = getattr(__import__('models.' + models), models).GAN
parser = GAN.add_model_specific_args(parser)

# Read json file and update it
with open('env/jsn/' + parser.parse_args().jsn + '.json', 'rt') as f:
    t_args = argparse.Namespace()
    t_args.__dict__.update(json.load(f)['train'])
    args = parser.parse_args(namespace=t_args)

# environment file
if args.env is not None:
    load_dotenv('env/.' + args.env)
else:
    load_dotenv('env/.t09')

# Finalize Arguments and create files for logging
args = prepare_log(args)

print(args)

# Load Dataset and DataLoader
from env.custom_data_utils import customize_data_split
folder, train_index, test_index = customize_data_split(dataset=args.dataset, split=args.split)

train_set = Dataset(root=os.environ.get('DATASET') + args.dataset + folder,
                    path=args.direction,
                    opt=args, mode='train', index=train_index, filenames=True)
train_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batch_size, shuffle=True, pin_memory=True)

if test_index is not None:
    test_set = Dataset(root=os.environ.get('DATASET') + args.dataset + folder,
                       path=args.direction,
                       opt=args, mode='test', index=test_index, filenames=True)
    test_loader = DataLoader(dataset=test_set, num_workers=args.threads, batch_size=args.batch_size, shuffle=False, pin_memory=True)
else:
    test_loader = None


# preload
if args.preload:
    tini = time.time()
    print('Preloading...')
    for i, x in enumerate(tqdm(train_loader)):
        pass
    if test_loader is not None:
        for i, x in enumerate(tqdm(test_loader)):
            pass
    print('Preloading time: ' + str(time.time() - tini))


# Logger
if 1:
    from pytorch_lightning.loggers.neptune import NeptuneLogger
    logger = NeptuneLogger(
        api_key="ANONYMOUS",
        project="test")
else:
    logger = pl_loggers.TensorBoardLogger(os.environ.get('LOGS') + args.dataset + '/', name=args.prj)


# Trainer
checkpoints = os.path.join(os.environ.get('LOGS'), args.dataset, args.prj, 'checkpoints')
os.makedirs(checkpoints, exist_ok=True)
net = GAN(hparams=args, train_loader=train_loader, test_loader=test_loader, checkpoints=checkpoints)
trainer = pl.Trainer(gpus=-1, strategy='ddp',
                     max_epochs=args.n_epochs + 1, progress_bar_refresh_rate=20, logger=logger,
                     enable_checkpointing=False)
print(args)
trainer.fit(net, train_loader, test_loader)  # test loader not used during training


# Examples of  Usage
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --jsn womac3 --prj 3D/descar3/GdsmcDbpatch16  --models descar3 --netG dsmc --netD bpatch_16 --direction ap_bp --final none -b 1 --split moaks --final none --n_epochs 400