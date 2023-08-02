from __future__ import print_function
import argparse
import glob

import torch
from torch.utils.data import DataLoader
import os, shutil, time, sys
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
import tifffile as tiff

from utils.make_config import load_json, save_json
import json
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from dataloader.data_multi import MultiData as Dataset

def prepare_log(args):
    """
    finalize arguments, creat a folder for logging, save argument in json
    """
    args.not_tracking_hparams = []#'mode', 'port', 'epoch_load', 'legacy', 'threads', 'test_batch_size']
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
parser.add_argument('-m', dest='message', type=str, help='message')
parser.add_argument('--models', dest='models', type=str, help='use which models')
# Data
parser.add_argument('--dataset', type=str)
parser.add_argument('--preload', action='store_true')
parser.add_argument('--split', type=str, help='split of data')
parser.add_argument('--flip', action='store_true', dest='flip', help='image flip left right')
parser.add_argument('--resize', type=int, help='size for resizing before cropping, 0 for no resizing')
parser.add_argument('--cropsize', type=int, help='size for cropping, 0 for no crop')
parser.add_argument('--cropz', type=int, default=0)
parser.add_argument('--direction', type=str, help='paired: a_b, unpaired a%b ex:(a_b%c_d)')
parser.add_argument('--nm', type=str, help='normalization method for dataset')
parser.add_argument('--gray', action='store_true', dest='gray', default=False, help='dont copy img to 3 channel')
parser.add_argument('--spd', action='store_true', dest='spd', default=False, help='USE SPADE?')
parser.add_argument('--permute', action='store_true', dest='permute', default=False, help='do interpolation and permutation')
parser.add_argument('--load_3D', action='store_true', dest='load_3D', default=False, help='load 3D cube')
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
parser.add_argument('--trd', type=int, dest='trd', nargs='+',  help='threshold of images')
# Training
parser.add_argument('-b', dest='batch_size', type=int, help='training batch size')
parser.add_argument('--resume', action='store_true', default=False, help='resume last time training')
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
    t_args.__dict__.update(json.load(f))
    args = parser.parse_args(namespace=t_args)

# environment file
if args.env is not None:
    load_dotenv('env/.' + args.env)
else:
    load_dotenv('env/.t09')

# Finalize Arguments and create files for logging
args.bash = ' '.join(sys.argv)
args = prepare_log(args)

print(args)

# Load Dataset and DataLoader
all_img_files = len(glob.glob(os.environ.get('DATASET') + args.dataset + args.direction.split('_')[0] + '/*'))
train_index, test_index = train_test_split(range(all_img_files), test_size=0.3, random_state=42)

# train_index, test_index = train_index[:7], test_index[:3]
print('train set:', len(train_index))

train_set = Dataset(root=os.environ.get('DATASET') + args.dataset, path=args.direction,
                    opt=args, mode='train', index=train_index, filenames=False)

train_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batch_size, shuffle=True, pin_memory=True)

# if test_index is not None:
# test_set = Dataset(root=os.environ.get('DATASET') + args.dataset,
#                    path=args.direction, opt=args, mode='test', index=test_index, filenames=False)
#
# test_loader = DataLoader(dataset=test_set, num_workers=args.threads, batch_size=args.batch_size, shuffle=False, pin_memory=True)
# else:
test_loader = None


# preload
if args.preload:
    tini = time.time()
    print('Preloading...')
    for i, x in enumerate((train_loader)):
        pass
    if test_loader is not None:
        for i, x in enumerate(tqdm(test_loader)):
            pass
    print('Preloading time: ' + str(time.time() - tini))


# Logger
logger = pl_loggers.TensorBoardLogger(os.environ.get('LOGS') + args.dataset + '/', name=args.prj)

# Trainer
checkpoints = os.path.join(os.environ.get('LOGS'), args.dataset, args.prj, 'checkpoints')
os.makedirs(checkpoints, exist_ok=True)

# continue training
if args.resume:
    if os.path.isfile(checkpoints+'/final.ckpt'):
        trainer = pl.Trainer(gpus=-1, strategy='ddp', resume_from_checkpoint=checkpoints+'/final.ckpt')
        resume_checkpoint = torch.load(checkpoints+'/final.ckpt')
        net = GAN(hparams=args, train_loader=train_loader, test_loader=test_loader, checkpoints=checkpoints, resume_ep=resume_checkpoint['epoch'])
        print('load checkpoints')
    else:
        print('no checkpoint for resume')
else:
    trainer = pl.Trainer(gpus=-1, strategy='ddp',
                         max_epochs=args.n_epochs + 1,# progress_bar_refresh_rate=20,
                         logger=logger,
                         enable_checkpointing=False, log_every_n_steps=200,
                         check_val_every_n_epoch=1)
    net = GAN(hparams=args, train_loader=train_loader, test_loader=test_loader, checkpoints=checkpoints, resume_ep=0)

print(args)

trainer.fit(net, train_loader, test_loader)  # test loader not used during training
trainer.save_checkpoint(checkpoints+'/final.ckpt')

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --jsn womac4min0 -b 1 --prj 0518 --models cyc --netG descargan --lamb 0 --lambI 0

# Examples of  Usage XXXYYYZZ
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --jsn womac3 --prj 3D/descar3/GdsmcDbpatch16  --models descar3 --netG dsmc --netD bpatch_16 --direction ap_bp --final none -b 1 --split moaks --final none --n_epochs 400