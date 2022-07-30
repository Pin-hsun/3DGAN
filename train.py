from __future__ import print_function
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
import os, shutil
from dotenv import load_dotenv

from utils.make_config import load_json, save_json
import json
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import pandas as pd


def prepare_log(args):
    """
    finalize arguments, creat a folder for logging, save argument in json
    """
    args.not_tracking_hparams = ['mode', 'port', 'epoch_load', 'legacy', 'threads', 'test_batch_size']
    os.makedirs(os.environ.get('LOGS') + args.dataset + '/', exist_ok=True)
    os.makedirs(os.environ.get('LOGS') + args.dataset + '/' + args.prj + '/', exist_ok=True)
    save_json(args, os.environ.get('LOGS') + args.dataset + '/' + args.prj + '/' + '0.json')
    shutil.copy('engine/' + args.engine + '.py', os.environ.get('LOGS') + args.dataset + '/' + args.prj + '/' + args.engine + '.py')
    return args


# Arguments
parser = argparse.ArgumentParser()#add_help=False)
# Env
parser.add_argument('--jsn', type=str, default='default', help='name of ini file')
parser.add_argument('--env', type=str, default=None, help='environment_to_use')
# Project name
parser.add_argument('--prj', type=str, help='name of the project')
parser.add_argument('--engine', dest='engine', type=str, help='use which engine')
# Data
parser.add_argument('--dataset', type=str)
parser.add_argument('--bysubject', action='store_true', dest='bysubject', default=False)
parser.add_argument('--index', action='store_true', dest='index', default=False, help='use train_index')
parser.add_argument('--direction', type=str, help='a2b or b2a')
parser.add_argument('--flip', action='store_true', dest='flip', help='image flip left right')
parser.add_argument('--resize', type=int, help='size for resizing before cropping, 0 for no resizing')
parser.add_argument('--cropsize', type=int, help='size for cropping, 0 for no crop')
parser.add_argument('--n01', dest='n01', action='store_true')
parser.add_argument('--n11', dest='n01', action='store_false')
parser.set_defaults(n01=False)
parser.add_argument('--gray', action='store_true', dest='gray', default=False, help='dont copy img to 3 channel')
parser.add_argument('--spd', action='store_true', dest='spd', default=False)
# Model
parser.add_argument('--gan_mode', type=str, help='gan mode')
parser.add_argument('--netG', type=str, help='netG model')
parser.add_argument('--norm', type=str, help='normalization in generator')
parser.add_argument('--mc', action='store_true', dest='mc', default=False, help='monte carlo dropout for pix2pix generator')
parser.add_argument('--netD', type=str, help='netD model')
parser.add_argument('--input_nc', type=int, help='input image channels')
parser.add_argument('--output_nc', type=int, help='output image channels')
parser.add_argument('--ngf', type=int, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, help='discriminator filters in first conv layer')
parser.add_argument("--n_attrs", type=int, default=1)
parser.add_argument('--final', type=str, dest='final', help='activation of final layer')
parser.add_argument('--cmb', dest='cmb', help='method to combine the outputs to the original')
parser.add_argument('--trd', type=float, dest='trd', help='threshold of images')
# Training
parser.add_argument('-b', dest='batch_size', type=int, help='training batch size')
parser.add_argument('--test_batch_size', type=int, help='testing batch size')
parser.add_argument('--n_epochs', type=int, help='# of iter at starting learning rate')
parser.add_argument('--lr', type=float, help='initial learning rate f -or adam')
parser.add_argument('--beta1', type=float, help='beta1 for adam. default=0.5')
parser.add_argument('--threads', type=int,  help='number of threads for data loader to use')
parser.add_argument('--epoch_count', type=int, help='the starting epoch count')
parser.add_argument('--epoch_load', type=int, help='to load checkpoint form the epoch count')
parser.add_argument('--n_epochs_decay', type=int, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr_policy', type=str, help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, help='multiply by a gamma every lr_decay_iters iterations')
# Loss
parser.add_argument('--lamb', type=int, help='weight on L1 term in objective')
# Misc
parser.add_argument('--seed', type=int, help='random seed to use. Default=123')
parser.add_argument('--legacy', action='store_true', dest='legacy', default=False, help='legacy pytorch')
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--port', type=str, default='dummy')

# Model-specific Arguments
engine = parser.parse_known_args()[0].engine
GAN = getattr(__import__('engine.' + engine), engine).GAN
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
print(os.environ.get('LOGS'))

# Finalize Arguments and create files for logging
args = prepare_log(args)

#  Define Dataset Class
from dataloader.data_multi import MultiData as Dataset

# Load Dataset and DataLoader
# THIS IS TRASH. I WAS TRYING TO SPLIT OAI DATA ON THE FLY
if args.index:  # if use customized index
    folder = '/full/'
    # train_index = range(*args.train_index)
    # new index
    #df = pd.read_csv(os.getenv("HOME") + '/Dropbox/TheSource/scripts/OAI_pipelines/meta/subjects_unipain_womac3.csv')
    df = pd.read_csv('env/subjects_unipain_womac3.csv')
    train_index = [x for x in range(df.shape[0]) if not df['has_moaks'][x]]
    test_index = [x for x in range(df.shape[0]) if df['has_moaks'][x]]
    # train_index = range(213, 710)
    # eval_index = range(0, 213)
else:
    folder = '/train/'
    train_index = None

train_set = Dataset(root=os.environ.get('DATASET') + args.dataset + folder,
                    path=args.direction,
                    opt=args, mode='train', index=train_index, filenames=True)
train_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batch_size, shuffle=True)

if args.index:
    test_set = Dataset(root=os.environ.get('DATASET') + args.dataset + folder,
                        path=args.direction,
                        opt=args, mode='train', index=test_index, filenames=True)
    test_loader = DataLoader(dataset=test_set, num_workers=args.threads, batch_size=args.batch_size, shuffle=False)
else:
    test_loader = None

#val_set = Dataset(root=os.environ.get('DATASET') + args.dataset + '/test/',
#                  path=args.direction,
#                  opt=args, mode='test')
#val_loader = DataLoader(dataset=val_set, num_workers=args.threads, batch_size=args.batch_size, shuffle=False)

# Logger
if 1:
    logger = pl_loggers.TensorBoardLogger(os.environ.get('LOGS') + args.dataset + '/', name=args.prj)
else:
    from pytorch_lightning.loggers.neptune import NeptuneLogger
    logger = NeptuneLogger(
        api_key="ANONYMOUS",
        project="shared/pytorch-lightning-integration")

# Trainer
checkpoints = os.path.join(os.environ.get('LOGS'), args.dataset, args.prj, 'checkpoints')
os.makedirs(checkpoints, exist_ok=True)
net = GAN(hparams=args, train_loader=train_loader, test_loader=test_loader, checkpoints=checkpoints)
trainer = pl.Trainer(gpus=-1, strategy='ddp',
                     max_epochs=args.n_epochs, progress_bar_refresh_rate=20, logger=logger,
                     enable_checkpointing=False)
print(args)
trainer.fit(net, train_loader, test_loader)  # test loader not used during training


# Example Usage
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset TSE_DESS -b 16 --prj VryCycle --direction a_b --resize 286 --engine cyclegan --lamb 10 --unpaired
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset pain -b 16 --prj VryNS4B --direction aregis1_b --resize 286 --engine NS4 --netG attgan
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset FlyZ -b 16 --prj WpWn286B --direction xyweak%zyweak --resize 286 --engine cyclegan --lamb 10
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset FlyZ -b 16 --prj WpOp256Mask --direction xyweak_xyorisb --resize 256 --engine pix2pixNS

# CUDA_VISIBLE_DEVICES=0 python train.py --jsn womac3 --prj mcfix/descar2/Gunet128 --engine descar2 --netG unet_128 --mc --direction areg_b --index
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset womac3 -b 1 --prj bysubjectright/descar2/GDdescars --direction areg_b --cropsize 256 --engine descar2 --netG descars --netD descar --n01 --final sigmoid --cmb mul --bysubject