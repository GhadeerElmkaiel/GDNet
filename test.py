import numpy as np
import os
import time
from tqdm import tqdm

from PIL import Image

import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers


from misc import check_mkdir, crf_refine
from gdnet import GDNet, LitGDNet
from dataset import ImageFolder
from arguments import get_args
from utils.loss import lovasz_hinge

import wandb


#######################################
# Initializing the arguments for testing
def init_args(args):
    args.train = True
    args.batch_size = 1
    args.developer_mode = True
    args.load_model = True
    args.fast_dev_run = False
    args.crf = True
    args.device_ids = [0, 1]
    args.val_every = 5
    args.wandb = False

args = get_args()



#######################################
# Checkpoint call back for saving the best models
# 
checkpoint_callback = ModelCheckpoint(
    monitor= args.monitor,
    dirpath= args.ckpt_path,
    filename= 'GDNet-{epoch:03d}-{val_loss:.2f}',
    save_top_k= args.save_top,
    mode='min',
)

tb_logger = pl_loggers.TensorBoardLogger(save_dir = args.log_path,
                                        name = args.log_name)

# change the argumnets for testing
init_args(args)

###############################
# Defining the transoformations
img_transform = transforms.Compose([
    transforms.Resize((args.scale, args.scale)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
mask_transform = transforms.Compose([
    transforms.Resize((args.scale, args.scale)),
    transforms.ToTensor()
])

to_pil = transforms.ToPILImage()

###############################
# Initializing random seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    pl.utilities.seed.seed_everything(seed = args.seed)


# device_ids = [0]
device_ids = args.device_ids
if args.cuda:
    torch.cuda.set_device(device_ids[0])

# Init Weights and Biases
if args.wandb:
    wandb.init(project='GDNet')
    config = wandb.config

#TODO
# Test the following code
def main():
    net = GDNet().cuda(device_ids[0])
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=args.betas)

    if args.load_model:
        # print(os.path.join(args.root_path + args.ckpt_path, args.exp_name, args.snapshot + '.pth'))
        print('Load snapshot {} for testing'.format(args.snapshot))
        net.load_state_dict(torch.load(os.path.join(args.ckpt_path, args.exp_name, args.snapshot + '.pth')))
        # net.load_state_dict(torch.load(os.path.join(args.ckpt_path, args.exp_name, args.snapshot + '.pth')))
        print('Load {} succeed!'.format(os.path.join(args.ckpt_path, args.exp_name, args.snapshot + '.pth')))

    if not args.train:
        net.eval()
        data_path = args.msd_testing_root
    else:
        data_path = args.msd_training_root
        eval_path = args.msd_eval_root
        net.train()

    if args.developer_mode:
        # To include the real images and masks
        dataset = ImageFolder(data_path, img_transform= img_transform, target_transform=mask_transform, add_real_imgs=True)
        eval_dataset = ImageFolder(eval_path, img_transform= img_transform, target_transform=mask_transform, add_real_imgs=True)
    else:
        dataset = ImageFolder(data_path, img_transform= img_transform, target_transform=mask_transform)
        eval_dataset = ImageFolder(eval_path, img_transform= img_transform, target_transform=mask_transform)

    loader = DataLoader(dataset, batch_size= args.batch_size, shuffle=args.shuffle_dataset)
    eval_loader = DataLoader(eval_dataset, batch_size = 1, shuffle=False)


if __name__ == "__main__":
    main()