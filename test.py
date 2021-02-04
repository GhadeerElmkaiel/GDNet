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
    args.batch_size = 12
    args.val_every = 2
    args.developer_mode = True
    args.load_model = False
    args.fast_dev_run = False
    args.crf = True
    args.wandb = True
    args.gdd_training_root = args.root_path+"/GDNet/mini"
    args.gdd_eval_root = args.root_path+"/GDNet/mini_eval"
    args.epochs = 20


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
wb_logger = pl_loggers.WandbLogger(project = args.log_name)

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
    model_loggers = [tb_logger, wb_logger]
else:
    model_loggers = [tb_logger]


def main():

    net = LitGDNet(args)
    # net = net.load_from_checkpoint(args.root_path + args.ckpt_path + "/MirrorNet-epoch=16-val_loss=3.99.ckpt")
    if args.load_model:
        net = LitGDNet.load_from_checkpoint(args.ckpt_path+args.ckpt_name, args=args)
        print('Loading {} checkpoint.'.format(args.ckpt_path + args.ckpt_name))
        trainer = Trainer(gpus=args.device_ids,
                        fast_dev_run = args.fast_dev_run,
                        accelerator = 'dp',
                        max_epochs = args.epochs,
                        callbacks = [checkpoint_callback],
                        check_val_every_n_epoch = args.val_every,
                        logger = model_loggers,
                        resume_from_checkpoint = args.ckpt_path+args.ckpt_name)
        print("Checkpoint loaded successfully!")
    else:
        trainer = Trainer(gpus=args.device_ids,
                        fast_dev_run = args.fast_dev_run,
                        accelerator = 'dp',
                        max_epochs = args.epochs,
                        callbacks = [checkpoint_callback],
                        check_val_every_n_epoch = args.val_every,
                        logger = model_loggers)
                        # resume_from_checkpoint = args.root_path + args.ckpt_path + "/MirrorNet-epoch=16-val_loss=3.99.ckpt")
    
    if args.wandb:
        wandb.watch(net)

    if args.train:
        print("Training")

        trainer.fit(net)
        final_epoch_model_path = args.ckpt_path + "final_epoch.ckpt"
        trainer.save_checkpoint(final_epoch_model_path)

        print("Done")

    else:
        print("Testing")
        # trainer.test(model = net,
        #             ckpt_path = args.ckpt_path+args.ckpt_name)
        trainer.test(model = net)


if __name__ == "__main__":
    main()
