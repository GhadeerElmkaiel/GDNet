import numpy as np
import os

import torch
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import random as r


from gdnet import LitGDNet
from arguments import get_args

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
    # args.gdd_training_root = args.root_path+"/GDNet/mini"
    # args.gdd_eval_root = args.root_path+"/GDNet/mini_eval"
    # args.epochs = 100


args = get_args()

# change the argumnets for testing
init_args(args)

#######################################
# Checkpoint call back for saving the best models
# 
run_name = "L-"
for l in args.loss_funcs:
    run_name+= l+'-'
# Adding the run number to the name of the run
f = open("run_num.log", "r+")
run_num = int(f.read())
f.close()

f = open("run_num.log", "w")
f.write(str(run_num+1))
f.close()
run_name+= str(run_num)
args.log_name = run_name

# Creating folder for saving the images:
if args.developer_mode:
    folder_path = os.path.join(args.gdd_results_root, "Training",run_name)
    os.makedirs(folder_path)
    
ckpt_path = os.path.join(args.ckpt_path,run_name)
os.makedirs(ckpt_path)
args.ckpt_path = ckpt_path

checkpoint_callback = ModelCheckpoint(
    monitor= args.monitor,
    dirpath= ckpt_path,
    filename= 'GDNet-' + args.log_name + '-{epoch:03d}-{val_loss:.2f}',
    save_top_k= args.save_top,
    mode='min',
)

tb_logger = pl_loggers.TensorBoardLogger(save_dir = args.log_path,
                                        name = args.log_name)
wb_logger = pl_loggers.WandbLogger(project = args.log_name)


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
    wandb.init(project='GDNet', name= run_name)
    config = wandb.config
    model_loggers = [tb_logger, wb_logger]
else:
    model_loggers = [tb_logger]


def main():

    net = LitGDNet(args)
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
