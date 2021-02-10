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
    # args.train = True
    # args.infer = False
    # args.mode = "infer"
    args.mode = "train"
    args.batch_size = 12
    args.val_every = 2
    args.developer_mode = False
    args.load_model = False
    args.fast_dev_run = False
    args.crf = True
    args.wandb = True
    # args.gdd_training_root = [args.GD_dataset_path, args.Sber_dataset_path, args.no_glass_dataset_path]
    args.gdd_training_root = [args.Sber_dataset_path, args.GD_dataset_path]
    # args.gdd_training_root.append(args.Sber_dataset_path)
    args.epochs = 160
    args.ckpt_path = "ckpt/"
    args.ckpt_name = "Mixed-L-lovasz-97/Mixed-L-lovasz-97-epoch=129-val_loss=0.26.ckpt"
    args.gdd_eval_root = "GDNet/mini_eval"

    args.debugging = False

    if args.debugging :
        args.wandb = False


#######################################
# Creating the name of the run
def get_run_name(args):
    run_name = ""

    if args.load_model:
        if args.mode == "train":
            run_name+= "Fine-Tuning-"
            if not args.freeze_LCFI and not args.freeze_resnet:
                run_name+= "all-"
            if args.freeze_resnet:
                run_name+= "frz-res-"
            if args.freeze_LCFI:
                run_name+= "frz-LCFI-"
                
        else:
            run_name+= "Pretrained-"+args.mode

    # Add the datasets to the name
    if (len(args.gdd_training_root)>1) and isinstance(args.gdd_training_root, list):
        run_name += "Mixed-"
        for i in range(len(args.gdd_training_root)):

            folders = args.gdd_training_root[i].split('/')
            if folders[-1] == "train":
                name = folders[-2]
            else:
                name = folders[-1]
            run_name += name + "-"
        run_name += "L-"
    else:
        # Add the name of the dataset
        folders = args.gdd_training_root[0].split('/')
        if folders[-1] == "train":
            name = folders[-2]
        else:
            name = folders[-1]
        # print()
        run_name += name+"-L-"

    for l in args.loss_funcs:
        run_name+= l+'-'

    if args.mode == "train":
        # Adding the run number to the name of the run
        f = open("run_num.log", "r+")
        run_num = int(f.read())
        f.close()

        f = open("run_num.log", "w")
        f.write(str(run_num+1))
        f.close()

        run_name= str(run_num) +"-"+ run_name
        # if not args.train:
        # run_name = args.mode  + "-" + run_name
    else:
        if args.load_model:
            model = args.ckpt_name.split("/")
            names = model[-1]
            run_name = args.mode +"-" +names[:-5]
    args.log_name = run_name


    return run_name

################################################################################################

args = get_args()

# change the argumnets for testing
init_args(args)

# get the name of the current run
run_name = get_run_name(args)


# Creating folder for saving the images:
if args.developer_mode or args.mode == "infer":
    folder_path = os.path.join(args.gdd_results_root, args.mode ,run_name)
    os.makedirs(folder_path)
    args.gdd_results_root =  os.path.join(args.gdd_results_root, args.mode)


# If training create folder for saving checkpoints
if (args.mode == "train") and not args.debugging:
    this_ckpt_path = os.path.join(args.ckpt_path,run_name)
    os.makedirs(this_ckpt_path)
    # args.ckpt_path = ckpt_path


#######################################
# Checkpoint call back for saving the best models
#
checkpoint_callback = ModelCheckpoint(
    monitor= args.monitor,
    dirpath= os.path.join(args.ckpt_path,run_name),
    filename=  args.log_name + '-{epoch:03d}-{val_loss:.2f}',
    save_top_k= args.save_top,
    mode='min',
)

# If in debugging mode no callbacks
if args.debugging or args.mode =="infer":
    callbacks = []
else:
    callbacks = [checkpoint_callback]

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
                        callbacks = callbacks,
                        check_val_every_n_epoch = args.val_every,
                        logger = model_loggers,
                        resume_from_checkpoint = args.ckpt_path+args.ckpt_name)
        print("Checkpoint loaded successfully!")
    else:
        trainer = Trainer(gpus=args.device_ids,
                        fast_dev_run = args.fast_dev_run,
                        accelerator = 'dp',
                        max_epochs = args.epochs,
                        callbacks = callbacks,
                        check_val_every_n_epoch = args.val_every,
                        logger = model_loggers)
    
    if args.wandb:
        wandb.watch(net)

    if args.mode == "train":
        print("Training")

        trainer.fit(net)
        final_epoch_model_path = os.path.join(args.ckpt_path,run_name,args.log_name +"final-epoch.ckpt") 
        trainer.save_checkpoint(final_epoch_model_path)

        print("Done")

    elif args.mode == "test":
        print("Testing")
        trainer.test(model = net)
    elif args.mode == "infer":
        print("Inference")
        net.infer(args.infer_path)
        


if __name__ == "__main__":
    main()
