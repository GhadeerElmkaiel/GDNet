import argparse
import math
import torch
import os 


def get_args():
    root_path = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(description='GDNet')

    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--scale', type=int, default=416,
                        help='Image scale  (default: 416)')
    parser.add_argument('--crf', action='store_true', default=False,
                        help='use crf (default: False)')
    parser.add_argument('--developer_mode', action='store_true', default=False,
                        help='developer_mode is for the phase of code development and testing (default: False)')
    parser.add_argument('--wandb', action='store_false', default=True,
                        help='Use weights and biases (default: True)')

    # Logging parameters
    parser.add_argument('--num_log_img', type=int, default=5,
                        help='The Training batch size (default: 5)')
    parser.add_argument('--metric_log', nargs='+', type=str, default=["iou", "recall", "precision", "FBeta", "F1"],
                        help='The metrics to log (default ["iou", "recall", "precision", "FBeta", "F1"])')

    # Training parameters
    parser.add_argument('--train', action='store_true', default=False,
                        help='Train the model (default: False)')
    parser.add_argument('--mode', type=str, default="train",
                        help='The mode of the model (train, test, infer) (default: train)')
    parser.add_argument('--infer', action='store_true', default=False,
                        help='Use the model for inference (default: False)')
    parser.add_argument('--fast_dev_run', action='store_true', default=False,
                        help='Do fast eval run to test if every thing is Okay (default: False)')
    parser.add_argument('--load_model', action='store_true', default=False,
                        help='Load pretrained model "named in snapshot"(default: False)')
    parser.add_argument('--shuffle_dataset', action='store_false', default=True,
                        help='Shuffle thee dataset while training (default: True)')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='The Training batch size (default: 12)')
    parser.add_argument('--eval_batch_size', type=int, default=1,
                        help='The evaluation batch size (default: 1)')
    parser.add_argument('--test_batch_size', type=int, default=1,
                        help='The Testing batch size (default: 1)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='The number of epoches (default: 500)')
    parser.add_argument('--device_ids', nargs='+', type=int, default=[0, 1],
                        help = "GPU devices (default [0, 1])")
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--freeze_resnet', action='store_true', default=False,
                        help='Freeze the resnet101 backbone')
    parser.add_argument('--freeze_LCFI', action='store_true', default=False,
                        help='Freeze the LCFI modules')
    parser.add_argument('--loss_funcs', nargs='+', type=str, default=["lovasz", "BCE"],
                        help='The loss function to use (default ["lovasz", "BCE"])')
    parser.add_argument('--w_losses', nargs='+', type=float, default=[1, 1, 1],
                        help='Weights for the 3 output losses (default [1, 1, 1])')
    # parser.add_argument('--w_losses_function', nargs='+', type=int, default=[1, 1],
    #                     help='Weights for the used loss funtions (default [1, 1])')
    parser.add_argument('--val_every', type=int, default=5,
                        help='Do validation epoch after how many Training epoch (default: 5)')
    parser.add_argument('--save_top', type=int, default=5,
                        help='The number of the top models to save (default: 5)')
    parser.add_argument('--monitor', type=str, default="val_loss",
                        help='The Name of the value to monitor in order to save models (default: val_loss)')


    # Optimizer parameters
    parser.add_argument('--optim', type=str, default="adam",
                        help='Name of the optimizer to use (default: adam)')
    parser.add_argument('--betas', nargs='+', type=int, default=[0.9, 0.999],
                        help='Betas values for Adam optimizer(default: (0.9, 0.999))')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for Adam optimizer (default: 0.001)')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='The threshold for iou metric (default: 0.5)')


    # Paths and Names
    parser.add_argument('--root_path', type=str, default=root_path,
                        help='the root path (default: {})'.format(root_path))
    parser.add_argument('--ckpt_path', type=str, default="ckpt/",
                        help='Path to place where to save the checkpoints (default: ckpt/)')
    parser.add_argument('--ckpt_name', type=str, default="L-lovasz-42/GDNet-L-lovasz-42-epoch=141-val_loss=0.25.ckpt",
                        help='Name of the checkpoint to load (default: L-lovasz-42/GDNet-L-lovasz-42-epoch=141-val_loss=0.25.ckpt)')
    parser.add_argument('--exp_name', type=str, default="MirrorNet",
                        help='Name of the folder of the snapshot to load (default: GDNet)')
    parser.add_argument('--backbone_path', type=str, default=None,
                        help='Path to the backbone (default:None')
    # parser.add_argument('--backbone_path', type=str, default=root_path+"/backbone/resnext/resnext_101_32x4d.pth",
    #                     help='Path to the backbone (default:'+ root_path+'/backbone/resnext/resnext_101_32x4d.pth')
    parser.add_argument('--gdd_training_root', nargs='+', type=str, default=[root_path+"/GDNet/train"],
                        help='List of paths to the training data (default: ['+root_path+'/GDNet/train]')
    parser.add_argument('--gdd_eval_root', type=str, default=root_path+"/GDNet/eval",
                        help='Path to the evaluation data (default: '+root_path+'/GDNet/eval')
    parser.add_argument('--gdd_testing_root', type=str, default=root_path+"/GDNet/test",
                        help='Path to the testing data (default: '+root_path+'/GDNet/test')
    parser.add_argument('--gdd_results_root', type=str, default=root_path+"/GDNet/results",
                        help='Path to the results (default: '+root_path+'/GDNet/results')
    parser.add_argument('--GD_dataset_path', type=str, default="GDNet/train",
                        help='Path to GD default dataset (default: GDNet/train')
    parser.add_argument('--Sber_dataset_path', type=str, default="sber_ds",
                        help='Path to Sber dataset (default: sber_ds')
    parser.add_argument('--no_glass_dataset_path', type=str, default="/home/ghadeer/Projects/Datasets/no_glass",
                        help='Path to no Glass dataset (default: /home/ghadeer/Projects/Datasets/no_glass')
    parser.add_argument('--infer_path', type=str, default=root_path+"/test_images",
                        help='Path to the inference images (default: '+root_path+'/test_images')
    parser.add_argument('--log_path', type=str, default="logs/",
                        help='Path to the logs folder (default: logs/')
    parser.add_argument('--log_name', type=str, default="lightning_logs",
                        help='The name of TB logger (default: lightning_logs')
    parser.add_argument('--iter', type=int, default=1,
                        help='Starting value for iterator to use (default: 1)')

                        

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args
