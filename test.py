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


# from config import msd_testing_root, msd_results_root
from misc import check_mkdir, crf_refine
from gdnet import GDNet
# from mirrornet import MirrorNet, LitMirrorNet
# from dataset import ImageFolder

from arguments import get_args

# from utils.loss import lovasz_hinge