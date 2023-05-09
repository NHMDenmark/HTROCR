import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
from torch import nn
import pytorch_lightning as pl
from utils.scheduler import CosineWarmupScheduler
from torchvision.models import resnet50, resnet101
import math
import torch.nn.functional as F
from tqdm.notebook import tqdm
from torch import optim
import collections
from itertools import repeat
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CTCLoss
from torchmetrics import CharErrorRate
from itertools import groupby
import utils.device as devutils

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class BlockCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            self.block(1, 64, stride=(2, 2)),
            self.block(64, 128, stride=(2, 2)),
            self.block(128, 256, stride=(2, 1)),
            self.block(256, 512, stride=(4, 1)),
        )

    def block(self, in_channels, out_channels, stride=2):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            Swish(),
        )

    def forward(self, images, *args, **kwargs):
        batch_size = images.size(0)
        out = self.conv_blocks(images)
        out = out.permute(0,3,2,1)
        out = out.reshape(batch_size, -1, 512)
        return out