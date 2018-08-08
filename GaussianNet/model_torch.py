
import torch
from torch import nn


class GaussianNet(nn.Moudle):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, )

    def forward(self):
        
