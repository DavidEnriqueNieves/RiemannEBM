import sys
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import ipdb
import math
from utils.toy_dataset import GaussianMixture
from torch.optim.lr_scheduler import CosineAnnealingLR

## Set the netowrk architecture
class MLP_ELU_convex(nn.Module):
    def __init__(self):
        super(MLP_ELU_convex, self).__init__()
        self.f = nn.Sequential(
            nn.Linear(2, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 32)
            )
        self.f1 = nn.Linear(32, 1)
        self.f2 = nn.Linear(32, 1)
        self.f3 = nn.Linear(32, 1)
    
    def forward(self, x):
        out_feat = self.f(x)
        energy = self.f1(out_feat)*self.f2(out_feat) + self.f3(out_feat**2)
        return energy
