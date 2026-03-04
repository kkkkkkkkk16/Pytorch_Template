import torch
import torch.nn as nn
import torch.nn.functional as F

class MSEloss(nn.Module):
    def __init__(self):
        super(MSEloss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, input, target):
        return self.loss(input, target) 