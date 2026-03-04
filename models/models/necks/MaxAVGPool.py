import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxAVGPool(nn.Module):
    def __init__(self):
        super(MaxAVGPool, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        max_pool = self.max_pool(x)
        avg_pool = self.avg_pool(x)
        return max_pool, avg_pool