import torch
import torch.nn as nn


class HRM(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_network = nn.Linear()

    def forward(self, x):
        return x

