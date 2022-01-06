from __future__ import print_function, division
import torch
import torch.nn as nn


class blink_encoder(nn.Module):
    def __init__(self):
        super(blink_encoder, self).__init__()
        self.fc1 = nn.Linear(3, 3)
        self.fc2 = nn.Linear(3, 256)
        self.relu = nn.ReLU(True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        net = self.fc1(x)
        net = self.fc2(self.relu(net))
        dis1 = self.sig(net)
        return dis1

