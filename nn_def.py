# coding: utf8

import torch
import torch.nn.functional as F

from torch import nn
from config import *


class Actor(nn.Module):
    def __init__(self, fea_num, action_num):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(fea_num, MID_LAYER_NODE_NUM)
        self.fc2 = nn.Linear(MID_LAYER_NODE_NUM, MID_LAYER_NODE_NUM)
        self.fc3 = nn.Linear(MID_LAYER_NODE_NUM, action_num)

    def forward(self, x):
        out = F.leaky_relu(self.fc1(x), 0.001)
        out = F.leaky_relu(self.fc2(out), 0.001)
        out = torch.tanh(self.fc3(out))
        out = torch.mul(out, ACTION_SCALE)
        return out


class RewardNet(nn.Module):
    def __init__(self, nb_status, nb_actions):
        super(RewardNet, self).__init__()
        self.fc1 = nn.Linear(nb_status + nb_actions, MID_LAYER_NODE_NUM)
        self.fc2 = nn.Linear(MID_LAYER_NODE_NUM, MID_LAYER_NODE_NUM)
        self.fc3 = nn.Linear(MID_LAYER_NODE_NUM, 1)

    def forward(self, x):
        s, a = x
        out = F.leaky_relu(self.fc1(torch.cat([s, a], 1)), 0.001)
        out = F.leaky_relu(self.fc2(out), 0.001)
        out = F.leaky_relu(self.fc3(out), 0.001)
        return out
