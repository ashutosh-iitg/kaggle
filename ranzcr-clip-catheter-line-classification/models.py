import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from model.resnet import resnext50_32x4d

class ResNext(nn.Module):
    def __init__(self, params, pretrained:bool):
        super().__init__()
        self.model = resnext50_32x4d(pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, params.num_targets * 8)

        # Custom Layers for prediction here
        self.fcbn1 = nn.BatchNorm1d(params.num_targets * 8)
        self.fc2 = nn.Linear(params.num_targets * 8, params.num_targets)

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        x = F.relu(self.fcbn1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.fc2(x)
        return x

class RANZCRModel():
    def __init__(self, params, pretrained:bool=False):
        if params.mode == 'res':
            self.model = ResNext(params, pretrained)
        else:
            assert()

    def check_model(self):
        print(self.model)