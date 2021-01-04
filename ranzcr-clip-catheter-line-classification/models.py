import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from model.resnet import resnext50_32x4d

import numpy as np
from sklearn.metrics import roc_auc_score

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

def get_score(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[1]):
        score = roc_auc_score(y_true[:,i], y_pred[:,i])
        scores.append(score)
    avg_score = np.mean(scores)
    return avg_score

def accuracy(outputs, labels):
    '''
    Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: (np.ndarray) dimension batch_size x num_classes - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, ... num_classes]
    Returns: (float) accuracy in [0,1]
    '''
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    #'accuracy': accuracy,
    'mean_roc_auc_score': get_score,
    # could add more metrics such as accuracy for each token type
}