import torch
import torch.nn as nn


class BCELossDomain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        return self.loss(outputs, targets)
