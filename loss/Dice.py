import torch
import torch.nn as nn
import torch.nn.functional as F


class Dice(nn.Module):
    def __init__(self, dice_smooth=1e-10):
        super().__init__()
        self.dice_smooth = dice_smooth

    def forward(self, outputs, targets):
        pred_probs = torch.softmax(outputs, dim=1)
        target_onehot = F.one_hot(targets, num_classes=2).permute(0, 3, 1, 2).float()
        intersection = (pred_probs * target_onehot).sum(dim=(2, 3))
        union = pred_probs.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        dice = (2. * intersection + self.dice_smooth) / (union + self.dice_smooth)
        dice_loss = 1 - dice.mean()

        return dice_loss
