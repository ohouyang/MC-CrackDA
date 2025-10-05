import torch
import torch.nn as nn
import torch.nn.functional as F


class Dice_CE_Loss(nn.Module):
    def __init__(self, dice_smooth=1e-6, class_weights=(1.0, 1.5)):
        super().__init__()
        self.dice_smooth = dice_smooth
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).cuda())

    def forward(self, pred, target):
        ce = self.ce_loss(pred, target.long())

        pred_probs = torch.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes=2).permute(0, 3, 1, 2).float()

        intersection = (pred_probs * target_onehot).sum(dim=(2, 3))
        union = pred_probs.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        dice = (2. * intersection + self.dice_smooth) / (union + self.dice_smooth)
        dice_loss = 1 - dice.mean()

        return ce + dice_loss
