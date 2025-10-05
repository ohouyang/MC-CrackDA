import torch
import torch.nn as nn
import torch.nn.functional as F


class Weighted_CE_Dice(nn.Module):
    def __init__(self, dice_smooth=1e-10):
        super().__init__()
        self.dice_smooth = dice_smooth

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor,
                pixel_weights: torch.Tensor = None, class_weights=None):
        if class_weights is None:
            class_weights = [1., 1.5]
        if pixel_weights is not None:
            assert pixel_weights.shape == targets.shape
            pixel_weights = torch.Tensor(pixel_weights).to(outputs.device)
        if class_weights is not None:
            class_weights = torch.Tensor(class_weights).to(outputs.device)

        ce_loss_fn = nn.CrossEntropyLoss(reduction='none', weight=class_weights)
        ce_loss = ce_loss_fn(outputs, targets)
        if pixel_weights is not None:
            ce_loss = ce_loss * pixel_weights
        ce_loss = ce_loss.mean()


        pred_probs = torch.softmax(outputs, dim=1)
        target_onehot = F.one_hot(targets, num_classes=2).permute(0, 3, 1, 2).float()
        if pixel_weights is not None:
            pixel_weights_expanded = pixel_weights.unsqueeze(1).expand_as(pred_probs)
            intersection = (pred_probs * target_onehot * pixel_weights_expanded).sum(dim=(2, 3))
            union = (pred_probs * pixel_weights_expanded).sum(dim=(2, 3)) + \
                    (target_onehot * pixel_weights_expanded).sum(dim=(2, 3))
        else:
            intersection = (pred_probs * target_onehot).sum(dim=(2, 3))
            union = pred_probs.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        dice = (2. * intersection + self.dice_smooth) / (union + self.dice_smooth)
        dice_loss = 1. - dice.mean()

        return ce_loss + dice_loss
