import torch
from torch.optim.lr_scheduler import CosineAnnealingLR


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps=10, T_max=80, lr_min=1e-8, base_scheduler=None, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.lr_min = lr_min

        if base_scheduler is None:
            self.base_scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=lr_min)
        else:
            self.base_scheduler = base_scheduler
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            progress = (self.last_epoch + 1) / self.warmup_steps
            return [self.lr_min * (base_lr / self.lr_min) ** progress for base_lr in self.base_lrs]
        else:
            if self.base_scheduler is not None:
                return self.base_scheduler.get_last_lr()
            return self.base_lrs

    def step(self, epoch=None):
        if self.last_epoch < self.warmup_steps:
            super(WarmupScheduler, self).step(epoch)
        elif self.base_scheduler is not None:
            self.base_scheduler.step(epoch)
