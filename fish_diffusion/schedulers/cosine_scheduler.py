import math

import torch


class LambdaCosineScheduler:
    """
    note: use with a base_lr of 1.0
    """

    def __init__(self, lr_min, lr_max, max_decay_steps):
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_max_decay_steps = max_decay_steps
        self.last_lr = 0.0

    def schedule(self, n):
        t = n / self.lr_max_decay_steps
        t = min(t, 1.0)
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1 + math.cos(t * torch.pi)
        )
        self.last_lr = lr

        return lr

    def __call__(self, n):
        return self.schedule(n)
