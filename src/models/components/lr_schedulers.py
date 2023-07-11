import math

import torch


class WarmupCyclicLR(torch.optim.lr_scheduler._LRScheduler):
    """Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, total_epoch, warmup_epoch, min_lr):
        self.total_epoch = total_epoch
        self.warmup_epoch = warmup_epoch
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        curr_epoch = self.last_epoch + 1
        if curr_epoch < self.warmup_epoch:
            return [base_lr * (float(curr_epoch) / self.warmup_epoch) for base_lr in self.base_lrs]
        else:
            updated_lr = []
            for lr in self.base_lrs:
                lr = self.min_lr + (lr - self.min_lr) * 0.5 * (
                    1.0
                    + math.cos(
                        math.pi
                        * (curr_epoch - self.warmup_epoch)
                        / (self.total_epoch - self.warmup_epoch)
                    )
                )
                updated_lr.append(lr)
            return updated_lr


# class WarmupCyclicLR(torch.optim.lr_scheduler._LRScheduler):
#     """ Gradually warm-up(increasing) learning rate in optimizer.
#     Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
#     Args:
#         optimizer (Optimizer): Wrapped optimizer.
#         multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
#         total_epoch: target learning rate is reached at total_epoch, gradually
#         after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
#     """

#     def __init__(self, optimizer, total_epoch, warmup_epoch, base_lr, min_lr):
#         super().__init__(optimizer)
#         self.total_epoch = total_epoch
#         self.warmup_epoch = warmup_epoch
#         self.base_lr = base_lr
#         self.min_lr = min_lr

#     def get_lr(self):
#         if self.last_epoch < self.warmup_epoch:
#             lr = self.base_lr * (float(self.last_epoch) / self.warmup_epoch)
#         else:
#             lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * \
#                     (1. + math.cos(math.pi * (self.last_epoch - self.warmup_epoch) / (self.total_epoch - self.warmup_epoch)))
#         return [lr]
