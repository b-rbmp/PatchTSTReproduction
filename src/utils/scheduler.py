
from src.training.supervised.config import TrainingConfig
from torch.optim.lr_scheduler import _LRScheduler

def adjust_lr(optimizer, scheduler, epoch, config: TrainingConfig):
    """
    Adjust the optimizer's learning rate based on a predefined schedule.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer whose learning rate will be adjusted.
        scheduler (torch.optim.lr_scheduler._LRScheduler): A PyTorch learning rate scheduler 
            used in some schedules (e.g. 'TST').
        epoch (int): The current epoch number.
        config (TrainingConfig): Configuration object containing the learning rate adjustment
    """
    # Initialize a dictionary for the new learning rate, keyed by the current epoch if applicable.
    lr_adjust = {}

    # Use the base learning rate from args as a reference.
    base_lr = config.learning_rate

    # Define schedules based on the value in config.learning_rate_adjustment
    if config.learning_rate_adjustment == 'type1':
        # Halve the learning rate each epoch
        lr_adjust = {epoch: base_lr * (0.5 ** ((epoch - 1) // 1))}
    elif config.learning_rate_adjustment == 'type2':
        # Manually specify learning rate at particular epochs
        lr_adjust = {
            2: 5e-5,  4: 1e-5,  6: 5e-6,  8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif config.learning_rate_adjustment == 'type3':
        # Keep the base LR for the first 3 epochs, then multiply by (0.9^(epoch - 3))
        lr_adjust = {
            epoch: base_lr if epoch < 3 else base_lr * (0.9 ** ((epoch - 3) // 1))
        }
    elif config.learning_rate_adjustment == 'constant':
        # Keep the learning rate constant throughout
        lr_adjust = {epoch: base_lr}
    elif config.learning_rate_adjustment == '3':
        # Keep base LR until epoch 10, then reduce by 10x
        lr_adjust = {epoch: base_lr if epoch < 10 else base_lr * 0.1}
    elif config.learning_rate_adjustment == '4':
        # Keep base LR until epoch 15, then reduce by 10x
        lr_adjust = {epoch: base_lr if epoch < 15 else base_lr * 0.1}
    elif config.learning_rate_adjustment == '5':
        # Keep base LR until epoch 25, then reduce by 10x
        lr_adjust = {epoch: base_lr if epoch < 25 else base_lr * 0.1}
    elif config.learning_rate_adjustment == '6':
        # Keep base LR until epoch 5, then reduce by 10x
        lr_adjust = {epoch: base_lr if epoch < 5 else base_lr * 0.1}
    elif config.learning_rate_adjustment == 'TST':
        # Use the learning rate from the scheduler's last update
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}

    # If the epoch is specified in the adjustment schedule, update the LR
    if epoch in lr_adjust:
        new_lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
            #print(f"Updating learning rate to {new_lr}")

class ExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a number of iterations.

    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float): the final learning rate.
        num_iter (int): the number of iterations over which the test occurs.
        last_epoch (int, optional): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.last_epoch = last_epoch
        if num_iter <= 1: raise ValueError("`num_iter` must be larger than 1")
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):  
        r = (self.last_epoch+1) / (self.num_iter - 1)
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]
