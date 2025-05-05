import torch
import numpy as np

class EarlyStopping:
    """
    Early stops the training if the monitored metric doesn't improve after a given patience.

    Args:
        patience (int): How many epochs to wait after last time the monitored metric improved.
                        Default: 7
        verbose (bool): If True, prints a message for each metric improvement.
                        Default: False
        delta (float): Minimum change in the monitored metric to qualify as an improvement.
                       Default: 0
        path (str): Path for the checkpoint to be saved to.
                    Default: 'checkpoint.pt'
        trace_func (function): Trace print function.
                               Default: print
        best_metric (float or None): Initial best metric value. If None, it will be set based on mode.
        mode (str): 'min' to minimize the metric, 'max' to maximize the metric.
    """

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path="checkpoint.pt",
        trace_func=print,
        best_metric=None,
        mode='min'
    ):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.mode = mode.lower()

        if self.mode == 'min':
            self.best_metric = np.Inf if best_metric is None else best_metric
            self.monitor_op = lambda current, best: current < best - self.delta
        elif self.mode == 'max':
            self.best_metric = -np.Inf if best_metric is None else best_metric
            self.monitor_op = lambda current, best: current > best + self.delta
        else:
            raise ValueError(f"Mode '{self.mode}' is not supported! Use 'min' or 'max'.")

        self.counter = 0
        self.early_stop = False
        self.best_epoch = None

    def __call__(
        self,
        current_metric,
        model,
        optimizer,
        epoch,
    ):
        """
        Call this method at the end of each epoch to check if training should be stopped.

        Args:
            current_metric (float): The current value of the monitored metric.
            model (nn.Module): The model being trained.
            optimizer (Optimizer): The optimizer used in training.
            epoch (int): The current epoch number.
        """
        if self.monitor_op(current_metric, self.best_metric):
            if self.verbose:
                improvement = (
                    f"{self.best_metric:.6f} --> {current_metric:.6f}"
                )
                self.trace_func(
                    f"Monitored metric improved: {improvement}. Saving model..."
                )
            self.best_metric = current_metric
            self.best_epoch = epoch
            self.save_checkpoint(
                current_metric,
                model,
                optimizer,
            )
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"Early Stopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    self.trace_func("Early stopping triggered.")

    def save_checkpoint(
        self,
        current_metric,
        model,
        optimizer,
    ):
        """Saves the model checkpoint when the monitored metric improves."""
        # Save the model checkpoint
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_metric": current_metric,
            },
            self.path,
        )
