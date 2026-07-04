"""Script to define early stopping."""

import numpy as np
import torch
from torch import nn


class EarlyStopping:
    """Class to implement early stopping during training."""

    def __init__(self, patience: int, delta: float = 1e-3) -> None:
        """Constructor of the class.

        Args:
            patience: Epochs allowed without improving validation loss until the
                training is stopped.
            delta: Minimum change in the validation loss to consider an improvement.
        """

        self.patience = patience
        self.delta = delta
        self.epochs_without_improvement = 0
        self.best_validation_loss = np.inf
        self.apply_early_stop = False

    def __call__(self, val_loss: float, model: nn.Module, path: str) -> None:
        """Call method.

        Args:
            val_loss: New value of the validation loss.
            model: Model to which early stopping is applied.
            path: Path where the parameters of the model are saved if the new validation
                loss is the best one obtained.
        """

        if val_loss >= self.best_validation_loss - self.delta:  # we do not improve
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                self.apply_early_stop = True
        else:  # we improve
            self.best_validation_loss = val_loss
            self.save_parameters(model, path)
            self.epochs_without_improvement = 0

    @staticmethod
    def save_parameters(model: nn.Module, path: str) -> None:
        """Saves the parameters of the model.

        Args:
            model: Model to which early stopping is applied.
            path: Path where the parameters of the model are saved.
        """

        torch.save(model.state_dict(), path)
