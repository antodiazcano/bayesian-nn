"""
Script for some utilities.
"""

import torch
from torch import nn
import torch.nn.functional as F

from src.bnn import BayesianLinear


def bayesian_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    bayesian_layers: list[BayesianLinear],
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the customized bayesian loss using KL with the prior of the parameters
    and the cross entropy of the predictions.

    Parameters
    ----------
    y_true          : True labels.
    y_pred          : Predicted probabilities.
    bayesian_layers : List with the bayesian layers of the network.
    beta            : Parameter to be more or less flexible with the prior.

    Returns
    -------
    kl_loss         : KL loss of the prior of the parameters.
    prediction_loss : Cross entropy loss.
    """

    prediction_loss = F.cross_entropy(y_pred, y_true)

    kl_loss = torch.tensor(0.0)
    for layer in bayesian_layers:
        for mu, sigma in ([layer.w_mu, layer.w_sigma], [layer.b_mu, layer.b_sigma]):
            mu_prior = torch.zeros_like(mu)
            sigma_prior = 0.1 * torch.ones_like(sigma)
            kl_loss += 0.5 * torch.mean(
                0.5
                * (torch.log1p(torch.exp(sigma)) ** 2 + (mu - mu_prior) ** 2)
                / sigma_prior**2
                - 0.5
                + torch.log(sigma_prior / torch.log1p(torch.exp(sigma)))
            )  # the first 0.5 is because we sum twice per layer
    kl_loss /= len(bayesian_layers)

    return beta * kl_loss, prediction_loss


class EarlyStopping:
    """
    Class to perform early stopping.
    """

    def __init__(self, patience: int, path: str) -> None:
        """
        Constructor of the class.

        Parameters
        ----------
        patience : Epochs allowed without improving validation loss.
        path     : Path to save model parameters.
        """

        self.patience = patience
        self.counter = 0  # to count epochs without improvement
        self.best_score = torch.inf  # best loss achieved
        self.early_stop = False  # flag to know when to stop
        self.path = path

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        """
        Call the class.

        Parameters
        ----------
        val_loss : New value of the loss.
        model    : Model to which the early stopping is applied.
        """

        if val_loss >= self.best_score:  # we do not improve
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:  # we improve
            self.best_score = val_loss
            self.save_weights(model)
            self.counter = 0

    def save_weights(self, model: nn.Module) -> None:
        """
        Saves the weights of the model.

        Parameters
        ----------
        model : Model to train.
        """

        torch.save(model.state_dict(), self.path)
