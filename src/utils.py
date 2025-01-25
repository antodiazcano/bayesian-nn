"""
Script for some utilities.
"""

import torch
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
