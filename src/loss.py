"""Script to define the loss function."""

import torch
import torch.nn.functional as F
from torch import nn

from src.bnn import BayesianLinear


class BayesianLoss(nn.Module):
    """Class for the bayesian loss function."""

    def __init__(self, beta: float = 1.0) -> None:
        """Constructor of the class.

        Args:
            beta: Parameter to be more or less flexible with the prior.
        """

        super().__init__()

        self.beta = beta

    @staticmethod
    def cross_entropy_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Computes the cross entropy loss.

        Args:
            y_true: True labels. Dimensions: [batch].
            y_pred: Predicted probabilities. Dimensions: [batch].

        Returns:
            Cross entropy loss.
        """

        return F.cross_entropy(y_pred, y_true)

    @staticmethod
    def kl_loss(bayesian_layers: list[BayesianLinear]) -> torch.Tensor:
        """Computes the KL loss.

        Args:
            bayesian_layers: Bayesian linear layers of the model.

        Returns:
            KL loss.
        """

        kl_loss = torch.tensor(0.0)

        for layer in bayesian_layers:
            for mu, sigma in ([layer.w_mu, layer.w_sigma], [layer.b_mu, layer.b_sigma]):
                mu_prior = torch.zeros_like(mu)
                sigma_prior = layer.scale * torch.ones_like(sigma)
                kl_loss += (
                    torch.mean(
                        torch.log(sigma_prior / torch.log(1 + torch.exp(sigma)))
                        + (torch.log(1 + torch.exp(sigma)) ** 2 + (mu - mu_prior) ** 2)
                        / (2 * sigma_prior**2)
                        - 0.5
                    )
                    / 2
                )  # the / 2 is because we sum twice per layer (weights and biases)

        return kl_loss / len(bayesian_layers)

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        bayesian_layers: list[BayesianLinear],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            y_true: True labels. Dimensions: [batch].
            y_pred: Predicted probabilities. Dimensions: [batch].
            bayesian_layers: Bayesian linear layers of the model.

        Returns:
            Cross entropy, KL and bayesian loss.
        """

        prediction_loss = self.cross_entropy_loss(y_true, y_pred)
        layers_loss = self.beta * self.kl_loss(bayesian_layers)
        bayesian_loss = prediction_loss + layers_loss

        return prediction_loss, layers_loss, bayesian_loss
