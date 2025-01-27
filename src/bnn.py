"""
Script to build a Bayesian Neural Network (BNN).
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore


class BayesianLinear(nn.Module):
    """
    Class to build a Bayesian Linear Layer.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        """
        Constructor of the class.

        Parameters
        ----------
        in_dim  : Dimension of the input.
        out_dim : Dimension of the output.
        """

        super().__init__()

        # In and out dimensions
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Mean and std of weights
        self.w_mu = nn.Parameter(0.1 * torch.randn(out_dim, in_dim))
        self.w_sigma = nn.Parameter(
            torch.log(torch.exp(torch.tensor(0.1)) - 1)
            + 0.1 * torch.randn(out_dim, in_dim)
        )
        # Mean and std of biases
        self.b_mu = nn.Parameter(0.1 * torch.randn(out_dim))
        self.b_sigma = nn.Parameter(
            torch.log(torch.exp(torch.tensor(0.1)) - 1) + 0.1 * torch.randn(out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Input tensor. Dimensions: [batch, self.in_dim].

        Returns
        -------
        Output tensor. Dimensions: [batch, self.out_dim].
        """

        # Reparametrization trick for backpropagation
        w_epsilon = torch.randn_like(self.w_mu)
        b_epsilon = torch.randn_like(self.b_mu)

        # Sample from a normal distribution
        weight = self.w_mu + torch.log1p(torch.exp(self.w_sigma)) * w_epsilon
        bias = self.b_mu + torch.log1p(torch.exp(self.b_sigma)) * b_epsilon

        return torch.matmul(x, weight.t()) + bias


class BayesianNN(nn.Module):
    """
    Class to build a Bayesian Neural Network.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_sizes: list[int] | None = None,
        p: float = 0.2,
    ) -> None:
        """
        Constructor of the class.

        Parameters
        ----------
        in_dim       : Dimension of the input.
        out_dim      : Dimension of the output.
        hidden_sizes : Dimension of the hidden sizes.
        p            : Probability of dropout.
        """

        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]

        self.out_dim = out_dim

        fc_layers: list[nn.Module] = []
        # First layer
        fc_layers.append(BayesianLinear(in_dim, hidden_sizes[0]))
        fc_layers.append(nn.ReLU())
        # Middle layers
        for i in range(len(hidden_sizes) - 1):
            fc_layers.append(BayesianLinear(hidden_sizes[i], hidden_sizes[i + 1]))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(p=p))
        # Last layer
        fc_layers.append(BayesianLinear(hidden_sizes[-1], out_dim))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Input tensor. Dimensions: [batch, in_dim].

        Returns:
        Output tensor. Dimensions: [batch, out_dim].
        """

        return self.fc(x)

    def predict_proba(
        self, x: torch.Tensor, n_samples: int = 10, save_fig: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Obtains the mean and std of the predictions of the network for each class.

        Parameters
        ----------
        x         : Input tensor. Dimensions: [batch, in_dim].
        n_samples : Number of predictions to generate.
        save_fig  : True to save a figure of the distributions and False otherwise.

        Returns
        -------
        Mean and std of the predictions. Dimensions: [batch, n_classes], [batch,
        n_classes].
        """

        self.eval()
        n_classes = self.fc[-1].out_dim
        with torch.no_grad():
            preds = torch.zeros(x.shape[0], n_samples, n_classes)
            # Note that we have to do a loop instead of using parallelization in order
            # for the weights of the network to be sampled differently.
            for i in range(n_samples):
                if self.out_dim == 1:
                    preds[:, i, :] = F.sigmoid(self.forward(x))
                else:
                    preds[:, i, :] = F.softmax(self.forward(x), dim=-1)

        if save_fig:
            fig, axs = plt.subplots(1, 2, figsize=(16, 8))

            for i in range(n_classes):
                y_pred = preds[:, :, i].flatten()
                if isinstance(axs, np.ndarray):  # to pass mypy
                    label = "Class 1" if self.out_dim == 1 else f"Class {i}"
                    axs[0].scatter([i] * len(y_pred), y_pred, alpha=0.1, label=label)
                    sns.kdeplot(y_pred, ax=axs[1], label=f"Class {i}")

            if isinstance(axs, np.ndarray):  # to pass mypy
                axs[0].set_title("Distribution of the Predictions")
                axs[0].set_xlabel("Class")
                axs[0].set_ylabel("Probability")
                axs[0].legend()
                axs[1].set_title("Distribution of the Predictions")
                axs[1].set_xlabel("Probability")
                axs[1].set_ylabel("Density")
                axs[1].legend()

            fig.savefig("images/histogram_predictions.png")
            plt.close(fig)

        return torch.mean(preds, dim=1), torch.std(preds, dim=1)

    def predict(self, x: torch.Tensor, n_samples: int = 10) -> torch.Tensor:
        """
        Obtains the predictions of the network.

        Parameters
        ----------
        x         : Input tensor. Dimensions: [batch, in_dim].
        n_samples : Number of predictions to generate.

        Returns
        -------
        Prediction. Dimensions: [batch].
        """

        return torch.argmax(self.predict_proba(x, n_samples)[0], dim=1)

    def explore_weights(self) -> None:
        """
        Plots a histogram of mu and sigma of the weights and biases of the model.
        """

        bayesian_layers = [
            layer for layer in self.modules() if isinstance(layer, BayesianLinear)
        ]
        fig, axs = plt.subplots(
            len(bayesian_layers),
            4,
            figsize=(6 * len(bayesian_layers), 4 * len(bayesian_layers)),
            constrained_layout=True,
        )

        for i, layer in enumerate(bayesian_layers):
            for j, (mu, sigma) in enumerate(
                [[layer.w_mu, layer.w_sigma], [layer.b_mu, layer.b_sigma]]
            ):
                title = f"Weights for layer {i}" if j == 0 else f"Bias for layer {i}"
                if isinstance(axs, np.ndarray):  # to pass mypy
                    axs[i, 2 * j].hist(mu.flatten().detach().numpy(), bins=20)
                    axs[i, 2 * j].set_title(title)
                    axs[i, 2 * j].set_xlabel(r"$\mu$")
                    axs[i, 2 * j].set_ylabel("Frequency")
                    axs[i, 2 * j + 1].hist(
                        torch.log1p(torch.exp(sigma.flatten())).detach().numpy(),
                        bins=20,
                    )
                    axs[i, 2 * j + 1].set_title(title)
                    axs[i, 2 * j + 1].set_xlabel(r"$\sigma$")
                    axs[i, 2 * j + 1].set_ylabel("Frequency")

        fig.savefig("images/model_mu_sigma.png")
        plt.close(fig)
