"""Script to train a Bayesian Neural Network (BNN)."""

from typing import Literal

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.bnn import BayesianLinear
from src.config import config
from src.early_stopping import EarlyStopping
from src.loss import BayesianLoss


class Trainer:
    """Class to train a BNN."""

    def __init__(self, model: torch.nn.Module) -> None:
        """Constructor of the class.

        Args:
            model: Model to train.
        """

        self.model = model.to(config.training.device)
        self.optimizer = config.training.optim(
            self.model.parameters(), config.training.lr
        )
        self.criterion = BayesianLoss(beta=config.training.beta)
        self.early_stopping = EarlyStopping(
            config.training.patience_early_stopping,
            delta=config.training.delta_early_stopping,
        )
        self.metrics: dict[str, dict[str, list[float]]] = {
            "train": {"cross_entropy": [], "kl": [], "loss": []},
            "valid": {"cross_entropy": [], "kl": [], "loss": []},
        }

    def _append_losses(
        self,
        losses_dict: dict[str, float],
        mode: Literal["train", "valid"],
        n_samples: int,
    ) -> None:
        """Appends all the losses to the training history.

        Args:
            losses_dict: Dict that contains all different losses.
            mode: Train or validation.
            n_samples: Number of batches used.
        """

        for value, loss_type in zip(losses_dict.values(), losses_dict.keys()):
            self.metrics[mode][loss_type].append(value / n_samples)

    def _train_one_epoch(self, train_loader: DataLoader) -> None:
        """Makes the training of an epoch.

        Args:
            train_loader: Train loader.
        """

        self.model.train()
        losses_dict = {"cross_entropy": 0.0, "kl": 0.0, "total": 0.0}

        for inputs, labels in train_loader:
            # Forward
            inputs = inputs.to(config.training.device)
            labels = labels.to(config.training.device)
            out = self.model(inputs)
            # Compute loss
            bayesian_layers = [
                layer
                for layer in self.model.modules()
                if isinstance(layer, BayesianLinear)
            ]
            cross_entropy, kl, loss = self.criterion(labels, out, bayesian_layers)
            losses_dict["cross_entropy"] += cross_entropy.item()
            losses_dict["kl"] += kl.item()
            losses_dict["loss"] += loss.item()
            # Optimize
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        self._append_losses(losses_dict, "train", len(train_loader))

    def _valid_one_epoch(self, val_loader: DataLoader) -> None:
        """Makes the validation of an epoch.

        Args:
            val_loader: Validation loader.
        """

        self.model.eval()
        losses_dict = {"cross_entropy": 0.0, "kl": 0.0, "total": 0.0}

        for inputs, labels in val_loader:
            # Forward
            inputs = inputs.to(config.training.device)
            labels = labels.to(config.training.device)
            out = self.model(inputs)
            # Compute loss
            bayesian_layers = [
                layer
                for layer in self.model.modules()
                if isinstance(layer, BayesianLinear)
            ]
            cross_entropy, kl, loss = self.criterion(labels, out, bayesian_layers)
            losses_dict["cross_entropy"] += cross_entropy.item()
            losses_dict["kl"] += kl.item()
            losses_dict["loss"] += loss.item()

        self._append_losses(losses_dict, "valid", len(val_loader))

    def fit(
        self,
        epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        path_weights: str = config.paths.weights,
        path_figure: str = config.paths.train_evolution,
    ) -> None:
        """Trains the model.

        Args:
            epochs: Number of epochs to train.
            train_loader: Train loader.
            val_loader: Validation loader.
            path_weights: Path where the weights of the model are saved.
            path_figure: Path where the figure is saved.
        """

        print_every = max(1, epochs // 10)

        for epoch in range(1, epochs + 1):
            self._train_one_epoch(train_loader)
            self._valid_one_epoch(val_loader)

            self.early_stopping(
                self.metrics["valid"]["loss"][-1], self.model, path_weights
            )
            if self.early_stopping.apply_early_stop:
                break

            if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
                last_loss_train = self.metrics["train"]["loss"][-1]
                last_loss_valid = self.metrics["valid"]["loss"][-1]
                print(
                    f"Epoch: {epoch}. Loss train: {last_loss_train:.3f}; Loss "
                    f"validation: {last_loss_valid:.3f}."
                )

        self.save_training_figure(path_figure)

    def save_training_figure(self, path_figure: str) -> None:
        """Shows the evolution of the metrics during the training.

        Args:
            path_figure: Path where the figure is saved.
        """

        fig, axs = plt.subplots(1, 1, figsize=(14, 6))
        epochs = len(self.metrics["train"]["loss"])
        x = range(1, epochs + 1)

        axs.plot(x, self.metrics["train"]["loss"], label="Loss Train")
        axs.plot(x, self.metrics["valid"]["loss"], label="Loss Validation")
        axs.plot(x, self.metrics["train"]["cross_entropy"], label="Cross Entropy Train")
        axs.plot(
            x,
            self.metrics["valid"]["cross_entropy"],
            label="Cross Entropy Validation",
        )
        axs.plot(x, self.metrics["train"]["kl"], label="KL Train")
        axs.plot(x, self.metrics["valid"]["kl"], label="KL Validation")
        axs.set_xlabel("Epoch")
        axs.set_ylabel("Loss")
        axs.grid()
        axs.legend()

        fig.savefig(path_figure)
        plt.close(fig)
