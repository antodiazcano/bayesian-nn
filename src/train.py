"""
Script to train a Bayesian Neural Network (BNN).
"""

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

from src.bnn import BayesianLinear
from src.utils import bayesian_loss
from src.constants import (
    BETA,
    PRINT_EVERY,
    CLIP_GRADIENTS,
    SCHEDULE_LR,
    BATCH_SIZE,
    LR,
    EARLY_STOPPING,
    OPTIM,
)


class Trainer:
    """
    Class to train a PyTorch model.
    """

    def __init__(
        self,
        model: torch.nn.Module,
    ) -> None:
        """
        Constructor of the class.
        Parameters
        ----------
        model : Model to train.
        """
        self.model = model
        self.optim = OPTIM(model.parameters(), LR)
        self.criterion = bayesian_loss
        self.lr_scheduler = (
            ReduceLROnPlateau(self.optim, factor=0.1, patience=5)
            if SCHEDULE_LR
            else None
        )
        self.early_stopping = EARLY_STOPPING
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # to use GPU if possible
        self.model.to(self.device)
        # Lists to store training metrics
        self.metrics: dict[str, dict[str, list[float]]] = {
            "train": {
                "loss": [],
                "cross_entropy": [],
                "kl": [],
                "accuracy": [],
            },
            "valid": {
                "loss": [],
                "cross_entropy": [],
                "kl": [],
                "accuracy": [],
            },
        }

    def _make_train(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Makes the training of an epoch.
        Parameters
        ----------
        x : Training set.
        y : Labels of the training set.
        """
        self.model.train()
        idx = np.random.permutation(x.shape[0])
        num_batches = x.shape[0] // BATCH_SIZE
        total_entropy_loss = 0.0
        total_kl_loss = 0.0
        for i in range(num_batches):
            self.optim.zero_grad()
            idx_batch = idx[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            embeddings, labels = x[idx_batch], y[idx_batch]
            embeddings, labels = embeddings.to(self.device), labels.to(self.device)
            out = self.model(embeddings)
            kl_loss, prediction_loss = self.criterion(
                labels,
                out,
                [
                    layer
                    for layer in self.model.modules()
                    if isinstance(layer, BayesianLinear)
                ],
                beta=BETA,
            )
            loss = kl_loss + prediction_loss
            loss.backward()
            if CLIP_GRADIENTS:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.optim.step()
            total_entropy_loss += prediction_loss.item()
            total_kl_loss += kl_loss.item()
        total_loss = (total_entropy_loss + total_kl_loss) / num_batches
        self.metrics["train"]["loss"].append(total_loss)
        self.metrics["train"]["cross_entropy"].append(total_entropy_loss / num_batches)
        self.metrics["train"]["kl"].append(total_kl_loss / num_batches)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(total_loss)

    def _make_validation(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Makes the validation of an epoch.
        Parameters
        ----------
        x : Validation set.
        y : Labels of the validation set.
        """
        self.model.eval()
        idx = np.random.permutation(x.shape[0])
        num_batches = x.shape[0] // BATCH_SIZE
        total_entropy_loss = 0.0
        total_kl_loss = 0.0
        with torch.no_grad():
            for i in range(num_batches):
                idx_batch = idx[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
                embeddings, labels = x[idx_batch], y[idx_batch]
                embeddings, labels = embeddings.to(self.device), labels.to(self.device)
                out = self.model(embeddings)
                kl_loss, prediction_loss = self.criterion(
                    labels,
                    out,
                    [
                        layer
                        for layer in self.model.modules()
                        if isinstance(layer, BayesianLinear)
                    ],
                    beta=BETA,
                )
                total_entropy_loss += prediction_loss.item()
                total_kl_loss += kl_loss.item()
        total_loss = (total_entropy_loss + total_kl_loss) / num_batches
        self.metrics["valid"]["loss"].append(total_loss)
        self.metrics["valid"]["cross_entropy"].append(total_entropy_loss / num_batches)
        self.metrics["valid"]["kl"].append(total_kl_loss / num_batches)

    def _save_train_metrics(self, y: torch.Tensor, y_pred: torch.Tensor) -> None:
        """
        Saves the training metrics of an epoch.
        Parameters
        ----------
        y       : Actual labels.
        y_pred  : Labels predicted by the model.
        y_probs : Probabilities predicted by the model.
        """
        self.metrics["train"]["accuracy"].append(accuracy_score(y, y_pred))

    def _save_validation_metrics(self, y: torch.Tensor, y_pred: torch.Tensor) -> None:
        """
        Saves the training metrics of an epoch.
        Parameters
        ----------
        y       : Actual labels.
        y_pred  : Labels predicted by the model.
        y_probs : Probabilities predicted by the model.
        """
        self.metrics["valid"]["accuracy"].append(accuracy_score(y, y_pred))

    def fit(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
        epochs: int = 100,
    ) -> None:
        """
        Training of the network.
        Parameters
        ----------
        x_train     : Tensor in which each row represents an instance of the train set.
        y_train     : Tensor in which each row represents a label of the train set.
        x_valid     : Tensor in which each row represents an instance of the valid set.
        y_train     : Tensor in which each row represents a label of the valid set.
        epochs      : Number of epochs to train.
        print_every : Number of epochs to print on screen the situation of the training.
        """
        print_every = epochs // 10 if PRINT_EVERY is None else PRINT_EVERY
        for epoch in range(1, epochs + 1):
            # Train and validation
            self._make_train(x_train, y_train)
            self._make_validation(x_valid, y_valid)
            # Save metrics and print on screen
            y_pred = self.model.predict(x_train)
            self._save_train_metrics(y_train, y_pred)
            y_pred = self.model.predict(x_valid)
            self._save_validation_metrics(y_valid, y_pred)
            if epoch % print_every == 0 or epoch == 1:
                print(f"\nEpoch {epoch}\n")
                print(f"Training loss: {self.metrics["train"]["loss"][-1]:.3f}")
                print(f"Validation loss: {self.metrics["valid"]["loss"][-1]:.3f}")
            # Early stopping
            self.early_stopping(self.metrics["valid"]["loss"][-1], self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                self.model.load_state_dict(
                    torch.load(self.early_stopping.path, weights_only=False)
                )
                break
        torch.save(self.model.state_dict(), self.early_stopping.path)

    def show_training(self) -> None:
        """
        Shows the evolution of the metrics during the training.
        """
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        epochs = len(self.metrics["train"]["loss"])
        x = range(1, epochs + 1)
        if isinstance(axs, np.ndarray):  # to pass mypy
            # Bayesian loss
            axs[0].plot(x, self.metrics["train"]["loss"], label="Loss Train")
            axs[0].plot(x, self.metrics["valid"]["loss"], label="Loss Validation")
            axs[0].plot(
                x, self.metrics["train"]["cross_entropy"], label="Cross Entropy Train"
            )
            axs[0].plot(
                x,
                self.metrics["valid"]["cross_entropy"],
                label="Cross Entropy Validation",
            )
            axs[0].plot(x, self.metrics["train"]["kl"], label="KL Train")
            axs[0].plot(x, self.metrics["valid"]["kl"], label="KL Validation")
            axs[0].set_xlabel("Epoch")
            axs[0].set_ylabel("Loss")
            axs[0].legend()
            # Accuracy
            axs[1].plot(x, self.metrics["train"]["accuracy"], label="Train")
            axs[1].plot(x, self.metrics["valid"]["accuracy"], label="Validation")
            axs[1].set_xlabel("Epoch")
            axs[1].set_ylabel("Accuracy")
            axs[1].legend()
        fig.savefig("images/train.png")
        plt.close(fig)

    def test(self, x_test: torch.Tensor, y_test: torch.Tensor) -> float:
        """
        Performs a test with the trained model.
        Parameters
        ----------
        x_test : Instances of the test.
        y_test : Labels of the test.
        Returns
        -------
        Accuracy in the test.
        """
        self.model.eval()
        return accuracy_score(y_test, self.model.predict(x_test))
