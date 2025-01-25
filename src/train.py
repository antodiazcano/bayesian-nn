"""
Script to train a Bayesian Neural Network (BNN).
"""

import torch
from torch import optim, nn
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
import numpy as np

from src.bnn import BayesianNN, BayesianLinear, bayesian_loss


def train() -> nn.Module:
    """
    Main function to train the model.

    Returns
    -------
    model : Trained BNN.
    """

    # Load model and data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_train = torch.tensor(pd.read_csv("data/X_train.csv").values, dtype=torch.float32)
    x_test = torch.tensor(pd.read_csv("data/X_test.csv").values, dtype=torch.float32)
    y_train = torch.tensor(pd.read_csv("data/y_train.csv").values, dtype=torch.float32)
    y_train = torch.tensor([int(x.item()) for x in y_train])
    y_test = torch.tensor(pd.read_csv("data/y_test.csv").values, dtype=torch.float32)
    y_test = torch.tensor([int(x.item()) for x in y_test])
    n_features = x_train.shape[1]
    n_classes = 3
    model = BayesianNN(n_features, n_classes, hidden_sizes=[128, 64])
    model = model.to(device)
    bayesian_layers = [
        layer for layer in model.modules() if isinstance(layer, BayesianLinear)
    ]

    # Train hyperparameters
    epochs = 100
    batch_size = 128
    learning_rate = 5e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    beta = 1

    # To store training data
    # Note that there is only one KL list because they are the same in train and test
    # (in test parameters are not updated).
    kl_losses = []
    prediction_losses_train = []
    train_losses = []
    train_accs = []
    prediction_losses_test = []
    test_losses = []
    test_accs = []

    # Trainloop
    for epoch in range(epochs):
        model.train()
        loss_epoch = 0.0
        prediction_loss_epoch = 0.0

        # Train
        idx = torch.randperm(x_train.shape[0])
        n_batches = len(x_train) // batch_size
        for i in range(n_batches):
            optimizer.zero_grad()
            idx_batch = idx[i * batch_size : (i + 1) * batch_size]
            x_batch = x_train[idx_batch].to(device)
            y_batch = y_train[idx_batch].to(device)
            y_pred = model(x_batch)

            kl_loss, prediction_loss = bayesian_loss(
                y_batch, y_pred, bayesian_layers, beta=beta
            )
            prediction_loss_epoch += prediction_loss.item()
            loss = kl_loss + prediction_loss
            loss_epoch += loss.item()

            loss.backward()
            optimizer.step()

        y_pred = model(x_train)
        acc = (y_pred.argmax(dim=1) == y_train).float().mean().item()
        prediction_losses_train.append(prediction_loss_epoch / n_batches)
        train_losses.append(loss_epoch / n_batches)
        train_accs.append(acc)

        # Validation
        model.eval()
        with torch.no_grad():
            y_pred = model(x_test.to(device))

            kl_loss, prediction_loss = bayesian_loss(
                y_test, y_pred, bayesian_layers, beta=beta
            )
            test_loss = kl_loss + prediction_loss
            kl_losses.append(kl_loss.item())
            prediction_losses_test.append(prediction_loss.item())

            acc = (y_pred.argmax(dim=1) == y_test).float().mean().item()
            test_losses.append(test_loss.item())
            test_accs.append(acc)

        if (epoch + 1) % 25 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}]")

    # Save training evolution
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    x = range(1, epochs + 1)
    if isinstance(axs, np.ndarray):  # to pass mypy
        axs[0].plot(x, prediction_losses_train, label="Prediction train")
        axs[0].plot(x, kl_losses, label="KL")
        axs[0].plot(x, prediction_losses_test, label="Prediction Test")
        axs[0].plot(x, train_losses, label="Total Train")
        axs[0].plot(x, test_losses, label="Total Test")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].legend()
        axs[1].plot(x, train_accs, label="Train")
        axs[1].plot(x, test_accs, label="Test")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Accuracy")
        axs[1].legend()
    fig.savefig("images/bnn/train.png")
    plt.close(fig)

    return model
