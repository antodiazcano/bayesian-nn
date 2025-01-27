"""
Tests for the training.
"""

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from src.bnn import BayesianNN
from src.train import Trainer


N_SAMPLES = 5_000
IN_DIM = 10
N_CLASSES = 3

bnn = BayesianNN(IN_DIM, N_CLASSES)
x, y = make_classification(
    N_SAMPLES, IN_DIM, n_classes=N_CLASSES, n_informative=IN_DIM // 2
)
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.4)
x_valid, x_test, y_valid, y_test = train_test_split(x_valid, y_valid, test_size=0.5)
x_train = torch.tensor(x_train, dtype=torch.float32)
x_valid = torch.tensor(x_valid, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train)
y_valid = torch.tensor(y_valid)
y_test = torch.tensor(y_test)


def test_make_train() -> None:
    """
    Test for _make_train.
    """

    trainer = Trainer(bnn)
    trainer._make_train(x_train, y_train)

    assert len(trainer.metrics["train"]["loss"]) == 1
    assert len(trainer.metrics["train"]["cross_entropy"]) == 1
    assert len(trainer.metrics["train"]["kl"]) == 1


def test_make_validation() -> None:
    """
    Test for _make_validation.
    """

    trainer = Trainer(bnn)
    trainer._make_validation(x_train, y_train)

    assert len(trainer.metrics["valid"]["loss"]) == 1
    assert len(trainer.metrics["valid"]["cross_entropy"]) == 1
    assert len(trainer.metrics["valid"]["kl"]) == 1


def test_save_train_metrics() -> None:
    """
    Test for _save_train_metrics.
    """

    trainer = Trainer(bnn)
    y_true = torch.tensor([0, 1, 1, 0])
    y_pred = torch.tensor([0, 1, 0, 0])
    trainer._save_train_metrics(y_true, y_pred)

    assert len(trainer.metrics["train"]["accuracy"]) == 1
    assert np.isclose(trainer.metrics["train"]["accuracy"][-1], 0.75)


def test_save_validation_metrics() -> None:
    """
    Test for _save_validation_metrics.
    """

    trainer = Trainer(bnn)
    y_true = torch.tensor([0, 1, 1, 0])
    y_pred = torch.tensor([0, 1, 1, 1])
    trainer._save_validation_metrics(y_true, y_pred)

    assert len(trainer.metrics["valid"]["accuracy"]) == 1
    assert np.isclose(trainer.metrics["valid"]["accuracy"][-1], 0.75)


def test_fit() -> None:
    """
    Test for the fit, show_training and test functions.
    """

    hidden_sizes = [256, 128, 64]
    model = BayesianNN(IN_DIM, N_CLASSES, hidden_sizes=hidden_sizes)
    epochs = 100

    trainer = Trainer(model)
    trainer.fit(x_train, y_train, x_valid, y_valid, epochs=epochs)
    trainer.show_training()
    accuracy = trainer.test(x_test, y_test)

    assert len(trainer.metrics["train"]["loss"]) <= epochs
    assert len(trainer.metrics["valid"]["loss"]) <= epochs
    assert 0.75 <= accuracy <= 1.0
