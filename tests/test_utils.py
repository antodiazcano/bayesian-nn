"""
Test for the utils script.
"""

import os
import torch
from torch import nn
import torch.nn.functional as F

from src.bnn import BayesianLinear
from src.utils import bayesian_loss, EarlyStopping


def test_bayesian_loss() -> None:
    """
    Test for the bayesian_loss.
    """

    y_true = torch.tensor([0, 1, 2])
    y_pred = torch.tensor(
        [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]], dtype=torch.float32
    )
    bayesian_layers = [BayesianLinear(2, 2)]
    kl_loss, prediction_loss = bayesian_loss(y_true, y_pred, bayesian_layers, beta=1.0)

    # KL loss
    assert kl_loss.dim() == 0, "Loss must be a scalar"
    assert kl_loss.item() != 0.0, "KL loss must be greater than zero"
    assert torch.isclose(
        bayesian_loss(y_true, y_pred, bayesian_layers, beta=0.5)[0], 0.5 * kl_loss
    ), "Incorrect beta behavior"

    # Prediction loss
    assert prediction_loss.dim() == 0, "Loss must be a scalar"
    assert torch.isclose(
        prediction_loss, F.cross_entropy(y_pred, y_true)
    ), "Loss must be the same as the Cross Entropy"


def test_improved_loss() -> None:
    """
    Test to check the correct work when the loss improves.
    """

    e_s = EarlyStopping(patience=10, path="tests/weights.pt")
    simple_model = nn.Linear(10, 1)

    val_loss = 0.5
    e_s(val_loss, simple_model)

    assert e_s.best_score == val_loss
    assert e_s.counter == 0
    assert not e_s.early_stop


def test_no_improvement() -> None:
    """
    Test to check the correct work when the loss does not improve.
    """

    e_s = EarlyStopping(patience=10, path="weights/test_weights.pt")
    simple_model = nn.Linear(10, 1)

    e_s.best_score = 0.5
    val_loss = 0.6
    e_s(val_loss, simple_model)

    assert e_s.counter == 1
    assert not e_s.early_stop


def test_early_stop_triggered() -> None:
    """
    Test for when the early stopping has to be activated.
    """

    e_s = EarlyStopping(patience=10, path="weights/test_weights.pt")
    simple_model = nn.Linear(10, 1)

    e_s.best_score = 0.5
    val_loss = 0.6

    for _ in range(e_s.patience):
        e_s(val_loss, simple_model)

    assert e_s.counter == e_s.patience
    assert e_s.early_stop


def test_save_weights() -> None:
    """
    Test for the correct storage of weights.
    """

    path = "weights/test_weights.pt"
    e_s = EarlyStopping(patience=10, path=path)
    simple_model = nn.Linear(10, 1)

    e_s.save_weights(simple_model)
    assert os.path.exists(path)

    loaded_model = nn.Linear(10, 1)
    loaded_model.load_state_dict(torch.load(path, weights_only=True))
    assert all(
        torch.equal(p1, p2)
        for p1, p2 in zip(simple_model.parameters(), loaded_model.parameters())
    )
