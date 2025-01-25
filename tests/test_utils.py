"""
Test for the utils script.
"""

import torch
import torch.nn.functional as F

from src.bnn import BayesianLinear
from src.utils import bayesian_loss


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
