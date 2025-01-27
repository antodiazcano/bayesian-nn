"""
Script for the tests of the Bayesian Neural Network.
"""

import torch

from src.bnn import BayesianNN, BayesianLinear


def test_bayesian_linear() -> None:
    """
    Test for the BayesianLinear forward.
    """

    bl = BayesianLinear(5, 3)
    x = torch.randn(10, 5)
    out = bl(x)

    assert out.shape == (10, 3), "Incorrect shape of output"
    assert not torch.allclose(
        out, bl(x)
    ), "Weights and biases are sampled, two predictions must not be close"
    assert not torch.allclose(out, bl(x + 0.1)), "Two different inputs should defer"


def test_bnn() -> None:
    """
    Test for the BayesianNN forward.
    """

    bnn = BayesianNN(5, 3)
    x = torch.randn(10, 5, dtype=torch.float32)
    out = bnn(x)

    assert out.shape == (10, 3), "Incorrect shape of output"
    assert not torch.allclose(
        out, bnn(x)
    ), "Weights and biases are sampled, two predictions must not be close"
    assert not torch.allclose(out, bnn(x + 0.1)), "Two different inputs should defer"


def test_predictions() -> None:
    """
    Test for the predictions of a network.
    """

    in_dim = 5
    x = torch.randn([10, in_dim])

    for out_dim in [1, 3]:
        bnn = BayesianNN(in_dim, out_dim)
        mean_prediction, std_prediction = bnn.predict_proba(x, save_fig=True)
        assert isinstance(mean_prediction, torch.Tensor) and isinstance(
            std_prediction, torch.Tensor
        ), "Incorrect output"
        correct_shape = torch.Size([x.shape[0], out_dim])
        assert (
            mean_prediction.shape == correct_shape
            and std_prediction.shape == correct_shape
        ), "Incorrect dimension of the output"
        predictions = bnn.predict(x)
        assert isinstance(predictions, torch.Tensor), "Incorrect output"
        assert predictions.shape == torch.Size([x.shape[0]])


def test_explore_model() -> None:
    """
    Test for the visualization of the network.
    """

    bnn = BayesianNN(5, 3)
    bnn.explore_weights()
