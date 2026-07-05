"""Script to define the configuration of the project."""

from dataclasses import dataclass

import torch


@dataclass
class PathsConfig:
    """Class to define the paths where the artifacts are saved."""

    weights = "weights/weights.pt"
    train_evolution = "images/train_evolution.png"
    predictions = "images/predictions.png"
    weights_figure = "images/weights.png"


@dataclass
class ModelConfig:
    """Class to define the configuration of the model."""

    in_dim = 5
    out_dim = 3
    hidden_sizes = [64, 32]


@dataclass
class TrainingConfig:
    """Class to define the configuration of the training."""

    beta = 0.5
    optim = torch.optim.Adam
    batch_size = 64
    lr = 1e-3
    epochs = 100
    patience_early_stopping = 10
    delta_early_stopping = 5e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    """Class to join all the configurations."""

    paths = PathsConfig()
    model = ModelConfig()
    training = TrainingConfig()


config = Config()
