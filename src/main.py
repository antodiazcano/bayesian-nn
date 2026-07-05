"""Script to train and evaluate a Bayesian Neural Network on a 3-class problem."""

import torch
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.bnn import BayesianNN
from src.config import config
from src.train import Trainer


def generate_data(
    n_samples: int = 10_000, n_features: int = 5, n_classes: int = 3, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates a synthetic 3-class dataset.

    Args:
        n_samples: Number of samples.
        n_features: Number of features of the dataset.
        n_classes: Number of classes of the dataset.
        seed: Random seed for reproducibility.

    Returns:
        x: Input tensor. Dimensions: [n_samples, 2].
        y: Label tensor. Dimensions: [n_samples].
    """

    x, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=seed,
    )
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


@torch.no_grad()
def evaluate(model: BayesianNN, test_loader: DataLoader, n_samples: int = 50) -> float:
    """Evaluates the model on the test set and returns accuracy and predictive
    probabilities.

    Args:
        model: Trained Bayesian NN.
        test_loader: Test data loader.
        n_samples: Number of MC samples per input.

    Returns:
        Test accuracy.
    """

    model.eval()
    correct = 0
    total = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(config.training.device)
        labels = labels.to(config.training.device)
        mean_preds = model.predict_proba(inputs, n_samples=n_samples)[0]
        pred_labels = torch.argmax(mean_preds, dim=1)
        correct += (pred_labels == labels).sum().item()
        total += labels.size(0)

    return correct / total


def main() -> None:
    """Main function: generates data, trains the model and evaluates it."""

    # Dataset
    x, y = generate_data()
    dataset = TensorDataset(x, y)
    n_train = int(0.7 * len(dataset))
    n_val = int(0.15 * len(dataset))
    n_test = len(dataset) - n_train - n_val
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test]
    )

    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.training.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.training.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.training.batch_size, shuffle=False
    )

    # Training
    model = BayesianNN(
        in_dim=config.model.in_dim,
        out_dim=config.model.out_dim,
        hidden_sizes=config.model.hidden_sizes,
    )
    trainer = Trainer(model)
    trainer.fit(
        epochs=config.training.epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        path_weights=config.paths.weights,
        path_figure=config.paths.train_evolution,
    )

    # Evaluation
    accuracy = evaluate(model, test_loader, n_samples=50)
    print(f"\nTest accuracy: {accuracy:.3f}")
    model.predict_proba(x.to(config.training.device), path=config.paths.predictions)
    model.explore_weights(path=config.paths.weights_figure)


if __name__ == "__main__":
    main()
