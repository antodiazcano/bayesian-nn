"""
Script to define constants.
"""

import torch

from src.utils import EarlyStopping


# Others
PRINT_EVERY = 10
PATH = "weights/best_model.pt"

# Hyperparameters
BETA = 0.1
OPTIM = torch.optim.Adam
CLIP_GRADIENTS = True
SCHEDULE_LR = True
BATCH_SIZE = 64
LR = 1e-3
EARLY_STOPPING = EarlyStopping(patience=10, path=PATH)
