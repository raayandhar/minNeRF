# src/config.py

from typing import Tuple
import os

DATASET_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../data/nerf_synthetic/lego")
)

BATCH_SIZE: int = 1
NUM_EPOCHS: int = 10

# NeRF model configurations
MODEL_LAYER_SIZES: Tuple[int, ...] = (6, 256, 256, 256, 256, 256, 256, 256, 256, 4)
NUM_SAMPLES: int = 64
NEAR: float = 2.0
FAR: float = 6.0

# Optimizer configurations
LEARNING_RATE: float = 1e-4

# Miscellaneous
RANDOM_SEED: int = 42
