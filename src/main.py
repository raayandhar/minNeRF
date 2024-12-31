# src/main.py

from jax import random
from model import create_nerf_model
from data import data_loader
from train import create_optimizer, train_model
from config import DATASET_PATH, BATCH_SIZE, RANDOM_SEED, MODEL_LAYER_SIZES


def main():
    # Set random seed for reproducibility
    key = random.key(RANDOM_SEED)

    # Initialize NeRF model
    key, model_key = random.split(key)
    model = create_nerf_model(model_key, MODEL_LAYER_SIZES)

    # Initialize optimizer
    optimizer, opt_state = create_optimizer(model)

    # Prepare data loader
    data_iter = data_loader(DATASET_PATH, BATCH_SIZE)

    # Start training
    train_model(model, optimizer, opt_state, data_iter, key)


if __name__ == "__main__":
    main()
