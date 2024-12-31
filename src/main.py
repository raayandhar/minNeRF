# src/main.py

from jax import random
from model import get_nerf_model
from data import data_loader
from train import create_optim, train
from config import DATASET_PATH, BATCH_SIZE, RANDOM_SEED, MODEL_LAYER_SIZES


def main():
    # Set random seed for reproducibility
    key = random.PRNGKey(RANDOM_SEED)

    # Initialize NeRF model
    key, model_key = random.split(key)
    model = get_nerf_model(model_key, MODEL_LAYER_SIZES)

    # Initialize optimizer
    optimizer, opt_state = create_optim(model)

    # Prepare data loader
    data_iter = data_loader(DATASET_PATH, BATCH_SIZE)
    test_model(model, data_iter, key)
    # Start training
    # train(model, optimizer, opt_state, data_iter, key)


def test_model(model, data_iter, key):
    rays, target_rgb = next(data_iter)
    print(f"Input rays shape: {rays.shape}")
    print(f"Target RGB shape: {target_rgb.shape}")

    key, subkey = random.split(key)
    output = model(rays)
    print(f"Model output shape: {output.shape}")
    assert output.shape[0] == rays.shape[0], "Output batch size mismatch."


if __name__ == "__main__":
    main()
