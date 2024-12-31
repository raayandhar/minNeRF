# src/train.py

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from typing import Tuple
from jaxtyping import Float, Array
from model import NeRF
from data import data_loader
from config import (
    LEARNING_RATE,
    NUM_EPOCHS,
    BATCH_SIZE,
    NUM_SAMPLES,
    NEAR,
    FAR,
    DATASET_PATH,
    RANDOM_SEED,
)

"""
Dimension key:

R: num_rays
S: num_samples
B: batch size

"""

Array_B3f = Float[Array, "batch 3"]
Array_R3f = Float[Array, "num_rays 3"]
Array_RS3f = Float[Array, "num_rays num_samples 3"]
PRNGKeyArray = Array


def compute_loss(
    model: NeRF,
    rays_origins: Array_B3f,
    rays_directions: Array_B3f,
    target_rgb: Array_B3f,
    key: PRNGKeyArray,
) -> Float[Array, ()]:
    points = sample_points(rays_origins, rays_directions, NUM_SAMPLES, NEAR, FAR, key)
    rendered_rgb = render_rays(NeRF, points, rays_directions)
    loss = jnp.mean((rendered_rgb - target_rgb) ** 2)
    return loss


def sample_points(
    rays_origins: Array_R3f,
    rays_directions: Array_R3f,
    num_samples: int,
    near: float,
    far: float,
    key: PRNGKeyArray,
) -> Array_RS3f:
    pass


def render_rays(model: NeRF, points: Array3f, directions: Array3f) -> Array3f:
    pass
