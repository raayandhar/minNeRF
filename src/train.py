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


def step(
    model: NeRF,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    rays_origins: Array_B3f,
    rays_directions: Array_B3f,
    target_rgb: Array_B3f,
    key: PRNGKeyArray,
) -> Tuple[NeRF, optax.OptState, Float[Array, ()]]:
    loss_fn = lambda m: compute_loss(
        model, rays_origins, rays_directions, target_rgb, key
    )
    loss, grads = jax.value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def sample_points(
    rays_origins: Array_R3f,
    rays_directions: Array_R3f,
    num_samples: int,
    near: float,
    far: float,
    key: PRNGKeyArray,
) -> Array_RS3f:
    t_vals = jnp.linspace(0.0, 1.0, num_samples)
    z_vals = near * (1.0 - t_vals) + far * t_vals  # uniform sampling
    points_RS3 = (
        rays_origins[:, None, :] + rays_directions[:, None, :] * z_vals[None, :, None]
    )
    return points_RS3


def render_rays(model: NeRF, points: Array_RS3f, directions: Array_R3f) -> Array_R3f:
    num_rays, num_samples, _ = points.shape
    points_flat = points.reshape(-1, 3)  # RS3 -> (R*S)3
    directions_expanded = jnp.tile(directions, (num_samples, 1))  # RS3 -> (R*S)3
    inputs = jnp.concatenate([points_flat, directions_expanded], axis=-1)  # (R*S)6
    outputs = model(inputs)  # (R*S)4
    outputs = outputs.reshape(num_rays, num_samples, 4)
    rgb = outputs[..., :3]
    density = jax.nn.softplus(outputs[..., 3])  # TODO: replace with own definition

    # Compute alpha, weights for compositing
    deltas = jnp.diff(jnp.linspace(NEAR, FAR, num_samples), axis=-1)
    delta_inf = 1e10
    deltas = jnp.concatenate([deltas, jnp.full((num_rays, 1), delta_inf)], axis=-1)

    alpha = 1.0 - jnp.exp(-density * deltas)
    weights = (
        alpha
        * jnp.cumprod(
            jnp.concatenate([jnp.ones((num_rays, 1)), 1.0 - alpha + 1e-10], axis=-1),
            axis=-1,
        )[:, :-1]
    )
    rgb_map = jnp.sum(weights[..., None] * rgb, axis=1)
    return rgb_map


def create_optim(
    params: eqx.Module,
) -> Tuple[optax.GradientTransformation, optax.OptState]:
    optim = optax.adam(learning_rate=LEARNING_RATE)
    opt_state = optim.init(eqx.filter(params, eqx.is_array))
    return optim, opt_state


def train(
    model: NeRF,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    data_iter,
    key: PRNGKeyArray,
):
    for epoch in range(NUM_EPOCHS):
        for batch_idx, (rays_o, rays_d, target_rgb) in enumerate(data_iter):
            key, subkey = jax.random.split(key)
            model, opt_state, loss = step(
                model,
                optimizer,
                opt_state,
                rays_o,
                rays_d,
                target_rgb,
                subkey,
            )
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss:.4f}")
