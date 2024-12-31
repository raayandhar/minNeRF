# src/utils.py

import jax.numpy as jnp
from jaxtyping import Float, Array
import matplotlib.pyplot as plt

Array_BDf = Float[Array, "batch dim"]
Array_B_D2Lf = Float[Array, "batch dim2L"]
Array_HW3f = Float[Array, "height width 3"]


def positional_encoding(x: Array_BDf, L: int = 10) -> Array_B_D2Lf:
    frequencies = 2.0 ** jnp.arange(L)
    x = x[..., None] * frequencies  # Shape: (batch, dim, L)
    return jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1).reshape(x.shape[0], -1)


def visualize_rendered_image(
    rendered_rgb: Array_HW3f,
    target_rgb: Array_HW3f,
):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Rendered Image")
    plt.imshow(rendered_rgb)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Ground Truth")
    plt.imshow(target_rgb)
    plt.axis("off")

    plt.show()
