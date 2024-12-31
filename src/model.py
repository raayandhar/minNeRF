# src/model.py

import jax
import equinox as eqx

from typing import Sequence
from jaxtyping import Float, Array

PRNGKeyArray = Array
BatchDim = Float[Array, "batch_dim"]
BatchOut = Float[Array, "batch_out"]


class NeRF(eqx.Module):
    layers: Sequence[eqx.nn.Linear]

    def __init__(self, layer_sizes: Sequence[int], key: PRNGKeyArray):
        keys = jax.random.split(key, len(layer_sizes))
        self.layers = [
            eqx.nn.Linear(in_features=in_size, out_features=out_size, key=key_i)
            for in_size, out_size, key_i in zip(layer_sizes[:-1], layer_sizes[1:], keys)
        ]

    @eqx.filter_jit
    def __call__(self, x: BatchDim) -> BatchOut:
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))

        x = self.layers[-1](x)
        return x


def get_nerf_model(key: PRNGKeyArray, layer_sizes: Sequence[int]) -> NeRF:
    return NeRF(layer_sizes=layer_sizes, key=key)
