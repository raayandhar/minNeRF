# src/data.py

import os
import jax.numpy as jnp
import numpy as np
import imageio

from typing import Iterator, Tuple
from jaxtyping import Float, Array

Array3f = Float[Array, "3"]
NPArray = np.ndarray
JNPArray = jnp.ndarray


def load_poses_bounds(pose_bounds_path: str) -> Tuple[NPArray, NPArray]:
    data = np.load(pose_bounds_path)
    poses = data["poses"]
    bounds = data["bounds"]
    return poses, bounds


def load_images(images_dir: str) -> Array3f:
    image_files = sorted(
        [
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.endswith(".png") or f.endswith(".jpg")
        ]
    )
    images = [imageio.imread(f) for f in image_files]
    images = jnp.array(images).astype(jnp.float32) / 255.0
    return images


def generate_rays(
    poses: NPArray, intrinsics: NPArray, image_shape: Tuple[int, int]
) -> Tuple[JNPArray, JNPArray]:
    height, width = image_shape
    i, j = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    i = i.flatten()
    j = j.flatten()

    dirs = np.stack(
        [
            (i - intrinsics[0, 2]) / intrinsics[0, 0],
            (j - intrinsics[1, 2]) / intrinsics[1, 1],
            np.ones_like(i),
        ],
        axis=-1,
    )  # (num_pixels, 3)

    rays_d = np.einsum(
        "ijk,ik->ij", poses[:, :3, :3], dirs.T
    ).T  # (num_images, num_pixels, 3)
    rays_o = np.broadcast_to(
        poses[:, :3, :3], rays_d.shape
    )  # (num_images, num_pixels, 3)

    rays_o = jnp.array(rays_o)
    rays_d = jnp.array(rays_d)
    return rays_o, rays_d


def data_loader(
    dataset_path: str, batch_size: int
) -> Iterator[Tuple[JNPArray, JNPArray, JNPArray]]:
    images_dir = os.path.join(dataset_path, "images")
    poses_bound_path = os.path.join(dataset_path, "pose_bounds.npy")

    images = load_images(images_dir)
    poses, bounds = load_poses_bounds(poses_bound_path)

    instrinsics = poses[0, :3, :3]

    rays_o, rays_d = generate_rays(poses, instrinsics, images.shape[1:3])

    num_images, num_pixels, _ = rays_o.shape
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    target_rgb = images.reshape(-1, 3)

    num_batches = rays_o.shape[0] // batch_size
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        yield rays_o[start:end], rays_d[start:end], target_rgb[start:end]
