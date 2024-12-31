import os
import jax.numpy as jnp
import numpy as np
import imageio
import json
from typing import Iterator, Tuple
from jaxtyping import Float, Array
from utils import positional_encoding

Array3f = Float[Array, "3"]
NPArray = np.ndarray
JNPArray = jnp.ndarray


def load_poses_bounds(json_path: str) -> Tuple[NPArray, NPArray]:
    with open(json_path, "r") as f:
        metadata = json.load(f)

    poses = [
        np.array(frame["transform_matrix"], dtype=np.float32)
        for frame in metadata["frames"]
    ]
    bounds = np.array(
        [metadata.get("bounds", [2.0, 6.0]) for _ in metadata["frames"]],
        dtype=np.float32,
    )
    return np.array(poses), bounds


def load_images(json_path: str) -> Array3f:
    with open(json_path, "r") as f:
        metadata = json.load(f)

    image_files = [frame["file_path"] for frame in metadata["frames"]]
    image_files = [
        os.path.join(os.path.dirname(json_path), f"{path}.png") for path in image_files
    ]
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
    )

    rays_o_list = []
    rays_d_list = []
    for pose in poses:
        rays_d = (pose[:3, :3] @ dirs.T).T
        rays_o = np.broadcast_to(pose[:3, 3], rays_d.shape)
        rays_o_list.append(rays_o)
        rays_d_list.append(rays_d)

    rays_o = jnp.array(np.concatenate(rays_o_list, axis=0))
    rays_d = jnp.array(np.concatenate(rays_d_list, axis=0))
    return rays_o, rays_d


def data_loader(
    dataset_path: str, batch_size: int, L: int = 10
) -> Iterator[Tuple[JNPArray, JNPArray]]:
    json_path = os.path.abspath(os.path.join(dataset_path, "transforms_train.json"))

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Metadata file not found: {json_path}")

    images = load_images(json_path)[..., :3]
    poses, _ = load_poses_bounds(json_path)
    intrinsics = poses[0, :3, :3]
    height, width = images.shape[1:3]
    num_pixels = height * width

    for img_idx in range(images.shape[0]):
        rays_o, rays_d = generate_rays(
            poses[img_idx : img_idx + 1], intrinsics, (height, width)
        )
        rays_o = rays_o.reshape(num_pixels, 3)
        rays_d = rays_d.reshape(num_pixels, 3)
        target_rgb = images[img_idx].reshape(num_pixels, 3)

        inputs = jnp.concatenate([rays_o, rays_d], axis=-1)

        num_batches = inputs.shape[0] // batch_size
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            yield inputs[start:end], target_rgb[start:end]
