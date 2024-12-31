import os
import json
import math
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt

from jaxtyping import Float, Array
from typing import Sequence, Tuple, Any, Callable
from tqdm import tqdm

# ==================== CONSTANTS ====================

"""
:param DATASET_PATH : Path to the dataset; assuming root-level
:param IMAGE_DIR : Directory to store rendered images; assuming root-level

:param EMBEDDING_DIM_POS : Dimension factor for positional encoding
:param EMBEDDING_DIM_DIR : Dimension factor for direction encoding

:param NUM_FREQS_POS : Number of frequency bands for positional encoding
:param NUM_FREQS_DIR : Number of frequency bands for direction encoding

Sampling [HN, HF] (near-far planes)
Recall that the ray is parameterized by t in [HN, HF]
:param NB_BINS : How many points are sampled along each ray for rendering
:param HN : Near plane distance (how close we begin sampling from the camera)
:param HF : Far plane distance (how far we continue sampling)

Learning and optimization parameters
:param LEARNING_RATE : Optimizer learning rate
:param BATCH_SIZE : Batch size
:param NB_EPOCHS : Number of epochs
:param SEED : Random seed
"""

DATASET_PATH = "data/nerf_synthetic/lego/"
IMAGE_DIR = "images"

EMBEDDING_DIM_POS = 10
EMBEDDING_DIM_DIR = 4

NUM_FREQS_POS = 10
NUM_FREQS_DIR = 4

NB_BINS = 192
HN = 0.2
HF = 6.0

LEARNING_RATE = 5e-4
BATCH_SIZE = 1024
NB_EPOCHS = 10
SEED = 42

# ==================== STATIC LOGSPACE ENCODING ====================

"""
These were originally under the NeRFModel, but there were some issues with
how it interacted in equinox. I couldn't get it to work.

In a NeRF (Neural Radiance Field), we need to encode 3D positions and directions
into a higher-dimensional space using sinusoidal functions. This helps the MLP
represent high-frequency details in the scene.

- We take an input x of shape (batch, 3) where each row is a 3D vector (e.g. a position or a direction).
- We then create 'freq_bands' via logspace, giving us multiple frequency scales (0..9 means up to 10 bands).
- For each frequency band, we compute sin(...) and cos(...).
- We then concatenate these sin/cos features, effectively mapping x -> [sin(ωx), cos(ωx)] across a range of ω values.
- Finally, we reshape so the output is (batch, encoded_dim), ready for the MLP.

positional_encoding_pos  => For encoding 3D positions (x, y, z).
positional_encoding_dir  => For encoding 3D directions (dx, dy, dz).

By applying these sinusoidal encodings, the model can learn higher-frequency variations.
This allows the detailed geometry and textures to be represented :)
"""


@jax.jit
def positional_encoding_pos(
    x: Float[Array, "batch 3"]
) -> Float[Array, "batch encoded_dim"]:
    freq_bands = jnp.logspace(0.0, 9.0, num=NUM_FREQS_POS)
    x = jnp.expand_dims(x, -1)
    sin = jnp.sin(x * freq_bands) / jnp.sqrt(freq_bands)
    cos = jnp.cos(x * freq_bands) / jnp.sqrt(freq_bands)
    encoded = jnp.concatenate([sin, cos], axis=-1)  # (batch, 3, 2*NUM_FREQS_POS)
    return encoded.reshape(x.shape[0], -1)


@jax.jit
def positional_encoding_dir(
    x: Float[Array, "batch 3"]
) -> Float[Array, "batch encoded_dim"]:
    freq_bands = jnp.logspace(0.0, 9.0, num=NUM_FREQS_DIR)
    x = jnp.expand_dims(x, -1)
    sin = jnp.sin(x * freq_bands) / jnp.sqrt(freq_bands)
    cos = jnp.cos(x * freq_bands) / jnp.sqrt(freq_bands)
    encoded = jnp.concatenate([sin, cos], axis=-1)
    return encoded.reshape(x.shape[0], -1)


# ==================== DATA LOADING ====================


def load_transforms(json_path: str) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)


def load_image(image_path: str) -> np.ndarray:
    img = imageio.imread(image_path).astype(np.float32) / 255.0
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img


def compute_focal_length(camera_angle_x: float, width: int) -> float:
    return 0.5 * width / math.tan(0.5 * camera_angle_x)


def get_ray_directions(H: int, W: int, focal: float) -> Float[Array, "H W 3"]:
    i, j = jnp.meshgrid(
        jnp.linspace(0, W - 1, W),
        jnp.linspace(0, H - 1, H),
    )
    i = i - W * 0.5
    j = j - H * 0.5
    dirs = jnp.stack([(i) / focal, -(j) / focal, -jnp.ones_like(i)], axis=-1)
    return dirs


def generate_rays(
    c2w: np.ndarray, H: int, W: int, focal: float
) -> Tuple[np.ndarray, np.ndarray]:
    dirs = get_ray_directions(H, W, focal).reshape(-1, 3)
    rays_d = dirs @ c2w[:3, :3].T
    rays_d /= np.linalg.norm(rays_d, axis=-1, keepdims=True)
    rays_o = np.broadcast_to(c2w[:3, -1], rays_d.shape)
    return rays_o.astype(np.float32), rays_d.astype(np.float32)


def prepare_dataset() -> Tuple[np.ndarray, ...]:
    trn = load_transforms(os.path.join(DATASET_PATH, "transforms_train.json"))
    val = load_transforms(os.path.join(DATASET_PATH, "transforms_val.json"))
    tst = load_transforms(os.path.join(DATASET_PATH, "transforms_test.json"))

    def load_subset(transforms):
        cax = transforms["camera_angle_x"]
        ro_list, rd_list = [], []
        for fr in transforms["frames"]:
            fp = fr["file_path"]
            img_path = os.path.join(DATASET_PATH, fp + ".png")
            img = load_image(img_path)
            H, W = img.shape[:2]
            focal = compute_focal_length(cax, W)
            c2w = np.array(fr["transform_matrix"], dtype=np.float32)
            ro, rd = generate_rays(c2w, H, W, focal)
            ro_list.append(ro)
            rd_list.append(rd)
        ro_list = np.concatenate(ro_list, axis=0)
        rd_list = np.concatenate(rd_list, axis=0)
        return ro_list, rd_list

    trn_ro, trn_rd = load_subset(trn)
    val_ro, val_rd = load_subset(val)
    tst_ro, tst_rd = load_subset(tst)
    return trn_ro, trn_rd, val_ro, val_rd, tst_ro, tst_rd


# ==================== MODEL  ====================


class NeRFModel(eqx.Module):
    # NeRF models are so simple?
    layers: Sequence[eqx.nn.Linear]

    def __init__(self, layer_sizes: Sequence[int], key: jax.random.PRNGKey):
        """
        layer_sizes: e.g. [84, 256, 256, 256, 256, 4].
        weight shapes =>
            layers[0].weight: (256, 84)
            layers[1].weight: (256, 256)
            ...
        """
        assert layer_sizes[-1] == 4
        keys = jax.random.split(key, len(layer_sizes))
        self.layers = [
            eqx.nn.Linear(in_features=inf, out_features=outf, key=k)
            for inf, outf, k in zip(layer_sizes[:-1], layer_sizes[1:], keys)
        ]

    # Strangely, we cannot do @eqx.filter_jit here.
    # TODO: Investigate why? This sucks :(
    def __call__(
        self, o: Float[Array, "batch 3"], d: Float[Array, "batch 3"]
    ) -> Tuple[Float[Array, "batch 3"], Float[Array, "batch"]]:
        """
        We do shape => (batch, 84). Then for each layer, we transpose => shape => (in_features, batch).
        Then do weight @ x => shape => (out_features, batch). Then transpose back => (batch, out_features).
        """
        emb_o = positional_encoding_pos(
            o
        )  # shape (batch, 3*(2*NUM_FREQS_POS)) e.g. (batch, 60)
        emb_d = positional_encoding_dir(
            d
        )  # shape (batch, 3*(2*NUM_FREQS_DIR)) e.g. (batch, 24)
        x = jnp.concatenate([emb_o, emb_d], axis=-1)  # (batch, 84)

        # Strangely, I need to transpose everything here??
        # TODO: Investigate why? This sucks :(
        for layer_i, layer in enumerate(self.layers[:-1]):
            # shape => (84, batch)
            xT = x.T
            # shape => (out_features, batch)
            outT = layer.weight @ xT + layer.bias[:, None]
            # shape => (batch, out_features)
            x = jax.nn.relu(outT.T)

        xT = x.T  # shape => (in_features, batch)
        outT = self.layers[-1].weight @ xT + self.layers[-1].bias[:, None]
        out = outT.T  # shape => (batch, 4)

        # Split color + sigma
        c, sigma = out[:, :3], jax.nn.relu(out[:, 3])
        return c, sigma


# ==================== RENDERING  ====================


def compute_accumulated_transmittance(
    alpha: Float[Array, "batch nb_bins"]
) -> Float[Array, "batch nb_bins"]:
    epsilon = 1e-10
    cumprod = jnp.cumprod(1.0 - alpha + epsilon, axis=1)
    T_i = jnp.concatenate([jnp.ones((alpha.shape[0], 1)), cumprod[:, :-1]], axis=1)
    return T_i


def make_render_rays(
    nb_bins: int, hn: float, hf: float
) -> Callable[..., Float[Array, "batch 3"]]:

    @jax.jit
    def _render_rays(
        model: NeRFModel,
        ray_origins: Float[Array, "batch 3"],
        ray_directions: Float[Array, "batch 3"],
        key: jax.random.PRNGKey,
    ) -> Float[Array, "batch 3"]:
        batch_size = ray_origins.shape[0]
        t_lin = jnp.linspace(hn, hf, nb_bins)
        t_lin = jnp.broadcast_to(t_lin, (batch_size, nb_bins))

        mid = 0.5 * (t_lin[:, :-1] + t_lin[:, 1:])
        lower = jnp.concatenate([t_lin[:, :1], mid], axis=1)
        upper = jnp.concatenate([mid, t_lin[:, -1:]], axis=1)

        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=(batch_size, nb_bins))
        t = lower + (upper - lower) * u

        delta = jnp.concatenate(
            [t[:, 1:] - t[:, :-1], jnp.full((batch_size, 1), 1e10)], axis=1
        )

        x = ray_origins[:, None, :] + t[:, :, None] * ray_directions[:, None, :]
        # x_flat => (batch_size*nb_bins, 3)
        x_flat = x.reshape(-1, 3)
        # d_flat => (batch_size*nb_bins, 3)
        d_flat = jnp.repeat(ray_directions, nb_bins, axis=0)

        # shape => (batch_size*nb_bins, 3) for positions, directions
        colors_flat, sigma_flat = model(
            x_flat, d_flat
        )  # => each = (batch_size*nb_bins, 3/1)
        colors = colors_flat.reshape(batch_size, nb_bins, 3)
        sigma = sigma_flat.reshape(batch_size, nb_bins)

        alpha = 1.0 - jnp.exp(-sigma * delta)
        T_i = compute_accumulated_transmittance(alpha)
        weights = T_i * alpha

        c = jnp.sum(weights[:, :, None] * colors, axis=1)
        weight_sum = jnp.sum(weights, axis=1)
        c += (1.0 - weight_sum)[:, None]
        return c

    return _render_rays


# ====================TRAINING ====================


def make_update_step(
    optimizer: optax.GradientTransformation,
    static_model: Any,
    render_rays_fn: Callable[..., Float[Array, "batch 3"]],
):
    @jax.jit
    def _update_step(
        float_params: Any,
        opt_state: Any,
        ray_origins: Float[Array, "batch 3"],
        ray_directions: Float[Array, "batch 3"],
        target: Float[Array, "batch 3"],
        key: jax.random.PRNGKey,
    ) -> Tuple[Any, Any, Float[Array, ""]]:
        def loss_fn(fp):
            model_ = eqx.combine(fp, static_model)
            pred = render_rays_fn(
                model_,
                ray_origins,
                ray_directions,
                key,
            )
            return jnp.mean((pred - target) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(float_params)
        updates, opt_state = optimizer.update(grads, opt_state, float_params)
        float_params = optax.apply_updates(float_params, updates)
        return float_params, opt_state, loss

    return _update_step


def train(
    full_model: NeRFModel,
    optimizer: optax.GradientTransformation,
    train_rays_o: np.ndarray,
    train_rays_d: np.ndarray,
    train_colors: np.ndarray,
    render_rays_fn: Callable[..., Float[Array, "batch 3"]],
):
    float_params = eqx.filter(full_model, eqx.is_array)
    static_model = eqx.filter(full_model, lambda x: not eqx.is_array(x))
    opt_state = optimizer.init(float_params)

    update_step = make_update_step(optimizer, static_model, render_rays_fn)

    # Convert data
    rays_o = jnp.array(train_rays_o, dtype=jnp.float32)
    rays_d = jnp.array(train_rays_d, dtype=jnp.float32)
    colors = jnp.array(train_colors, dtype=jnp.float32).reshape(-1, 3)

    num_rays = rays_o.shape[0]
    num_batches = max(1, num_rays // BATCH_SIZE)

    loss_log = []
    for epoch in range(NB_EPOCHS):
        perm_key = jax.random.PRNGKey(SEED + epoch)
        perm = jax.random.permutation(perm_key, num_rays)

        ro_epoch = rays_o[perm]
        rd_epoch = rays_d[perm]
        c_epoch = colors[perm]

        epoch_loss = 0.0
        for i in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{NB_EPOCHS}"):
            start_i = i * BATCH_SIZE
            end_i = min(start_i + BATCH_SIZE, num_rays)

            batch_o = ro_epoch[start_i:end_i]
            batch_d = rd_epoch[start_i:end_i]
            batch_c = c_epoch[start_i:end_i]

            step_key = jax.random.PRNGKey(SEED + epoch * 10000 + i)
            float_params, opt_state, loss_val = update_step(
                float_params,
                opt_state,
                batch_o,
                batch_d,
                batch_c,
                step_key,
            )
            epoch_loss += loss_val.item()

        avg_loss = epoch_loss / num_batches
        loss_log.append(avg_loss)
        print(f"Epoch {epoch+1}/{NB_EPOCHS}: Loss={avg_loss:.6f}")

    final_model = eqx.combine(float_params, static_model)
    return final_model, opt_state, loss_log


# ==================== TEST / RENDER ====================


def test_render(
    render_rays_fn: Callable[..., Float[Array, "batch 3"]],
    model: NeRFModel,
    dataset_o: np.ndarray,
    dataset_d: np.ndarray,
    img_index: int,
    H: int,
    W: int,
    image_dir: str,
    key: jax.random.PRNGKey,
):
    os.makedirs(image_dir, exist_ok=True)

    start = img_index * H * W
    end = (img_index + 1) * H * W
    ro = jnp.array(dataset_o[start:end], dtype=jnp.float32)
    rd = jnp.array(dataset_d[start:end], dtype=jnp.float32)

    data = []
    chunk_size = 10
    num_chunks = math.ceil(H / chunk_size)

    for i in tqdm(range(num_chunks), desc=f"Rendering Image {img_index}"):
        chunk_start = i * W * chunk_size
        chunk_end = min((i + 1) * W * chunk_size, H * W)

        batch_o = ro[chunk_start:chunk_end]
        batch_d = rd[chunk_start:chunk_end]

        rendered = render_rays_fn(
            model,
            batch_o,
            batch_d,
            key,
        )
        data.append(rendered)

    img = jnp.concatenate(data, axis=0).reshape(H, W, 3)
    img_np = jax.device_get(img)

    plt.figure(figsize=(W / 100, H / 100), dpi=100)
    plt.imshow(img_np)
    plt.axis("off")
    plt.tight_layout()
    os.makedirs(image_dir, exist_ok=True)
    plt.savefig(
        os.path.join(image_dir, f"img_{img_index}.png"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


# ==================== MAIN ====================


def main():
    trn_ro, trn_rd, val_ro, val_rd, tst_ro, tst_rd = prepare_dataset()
    train_colors = np.ones_like(trn_ro, dtype=np.float32)

    #   Determine input dimension:
    #   For positions: 3 * (2 * NUM_FREQS_POS) = e.g. 60
    #   For directions: 3 * (2 * NUM_FREQS_DIR) = e.g. 24
    #   total = 84
    pos_enc_size = 3 * (2 * NUM_FREQS_POS)
    dir_enc_size = 3 * (2 * NUM_FREQS_DIR)
    input_dim = pos_enc_size + dir_enc_size

    layer_sizes = [input_dim, 256, 256, 256, 256, 4]
    key = jax.random.PRNGKey(SEED)
    full_model = NeRFModel(layer_sizes=layer_sizes, key=key)

    # Make jitted rendering function with static nb_bins, hn, hf
    render_rays_fn = make_render_rays(NB_BINS, HN, HF)

    optimizer = optax.adam(LEARNING_RATE)

    print("Starting training...")
    trained_model, opt_state, loss_history = train(
        full_model,
        optimizer,
        trn_ro,
        trn_rd,
        train_colors,
        render_rays_fn,
    )
    print("Training complete.")

    os.makedirs(IMAGE_DIR, exist_ok=True)
    np.save(os.path.join(IMAGE_DIR, "training_loss.npy"), np.array(loss_history))

    # Validate if enough rays
    H, W = 400, 400
    if len(val_ro) >= H * W:
        print("Rendering validation image 0...")
        test_render(
            render_rays_fn,
            trained_model,
            val_ro,
            val_rd,
            img_index=0,
            H=H,
            W=W,
            image_dir=IMAGE_DIR,
            key=jax.random.PRNGKey(SEED + 999),
        )


if __name__ == "__main__":
    main()
