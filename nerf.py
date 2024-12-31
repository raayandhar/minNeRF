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
import wandb

from jaxtyping import Float, Array
from typing import Sequence, Tuple, Any, Callable
from tqdm import tqdm

# ==================== CONSTANTS ====================

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


@jax.jit
def positional_encoding_pos(
    x: Float[Array, "batch 3"]
) -> Float[Array, "batch encoded_dim"]:
    freq_bands = jnp.logspace(0.0, 9.0, num=NUM_FREQS_POS)
    x = jnp.expand_dims(x, -1)
    sin = jnp.sin(x * freq_bands) / jnp.sqrt(freq_bands)
    cos = jnp.cos(x * freq_bands) / jnp.sqrt(freq_bands)
    encoded = jnp.concatenate([sin, cos], axis=-1)
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
    """Loads the JSON transforms file with error checking."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Transforms file not found at: {json_path}")
    with open(json_path, "r") as f:
        return json.load(f)


def load_image(image_path: str) -> np.ndarray:
    """
    Loads an image with original [0..255] pixel values (float32).
    Throws error if file is missing.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
    img = imageio.imread(image_path).astype(np.float32)  # no / 255 => keep raw
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
    print("Preparing dataset...")

    trn_json = os.path.join(DATASET_PATH, "transforms_train.json")
    val_json = os.path.join(DATASET_PATH, "transforms_val.json")
    tst_json = os.path.join(DATASET_PATH, "transforms_test.json")

    trn = load_transforms(trn_json)
    val = load_transforms(val_json)
    tst = load_transforms(tst_json)

    def load_subset(transforms, subset_name=""):
        cax = transforms["camera_angle_x"]
        ro_list, rd_list, px_list = [], [], []

        for idx, fr in enumerate(transforms["frames"]):
            # Print the file_path from JSON
            fp = fr["file_path"]
            img_path = os.path.join(DATASET_PATH, fp + ".png")
            print(
                f"[{subset_name}] idx={idx}, JSON file_path={fp}, full path={img_path}"
            )

            # Load the image
            img = load_image(img_path)
            print(
                f"Loaded {img_path} => shape={img.shape}, dtype={img.dtype}, "
                f"min={img.min()}, max={img.max()}"
            )

            # Print raw pixel values for the first 10
            flat_img = img.reshape(-1, 3)
            print("First 10 pixels:", flat_img[:10])

            H, W = img.shape[:2]
            focal = compute_focal_length(cax, W)
            c2w = np.array(fr["transform_matrix"], dtype=np.float32)

            # Debug: Print the first transform
            if idx == 0:
                print(f"[{subset_name}] c2w[0]:\n{c2w}")

            # Generate rays
            ro, rd = generate_rays(c2w, H, W, focal)
            ro_list.append(ro)
            rd_list.append(rd)

            # Flatten image row-major
            px = img.reshape(-1, 3)  # still in [0..255]
            px_list.append(px)

            # Save small patch of the first few images
            if idx < 2 and subset_name != "test":  # let's skip test set
                patch = img[:5, :5, :].astype(np.uint8)
                plt.figure(figsize=(5, 5))
                plt.imshow(patch)
                plt.title(f"Patch from {subset_name} idx={idx}")
                plt.axis("off")
                patch_path = os.path.join(IMAGE_DIR, f"patch_{subset_name}_{idx}.png")
                plt.savefig(patch_path)
                plt.close()
                print(f"Saved patch from {subset_name}, idx={idx} => {patch_path}")

        ro_list = np.concatenate(ro_list, axis=0)
        rd_list = np.concatenate(rd_list, axis=0)
        px_list = np.concatenate(px_list, axis=0)

        print(
            f"[{subset_name}] Final shapes => ro: {ro_list.shape}, rd: {rd_list.shape}, px: {px_list.shape}"
        )
        return ro_list, rd_list, px_list

    trn_ro, trn_rd, trn_px = load_subset(trn, subset_name="train")
    val_ro, val_rd, val_px = load_subset(val, subset_name="val")
    tst_ro, tst_rd, tst_px = load_subset(tst, subset_name="test")

    print("Dataset prepared successfully.")
    return trn_ro, trn_rd, trn_px, val_ro, val_rd, val_px, tst_ro, tst_rd, tst_px


# ==================== MODEL ====================


class NeRFModel(eqx.Module):
    layers: Sequence[eqx.nn.Linear]

    def __init__(self, layer_sizes: Sequence[int], key: jax.random.PRNGKey):
        assert layer_sizes[-1] == 4
        keys = jax.random.split(key, len(layer_sizes))
        self.layers = [
            eqx.nn.Linear(in_features=inf, out_features=outf, key=k)
            for inf, outf, k in zip(layer_sizes[:-1], layer_sizes[1:], keys)
        ]

    def __call__(
        self, o: Float[Array, "batch 3"], d: Float[Array, "batch 3"]
    ) -> Tuple[Float[Array, "batch 3"], Float[Array, "batch"]]:
        emb_o = positional_encoding_pos(o)
        emb_d = positional_encoding_dir(d)
        x = jnp.concatenate([emb_o, emb_d], axis=-1)
        for layer in self.layers[:-1]:
            xT = x.T
            outT = layer.weight @ xT + layer.bias[:, None]
            x = jax.nn.relu(outT.T)

        xT = x.T
        outT = self.layers[-1].weight @ xT + self.layers[-1].bias[:, None]
        out = outT.T  # (batch,4)

        c = jax.nn.sigmoid(out[:, :3])  # [0..1]
        sigma = jax.nn.relu(out[:, 3])  # >= 0
        return c, sigma


# ==================== RENDERING ====================


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
            [t[:, 1:] - t[:, :-1], jnp.full((batch_size, 1), 1e10)],
            axis=1,
        )

        x = ray_origins[:, None, :] + t[:, :, None] * ray_directions[:, None, :]
        x_flat = x.reshape(-1, 3)
        d_flat = jnp.repeat(ray_directions, nb_bins, axis=0)

        colors_flat, sigma_flat = model(x_flat, d_flat)
        colors = colors_flat.reshape(batch_size, nb_bins, 3)
        sigma = sigma_flat.reshape(batch_size, nb_bins)

        alpha = 1.0 - jnp.exp(-sigma * delta)
        T_i = compute_accumulated_transmittance(alpha)
        weights = T_i * alpha

        c = jnp.sum(weights[:, :, None] * colors, axis=1)
        weight_sum = jnp.sum(weights, axis=1)
        c += (1.0 - weight_sum)[:, None]  # white BG
        return c

    return _render_rays


# ==================== TRAINING ====================


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
            pred = render_rays_fn(model_, ray_origins, ray_directions, key)
            # pred in [0..1], target in [0..255], so scale pred
            return jnp.mean((pred * 255.0 - target) ** 2)

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
    layer_sizes: list,
    val_ro: np.ndarray,
    val_rd: np.ndarray,
):
    float_params = eqx.filter(full_model, eqx.is_array)
    static_model = eqx.filter(full_model, lambda x: not eqx.is_array(x))
    opt_state = optimizer.init(float_params)

    update_fn = make_update_step(optimizer, static_model, render_rays_fn)

    rays_o = jnp.array(train_rays_o, dtype=jnp.float32)
    rays_d = jnp.array(train_rays_d, dtype=jnp.float32)
    colors = jnp.array(train_colors, dtype=jnp.float32).reshape(-1, 3)  # [0..255]

    num_rays = rays_o.shape[0]
    num_batches = max(1, num_rays // BATCH_SIZE)

    loss_log = []

    run_name = input("Enter the wandb run name: ")
    wandb.init(
        project="minNeRF",
        config={
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "num_epochs": NB_EPOCHS,
            "nb_bins": NB_BINS,
            "hn": HN,
            "hf": HF,
            "num_freqs_pos": NUM_FREQS_POS,
            "num_freqs_dir": NUM_FREQS_DIR,
            "embedding_dim_pos": EMBEDDING_DIM_POS,
            "embedding_dim_dir": EMBEDDING_DIM_DIR,
            "layer_sizes": layer_sizes,
            "seed": SEED,
        },
        name=run_name,
        reinit=True,
    )

    print("Sample train colors:", train_colors[:10])
    print("Train colors range => min:", train_colors.min(), "max:", train_colors.max())

    step = 0
    for epoch in range(NB_EPOCHS):
        perm_key = jax.random.PRNGKey(SEED + epoch)
        perm = jax.random.permutation(perm_key, num_rays)

        ro_epoch = rays_o[perm]
        rd_epoch = rays_d[perm]
        c_epoch = colors[perm]

        epoch_loss = 0.0
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{NB_EPOCHS}")
        for i in pbar:
            start_i = i * BATCH_SIZE
            end_i = min(start_i + BATCH_SIZE, num_rays)

            batch_o = ro_epoch[start_i:end_i]
            batch_d = rd_epoch[start_i:end_i]
            batch_c = c_epoch[start_i:end_i]  # [0..255]

            step_key = jax.random.PRNGKey(SEED + epoch * 10000 + i)
            float_params, opt_state, loss_val = update_fn(
                float_params, opt_state, batch_o, batch_d, batch_c, step_key
            )
            epoch_loss += loss_val.item()

            wandb.log(
                {"batch_loss": loss_val.item(), "epoch": epoch + 1, "step": step},
                step=step,
            )
            step += 1
            pbar.set_postfix(loss=loss_val.item())

            # Optionally render a validation image halfway
            if i == num_batches // 4:
                print(f"Rendering val image at epoch {epoch+1} halfway.")
                Hval, Wval = 400, 400
                if len(val_ro) >= Hval * Wval:
                    val_img = test_render(
                        render_rays_fn,
                        eqx.combine(float_params, static_model),
                        val_ro,
                        val_rd,
                        img_index=0,
                        H=Hval,
                        W=Wval,
                        image_dir=IMAGE_DIR,
                        key=jax.random.PRNGKey(SEED + 999),
                    )
                    wandb.log(
                        {
                            "rendered_image_half_epoch": epoch + 1,
                            "image_index": 0,
                            "rendered_image": wandb.Image(val_img),
                        },
                        step=step,
                    )

        avg_loss = epoch_loss / num_batches
        loss_log.append(avg_loss)
        print(f"Epoch {epoch+1}/{NB_EPOCHS} => Avg Loss: {avg_loss:.6f}")
        wandb.log({"epoch_loss": avg_loss, "epoch": epoch + 1}, step=step)

        # Render at end of epoch
        print(f"Rendering val image at epoch {epoch+1} end.")
        Hval, Wval = 400, 400
        if len(val_ro) >= Hval * Wval:
            val_img = test_render(
                render_rays_fn,
                eqx.combine(float_params, static_model),
                val_ro,
                val_rd,
                img_index=0,
                H=Hval,
                W=Wval,
                image_dir=IMAGE_DIR,
                key=jax.random.PRNGKey(SEED + 999),
            )
            wandb.log(
                {
                    "rendered_image_epoch": epoch + 1,
                    "image_index": 0,
                    "rendered_image": wandb.Image(val_img),
                },
                step=step,
            )

    final_model = eqx.combine(float_params, static_model)

    os.makedirs(IMAGE_DIR, exist_ok=True)
    np.save(os.path.join(IMAGE_DIR, "training_loss.npy"), np.array(loss_log))
    wandb.log({"training_loss_history": wandb.Histogram(np.array(loss_log))})

    wandb.finish()

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
) -> np.ndarray:
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

        rendered = render_rays_fn(model, batch_o, batch_d, key)
        data.append(rendered)

    img = jnp.concatenate(data, axis=0).reshape(H, W, 3)
    img_np = jax.device_get(img)

    plt.figure(figsize=(W / 100, H / 100), dpi=100)
    plt.imshow(img_np)  # [0..1] from model
    plt.axis("off")
    plt.tight_layout()
    os.makedirs(image_dir, exist_ok=True)
    out_path = os.path.join(image_dir, f"img_{img_index}.png")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    return img_np


# ==================== MAIN ====================


def main():
    print("Beginning dataset checks and loading...")

    (
        trn_ro,
        trn_rd,
        trn_px,
        val_ro,
        val_rd,
        val_px,
        tst_ro,
        tst_rd,
        tst_px,
    ) = prepare_dataset()

    # Confirm the shape/range of train px
    print("train px shape:", trn_px.shape, "dtype:", trn_px.dtype)
    print("First 10 train pixels:\n", trn_px[:10])
    print("Range in train px => min:", trn_px.min(), ", max:", trn_px.max())

    # Optionally visualize entire first train image if (H=400, W=400)
    H, W = 400, 400
    if len(trn_px) >= H * W:
        first_img = trn_px[: H * W].reshape(H, W, 3).astype(np.uint8)
        plt.figure()
        plt.imshow(first_img)
        plt.title("First Train Image (Raw 0..255)")
        plt.axis("off")
        sample_path = os.path.join(IMAGE_DIR, "first_train_image_raw.png")
        plt.savefig(sample_path)
        plt.close()
        print(f"Saved entire first training image => {sample_path}")

    # Setup model
    pos_enc_size = 3 * (2 * NUM_FREQS_POS)
    dir_enc_size = 3 * (2 * NUM_FREQS_DIR)
    input_dim = pos_enc_size + dir_enc_size
    layer_sizes = [input_dim, 256, 256, 256, 256, 4]

    key = jax.random.PRNGKey(SEED)
    full_model = NeRFModel(layer_sizes=layer_sizes, key=key)

    render_rays_fn = make_render_rays(NB_BINS, HN, HF)
    optimizer = optax.adam(LEARNING_RATE)

    print("Starting training now, with unnormalized images in [0..255].")
    trained_model, opt_state, loss_history = train(
        full_model,
        optimizer,
        trn_ro,
        trn_rd,
        trn_px,  # pass raw [0..255]
        render_rays_fn,
        layer_sizes,
        val_ro,
        val_rd,
    )
    print("Training completed.")


if __name__ == "__main__":
    main()
