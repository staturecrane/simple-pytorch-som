import glob
import os
import random

import click
import torch
import torchvision.transforms.functional as F
import tqdm
from diffusers import AutoModel, Flux2Pipeline
from PIL import Image
from torchvision.utils import save_image

ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png"]


def image_to_tensor(image_filepath: str, vae) -> torch.Tensor:
    pil_image = Image.open(image_filepath).convert("RGB")

    image_tensor = F.to_tensor(pil_image)
    image_tensor = image_tensor.unsqueeze(0) * 2.0 - 1.0
    image_tensor = image_tensor.to(vae.device, dtype=vae.dtype)
    image_tensor = F.center_crop(F.resize(image_tensor, [128, 128]), [128, 128])

    return image_tensor


@torch.inference_mode()
def save_som_grid(weights, vae, filepath="som_grid.png"):
    rows, cols = weights.shape[:2]
    images = []

    for latent in weights.flatten(0, 1):
        decoded = vae.decode(
            latent.unsqueeze(0).to(device=vae.device, dtype=vae.dtype),
            return_dict=False,
        )
        images.append((decoded.float().cpu().clamp(-1, 1) + 1) / 2)

    save_image(
        torch.cat(images),
        filepath,
        nrow=cols,
        padding=2,
    )


def decay(start, end, progress):
    return start * (1 - progress) + end * progress


@click.command()
@click.argument(
    "image-directory",
    type=click.Path(dir_okay=True, file_okay=False, exists=True),
    required=True,
)
@click.option("-r", "--rows", type=int, default=10)
@click.option("-c", "--columns", type=int, default=10)
@click.option("-e", "--epochs", type=int, default=10)
@click.option("-s", "--sigma", type=int, default=2, help="Initial size of neighborhood")
@click.option("-fs", "--final-sigma", type=int, default=0.5, help="Final size of neighborhood")
@click.option(
    "-lr", "--learning_rate", type=float, default=0.1, help="Initial learning rate"
)
@click.option(
    "-flr", "--final_learning_rate", type=float, default=0.01, help="Final learning rate"
)
def main(
    image_directory: str,
    rows: int,
    columns: int,
    epochs: int,
    sigma: int,
    final_sigma: int,
    learning_rate: float,
    final_learning_rate: float
):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )
    tiny_vae = AutoModel.from_pretrained(
        "fal/FLUX.2-Tiny-AutoEncoder",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)

    with torch.no_grad():
        weights = torch.randn((rows, columns, 128, 8, 8)).to(device=device)

    images = []
    for ext in ALLOWED_EXTENSIONS:
        glob_pattern = os.path.join(image_directory, f"**/*{ext}")
        images += glob.glob(glob_pattern)

    random.shuffle(images)

    total_steps = len(images) * epochs

    step = 0
    for epoch in range(epochs):
        for image in tqdm.tqdm(images):
            if epoch > 0:
                progress = step / max(total_steps - 1, 1)

                current_learning_rate = decay(
                    learning_rate, final_learning_rate, progress
                )
                current_sigma = decay(sigma, final_sigma, progress)
            else:
                current_learning_rate = learning_rate
                current_sigma = sigma

            with torch.inference_mode():
                image_tensor = image_to_tensor(image, tiny_vae)
                latents = tiny_vae.encode(image_tensor, return_dict=False)
                latents = torch.squeeze(latents, 0)

            with torch.no_grad():
                distances = (
                    (weights - latents).square().flatten(start_dim=2).sum(dim=-1)
                )
                winner = distances.argmin()
                winner_row, winner_col = (
                    winner // weights.shape[1],
                    winner % weights.shape[1],
                )

                update_rows, update_cols = torch.meshgrid(
                    torch.arange(weights.shape[0], device=weights.device),
                    torch.arange(weights.shape[1], device=weights.device),
                    indexing="ij",
                )
                grid_dist_sq = (update_rows - winner_row).square() + (
                    update_cols - winner_col
                ).square()
                neighborhood = torch.exp(-grid_dist_sq / (2 * current_sigma**2))

                neighborhood = neighborhood.reshape(rows, columns, 1, 1, 1)
                weights += current_learning_rate * neighborhood * (latents - weights)

                if epoch > 0:
                    step += 1

    save_som_grid(weights, tiny_vae)


if __name__ == "__main__":
    main()
