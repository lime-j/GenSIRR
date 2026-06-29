import os
from pathlib import Path

os.environ["TFDS_DATA_DIR"] = "gs://trc-2/"

import lpips
import torch
import torch.nn.functional as F
import torch.optim as optim
import webdataset as wds
from diffusers import AutoencoderKL
from safetensors.torch import load_file
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parent
VAE_WEIGHTS_PATH = ROOT_DIR / "flux_vae.safetensors"


def create_vae():
    """Create VAE model with FLUX-dev configuration and load weights."""
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=[
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ],
        up_block_types=[
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ],
        block_out_channels=[128, 256, 512, 512],
        latent_channels=16,
        layers_per_block=2,
        act_fn="silu",
        norm_num_groups=32,
        sample_size=1024,
        scaling_factor=0.3611,
        shift_factor=0.1159,
        force_upcast=True,
        mid_block_add_attention=True,
        use_quant_conv=False,
        use_post_quant_conv=False,
    )

    state_dict = load_file(str(VAE_WEIGHTS_PATH), device="cpu")
    vae.load_state_dict(state_dict)
    return vae


def create_dataset():
    """Load dataset from gs://trc-2/pd12m/ with tar files 00155 to 02480."""
    urls = [f"pipe:gsutil cat gs://trc-2/pd12m/{i:05d}.tar" for i in range(155, 2481)]

    image_transform = transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.RandomResizedCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = (
        wds.WebDataset(urls, shardshuffle=True)
        .shuffle(1000)
        .decode("pil")
        .map_dict(jpg=image_transform)
        .to_tuple("jpg")
    )

    return dataset


def train_step(model, batch, optimizer, device, lpips_loss_fn):
    """Single training step."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    images = batch[0].to(device, non_blocking=True)
    images2 = images[torch.randperm(images.size(0), device=images.device)]

    posterior1 = model.encode(images).latent_dist
    latents1 = posterior1.sample()
    reconstructed1 = model.decode(latents1).sample

    alpha = torch.rand(images.size(0), 1, 1, 1, device=device)
    mixed_images = alpha * images + (1 - alpha) * images2

    posterior_mixed = model.encode(mixed_images).latent_dist
    latents_mixed = posterior_mixed.sample()
    reconstructed_mixed = model.decode(latents_mixed).sample

    recon_loss = F.mse_loss(reconstructed1, images, reduction="mean")
    lpips_loss = lpips_loss_fn(reconstructed1, images).mean()

    mixed_reconstructions_target = alpha * reconstructed1 + (1 - alpha) * reconstructed1.flip(0)
    interp_loss = F.mse_loss(reconstructed_mixed, mixed_reconstructions_target, reduction="mean")

    loss = recon_loss + 0.1 * lpips_loss + interp_loss
    loss.backward()
    optimizer.step()

    return {
        "total_loss": loss.detach(),
        "recon_loss": recon_loss.detach(),
        "lpips_loss": lpips_loss.detach(),
        "interp_loss": interp_loss.detach(),
    }


def save_checkpoint(model, optimizer, step, loss):
    """Save model and optimizer checkpoint."""
    save_directory = ROOT_DIR / f"checkpoint_step_{step}"
    save_directory.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(save_directory)

    optimizer_checkpoint = {
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "loss": loss,
    }
    torch.save(optimizer_checkpoint, save_directory / "optimizer.pt")

    print(f"Checkpoint saved in {save_directory}")


def main():
    """Main function for single-process GPU training."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")
    if not VAE_WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Missing VAE weights: {VAE_WEIGHTS_PATH}")

    device = torch.device("cuda")
    print(f"Using device: {device}")

    model = create_vae().to(device)
    lpips_loss_fn = lpips.LPIPS(net="alex").to(device)

    dataset = create_dataset()
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=0.01,
    )
    total_steps = 100000
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    progress = tqdm(enumerate(dataloader), total=total_steps, desc="training", dynamic_ncols=True)
    for batch_idx, batch in progress:
        losses = train_step(model, batch, optimizer, device, lpips_loss_fn)

        if batch_idx % 100 == 0:
            progress.set_postfix(
                loss=f"{losses['total_loss'].item():.4f}",
                recon=f"{losses['recon_loss'].item():.4f}",
                lpips=f"{losses['lpips_loss'].item():.4f}",
                interp=f"{losses['interp_loss'].item():.4f}",
            )

        if batch_idx % 10000 == 0:
            save_checkpoint(model, optimizer, batch_idx, losses["total_loss"].item())

        scheduler.step()

        if batch_idx + 1 >= total_steps:
            break

    print("Training completed!")


if __name__ == "__main__":
    main()
