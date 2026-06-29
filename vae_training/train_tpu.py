import os
os.environ["TFDS_DATA_DIR"] = "gs://trc-2/"
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import logging
from tqdm import tqdm
import lpips
from torchvision import transforms
from peft import LoraConfig, get_peft_model
import webdataset as wds

# from model import create_vae
# from data import create_dataset
from diffusers import AutoencoderKL
# Setup logging

def create_vae():
    """Create VAE model with FLUX-dev configuration and load weights"""
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=[
            "DownEncoderBlock2D",
            "DownEncoderBlock2D", 
            "DownEncoderBlock2D",
            "DownEncoderBlock2D"
        ],
        up_block_types=[
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D", 
            "UpDecoderBlock2D"
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
        use_post_quant_conv=False
    )
    
    # Load weights from safetensors file
    from safetensors.torch import load_file
    state_dict = load_file("flux_vae.safetensors", device="cpu")
    vae.load_state_dict(state_dict)
    return vae

def create_dataset():
    """Load dataset from gs://trc-2/pd12m/ with tar files 00155 to 02480"""
    # Generate URLs for tar files from 00155 to 02480
    urls = [f"pipe:gsutil cat gs://trc-2/pd12m/{i:05d}.tar" for i in range(155, 2481)]

    # Define the image transformations
    image_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.RandomResizedCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    print(f"Rank {xm.get_ordinal()}, World size {xm.xrt_world_size()}")
    # Create webdataset and apply transformations
    dataset = (
        wds.WebDataset(urls, shardshuffle=True)
        .shuffle(1000)
        .slice(xm.get_ordinal(), xm.xrt_world_size())
        .decode("pil")
        .map_dict(jpg=image_transform)
        .to_tuple("jpg")
    )

    return dataset


def train_step(model, batch, optimizer, device, lpips_loss_fn):
    """Single training step"""
    model.train()
    optimizer.zero_grad()
    
    # Extract tensor from list and move to device
    images = batch[0].to(device)
    
    # Create a second batch by shuffling
    images2 = images[torch.randperm(images.size(0))]
    
    # --- VAE passes ---
    # Pass for images 1
    posterior1 = model.encode(images).latent_dist
    latents1 = posterior1.sample()
    reconstructed1 = model.decode(latents1).sample
    
    # # Pass for images 2
    # posterior2 = model.encode(images2).latent_dist
    # latents2 = posterior2.sample()
    # reconstructed2 = model.decode(latents2).sample

    # --- Interpolation ---
    # Generate random interpolation factor
    alpha = torch.rand(images.size(0), 1, 1, 1, device=device)
    
    # Mix images
    mixed_images = alpha * images + (1 - alpha) * images2
    
    # Pass for mixed images
    posterior_mixed = model.encode(mixed_images).latent_dist
    latents_mixed = posterior_mixed.sample()
    reconstructed_mixed = model.decode(latents_mixed).sample
    
    # --- Loss Calculation ---
    
    # 1. Reconstruction loss
    recon_loss = F.mse_loss(reconstructed1, images, reduction='mean')
    
    # 2. LPIPS perceptual loss
    lpips_loss = lpips_loss_fn(reconstructed1, images).mean()
    

    # 4. Interpolation loss
    # The target for the mixed reconstruction is the mixture of reconstructions
    mixed_reconstructions_target = alpha * reconstructed1 + (1 - alpha) * reconstructed1.flip(0)
    interp_loss = F.mse_loss(reconstructed_mixed, mixed_reconstructions_target, reduction='mean')
    
    # Total loss
    # Weight for the interpolation loss, can be tuned.
    interp_loss_weight = 1.0
    loss = recon_loss + 0.1 * lpips_loss + interp_loss_weight * interp_loss

    # Backward pass
    loss.backward()
    xm.optimizer_step(optimizer)
    
    return {
        'total_loss': loss,
        'recon_loss': recon_loss,
        'lpips_loss': lpips_loss,
        'interp_loss': interp_loss
    }

def save_checkpoint(model, optimizer, epoch, loss, device):
    """Save LoRA adapter checkpoint"""
    if xm.is_master_ordinal():
        save_directory = f'lora_checkpoint_epoch_{epoch}'
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the LoRA adapter weights and config
        model.save_pretrained(save_directory)
        
        # Save optimizer state for resuming training
        optimizer_checkpoint = {
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss
        }
        torch.save(optimizer_checkpoint, os.path.join(save_directory, 'optimizer.pt'))
        
        print(f"LoRA checkpoint saved in {save_directory}")

def _mp_fn(rank):
    """Main training function for multiprocessing"""
    # Initialize TPU
    print('init')
    device = 'xla' # xm.xla_device()
    print(device, rank)    
    # Create model
    model = create_vae()
    
    # --- LoRA Configuration ---
    '''
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "conv_in", "conv_out"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    if xm.is_master_ordinal():
        model.print_trainable_parameters()
    # --------------------------
    '''
    model = model.to(device)
    
    # Initialize LPIPS loss function
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)  # Use AlexNet as feature extractor
    
    # Create dataset and dataloader
    dataset = create_dataset()
    dataloader = DataLoader(
        dataset, 
        batch_size=4,  # Adjust based on TPU memory
        shuffle=False,
        num_workers=0,  # Changed to 0 to debug KeyboardInterrupt hang
        drop_last=True
    )
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=0.01)
    total_steps = 100000  # Assume 1000 steps per epoch for 100 epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    

    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_lpips_loss = 0
    total_interp_loss = 0
    num_batches = 0
    # don't use progress bar on TPUs
    for batch_idx, batch in enumerate(dataloader):
        losses = train_step(model, batch, optimizer, device, lpips_loss_fn)

        num_batches += 1
        
        # Log every 100 batches
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}: Loss={losses['total_loss'].item():.4f}, "
                       f"Recon={losses['recon_loss'].item():.4f}, "
                       f"LPIPS={losses['lpips_loss'].item():.4f}, "
                       f"Interp={losses['interp_loss'].item():.4f}")
        if xm.is_master_ordinal() and batch_idx % 10000 == 0:
            save_checkpoint(model, optimizer, batch_idx, losses['total_loss'].item(), device)
        # Step scheduler
        scheduler.step()

    print("Training completed!")

def main():

    """Main function"""
    # Set environment variables for TPU
    # os.environ.setdefault('XLA_USE_BF16', '1')
    
    # Get TPU configuration
    # Un-comment the following lines to confirm the world size at startup
    # if xm.is_master_ordinal():
    #     print(f"Training on {xm.xrt_world_size()} TPU cores")
    
    # Start multiprocessing
    print('before launch')
    torch_xla.launch(_mp_fn, args=())

if __name__ == "__main__":
    main()
