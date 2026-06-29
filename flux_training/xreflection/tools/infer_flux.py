#!/usr/bin/env python3
"""Flux reflection removal inference script.

This utility loads a FluxKontext model that was fine-tuned with DeepSpeed ZeRO
stage 2 and runs reflection removal on arbitrary input images. It supports
checkpoints produced by Lightning+DeepSpeed (``last.ckpt`` directories) as well
as consolidated ``mp_rank_00_model_states.pt`` files.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Dict, Iterable, Sequence, Tuple

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
import torchvision.transforms.functional as TF

# Ensure the repository root is on sys.path when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from xreflection.archs.flux_arch import FluxKontext
except ModuleNotFoundError as exc:
    missing = exc.name or "an optional dependency"
    raise ModuleNotFoundError(
        f"Missing dependency '{missing}'. Install project requirements before running inference."
    ) from exc

try:
    from xreflection.archs.fluxtext_arch import FluxKontextText
except ModuleNotFoundError:
    FluxKontextText = None

LOGGER = logging.getLogger("flux_infer")
SUPPORTED_EXTENSIONS: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    """Create the CLI parser and return parsed arguments."""
    parser = argparse.ArgumentParser(description="Run Flux-based reflection removal on input images")
    parser.add_argument("--config", required=True, type=Path, help="YAML config used during training")
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to DeepSpeed ZeRO checkpoint directory or mp_rank_00_model_states.pt file",
    )
    parser.add_argument("--input", required=True, type=Path, help="Input image or directory")
    parser.add_argument("--output", default=Path("inference_results"), type=Path, help="Output directory")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cuda", "cpu"),
        help="Computation device",
    )
    parser.add_argument("--steps", type=int, default=28, help="Number of diffusion steps")
    parser.add_argument("--guidance-scale", type=float, default=3.5, dest="guidance", help="CFG guidance scale")
    parser.add_argument(
        "--short-side",
        type=int,
        choices=(512, 768),
        default=768,
        help="Resize the input short side before full-resolution inference",
    )
    parser.add_argument(
        "--round-multiple",
        type=int,
        default=16,
        help="Pad images so height/width are divisible by this value",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for images when the input is a directory",
    )
    parser.add_argument(
        "--keep-padding",
        action="store_true",
        help="Keep padded pixels instead of cropping back to the original size",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
        help="Override model precision before inference",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples per input when --seeds is not specified",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Explicit random seeds to use; one output is generated per seed per image",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=None,
        help="Starting seed used to derive seeds for --num-samples when --seeds is omitted",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load model and checkpoint without saving outputs (useful for smoke tests)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    """Configure the root logger for CLI output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=level)


def load_config(config_path: Path) -> Dict:
    """Load the training configuration YAML."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model(network_cfg: Dict, device: str, precision: str) -> FluxKontext:
    """Instantiate the FluxKontext model from configuration."""
    # if network_cfg.get("type") != "FluxKontext":
    #     raise ValueError(f"Only FluxKontext networks are supported, got: {network_cfg.get('type')}")

    model_path = network_cfg.get("model_path")
    if model_path is None:
        raise ValueError("`model_path` must be provided in the network config")
    if network_cfg.get("type") == "FluxKontext":
        model = FluxKontext(
            model_path=model_path,
            train_dit=network_cfg.get("train_dit", False),
            vae_path=network_cfg.get("vae_path"),
        )
    else:
        if FluxKontextText is None:
            raise ValueError("FluxKontextText is not available in this release.")
        model = FluxKontextText(model_path=model_path, train_dit=network_cfg.get("train_dit", False))

    model_device = torch.device(device)
    if precision == "auto":
        return model.to(device=model_device)

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[precision]
    return model.to(device=model_device, dtype=dtype)


def find_zero_checkpoint(checkpoint_root: Path) -> Path:
    """Resolve the actual tensor file inside a DeepSpeed ZeRO checkpoint structure."""
    if checkpoint_root.is_file():
        return checkpoint_root

    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_root}")

    # Typical Lightning+DeepSpeed layout: last.ckpt/checkpoint/mp_rank_00_model_states.pt
    candidate = checkpoint_root / "checkpoint" / "mp_rank_00_model_states.pt"
    if candidate.is_file():
        return candidate

    # Allow passing the inner "checkpoint" directory directly.
    candidate = checkpoint_root / "mp_rank_00_model_states.pt"
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        "Could not locate mp_rank_00_model_states.pt under the provided checkpoint path"
    )


def load_deepspeed_weights(model: FluxKontext, checkpoint_path: Path) -> None:
    """Load LoRA weights from a DeepSpeed ZeRO Stage 2 checkpoint into the model."""
    tensor_path = find_zero_checkpoint(checkpoint_path)
    LOGGER.info("Loading ZeRO checkpoint from %s", tensor_path)
    raw_state = torch.load(tensor_path, map_location="cpu")
    module_state: Dict[str, torch.Tensor] = raw_state.get("module")
    if module_state is None:
        raise KeyError("Checkpoint is missing the 'module' state dict")

    # Remove the Lightning prefix so it matches the FluxKontext state dict.
    cleaned_state = {key[len("net_g."):]: value for key, value in module_state.items() if key.startswith("net_g.")}

    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
    if missing:
        LOGGER.warning("Missing keys when loading checkpoint: %s", missing)
    if unexpected:
        LOGGER.warning("Unexpected keys when loading checkpoint: %s", unexpected)


def iter_images(input_path: Path, recursive: bool) -> Iterable[Path]:
    """Yield image files from a path (file or directory)."""
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path}")
        yield input_path
        return

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    pattern = "**/*" if recursive else "*"
    for candidate in sorted(input_path.glob(pattern)):
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield candidate


def pad_for_model(image: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad the tensor image so height/width are divisible by ``multiple``."""
    if multiple <= 0:
        raise ValueError("round_multiple must be > 0")

    height, width = image.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return image, (0, 0)

    padded = F.pad(image.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect").squeeze(0)
    return padded, (pad_h, pad_w)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a [C,H,W] tensor in [0,1] to a PIL image."""
    # tensor = tensor.clamp(-1.0, 1.0) / 2.0 + 0.5
    tensor = tensor.clamp(0.0, 1.0)
    array = tensor.mul(255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


def run_inference(args: argparse.Namespace) -> None:
    """Main inference loop."""
    setup_logging(args.verbose)

    config = load_config(args.config)
    network_cfg = config.get("network_g")
    if network_cfg is None:
        raise KeyError("network_g section is required in the config")

    device = torch.device(args.device)
    model = build_model(network_cfg, args.device, args.precision)
    model.eval()

    load_deepspeed_weights(model, args.checkpoint)

    image_paths = list(iter_images(args.input, args.recursive))
    if not image_paths:
        LOGGER.warning("No input images found under %s", args.input)
        return

    if args.seeds:
        seed_values = args.seeds
    else:
        if args.num_samples < 1:
            raise ValueError("--num-samples must be >= 1")
        if args.num_samples == 1 and args.base_seed is None:
            seed_values = [None]
        else:
            base_seed = args.base_seed or 0
            seed_values = [base_seed + idx for idx in range(args.num_samples)]

    args.output.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        size = args.short_side
        LOGGER.info("Processing %s", image_path)
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            # Resize by short side while preserving aspect ratio.
            if img.width < img.height:
                img = img.resize((size, int(size * img.height / img.width)))
            else:
                img = img.resize((int(size * img.width / img.height), size))
            tensor = TF.to_tensor(img)
        # tensor = (tensor - 0.5) / 0.5
        original_size = tensor.shape[-2:]
        padded_tensor, padding = pad_for_model(tensor, args.round_multiple)
        batch_cpu = padded_tensor.unsqueeze(0)
        batch_device = batch_cpu.to(device)

        for sample_idx, seed in enumerate(seed_values):
            generator = None
            if seed is not None:
                generator = torch.Generator(device=device)
                generator.manual_seed(int(seed))
            LOGGER.info("\tseed=%s", "default" if seed is None else seed)

            with torch.inference_mode():
                output = model(
                    image=batch_device,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    generator=generator,
                )
            if isinstance(output, tuple):
                output_tensor = output[0]
            else:
                output_tensor = output

            output_tensor = output_tensor.squeeze(0).detach().cpu()
            if not args.keep_padding and any(padding):
                h, w = original_size
                output_tensor = output_tensor[..., :h, :w]

            output_image = tensor_to_image(output_tensor)

            if args.dry_run:
                LOGGER.debug(
                    "Dry run enabled; skipping save for %s (seed=%s)",
                    image_path,
                    "default" if seed is None else seed,
                )
                continue

            if seed is None:
                suffix = f"_sample{sample_idx:02d}" if len(seed_values) > 1 else ""
            else:
                suffix = f"_seed{seed}"

            out_path = args.output / f"{image_path.stem}{suffix}_clean.png"
            output_image.save(out_path)
            LOGGER.info("\tSaved %s", out_path)


def main() -> None:
    """Entry point."""
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
