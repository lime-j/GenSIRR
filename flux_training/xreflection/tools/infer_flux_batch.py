#!/usr/bin/env python3
"""Batch multi-seed Flux inference.

Runs the Flux manual sampler once per batch of seeds so that all seeds in the batch
share the same diffusion trajectory (executed in parallel on the GPU)."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from PIL import Image
import torchvision.transforms.functional as TF

from xreflection.tools.infer_flux import (
    FluxKontext,
    build_model,
    load_config,
    load_deepspeed_weights,
    pad_for_model,
    iter_images,
)
from xreflection.tools import infer_flux_manual as manual

LOGGER = logging.getLogger("flux_batch")
SUPPORTED_EXTENSIONS = manual.SUPPORTED_EXTENSIONS
DEFAULT_METRICS = ("mean", "std")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch multi-seed Flux inference")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="Seeds to evaluate in batches")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of seeds processed simultaneously")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--round-multiple", type=int, default=16)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")

    parser.add_argument("--shared-prefix-steps", type=int, default=0)
    parser.add_argument("--branch-noise-scale", type=float, default=0.01)
    parser.add_argument("--coarse-steps", type=int, default=0)

    parser.add_argument("--log-metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--print-every", type=int, default=0)

    parser.add_argument("--keep-padding", action="store_true")
    parser.add_argument("--save-reflection", action="store_true")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=level)


def chunked(seq: Sequence[int], size: int) -> Iterable[List[int]]:
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.coarse_steps < 0:
        raise ValueError("--coarse-steps must be >= 0")

    setup_logging(args.verbose)
    output_root = args.output
    output_root.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    network_cfg = cfg["network_g"]

    device = torch.device(args.device)
    model: FluxKontext = build_model(network_cfg, args.device, args.precision)
    model.eval()
    load_deepspeed_weights(model, args.checkpoint)

    seed_values = args.seeds

    for image_path in iter_images(args.input, args.recursive):
        LOGGER.info("Processing %s", image_path)
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            tensor = TF.to_tensor(img)
        tensor = (tensor - 0.5) / 0.5
        original_size = tensor.shape[-2:]
        pad_multiple = max(args.round_multiple, 64)
        padded_tensor, padding = pad_for_model(tensor, pad_multiple)

        shared_prefix_cache = None
        batched_outputs = []

        for seeds_chunk in chunked(seed_values, args.batch_size):
            batch_tensor = padded_tensor.unsqueeze(0).repeat(len(seeds_chunk), 1, 1, 1)
            batch_device = batch_tensor.to(device)

            generators = [torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds_chunk]

            latents, _, prefix_cache_out = manual.manual_sample(
                model,
                batch_device,
                guidance_scale=args.guidance_scale,
                steps=args.steps,
                generator=generators,
                metrics=args.log_metrics,
                print_every=args.print_every,
                shared_prefix_cache=shared_prefix_cache,
                shared_prefix_steps=args.shared_prefix_steps,
                branch_noise_scale=args.branch_noise_scale,
                coarse_steps=args.coarse_steps,
            )

            if shared_prefix_cache is None and prefix_cache_out is not None:
                shared_prefix_cache = prefix_cache_out

            batched_outputs.append((latents.detach().cpu(), list(seeds_chunk)))
            del batch_device, latents
            torch.cuda.empty_cache()

        mixture_tensor = padded_tensor.clone()

        for latents_cpu, seeds_chunk in batched_outputs:
            for latent_tensor, seed in zip(latents_cpu, seeds_chunk):
                latent_tensor = latent_tensor.unsqueeze(0).to(device=device, dtype=model.transformer.dtype)
                with torch.inference_mode():
                    decoded = manual.decode_latents(model, latent_tensor, padded_tensor.shape[-2], padded_tensor.shape[-1])
                decoded = decoded.squeeze(0).detach().cpu()

                save_tensor = decoded
                if not args.keep_padding and any(padding):
                    h, w = original_size
                    save_tensor = save_tensor[..., :h, :w]

                output_image = manual.tensor_to_image(save_tensor)
                output_path = output_root / f"{image_path.stem}_seed{seed}_clean.png"
                output_image.save(output_path)
                LOGGER.info("Saved %s", output_path)

                if args.save_reflection:
                    reflection_tensor = mixture_tensor - decoded
                    if not args.keep_padding and any(padding):
                        h, w = original_size
                        reflection_tensor = reflection_tensor[..., :h, :w]
                    reflection_image = manual.tensor_to_image(reflection_tensor)
                    reflection_path = output_root / f"{image_path.stem}_seed{seed}_reflection.png"
                    reflection_image.save(reflection_path)

                torch.cuda.empty_cache()

        del shared_prefix_cache
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
