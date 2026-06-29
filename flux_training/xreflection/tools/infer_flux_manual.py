#!/usr/bin/env python3
"""Flux manual sampling script.

Re-implements the FluxKontext sampling loop in Python so that every step can be
instrumented without relying on the pipeline callback API.  Supports optional
multi-GPU inference via ``--devices`` for large batch runs.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from PIL import Image
import torchvision.transforms.functional as TF
from diffusers import AutoencoderKL
# Ensure repository root on sys.path for local imports.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from xreflection.tools.infer_flux import (  # noqa: E402
        FluxKontext,
        build_model,
        iter_images,
        load_config,
        load_deepspeed_weights,
        pad_for_model,
    )
    from xreflection.archs.flux_arch import (  # noqa: E402
        calculate_shift,
        retrieve_timesteps,
        retrieve_latents,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - defensive
    missing = exc.name or "an optional dependency"
    raise ModuleNotFoundError(
        f"Missing dependency '{missing}'. Install project requirements before running manual inference."
    ) from exc

def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a [C,H,W] tensor in [-1,1] to a PIL image."""
    tensor = tensor.clamp(-1.0, 1.0) * 0.5 + 0.5
    array = tensor.mul(255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


def encode_tensor_to_latents(model: FluxKontext, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Encode a normalized image tensor ([C,H,W] in [-1,1]) into VAE latents."""
    batch = tensor.unsqueeze(0).to(device=device, dtype=model.transformer.dtype)
    with torch.no_grad():
        encoded = model.vae.encode(batch)
    latents = retrieve_latents(encoded, generator=None, sample_mode="argmax")
    latents = (latents - model.vae.config.shift_factor) * model.vae.config.scaling_factor
    return latents.to(torch.float32)


def pack_latents(model: FluxKontext, latents: torch.Tensor) -> torch.Tensor:
    """Pack latents to the shape expected by the Flux transformer."""
    b, c, h, w = latents.shape
    return model._pack_latents(latents, b, c, h, w)

LOGGER = logging.getLogger("flux_manual")
SUPPORTED_EXTENSIONS: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
METRIC_CHOICES = ("mean", "std", "min", "max", "norm")


class TrajectoryRecorder:
    """Utility to persist intermediate denoising frames/tensors."""

    def __init__(
        self,
        model: FluxKontext,
        save_dir: Path,
        height: int,
        width: int,
        original_size: Tuple[int, int],
        keep_padding: bool,
        frequency: int,
        image_format: str,
        save_images: bool,
        save_tensors: bool,
    ) -> None:
        self.model = model
        self.save_dir = save_dir
        self.height = height
        self.width = width
        self.original_size = original_size
        self.keep_padding = keep_padding
        self.frequency = max(1, frequency)
        self.image_format = image_format
        self.save_images = save_images
        self.save_tensors = save_tensors
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        latents: torch.Tensor,
        step_idx: int,
        final_step: bool,
        timestep: torch.Tensor,
    ) -> None:
        if not (self.save_images or self.save_tensors):
            return

        if (step_idx % self.frequency != 0) and not final_step:
            return

        target_dtype = getattr(self.model.vae, "dtype", latents.dtype)
        latents_to_decode = latents.detach().to(dtype=target_dtype)
        with torch.inference_mode():
            image_tensor = decode_latents(self.model, latents_to_decode, self.height, self.width)

        image_tensor = image_tensor.detach().to(dtype=torch.float32)
        output_tensor = image_tensor.squeeze(0)
        if not self.keep_padding:
            h, w = self.original_size
            output_tensor = output_tensor[..., :h, :w]

        if self.save_images:
            image = tensor_to_image(output_tensor.clone())
            filename = self.save_dir / f"step_{step_idx:03d}.{self.image_format}"
            image.save(filename)

        if self.save_tensors:
            tensor_path = self.save_dir / f"step_{step_idx:03d}.pt"
            payload = {
                "latents": latents_to_decode.float().cpu(),
                "decoded": output_tensor.clone().cpu(),
                "timestep": float(timestep.item()) if hasattr(timestep, "item") else float(timestep),
            }
            torch.save(payload, tensor_path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Flux reflection removal with a manual sampling loop",
    )
    parser.add_argument("--config", required=True, type=Path, help="Training YAML config")
    parser.add_argument("--checkpoint", required=True, type=Path, help="DeepSpeed ZeRO checkpoint")
    parser.add_argument("--input", required=True, type=Path, help="Input image or directory")
    parser.add_argument("--output", default=Path("inference_results"), type=Path, help="Output directory")
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        help="Single-device inference target (e.g., 'cuda', 'cuda:1', or 'cpu')",
    )
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        help="List of CUDA device indices for multi-GPU inference",
    )
    parser.add_argument("--vae-path", type=Path, default="vae_merged", help="Path to the VAE checkpoint")
    parser.add_argument("--steps", type=int, default=28, help="Number of diffusion steps")
    parser.add_argument("--guidance-scale", type=float, default=3.5, dest="guidance", help="CFG guidance scale")
    parser.add_argument("--round-multiple", type=int, default=16, help="Pad so H/W divisible by this value")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories when input is a folder")
    parser.add_argument("--keep-padding", action="store_true", help="Skip cropping back to the original resolution")

    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seeds", type=int, nargs="+", help="Explicit seeds to evaluate per image")
    seed_group.add_argument("--num-samples", type=int, default=1, help="Number of samples when --seeds not provided")
    parser.add_argument("--base-seed", type=int, default=None, help="Starting seed when generating samples automatically")

    parser.add_argument(
        "--log-metrics",
        nargs="+",
        choices=METRIC_CHOICES,
        default=["mean", "std"],
        help="Per-step metrics to compute for latent inputs and noise predictions",
    )
    parser.add_argument("--log-file", type=Path, help="Optional JSONL to store per-step diagnostics")
    parser.add_argument("--print-every", type=int, default=1, help="Print diagnostics every N steps")
    parser.add_argument("--no-save", action="store_true", help="Skip writing cleaned outputs")
    parser.add_argument("--save-trajectory", action="store_true", help="Save intermediate denoising frames")
    parser.add_argument("--save-tensors", action="store_true", help="Persist input/trajectory tensors to .pt files")
    parser.add_argument("--shared-prefix-steps", type=int, default=0, help="Number of deterministic steps shared by all seeds before branching")
    parser.add_argument("--branch-noise-scale", type=float, default=0.01, help="Stddev of Gaussian noise applied when seeds branch from the shared prefix")
    parser.add_argument("--coarse-steps", type=int, default=0, help="Run the first N steps on a downsampled latent grid")
    parser.add_argument("--save-reflection", action="store_true", help="Save the inferred reflection layer alongside the selected transmission image")
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        help="Root directory for trajectory outputs (defaults to <output>/trajectories)",
    )
    parser.add_argument(
        "--trajectory-frequency",
        type=int,
        default=1,
        help="Save every N diffusion steps when --save-trajectory is enabled",
    )
    parser.add_argument(
        "--trajectory-format",
        choices=("png", "jpg", "jpeg", "webp"),
        default="png",
        help="Image format for saved trajectory frames",
    )
    parser.add_argument(
        "--trajectory-keep-padding",
        action="store_true",
        help="Keep padded pixels in trajectory frames even if final outputs are cropped",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging output")

    return parser.parse_args(argv)


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=level)


def compute_metrics(tensor: torch.Tensor, metrics: Sequence[str]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if tensor.numel() == 0:
        return {metric: float("nan") for metric in metrics}
    tensor_f32 = tensor.float()
    for metric in metrics:
        if metric == "mean":
            result["mean"] = tensor_f32.mean().item()
        elif metric == "std":
            result["std"] = tensor_f32.std(unbiased=False).item()
        elif metric == "min":
            result["min"] = tensor_f32.min().item()
        elif metric == "max":
            result["max"] = tensor_f32.max().item()
        elif metric == "norm":
            result["norm"] = tensor_f32.norm().item()
    return result


class StepLog(List[Dict[str, object]]):
    """Per-step log container that tracks aggregate timings without storing tensors."""

    __slots__ = ("elapsed", "steps", "final_record")

    def __init__(self) -> None:
        super().__init__()
        self.elapsed: float = 0.0
        self.steps: int = 0
        self.final_record: Optional[Dict[str, object]] = None


def select_latent_from_seeds(
    latents: Sequence[torch.Tensor],
    mixture_latent: torch.Tensor,
    cluster_ratio: float = 1.5,
) -> Tuple[torch.Tensor, int, Dict[int, int]]:
    """Per-element majority vote (patch size = 1) across seed latents."""

    if not latents:
        return mixture_latent.clone(), -1, {}

    latent_stack = torch.stack(latents, dim=0)  # [n, C, H, W]
    n = latent_stack.shape[0]
    if n == 1:
        total_votes = latent_stack[0].numel()
        return latent_stack[0].clone(), 0, {0: total_votes}

    flat = latent_stack.view(n, -1)
    median_flat = flat.median(dim=0).values
    diff = torch.abs(flat - median_flat)
    winner_idx = diff.argmin(dim=0)  # [num_elements]

    arange = torch.arange(flat.shape[1], device=flat.device)
    chosen_flat = flat[winner_idx, arange]
    chosen = chosen_flat.view_as(latent_stack[0])

    counts = torch.bincount(winner_idx, minlength=n)
    votes = {int(i): int(counts[i].item()) for i in range(n) if counts[i] > 0}
    best_seed = int(torch.argmax(counts).item()) if counts.numel() > 0 else -1
    return chosen, best_seed, votes


def resolve_seeds(args: argparse.Namespace) -> List[Optional[int]]:
    if args.seeds:
        return [int(seed) for seed in args.seeds]
    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")
    if args.num_samples == 1 and args.base_seed is None:
        return [None]
    base = args.base_seed or 0
    return [base + offset for offset in range(args.num_samples)]


def write_record(handle, record: Dict) -> None:
    if handle is None:
        return
    handle.write(json.dumps(record))
    handle.write("\n")
    handle.flush()


def iter_ordered_images(path: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    if path.is_file():
        yield path
    else:
        for candidate in sorted(path.glob(pattern)):
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield candidate


def manual_sample(
    model: FluxKontext,
    batch: torch.Tensor,
    guidance_scale: float,
    steps: int,
    generator: Optional[torch.Generator],
    metrics: Sequence[str],
    print_every: int,
    trajectory_recorder: Optional[TrajectoryRecorder] = None,
    shared_prefix_cache: Optional[Dict[str, torch.Tensor]] = None,
    shared_prefix_steps: int = 0,
    branch_noise_scale: float = 0.0,
    coarse_steps: int = 0,
    *,
    record_consumer: Optional[Callable[[Dict[str, object]], None]] = None,
    collect_records: bool = True,
) -> Tuple[torch.Tensor, StepLog, Optional[Dict[str, torch.Tensor]]]:
    """Run the rectified-flow sampler, optionally reusing a shared prefix."""

    device = batch.device
    dtype = model.transformer.dtype

    prompt = None
    batch_size = batch.shape[0]
    full_height = batch.shape[2]
    full_width = batch.shape[3]

    prompt_embeds = model.cached_prompt_embeds.detach().to(device=device, dtype=dtype)
    pooled_prompt_embeds = model.cached_pooled_prompt_embeds.detach().to(device=device, dtype=dtype)
    text_ids = model.cached_text_ids.detach().to(device=device, dtype=dtype)

    if model.text_encoder_2 is not None:
        model.text_encoder.max_position_embeddings = 77
        model.text_encoder_2.max_position_embeddings = 512

    model.check_inputs(
        prompt=prompt,
        prompt_2=prompt,
        height=batch.shape[2],
        width=batch.shape[3],
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    )
    num_images_per_prompt = 1
    latents_channels = model.transformer.config.in_channels // 4
    latents, image_latents, latent_ids, image_ids = model.prepare_latents(
        image=batch,
        batch_size=batch_size * num_images_per_prompt,
        num_channels_latents=latents_channels,
        height=batch.shape[2],
        width=batch.shape[3],
        dtype=dtype,
        device=device,
        generator=generator,
        latents=None,
    )
    if image_ids is not None:
        latent_ids = torch.cat([latent_ids, image_ids], dim=0)

    full_latent_ids = latent_ids.clone()
    full_image_latents = image_latents.clone() if image_latents is not None else None

    coarse_active = False
    coarse_transition_step = 0
    coarse_info: Dict[str, object] = {}

    if coarse_steps > 0 and shared_prefix_cache is None:
        latents_unpacked_full = model._unpack_latents(
            latents, full_height, full_width, model.vae_scale_factor
        )
        full_latent_h = latents_unpacked_full.shape[2]
        full_latent_w = latents_unpacked_full.shape[3]

        coarse_latent_h = max(2, full_latent_h // 2)
        coarse_latent_w = max(2, full_latent_w // 2)
        coarse_latent_h = max(2, (coarse_latent_h // 2) * 2)
        coarse_latent_w = max(2, (coarse_latent_w // 2) * 2)

        if coarse_latent_h < full_latent_h or coarse_latent_w < full_latent_w:
            coarse_active = True
            coarse_transition_step = coarse_steps
            coarse_info = {
                "coarse_h": coarse_latent_h,
                "coarse_w": coarse_latent_w,
                "full_h": full_latent_h,
                "full_w": full_latent_w,
                "full_latent_ids": full_latent_ids.clone().cpu(),
                "full_image_latents": full_image_latents.clone().cpu() if full_image_latents is not None else None,
            }
            latents_coarse_unpacked = F.interpolate(
                latents_unpacked_full,
                size=(coarse_latent_h, coarse_latent_w),
                mode="bilinear",
                align_corners=False,
            )
            latents = model._pack_latents(
                latents_coarse_unpacked,
                batch_size,
                latents_coarse_unpacked.shape[1],
                coarse_latent_h,
                coarse_latent_w,
            )
            latent_ids = model._prepare_latent_image_ids(
                batch_size,
                coarse_latent_h // 2,
                coarse_latent_w // 2,
                device,
                latents.dtype,
            )
            if image_latents is not None:
                image_unpacked_full = model._unpack_latents(
                    image_latents, full_height, full_width, model.vae_scale_factor
                )
                image_coarse_unpacked = F.interpolate(
                    image_unpacked_full,
                    size=(coarse_latent_h, coarse_latent_w),
                    mode="bilinear",
                    align_corners=False,
                )
                image_latents = model._pack_latents(
                    image_coarse_unpacked,
                    batch_size,
                    image_coarse_unpacked.shape[1],
                    coarse_latent_h,
                    coarse_latent_w,
                )
                image_ids_coarse = model._prepare_latent_image_ids(
                    batch_size,
                    coarse_latent_h // 2,
                    coarse_latent_w // 2,
                    device,
                    latents.dtype,
                )
                latent_ids = torch.cat([latent_ids, image_ids_coarse], dim=0)

        del latents_unpacked_full
        torch.cuda.empty_cache()

    mu = calculate_shift(
        latents.shape[1],
        model.scheduler.config.get("base_image_seq_len", 256),
        model.scheduler.config.get("max_image_seq_len", 4096),
        model.scheduler.config.get("base_shift", 0.5),
        model.scheduler.config.get("max_shift", 1.15),
    )

    timesteps, _ = retrieve_timesteps(
        model.scheduler,
        num_inference_steps=steps,
        device=device,
        timesteps=None,
        sigmas=None,
        mu=mu,
    )

    total_steps = len(timesteps)
    if coarse_active:
        coarse_transition_step = min(coarse_transition_step, total_steps)
        if coarse_transition_step == 0:
            coarse_active = False
    prefix_cache_out: Optional[Dict[str, torch.Tensor]] = None
    target_prefix_step = min(shared_prefix_steps, total_steps) if shared_prefix_steps > 0 else 0

    if model.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32).expand(latents.shape[0])
    else:
        guidance = None

    joint_kwargs = model.joint_attention_kwargs or {}
    start_step = 0
    if shared_prefix_cache is not None:
        start_step = int(shared_prefix_cache.get("start_step", 0))
        latents = shared_prefix_cache["latents"].to(device=device, dtype=dtype).clone()
        cached_image_latents = shared_prefix_cache.get("image_latents")
        if cached_image_latents is not None:
            image_latents = cached_image_latents.to(device=device, dtype=dtype).clone()
        latent_ids = shared_prefix_cache["latent_ids"].to(device=device, dtype=latent_ids.dtype).clone()
        if branch_noise_scale > 0:
            if isinstance(generator, list):
                noises = []
                for idx, gen in enumerate(generator):
                    shape = latents[idx : idx + 1].shape
                    noises.append(
                        torch.randn(
                            shape,
                            device=latents.device,
                            dtype=latents.dtype,
                            generator=gen,
                        )
                    )
                noise = torch.cat(noises, dim=0)
            elif isinstance(generator, torch.Generator):
                noise = torch.randn(
                    latents.shape,
                    device=latents.device,
                    dtype=latents.dtype,
                    generator=generator,
                )
            else:
                noise = torch.randn_like(latents)
            latents = latents + noise * branch_noise_scale
    else:
        start_step = 0

    model.scheduler.set_begin_index(start_step)

    records = StepLog()
    last_step_time = time.time()

    for step_idx, timestep in enumerate(timesteps[start_step:], start=start_step):
        latent_model_input = latents
        if image_latents is not None:
            latent_model_input = torch.cat([latents, image_latents], dim=1)

        noise_pred = model.transformer(
            hidden_states=latent_model_input,
            timestep=timestep.expand(latent_model_input.shape[0]).to(latent_model_input.dtype) / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_ids,
            joint_attention_kwargs=joint_kwargs,
            return_dict=False,
        )[0][:, : latents.size(1)]

        latents = model.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

        if coarse_active and (step_idx + 1) == coarse_transition_step:
            coarse_h = int(coarse_info.get("coarse_h", 0))
            coarse_w = int(coarse_info.get("coarse_w", 0))
            full_h = int(coarse_info.get("full_h", full_height))
            full_w = int(coarse_info.get("full_w", full_width))
            latents_unpacked = model._unpack_latents(
                latents,
                coarse_h,
                coarse_w,
                model.vae_scale_factor,
            )
            latents_unpacked = F.interpolate(
                latents_unpacked,
                size=(full_h, full_w),
                mode="bilinear",
                align_corners=False,
            )
            latents = model._pack_latents(
                latents_unpacked,
                batch_size,
                latents_unpacked.shape[1],
                full_h,
                full_w,
            )
            latent_ids = coarse_info["full_latent_ids"].to(device=device, dtype=latents.dtype)
            if coarse_info.get("full_image_latents") is not None:
                image_latents = coarse_info["full_image_latents"].to(device=device, dtype=latents.dtype)
            else:
                image_latents = full_image_latents.to(device=device, dtype=latents.dtype) if full_image_latents is not None else None
            coarse_active = False

        now = time.time()
        delta = now - last_step_time
        last_step_time = now

        if (
            shared_prefix_cache is None
            and target_prefix_step > 0
            and prefix_cache_out is None
            and (step_idx + 1) == target_prefix_step
        ):
            prefix_cache_out = {
                "latents": latents.detach().cpu().clone(),
                "image_latents": image_latents.detach().cpu().clone() if image_latents is not None else None,
                "latent_ids": latent_ids.detach().cpu().clone(),
                "start_step": target_prefix_step,
            }

        record: Dict[str, object] = {
            "step": int(step_idx),
            "timestep": float(timestep.item()) if hasattr(timestep, "item") else float(timestep),
            "latent_shape": list(latent_model_input.shape),
            "latent_dtype": str(latent_model_input.dtype),
            "noise_shape": list(noise_pred.shape),
            "noise_dtype": str(noise_pred.dtype),
            "step_time": delta,
            "total_steps": total_steps,
        }
        record.update({f"latent_{k}": v for k, v in compute_metrics(latent_model_input, metrics).items()})
        record.update({f"noise_{k}": v for k, v in compute_metrics(noise_pred, metrics).items()})
        record_copy = dict(record)
        records.elapsed += delta
        records.steps += 1
        records.final_record = record_copy

        if record_consumer is not None:
            record_consumer(record_copy)
        if collect_records:
            records.append(record_copy)

        final_step = step_idx == (total_steps - 1)

        if trajectory_recorder is not None and not coarse_active:
            trajectory_recorder.record(latents, step_idx, final_step, timestep)

        if print_every and (step_idx % print_every == 0):
            metric_parts = []
            for metric in metrics:
                latent_key = f"latent_{metric}"
                noise_key = f"noise_{metric}"
                if latent_key in record:
                    metric_parts.append(f"latent_{metric}={record[latent_key]:.4f}")
                if noise_key in record:
                    metric_parts.append(f"noise_{metric}={record[noise_key]:.4f}")
            LOGGER.info(
                "\t  step %03d/%d | t=%.3f | Δt=%.3fs | %s",
                step_idx,
                total_steps,
                record["timestep"],
                delta,
                ", ".join(metric_parts) if metric_parts else "no metrics",
            )

    return latents, records, prefix_cache_out


def decode_latents(model: FluxKontext, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    latents = model._unpack_latents(latents, height, width, model.vae_scale_factor)
    latents = (latents / model.vae.config.scaling_factor) + model.vae.config.shift_factor
    image = model.vae.decode(latents, return_dict=False)[0]
    return image


def worker_main(
    args: argparse.Namespace,
    device_id: str,
    image_paths: List[Path],
    worker_rank: int = 0,
) -> None:
    setup_logging(args.verbose)

    config = load_config(args.config)
    network_cfg = config.get("network_g")
    if network_cfg is None:
        raise KeyError("network_g section missing in config")

    device = torch.device(device_id)
    if device.type == "cuda":
        if device.index is None:
            torch.cuda.set_device(0)
        else:
            torch.cuda.set_device(device.index)
    model: FluxKontext = build_model(network_cfg, device_id, args.precision)
    model.eval()
    load_deepspeed_weights(model, args.checkpoint)
    model.vae = AutoencoderKL.from_pretrained(args.vae_path, torch_dtype=torch.bfloat16, device=model.transformer.device).to(model.transformer.device)
    seed_values = resolve_seeds(args)
    if not image_paths:
        LOGGER.info("Worker %d: no assigned images, exiting", worker_rank)
        return

    args.output.mkdir(parents=True, exist_ok=True)
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = args.log_file.open("a", encoding="utf-8")
    else:
        log_handle = None

    trajectory_root = None
    if args.save_trajectory or args.save_tensors or args.save_reflection:
        trajectory_root = args.trajectory_dir or (args.output / "trajectories")
        trajectory_root.mkdir(parents=True, exist_ok=True)

    try:
        for image_path in image_paths:
            LOGGER.info("Worker %d: processing %s", worker_rank, image_path)
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                if img.width < img.height:
                    img = img.resize((512, int(512 * img.height / img.width)))
                else:
                    img = img.resize((int(512 * img.width / img.height), 512))
            tensor = TF.to_tensor(img)
            tensor = (tensor - 0.5) / 0.5
            original_size = tensor.shape[-2:]
            pad_multiple = max(args.round_multiple, 64)
            padded_tensor, padding = pad_for_model(tensor, pad_multiple)
            batch_device = padded_tensor.unsqueeze(0).to(device)
            LOGGER.info(
                "Worker %d:\tinput tensor shape=%s -> padded=%s | dtype=%s",
                worker_rank,
                list(tensor.shape),
                list(batch_device.shape),
                str(batch_device.dtype),
            )

            if trajectory_root is not None and args.save_tensors:
                input_dir = trajectory_root / image_path.stem
                input_dir.mkdir(parents=True, exist_ok=True)
                input_tensor_path = input_dir / "input.pt"
                if not input_tensor_path.exists():
                    torch.save(
                        {
                            "original": tensor.clone().cpu(),
                            "padded": padded_tensor.clone().cpu(),
                            "original_size": original_size,
                            "padded_shape": list(padded_tensor.shape),
                        },
                        input_tensor_path,
                    )

            mix_latents_unpacked = encode_tensor_to_latents(model, padded_tensor, device)
            mix_latents_packed = pack_latents(model, mix_latents_unpacked).detach().cpu().float()
            mix_latent_flat = mix_latents_packed.view(-1)

            shared_prefix_cache = None
            final_latents_packed: List[torch.Tensor] = []
            seed_ids: List[str] = []

            for sample_idx, seed in enumerate(seed_values):
                generator = None
                if seed is not None:
                    generator = torch.Generator(device=device)
                    generator.manual_seed(int(seed))

                LOGGER.info(
                    "Worker %d:\tseed=%s | steps=%d | guidance=%.2f",
                    worker_rank,
                    "default" if seed is None else seed,
                    args.steps,
                    args.guidance,
                )

                trajectory_recorder = None
                if trajectory_root is not None:
                    seed_name = f"seed_{seed}" if seed is not None else f"sample_{sample_idx:02d}"
                    traj_dir = trajectory_root / image_path.stem / seed_name
                    traj_keep_padding = args.keep_padding or args.trajectory_keep_padding
                    trajectory_recorder = TrajectoryRecorder(
                        model=model,
                        save_dir=traj_dir,
                        height=batch_device.shape[2],
                        width=batch_device.shape[3],
                        original_size=original_size,
                        keep_padding=traj_keep_padding,
                        frequency=args.trajectory_frequency,
                        image_format=args.trajectory_format,
                        save_images=args.save_trajectory,
                        save_tensors=args.save_tensors,
                    )

                record_consumer = None
                if log_handle is not None:
                    def record_consumer(entry: Dict[str, object], *,
                                        _handle=log_handle,
                                        _image=image_path.name,
                                        _seed=seed,
                                        _sample_idx=sample_idx,
                                        _worker=worker_rank) -> None:
                        payload = dict(entry)
                        payload.update(
                            {
                                "image": _image,
                                "seed": _seed,
                                "sample_index": _sample_idx,
                                "worker": _worker,
                            }
                        )
                        write_record(_handle, payload)

                latents, step_records, prefix_cache_out = manual_sample(
                    model,
                    batch_device,
                    guidance_scale=args.guidance,
                    steps=args.steps,
                    generator=generator,
                    metrics=args.log_metrics,
                    print_every=args.print_every,
                    trajectory_recorder=trajectory_recorder,
                    shared_prefix_cache=shared_prefix_cache,
                    shared_prefix_steps=args.shared_prefix_steps,
                    branch_noise_scale=args.branch_noise_scale,
                    coarse_steps=args.coarse_steps,
                    record_consumer=record_consumer,
                    collect_records=False,
                )

                if shared_prefix_cache is None and prefix_cache_out is not None:
                    shared_prefix_cache = prefix_cache_out

                final_latent_cpu = latents.detach().cpu().float()
                final_latents_packed.append(final_latent_cpu)
                seed_ids.append(str(seed) if seed is not None else f"sample_{sample_idx:02d}")

                with torch.inference_mode():
                    image_tensor = decode_latents(model, latents, batch_device.shape[2], batch_device.shape[3])

                elapsed = step_records.elapsed
                steps_taken = step_records.steps
                final_metrics = step_records.final_record or {}
                metric_parts = []
                for metric in args.log_metrics:
                    key = f"latent_{metric}"
                    if key in final_metrics:
                        metric_parts.append(f"{metric}={final_metrics[key]:.4f}")
                LOGGER.info(
                    "Worker %d:\t  sample finished in %.2fs (%d steps) | final %s",
                    worker_rank,
                    elapsed,
                    steps_taken,
                    ", ".join(metric_parts) if metric_parts else "no metrics",
                )

                if args.no_save:
                    continue

                output_tensor = image_tensor.squeeze(0).detach().cpu()
                if not args.keep_padding and any(padding):
                    h, w = original_size
                    output_tensor = output_tensor[..., :h, :w]

                output_image = tensor_to_image(output_tensor)
                suffix = (
                    f"_seed{seed}" if seed is not None else (
                        f"_sample{sample_idx:02d}" if len(seed_values) > 1 else ""
                    )
                )
                out_path = args.output / f"{image_path.stem}{suffix}_clean.png"
                output_image.save(out_path)
                LOGGER.info("Worker %d:\tSaved %s", worker_rank, out_path)

                # explicit cleanup before next seed to avoid accumulation
                del latents, image_tensor, step_records
                torch.cuda.empty_cache()
                batch_device = padded_tensor.unsqueeze(0).to(device)

            if final_latents_packed:
                selected_latent_cpu, selected_idx, vote_counts = select_latent_from_seeds(
                    final_latents_packed, mix_latents_packed
                )
                selected_latent = selected_latent_cpu.to(
                    device=batch_device.device, dtype=model.transformer.dtype
                )
                with torch.inference_mode():
                    selected_image_tensor = decode_latents(
                        model, selected_latent, batch_device.shape[2], batch_device.shape[3]
                    ).squeeze(0)

                selected_image_to_store = selected_image_tensor
                if not args.keep_padding and any(padding):
                    h, w = original_size
                    selected_image_to_store = selected_image_to_store[..., :h, :w]
                    # reflection_tensor = reflection_tensor[..., :h, :w]

                if not args.no_save:
                    selected_image = tensor_to_image(selected_image_to_store.cpu())
                    selected_path = args.output / f"{image_path.stem}_selected.png"
                    selected_image.save(selected_path)
                    LOGGER.info(
                        "Worker %d:\tSaved %s (majority seed: %s; votes=%d/%d)",
                        worker_rank,
                        selected_path,
                        seed_ids[selected_idx] if 0 <= selected_idx < len(seed_ids) else "?",
                        vote_counts.get(selected_idx, 0),
                        sum(vote_counts.values()),
                    )

                if args.save_reflection:
                    reflection_tensor = padded_tensor.clone().to(selected_image_tensor.device)
                    reflection_tensor = reflection_tensor - selected_image_tensor
                    if not args.keep_padding and any(padding):
                        h, w = original_size
                        reflection_tensor = reflection_tensor[..., :h, :w]
                    reflection_path = args.output / f"{image_path.stem}_reflection.png"
                    tensor_to_image(reflection_tensor.cpu()).save(reflection_path)
                    if args.save_tensors and trajectory_root is not None:
                        sel_dir = trajectory_root / image_path.stem
                        sel_dir.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            {"reflection_tensor": reflection_tensor.cpu()},
                            sel_dir / "reflection_tensor.pt",
                        )

                if args.save_tensors and trajectory_root is not None:
                    sel_dir = trajectory_root / image_path.stem
                    sel_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "selected_latent": selected_latent_cpu,
                            "selected_index": selected_idx,
                            "vote_counts": vote_counts,
                        },
                        sel_dir / "selected_latent.pt",
                    )

                del selected_latent, selected_image_tensor
                torch.cuda.empty_cache()
    finally:
        if log_handle is not None:
            log_handle.close()


def run_manual(args: argparse.Namespace) -> None:
    if args.shared_prefix_steps < 0:
        raise ValueError("--shared-prefix-steps must be >= 0")
    if args.branch_noise_scale < 0:
        raise ValueError("--branch-noise-scale must be >= 0")
    if args.coarse_steps < 0:
        raise ValueError("--coarse-steps must be >= 0")
    if args.coarse_steps > 0:
        pass
    if args.devices:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for multi-GPU inference")
        unique_devices = list(dict.fromkeys(args.devices))
        if not unique_devices:
            raise ValueError("--devices must contain at least one device index")
        device_strings = [f"cuda:{idx}" for idx in unique_devices]
        image_paths = list(iter_ordered_images(args.input, args.recursive))
        if not image_paths:
            setup_logging(args.verbose)
            LOGGER.warning("No images found under %s", args.input)
            return

        world_size = len(device_strings)
        assignments: List[List[Path]] = [[] for _ in range(world_size)]
        for idx, image_path in enumerate(image_paths):
            assignments[idx % world_size].append(image_path)

        available = torch.cuda.device_count()
        for dev in unique_devices:
            if dev < 0 or dev >= available:
                raise ValueError(f"CUDA device index {dev} is out of range (available: 0-{available - 1})")

        ctx = mp.get_context("spawn")
        processes = []
        for rank, device_id in enumerate(device_strings):
            p = ctx.Process(
                target=worker_main,
                args=(args, device_id, assignments[rank], rank),
                daemon=False,
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Worker process exited with code {p.exitcode}")
    else:
        worker_main(args, args.device, list(iter_ordered_images(args.input, args.recursive)))


def main() -> None:
    args = parse_args()
    run_manual(args)


if __name__ == "__main__":
    main()
