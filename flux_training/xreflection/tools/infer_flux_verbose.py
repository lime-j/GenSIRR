#!/usr/bin/env python3
"""Verbose Flux sampling script.

This utility mirrors ``infer_flux.py`` but adds detailed logging of the
diffusion sampling loop so that experiments on new sampling strategies are
easier to run and analyse.  Per-step latent statistics (captured before the
scheduler update) can be printed to the console and/or written to JSON Lines
for offline study.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
import torchvision.transforms.functional as TF

# Re-use helpers from the base inference script to avoid duplication.
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
        tensor_to_image,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - defensive
    missing = exc.name or "an optional dependency"
    raise ModuleNotFoundError(
        f"Missing dependency '{missing}'. Install project requirements before running verbose inference."
    ) from exc


LOGGER = logging.getLogger("flux_verbose")
SUPPORTED_EXTENSIONS: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
METRIC_CHOICES = ("mean", "std", "min", "max", "norm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Flux reflection removal with verbose sampling diagnostics"
    )
    parser.add_argument("--config", required=True, type=Path, help="Training YAML config")
    parser.add_argument("--checkpoint", required=True, type=Path, help="DeepSpeed ZeRO checkpoint")
    parser.add_argument("--input", required=True, type=Path, help="Image file or directory")
    parser.add_argument("--output", default=Path("inference_results"), type=Path, help="Output directory")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cuda", "cpu"),
        help="Inference device",
    )
    parser.add_argument("--steps", type=int, default=28, help="Number of diffusion steps")
    parser.add_argument("--guidance-scale", type=float, default=3.5, dest="guidance", help="CFG guidance scale")
    parser.add_argument("--round-multiple", type=int, default=16, help="Pad so H/W divisible by this value")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--recursive", action="store_true", help="Recurse into sub-dirs when input is a folder")
    parser.add_argument("--keep-padding", action="store_true", help="Skip cropping back to the original size")

    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Explicit seeds to run per image",
    )
    seed_group.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples when explicit seeds are not provided",
    )
    parser.add_argument("--base-seed", type=int, default=None, help="Base seed for --num-samples runs")

    parser.add_argument(
        "--log-metrics",
        nargs="+",
        choices=METRIC_CHOICES,
        default=["mean", "std"],
        help="Per-step metrics to compute for diffusion latents",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Optional JSONL output capturing per-step diagnostics",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Print diagnostics every N steps (1 prints all steps)",
    )
    parser.add_argument("--no-save", action="store_true", help="Skip writing cleaned images to disk")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=level)


def compute_metrics(tensor: torch.Tensor, metrics: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if tensor.numel() == 0:
        return {metric: float("nan") for metric in metrics}
    tensor_f32 = tensor.float()
    for metric in metrics:
        if metric == "mean":
            out["mean"] = tensor_f32.mean().item()
        elif metric == "std":
            out["std"] = tensor_f32.std(unbiased=False).item()
        elif metric == "min":
            out["min"] = tensor_f32.min().item()
        elif metric == "max":
            out["max"] = tensor_f32.max().item()
        elif metric == "norm":
            out["norm"] = tensor_f32.norm().item()
    return out


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
        for cand in sorted(path.glob(pattern)):
            if cand.is_file() and cand.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield cand


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    config = load_config(args.config)
    network_cfg = config.get("network_g")
    if network_cfg is None:
        raise KeyError("network_g section missing in config")

    device = torch.device(args.device)
    model: FluxKontext = build_model(network_cfg, args.device, args.precision)
    model.eval()
    load_deepspeed_weights(model, args.checkpoint)

    seed_values = resolve_seeds(args)
    image_paths = list(iter_ordered_images(args.input, args.recursive))
    if not image_paths:
        LOGGER.warning("No images found under %s", args.input)
        return

    args.output.mkdir(parents=True, exist_ok=True)
    log_handle = args.log_file.open("a", encoding="utf-8") if args.log_file else None

    try:
        for image_path in image_paths:
            LOGGER.info("Processing %s", image_path)
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                if img.width < img.height:
                    img = img.resize((512, int(512 * img.height / img.width)))
                else:
                    img = img.resize((int(512 * img.width / img.height), 512))
                tensor = TF.to_tensor(img)

            original_size = tensor.shape[-2:]
            padded_tensor, padding = pad_for_model(tensor, args.round_multiple)
            batch_device = padded_tensor.unsqueeze(0).to(device)
            LOGGER.info(
                "\tinput tensor shape=%s -> padded=%s | dtype=%s",
                list(tensor.shape),
                list(batch_device.shape),
                str(batch_device.dtype),
            )

            for sample_idx, seed in enumerate(seed_values):
                generator = None
                if seed is not None:
                    generator = torch.Generator(device=device)
                    generator.manual_seed(int(seed))

                LOGGER.info(
                    "\tseed=%s | steps=%d | guidance=%.2f",
                    "default" if seed is None else seed,
                    args.steps,
                    args.guidance,
                )
                sample_start = time.time()
                step_records: List[Dict] = []
                last_step_time = [sample_start]

                def callback(_pipeline, step_idx: int, timestep, tensors: Dict[str, torch.Tensor]):
                    latents = tensors["latent_model_input"].detach()
                    now = time.time()
                    step_time = now - last_step_time[0]
                    last_step_time[0] = now
                    total_steps = getattr(_pipeline, "_num_timesteps", None)
                    record: Dict[str, object] = {
                        "image": image_path.name,
                        "seed": seed,
                        "sample_index": sample_idx,
                        "step": int(step_idx),
                        "timestep": float(timestep.item()) if hasattr(timestep, "item") else float(timestep),
                        "latents_shape": list(latents.shape),
                        "latents_dtype": str(latents.dtype),
                        "step_time": step_time,
                    }
                    if total_steps is not None:
                        record["total_steps"] = int(total_steps)
                    record.update({f"latents_{k}": v for k, v in compute_metrics(latents, args.log_metrics).items()})
                    step_records.append(record)
                    if args.print_every and (step_idx % args.print_every == 0):
                        metric_parts = []
                        for metric in args.log_metrics:
                            key = f"latents_{metric}"
                            if key in record:
                                metric_parts.append(f"{metric}={record[key]:.4f}")
                        LOGGER.info(
                            "\t  step %03d/%s | t=%.3f | Δt=%.3fs | %s",
                            step_idx,
                            record.get("total_steps", "?"),
                            record["timestep"],
                            step_time,
                            ", ".join(metric_parts) if metric_parts else "no metrics",
                        )
                    return {}

                with torch.inference_mode():
                    output = model(
                        image=batch_device,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance,
                        generator=generator,
                        callback_on_step_end=callback,
                        callback_on_step_end_tensor_inputs="latents",
                    )

                elapsed = time.time() - sample_start
                final_metrics = step_records[-1] if step_records else {}
                metric_parts = []
                for metric in args.log_metrics:
                    key = f"latents_{metric}"
                    if key in final_metrics:
                        metric_parts.append(f"{metric}={final_metrics[key]:.4f}")
                LOGGER.info(
                    "\t  sample finished in %.2fs (%d steps) | final %s",
                    elapsed,
                    len(step_records),
                    ", ".join(metric_parts) if metric_parts else "no metrics",
                )

                if log_handle is not None:
                    for rec in step_records:
                        write_record(log_handle, rec)

                if args.no-save:
                    continue

                output_tensor = output[0] if isinstance(output, tuple) else output
                output_tensor = output_tensor.squeeze(0).detach().cpu()
                if not args.keep_padding and any(padding):
                    h, w = original_size
                    output_tensor = output_tensor[..., :h, :w]

                image_out = tensor_to_image(output_tensor)
                suffix = (
                    f"_seed{seed}" if seed is not None else (
                        f"_sample{sample_idx:02d}" if len(seed_values) > 1 else ""
                    )
                )
                out_path = args.output / f"{image_path.stem}{suffix}_clean.png"
                image_out.save(out_path)
                LOGGER.info("\tSaved %s", out_path)
    finally:
        if log_handle is not None:
            log_handle.close()


if __name__ == "__main__":
    main()
