#!/usr/bin/env python3
"""Compute PSNR/SSIM on regions where input and GT differ beyond a threshold.

Mask is derived from the absolute difference between the original input and
ground truth; metrics are computed between prediction and ground truth inside
that mask.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Iterator, Sequence, Tuple

import cv2
import numpy as np

DEFAULT_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare in-mask PSNR/SSIM between two image folders. The mask is "
            "derived from absolute difference between input and ground truth."
        )
    )
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory of raw input images")
    parser.add_argument("--pred_dir", type=Path, required=True, help="Directory of predicted images")
    parser.add_argument("--gt_dir", type=Path, required=True, help="Directory of ground truth images")
    parser.add_argument("--threshold", type=float, default=10.0, help="Mask threshold on input-gt abs-diff (0-255 scale)")
    parser.add_argument(
        "--suffixes",
        nargs="*",
        default=list(DEFAULT_SUFFIXES),
        help="Image suffixes to include (default: common formats)",
    )
    parser.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    parser.add_argument("--quiet", action="store_true", help="Only print summary line")
    return parser.parse_args(argv)


def normalise_suffixes(suffixes: Iterable[str]) -> tuple[str, ...]:
    normalized = []
    for s in suffixes:
        s = s.strip()
        if not s:
            continue
        if not s.startswith('.'):
            s = f'.{s}'
        normalized.append(s.lower())
    return tuple(sorted(set(normalized)))


def iter_images(root: Path, suffixes: tuple[str, ...], recursive: bool) -> Iterator[Path]:
    paths = root.rglob('*') if recursive else root.glob('*')
    for p in paths:
        if p.is_file() and p.suffix.lower() in suffixes:
            yield p


def resolve_match(source_path: Path, source_root: Path, target_root: Path, suffixes: tuple[str, ...]) -> Path | None:
    rel = source_path.relative_to(source_root)
    candidate = target_root / rel
    if candidate.is_file():
        return candidate

    parent = target_root / rel.parent
    if parent.is_dir():
        matches = [q for q in parent.iterdir() if q.is_file() and q.stem == rel.stem and q.suffix.lower() in suffixes]
        if matches:
            return sorted(matches)[0]
    return None


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32)


def maybe_resize(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if img.shape[:2] == ref.shape[:2]:
        return img
    h, w = ref.shape[:2]
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def build_mask(inp: np.ndarray, gt: np.ndarray, threshold: float) -> np.ndarray:
    diff = np.abs(inp - gt)
    if diff.ndim == 3:
        diff = diff.max(axis=2)
    return diff > threshold


def psnr_masked(inp: np.ndarray, gt: np.ndarray, mask: np.ndarray, data_range: float = 255.0) -> float:
    count = mask.sum()
    if count == 0:
        return float('nan')
    diff = (inp - gt)[mask]
    mse = np.mean(diff ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10((data_range ** 2) / mse)


def ssim_masked(inp: np.ndarray, gt: np.ndarray, mask: np.ndarray, data_range: float = 255.0) -> float:
    count = mask.sum()
    if count == 0:
        return float('nan')

    # Expand mask to channels
    if inp.ndim == 3:
        mask_c = mask[..., None]
    else:
        mask_c = mask

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    masked_inp = inp * mask_c
    masked_gt = gt * mask_c

    # Means
    mu_x = masked_inp.sum(axis=(0, 1)) / count
    mu_y = masked_gt.sum(axis=(0, 1)) / count

    # Variances and covariance
    var_x = ((masked_inp - mu_x) ** 2).sum(axis=(0, 1)) / count
    var_y = ((masked_gt - mu_y) ** 2).sum(axis=(0, 1)) / count
    cov_xy = ((masked_inp - mu_x) * (masked_gt - mu_y)).sum(axis=(0, 1)) / count

    ssim_c = ((2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (var_x + var_y + C2))

    ssim_scalar = float(np.mean(ssim_c))
    return ssim_scalar


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if not args.quiet else logging.WARNING)

    if not args.input_dir.is_dir() or not args.gt_dir.is_dir() or not args.pred_dir.is_dir():
        logging.error("Input, prediction, or GT directory missing")
        return 1

    suffixes = normalise_suffixes(args.suffixes)
    inputs = list(iter_images(args.input_dir, suffixes, args.recursive))
    if not inputs:
        logging.error("No images found in %s", args.input_dir)
        return 1

    per_image: list[Tuple[str, float, float]] = []
    for inp_path in sorted(inputs):
        gt_path = resolve_match(inp_path, args.input_dir, args.gt_dir, suffixes)
        pred_path = resolve_match(inp_path, args.input_dir, args.pred_dir, suffixes)
        if gt_path is None or pred_path is None:
            logging.warning("Missing GT or prediction for %s", inp_path)
            continue
        print(inp_path, gt_path, pred_path)
        raw_inp = load_image(inp_path)
        pred = load_image(pred_path)
        gt = load_image(gt_path)
        # Align shapes to GT
        raw_inp = maybe_resize(raw_inp, gt)
        pred = maybe_resize(pred, gt)

        mask = build_mask(raw_inp, gt, args.threshold)
        psnr = psnr_masked(pred, gt, mask)
        ssim = ssim_masked(pred, gt, mask)

        per_image.append((str(inp_path.relative_to(args.input_dir)), psnr, ssim))
        if not args.quiet:
            print(
                f"{inp_path.relative_to(args.input_dir)}\tPSNR_mask: {psnr:.4f}\tSSIM_mask: {ssim:.4f}"
                f"\tMask pixels: {int(mask.sum())}"
            )

    if not per_image:
        logging.error("No valid pairs processed")
        return 1

    psnrs = [p for _, p, _ in per_image if not np.isnan(p)]
    ssims = [s for _, _, s in per_image if not np.isnan(s)]
    avg_psnr = float(np.mean(psnrs)) if psnrs else float('nan')
    avg_ssim = float(np.mean(ssims)) if ssims else float('nan')

    print(f"Average\tPSNR_mask: {avg_psnr:.4f}\tSSIM_mask: {avg_ssim:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
