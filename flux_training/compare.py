"""CLI utility to compare PSNR/SSIM for two image directories."""
from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import cv2
import numpy as np
from functools import partial
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


class Bandwise(object):
    def __init__(self, index_fn):
        self.index_fn = index_fn

    def __call__(self, X, Y):
        C = X.shape[-1]
        bwindex = []
        for ch in range(C):
            x = X[..., ch]
            y = Y[..., ch]
            index = self.index_fn(x, y)
            bwindex.append(index)
        return bwindex


def calculate_psnr(img, img2, **kwargs):
    cal_bwpsnr = Bandwise(partial(compare_psnr, data_range=255))
    psnr = np.mean(cal_bwpsnr(img2, img))
    return psnr


def calculate_ssim(img, img2, **kwargs):
    cal_bwssim = Bandwise(partial(compare_ssim, data_range=255))
    ssim = np.mean(cal_bwssim(img2, img))
    return ssim


# Default extensions we consider image files.
DEFAULT_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class ImagePair:
    """Bundle a predicted image path with its corresponding target path."""

    predicted: Path
    target: Path

    @property
    def name(self) -> str:
        """Return the relative identifier used for reporting."""
        return str(self.predicted)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare PSNR and SSIM between images in two directories. "
            "Targets are matched by filename regardless of suffix differences."
        )
    )
    parser.add_argument("--input_dir", type=Path, help="Directory containing predicted/input images.")
    parser.add_argument("--target_dir", type=Path, help="Directory containing target/ground-truth images.")
    parser.add_argument(
        "--suffixes",
        nargs="*",
        default=list(DEFAULT_SUFFIXES),
        help="Image suffixes to include (default: common formats).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively match files using relative paths (default: only top-level files).",
    )
    parser.add_argument(
        "--save_csv",
        type=Path,
        help="Optional path to save per-image metrics as a CSV file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the summary row; suppress per-image logging.",
    )
    return parser.parse_args(argv)


def normalise_suffixes(suffixes: Iterable[str]) -> tuple[str, ...]:
    """Return suffixes normalised to lowercase with leading dots."""
    normalised: list[str] = []
    for suffix in suffixes:
        suffix = suffix.strip()
        if not suffix:
            continue
        if not suffix.startswith('.'):
            suffix = f'.{suffix}'
        normalised.append(suffix.lower())
    return tuple(sorted(set(normalised)))


def iter_image_pairs(
    input_dir: Path,
    target_dir: Path,
    suffixes: tuple[str, ...],
    recursive: bool,
) -> Iterator[ImagePair]:
    """Yield `ImagePair` entries for files found under `input_dir` with matches in `target_dir`."""
    if recursive:
        input_paths = [p for p in input_dir.rglob('*') if p.is_file() and p.suffix.lower() in suffixes]
    else:
        input_paths = [p for p in input_dir.glob('*') if p.is_file() and p.suffix.lower() in suffixes]

    for predicted_path in sorted(input_paths):
        relative = predicted_path.relative_to(input_dir)
        target_path = resolve_target_path(relative, target_dir, suffixes)
        if target_path is None:
            logging.warning("Skipping %s (no matching target with same name)", predicted_path)
            continue
        yield ImagePair(predicted=predicted_path, target=target_path)


def resolve_target_path(relative: Path, target_dir: Path, suffixes: tuple[str, ...]) -> Path | None:
    """Resolve a target path by matching filename stem while ignoring suffix differences."""
    candidate = target_dir / relative
    if candidate.is_file():
        return candidate

    parent = target_dir / relative.parent
    if not parent.is_dir():
        return None

    stem = relative.stem
    matches = [
        path
        for path in parent.iterdir()
        if path.is_file() and path.stem == stem and path.suffix.lower() in suffixes
    ]
    if not matches:
        return None
    return sorted(matches)[0]


def load_image(path: Path) -> np.ndarray:
    """Load an image as float32 in [0, 255]; raise if loading fails."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    return image


def maybe_resize(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Resize image to the spatial size of reference if needed."""
    if image.shape[:2] == reference.shape[:2]:
        return image
    height, width = reference.shape[:2]
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized


def ensure_channel_match(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match the number of channels between `image` and `reference` when possible."""
    if image.ndim == reference.ndim:
        return image
    if reference.ndim == 3 and image.ndim == 2:
        return np.repeat(image[..., None], reference.shape[2], axis=2)
    if reference.ndim == 2 and image.ndim == 3 and image.shape[2] == 1:
        return image.reshape(image.shape[0], image.shape[1])
    raise ValueError(
        "Unable to match image channel dimensions: "
        f"predicted={image.shape}, target={reference.shape}"
    )


def compute_metrics(predicted: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Compute PSNR and SSIM given two aligned images."""
    psnr = float(calculate_psnr(predicted, target))
    ssim = float(calculate_ssim(predicted, target))
    return psnr, ssim


def write_csv(results: list[tuple[str, float, float]], csv_path: Path) -> None:
    """Persist per-image metrics as CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "psnr", "ssim"])
        writer.writerows(results)


def main(argv: Sequence[str] | None = None) -> int:
    """Entrypoint for script execution."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if not args.quiet else logging.WARNING)

    if not args.input_dir.is_dir():
        logging.error("Input directory does not exist: %s", args.input_dir)
        return 1
    if not args.target_dir.is_dir():
        logging.error("Target directory does not exist: %s", args.target_dir)
        return 1

    suffixes = normalise_suffixes(args.suffixes)
    pairs = list(iter_image_pairs(args.input_dir, args.target_dir, suffixes, args.recursive))
    if not pairs:
        logging.error("No image pairs found. Verify suffixes or directory structure.")
        return 1

    per_image_results: list[tuple[str, float, float]] = []
    for pair in pairs:
        predicted = load_image(pair.predicted)
        target = load_image(pair.target)

        predicted = maybe_resize(predicted, target)
        predicted = ensure_channel_match(predicted, target)

        try:
            psnr, ssim = compute_metrics(predicted, target.astype(np.float32))
        except ValueError as exc:
            logging.error("Skipping %s due to channel mismatch: %s", pair.predicted, exc)
            continue

        per_image_results.append((str(pair.predicted.relative_to(args.input_dir)), psnr, ssim))
        if not args.quiet:
            print(f"{pair.predicted.relative_to(args.input_dir)}\tPSNR: {psnr:.4f} dB\tSSIM: {ssim:.4f}")

    if not per_image_results:
        logging.error("No valid comparisons were computed.")
        return 1

    psnr_values = [item[1] for item in per_image_results]
    ssim_values = [item[2] for item in per_image_results]
    avg_psnr = float(np.mean(psnr_values))
    avg_ssim = float(np.mean(ssim_values))

    print(f"Average\tPSNR: {avg_psnr:.4f} dB\tSSIM: {avg_ssim:.4f}")

    if args.save_csv is not None:
        write_csv(per_image_results, args.save_csv)
        logging.info("Saved CSV results to %s", args.save_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
