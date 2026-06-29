#!/usr/bin/env python3
"""Gradio UI for labeling dereflection user-study comparisons."""

from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "false")

import gradio as gr
from PIL import Image, ImageOps


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_OPTIONS: Tuple[Tuple[str, str], ...] = (
    ("success", "Success"),
    ("failure_incomplete", "Failure (Incomplete Removal)"),
    ("failure_content_deletion", "Failure (Content Deletion)"),
    ("failure_artifacts", "Failure (Artifact Generation)"),
)

try:
    RESAMPLE_FILTER = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover
    RESAMPLE_FILTER = Image.LANCZOS


@dataclass(frozen=True)
class Sample:
    name: str
    input_path: str
    gt_path: str
    method_outputs: Dict[str, str]


@dataclass(frozen=True)
class MethodConfig:
    name: str
    output_dir: Path
    log_file: Path


class ImageDownsampler:
    def __init__(self, max_size: int, cache_dir: Path):
        self.max_size = max_size
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.size_cache: Dict[str, Tuple[int, int]] = {}

    def _get_size(self, source_path: str) -> Tuple[int, int]:
        cached = self.size_cache.get(source_path)
        if cached is not None:
            return cached
        with Image.open(source_path) as img:
            size = img.size
        self.size_cache[source_path] = size
        return size

    def smallest_size(self, paths: Sequence[str]) -> Tuple[int, int]:
        widths, heights = zip(*(self._get_size(p) for p in paths))
        return min(widths), min(heights)

    def prepare(self, source_path: str, target_size: Tuple[int, int] | None = None) -> str:
        needs_resize = self.max_size > 0 or target_size is not None
        if not needs_resize:
            return source_path

        width, height = self._get_size(source_path)
        target_w, target_h = target_size or (width, height)
        target_w = max(1, min(target_w, width))
        target_h = max(1, min(target_h, height))

        if self.max_size > 0:
            max_dim = max(target_w, target_h)
            if max_dim > self.max_size:
                scale = self.max_size / max_dim
                target_w = max(1, int(round(target_w * scale)))
                target_h = max(1, int(round(target_h * scale)))

        scale = min(target_w / width, target_h / height, 1.0)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        if new_w == width and new_h == height:
            return source_path

        src = Path(source_path)
        hashed = hashlib.sha1(f"{src}-{new_w}x{new_h}".encode("utf-8")).hexdigest()[:10]
        target = self.cache_dir / f"{src.stem}_{hashed}_{new_w}x{new_h}.png"

        if target.exists() and target.stat().st_mtime >= src.stat().st_mtime:
            return str(target)

        with Image.open(src) as img:
            img = ImageOps.exif_transpose(img)
            if scale != 1.0:
                img = img.resize((new_w, new_h), RESAMPLE_FILTER)
            img.save(target, format="PNG")
        return str(target)


def list_images(directory: Path) -> Dict[str, Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"Expected a directory: {directory}")
    mapping: Dict[str, Path] = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            mapping[path.stem] = path
    return mapping


def intersect_samples(
    input_dir: Path, gt_dir: Path, methods: Sequence[MethodConfig]
) -> List[Sample]:
    if not methods:
        raise ValueError("At least one method directory must be provided.")

    input_map = list_images(input_dir)
    gt_map = list_images(gt_dir)
    method_maps = {method.name: list_images(method.output_dir) for method in methods}

    common_names = set(input_map) & set(gt_map)
    for mapping in method_maps.values():
        common_names &= set(mapping)

    if not common_names:
        details = ", ".join(f"{name}: {len(mapping)}" for name, mapping in method_maps.items())
        raise ValueError(
            "No overlapping image names found across the provided folders. "
            f"Input: {len(input_map)}, GT: {len(gt_map)}, Methods: {details}"
        )

    method_names = [method.name for method in methods]
    samples: List[Sample] = []
    for name in sorted(common_names):
        method_outputs = {
            method_name: str(method_maps[method_name][name])
            for method_name in method_names
        }
        samples.append(
            Sample(
                name=name,
                input_path=str(input_map[name]),
                gt_path=str(gt_map[name]),
                method_outputs=method_outputs,
            )
        )
    return samples


def parse_label_file(label_file: Path) -> Dict[str, str]:
    if not label_file.exists():
        return {}

    labels: Dict[str, str] = {}
    with label_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if "," in line:
                name, value = line.split(",", 1)
            else:
                name, value = line.split(maxsplit=1)
            labels[name] = value
    return labels


def write_label_file(label_file: Path, labels: Dict[str, str]) -> None:
    label_file.parent.mkdir(parents=True, exist_ok=True)
    with label_file.open("w", encoding="utf-8") as handle:
        for sample_name in sorted(labels):
            handle.write(f"{sample_name},{labels[sample_name]}\n")


def load_label_state(methods: Sequence[MethodConfig]) -> Dict[str, Dict[str, str]]:
    return {method.name: parse_label_file(method.log_file) for method in methods}


def completed_samples(label_state: Dict[str, Dict[str, str]]) -> set[str]:
    if not label_state:
        return set()
    iterator = iter(label_state.values())
    try:
        done = set(next(iterator))
    except StopIteration:
        return set()
    for mapping in iterator:
        done &= set(mapping)
    return done


def find_next_index(
    samples: Sequence[Sample], label_state: Dict[str, Dict[str, str]], start: int = 0
) -> int:
    finished = completed_samples(label_state)
    for idx in range(max(start, 0), len(samples)):
        if samples[idx].name not in finished:
            return idx
    return len(samples)


def render_sample(
    samples: Sequence[Sample],
    index: int,
    downsampler: ImageDownsampler | None,
    method_names: Sequence[str],
) -> Tuple[str | None, str | None, List[str | None], str, int]:
    method_count = len(method_names)
    if not samples:
        return None, None, [None] * method_count, "No samples found.", 0
    if index >= len(samples):
        return None, None, [None] * method_count, f"All samples labeled ({len(samples)} total).", index

    sample = samples[index]
    method_paths = [sample.method_outputs[name] for name in method_names]
    status = f"{sample.name} ({index + 1}/{len(samples)})"

    if downsampler is None:
        return sample.input_path, sample.gt_path, method_paths, status, index

    all_paths = [sample.input_path, sample.gt_path, *method_paths]
    target_size = downsampler.smallest_size(all_paths)
    input_path = downsampler.prepare(sample.input_path, target_size)
    gt_path = downsampler.prepare(sample.gt_path, target_size)
    resized_methods = [downsampler.prepare(path, target_size) for path in method_paths]
    return input_path, gt_path, resized_methods, status, index


def record_label(
    label_value: str,
    method_name: str,
    index: int,
    samples: Sequence[Sample],
    label_state: Dict[str, Dict[str, str]],
    label_files: Dict[str, Path],
    method_names: Sequence[str],
    downsampler: ImageDownsampler | None,
) -> Tuple[str | None, str | None, List[str | None], str, int, Dict[str, Dict[str, str]]]:
    method_count = len(method_names)
    if not samples:
        return None, None, [None] * method_count, "No samples found.", 0, label_state

    if index >= len(samples):
        status = f"All samples labeled ({len(samples)} total)."
        return None, None, [None] * method_count, status, index, label_state

    sample = samples[index]
    updated_state = {name: dict(values) for name, values in label_state.items()}
    updated_state.setdefault(method_name, {})
    updated_state[method_name][sample.name] = label_value

    for name, mapping in updated_state.items():
        write_label_file(label_files[name], mapping)

    pending = [name for name in method_names if sample.name not in updated_state[name]]
    if pending:
        next_index = index
        base_status = f"Pending labels for {sample.name}: {', '.join(pending)}"
    else:
        next_index = find_next_index(samples, updated_state, index + 1)
        base_status = f"All methods labeled for {sample.name}."

    input_path, gt_path, method_paths, status_text, _ = render_sample(
        samples, next_index, downsampler, method_names
    )
    message = (
        f"Recorded '{label_value}' for {method_name} on {sample.name}.\n"
        f"{base_status}\n{status_text}"
    )
    return input_path, gt_path, method_paths, message, next_index, updated_state


def build_interface(
    samples: List[Sample],
    methods: Sequence[MethodConfig],
    label_state: Dict[str, Dict[str, str]],
    downsampler: ImageDownsampler | None,
    dataset_name: str,
    user_count: int,
) -> gr.Blocks:
    method_names = [method.name for method in methods]
    label_files = {method.name: method.log_file for method in methods}

    start_index = find_next_index(samples, label_state)
    initial_input, initial_gt, initial_methods, initial_status, _ = render_sample(
        samples, start_index, downsampler, method_names
    )

    log_lines = "<br>".join(
        f"{method.name}: `{method.log_file}`" for method in methods
    )

    with gr.Blocks() as demo:
        gr.Markdown(f"## Dereflection User Study — Dataset: **{dataset_name}**")
        gr.Markdown(
            f"User ID / count: **{user_count}**  \n"
            f"Methods: {', '.join(method_names)}"
        )
        gr.Markdown(
            "Label each method independently using the buttons below. "
            "A sample advances once every method receives a label."
        )
        if log_lines:
            gr.Markdown(f"Votes saved to:<br>{log_lines}")

        sample_state = gr.State(samples)
        index_state = gr.State(start_index)
        label_state_store = gr.State(label_state)

        with gr.Row():
            input_image = gr.Image(label="Input", value=initial_input, type="filepath")
            gt_image = gr.Image(label="Ground Truth", value=initial_gt, type="filepath")

        def make_handler(value: str, name: str):
            def _handler(idx, sample_list, current_state):
                input_path, gt_path, method_paths, message, next_index, new_state = record_label(
                    value,
                    name,
                    idx,
                    sample_list,
                    current_state,
                    label_files,
                    method_names,
                    downsampler,
                )
                return (input_path, gt_path, *method_paths, message, next_index, new_state)

            return _handler

        method_image_components: List[gr.Image] = []
        button_specs: List[Tuple[gr.Button, str, str]] = []
        with gr.Row():
            for name, path in zip(method_names, initial_methods):
                with gr.Column():
                    img = gr.Image(label=f"{name} Output", value=path, type="filepath")
                    method_image_components.append(img)
                    gr.Markdown(f"### {name} Labels")
                    for value, label in LABEL_OPTIONS:
                        btn = gr.Button(label)
                        button_specs.append((btn, value, name))

        status = gr.Markdown(value=initial_status)
        button_outputs = [input_image, gt_image, *method_image_components, status, index_state, label_state_store]

        for btn, value, name in button_specs:
            btn.click(
                fn=make_handler(value, name),
                inputs=[index_state, sample_state, label_state_store],
                outputs=button_outputs,
            )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio app for dereflection user studies")
    parser.add_argument("--input_dir", type=Path, required=True, help="Folder containing input images")
    parser.add_argument("--gt_dir", type=Path, required=True, help="Folder containing ground truth images")
    parser.add_argument(
        "--method",
        action="append",
        nargs=2,
        metavar=("NAME", "DIR"),
        required=True,
        help="Method name and folder containing its outputs. Provide at least two.",
    )
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset identifier for log files")
    parser.add_argument("--user_count", type=int, required=True, help="User id/count for logs and port computation")
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=None,
        help="Directory to store generated log files (default: current working directory)",
    )
    parser.add_argument(
        "--max_display_size",
        type=int,
        default=1024,
        help="Longest side (pixels) for preview images; set 0 to disable additional downsampling.",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
        help="Cache directory for resized previews (default: <log_dir>/.labeler_cache)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Optional port override (default: 7859 + user_count)",
    )
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")

    args = parser.parse_args()

    if len(args.method) < 2:
        parser.error("Provide at least two --method NAME DIR pairs for comparison.")

    method_names = [name for name, _ in args.method]
    if len(method_names) != len(set(method_names)):
        parser.error("Method names must be unique.")

    return args


def main() -> None:
    args = parse_args()

    log_root = args.log_dir or Path.cwd()
    log_root.mkdir(parents=True, exist_ok=True)

    method_configs = []
    for name, directory in args.method:
        output_dir = Path(directory)
        log_file = log_root / f"{name}_{args.dataset_name}_userstudy_user{args.user_count}.txt"
        method_configs.append(MethodConfig(name=name, output_dir=output_dir, log_file=log_file))

    samples = intersect_samples(args.input_dir, args.gt_dir, method_configs)
    label_state = load_label_state(method_configs)

    cache_dir = args.cache_dir or (log_root / ".labeler_cache")
    downsampler = ImageDownsampler(args.max_display_size, cache_dir)
    demo = build_interface(samples, method_configs, label_state, downsampler, args.dataset_name, args.user_count)

    port = args.port if args.port is not None else 7859 + args.user_count
    demo.launch(share=args.share, server_port=port, server_name="0.0.0.0")


if __name__ == "__main__":
    main()
