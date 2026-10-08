#!/usr/bin/env python3
"""Reproducible same-hardware FP32 efficiency benchmark for frozen checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ultralytics
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def parse_weights(values: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, received {value!r}")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        parsed.append((name, path))
    return parsed


@torch.inference_mode()
def timed_forward(model: torch.nn.Module, batch_size: int, device: torch.device) -> tuple[float, float, list[float]]:
    image = torch.zeros(batch_size, 3, 640, 640, dtype=torch.float32, device=device)
    warmup = 50 if batch_size == 1 else 20
    groups = 10 if batch_size == 1 else 5
    iterations = 50 if batch_size == 1 else 20
    for _ in range(warmup):
        model(image)
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(groups):
        started = time.perf_counter()
        for _ in range(iterations):
            model(image)
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - started) * 1000 / iterations / batch_size)
    latency = statistics.median(samples)
    del image
    return latency, 1000.0 / latency, samples


@torch.inference_mode()
def peak_memory(model: torch.nn.Module, device: torch.device, batch_size: int = 16) -> float:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    image = torch.zeros(batch_size, 3, 640, 640, dtype=torch.float32, device=device)
    model(image)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) - baseline
    del image
    torch.cuda.empty_cache()
    return peak / 1024**2


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen efficiency protocol")
    device = torch.device(args.device)
    rows = []
    metadata = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "ultralytics": ultralytics.__version__,
            "gpu": torch.cuda.get_device_name(device),
        },
        "protocol": {
            "input": "640x640",
            "precision": "FP32",
            "graph": "PyTorch fused Conv+BN raw model forward",
            "excluded": "file I/O, image preprocessing, NMS",
            "batch1": "50 warmup; 10 groups x 50 iterations; median group latency",
            "batch16": "20 warmup; 5 groups x 20 iterations; median per-image group latency",
        },
        "models": [],
    }
    for name, path in parse_weights(args.weights):
        print(f"Benchmarking {name}: {path}", flush=True)
        wrapper = YOLO(str(path))
        parameters = sum(parameter.numel() for parameter in wrapper.model.parameters())
        gflops = float(get_flops(wrapper.model, imgsz=640))
        model = wrapper.model.fuse(verbose=False).to(device).float().eval()
        batch1_ms, batch1_fps, batch1_samples = timed_forward(model, 1, device)
        batch16_ms, batch16_fps, batch16_samples = timed_forward(model, 16, device)
        peak_mib = peak_memory(model, device, 16)
        row = {
            "model": name,
            "weights": str(path),
            "parameters": parameters,
            "parameters_m": parameters / 1e6,
            "checkpoint_mib": path.stat().st_size / 1024**2,
            "gflops_640": gflops,
            "fp32_b1_latency_ms": batch1_ms,
            "fp32_b1_fps": batch1_fps,
            "fp32_b16_latency_ms_per_image": batch16_ms,
            "fp32_b16_throughput_fps": batch16_fps,
            "fp32_b16_peak_allocated_mib": peak_mib,
        }
        rows.append(row)
        metadata["models"].append(
            {**row, "batch1_group_ms": batch1_samples, "batch16_group_ms_per_image": batch16_samples}
        )
        del model, wrapper
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
