#!/usr/bin/env python3
"""Benchmark each unique screening inference graph once and expand graph aliases."""

from __future__ import annotations

import csv
import json
import platform
from pathlib import Path

import torch

from benchmark_frozen import ROOT, peak_memory, timed_forward

import ultralytics
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


RUN_ROOT = ROOT / "runs" / "ablation_screen_v1"
SUMMARY = RUN_ROOT / "screen_summary.csv"
OUTPUT = RUN_ROOT / "screen_efficiency.csv"
REFERENCE_WEIGHTS = {
    "yolo11n_ciou_s0_reference": ROOT / "runs/clean_v2_scratch_pair/yolo11n_seed0/weights/best.pt",
    "full_meda_author_focal_s0_reference": ROOT
    / "runs/clean_v2_scratch_pair/meda_w025_focaler_seed0/weights/best.pt",
}


def load_summary() -> list[dict[str, str]]:
    with SUMMARY.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rows = load_summary()
    graph_groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        graph_groups.setdefault(row["model_yaml"], []).append(row)
    device = torch.device("cuda:0")
    graph_results: dict[str, dict] = {}
    raw_groups: dict[str, dict] = {}
    for graph, aliases in graph_groups.items():
        representative = aliases[0]
        weight = Path(representative["best_weight"]) if representative["best_weight"] else REFERENCE_WEIGHTS[
            representative["experiment"]
        ]
        print(f"Benchmarking graph {graph} via {weight}", flush=True)
        wrapper = YOLO(str(weight))
        parameters = sum(parameter.numel() for parameter in wrapper.model.parameters())
        gflops = float(get_flops(wrapper.model, imgsz=640))
        model = wrapper.model.fuse(verbose=False).to(device).float().eval()
        b1_ms, b1_fps, b1_groups = timed_forward(model, 1, device)
        b16_ms, b16_fps, b16_groups = timed_forward(model, 16, device)
        peak_mib = peak_memory(model, device, 16)
        graph_results[graph] = {
            "representative_weight": str(weight.resolve()),
            "parameters": parameters,
            "parameters_m": parameters / 1e6,
            "gflops_640": gflops,
            "fp32_b1_latency_ms": b1_ms,
            "fp32_b1_fps": b1_fps,
            "fp32_b16_latency_ms_per_image": b16_ms,
            "fp32_b16_throughput_fps": b16_fps,
            "fp32_b16_peak_allocated_mib": peak_mib,
        }
        raw_groups[graph] = {"batch1_group_ms": b1_groups, "batch16_group_ms_per_image": b16_groups}
        del model, wrapper
        torch.cuda.empty_cache()

    expanded = []
    for row in rows:
        metrics = graph_results[row["model_yaml"]]
        expanded.append(
            {
                "model": row["experiment"],
                "graph": row["model_yaml"],
                **metrics,
                "alias_policy": "one benchmark per identical model_yaml inference graph",
            }
        )
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(expanded[0]))
        writer.writeheader()
        writer.writerows(expanded)
    OUTPUT.with_suffix(".json").write_text(
        json.dumps(
            {
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
                    "alias_policy": "loss-only runs and other identical YAML graphs reuse one measurement",
                },
                "graphs": {key: {**value, **raw_groups[key]} for key, value in graph_results.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
