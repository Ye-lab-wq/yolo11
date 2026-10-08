#!/usr/bin/env python3
"""Evaluate frozen seed-1 P2 and P2+MSEFPaper on the motion-blur benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from ultralytics import YOLO

from reproduction.ablation.evaluate_frozen import coco_metrics, image_paths, make_ground_truth, predictions


BENCHMARK = ROOT / "DetectDataset_clean_v2_motion_blur"
OUTPUT = ROOT / "reproduction/results/msef_motion_blur_seed1"
MODELS = OrderedDict(
    [
        ("P2", ROOT / "runs/ablation_200_v1/p2/train_seed1/weights/best.pt"),
        ("P2+MSEFPaper", ROOT / "runs/ablation_200_v1/p2_msef_paper/train_seed1/weights/best.pt"),
    ]
)
CONDITIONS = ("clean", "light", "moderate", "strong")
CLEAN_RESULTS = {
    "P2": ROOT / "runs/ablation_200_v1/p2/test_results/seed1/public_test.json",
    "P2+MSEFPaper": ROOT / "runs/ablation_200_v1/p2_msef_paper/test_results/seed1/public_test.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int, default=4)
    return parser.parse_args()


def now() -> str:
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def clean_metric(model_name: str) -> dict[str, object]:
    payload = json.loads(CLEAN_RESULTS[model_name].read_text(encoding="utf-8"))
    return next(iter(payload["models"].values()))["coco_style"]


def paths_for(condition: str) -> list[Path]:
    directory = (
        ROOT / "DetectDataset_clean_v2/test/images"
        if condition == "clean"
        else BENCHMARK / condition / "images"
    )
    return image_paths((directory,))


def row(model: str, condition: str, metrics: dict[str, object]) -> dict[str, object]:
    return {
        "model": model,
        "condition": condition,
        "images": int(metrics["images"]),
        "instances": int(metrics["instances"]),
        "AP50_95_percent": 100.0 * float(metrics["AP"]),
        "AP50_percent": 100.0 * float(metrics["AP50"]),
        "AP75_percent": 100.0 * float(metrics["AP75"]),
        "AP_tiny_lt16_percent": 100.0 * float(metrics["AP_tiny_lt16"]),
        "AP_small_16_32_percent": 100.0 * float(metrics["AP_small_16_32"]),
        "AP_medium_32_96_percent": 100.0 * float(metrics["AP_coco_medium_32_96"]),
        "AP_large_ge96_percent": 100.0 * float(metrics["AP_coco_large_ge96"]),
    }


def save_outputs(records: list[dict[str, object]], device: str, batch: int) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(
        OUTPUT / "results.json",
        {
            "protocol": "frozen seed-1 checkpoints; public test and deterministic derivative only",
            "device": device,
            "batch": batch,
            "confidence_floor": 0.001,
            "nms_iou": 0.7,
            "max_det": 300,
            "records": records,
        },
    )
    with (OUTPUT / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    by_key = {(str(item["model"]), str(item["condition"])): item for item in records}
    lines = [
        "# MSEFPaper motion-blur robustness audit",
        "",
        "Frozen seed-1 P2 and P2+MSEFPaper checkpoints were evaluated on the original public test split and its deterministic motion-blur derivatives. No retraining was performed.",
        "",
        "| Condition | P2 AP50:95/% | P2+MSEF AP50:95/% | MSEF-P2/pp | P2 AP75/% | P2+MSEF AP75/% | MSEF-P2/pp |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        p2 = by_key[("P2", condition)]
        msef = by_key[("P2+MSEFPaper", condition)]
        lines.append(
            f"| {condition} | {p2['AP50_95_percent']:.2f} | {msef['AP50_95_percent']:.2f} | "
            f"{msef['AP50_95_percent'] - p2['AP50_95_percent']:+.2f} | {p2['AP75_percent']:.2f} | "
            f"{msef['AP75_percent']:.2f} | {msef['AP75_percent'] - p2['AP75_percent']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "## Clean-relative degradation",
            "",
            "Positive robustness advantage means MSEF loses fewer points than P2 relative to each model's own clean result.",
            "",
            "| Condition | P2 AP change/pp | MSEF AP change/pp | MSEF robustness advantage/pp | P2 AP75 change/pp | MSEF AP75 change/pp | MSEF AP75 robustness advantage/pp |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    p2_clean = by_key[("P2", "clean")]
    msef_clean = by_key[("P2+MSEFPaper", "clean")]
    for condition in CONDITIONS[1:]:
        p2 = by_key[("P2", condition)]
        msef = by_key[("P2+MSEFPaper", condition)]
        p2_ap_change = p2["AP50_95_percent"] - p2_clean["AP50_95_percent"]
        msef_ap_change = msef["AP50_95_percent"] - msef_clean["AP50_95_percent"]
        p2_ap75_change = p2["AP75_percent"] - p2_clean["AP75_percent"]
        msef_ap75_change = msef["AP75_percent"] - msef_clean["AP75_percent"]
        lines.append(
            f"| {condition} | {p2_ap_change:+.2f} | {msef_ap_change:+.2f} | "
            f"{msef_ap_change - p2_ap_change:+.2f} | {p2_ap75_change:+.2f} | {msef_ap75_change:+.2f} | "
            f"{msef_ap75_change - p2_ap75_change:+.2f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: architecture-specific blur robustness requires a growing or consistently positive MSEF-minus-P2 advantage as blur severity increases. Absolute degradation alone only establishes that blur is difficult. This synthetic benchmark does not replace real blur labels.",
        ]
    )
    (OUTPUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = "0" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if not (BENCHMARK / "metadata.json").is_file():
        raise FileNotFoundError("build the motion-blur benchmark first")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    partial_path = OUTPUT / "partial_results.json"
    records = json.loads(partial_path.read_text(encoding="utf-8")) if partial_path.is_file() else []
    completed = {(str(item["model"]), str(item["condition"])) for item in records}

    for model_name, weight in MODELS.items():
        if (model_name, "clean") not in completed:
            records.append(row(model_name, "clean", clean_metric(model_name)))
            write_json(partial_path, records)
            completed.add((model_name, "clean"))
        pending = [condition for condition in CONDITIONS[1:] if (model_name, condition) not in completed]
        if not pending:
            continue
        model = YOLO(str(weight))
        for condition in pending:
            paths = paths_for(condition)
            if len(paths) != 625:
                raise RuntimeError(f"{condition}: expected 625 images, found {len(paths)}")
            print(f"{now()} | {model_name} | {condition} | device={device}", flush=True)
            ground_truth, path_to_id = make_ground_truth(paths)
            detections = predictions(model, paths, path_to_id, device, args.batch)
            metrics = coco_metrics(ground_truth, detections)
            records.append(row(model_name, condition, metrics))
            write_json(partial_path, records)
            completed.add((model_name, condition))
            print(f"AP50:95={100.0 * float(metrics['AP']):.2f}", flush=True)

    records.sort(key=lambda item: (list(MODELS).index(str(item["model"])), CONDITIONS.index(str(item["condition"]))))
    save_outputs(records, device, args.batch)
    write_json(OUTPUT / "status.json", {"status": "completed", "completed_at": now(), "device": device})
    print(OUTPUT, flush=True)


if __name__ == "__main__":
    main()
