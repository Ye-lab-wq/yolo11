#!/usr/bin/env python3
"""Resume matched seed-1/2, 60-epoch MSEF screens and then evaluate all on test."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import ultralytics
import yaml

from ultralytics import YOLO
from reproduction.ablation.run_screen import TRAIN_ARGS


RUN_ROOT = ROOT / "runs/msef_redesign_multiseed_60_v1"
RESULT_ROOT = ROOT / "reproduction/results/msef_redesign_60_s12"
MODELS = (
    ("p2_nearest_ciou", "p2_nearest.yaml", ROOT / "runs/ablation_screen_v1/p2_nearest_ciou_s0"),
    ("p2_msef_paper_ciou", "p2_msef_paper.yaml", ROOT / "runs/ablation_screen_v1/p2_msef_paper_ciou_s0"),
    ("p2_msef_highpass_ciou", "p2_msef_highpass.yaml", ROOT / "runs/msef_highpass_screen_v1/p2_msef_highpass_ciou_s0"),
    ("p2_msef_highpass_atrous_ciou", "p2_msef_highpass_atrous.yaml", ROOT / "runs/msef_atrous_screen_v1/p2_msef_highpass_atrous_ciou_s0"),
    ("p2_c3k2_msef_edge_addon_ciou", "p2_c3k2_msef_edge_addon.yaml", ROOT / "runs/msef_addon_screen_v1/p2_c3k2_msef_edge_addon_ciou_s0"),
)
SEEDS = (1, 2)
COMPARE_KEYS = tuple(key for key in TRAIN_ARGS if key not in {"project", "device", "seed"})


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def check_references() -> None:
    for name, filename, reference_dir in MODELS:
        model_yaml = ROOT / "reproduction/ablation/models" / filename
        assert model_yaml.is_file(), model_yaml
        reference_args = reference_dir / "args.yaml"
        reference_rows = read_rows(reference_dir / "results.csv")
        reference_best = reference_dir / "weights/best.pt"
        assert reference_args.is_file() and reference_best.is_file() and len(reference_rows) == 60, reference_dir
        saved = yaml.safe_load(reference_args.read_text(encoding="utf-8"))
        assert saved["seed"] == 0 and saved["bbox_loss_mode"] == "ciou", reference_dir
        for key in COMPARE_KEYS:
            if saved[key] != TRAIN_ARGS[key]:
                raise ValueError(f"Reference {name} differs at {key}: {saved[key]!r} != {TRAIN_ARGS[key]!r}")
    print("Five seed-0 references verified against frozen 60-epoch training controls", flush=True)


def check_run_args(run_dir: Path, seed: int) -> None:
    saved = yaml.safe_load((run_dir / "args.yaml").read_text(encoding="utf-8"))
    if saved["seed"] != seed or saved["bbox_loss_mode"] != "ciou":
        raise ValueError(f"Seed/loss mismatch: {run_dir}")
    for key in COMPARE_KEYS:
        if saved[key] != TRAIN_ARGS[key]:
            raise ValueError(f"Training mismatch at {key}: {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Read-only protocol check")
    parser.add_argument("--skip-test", action="store_true", help="Train and summarize validation only")
    args = parser.parse_args()
    check_references()
    if args.check:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; do not silently train this comparison on CPU")

    protocol = {
        "purpose": "Matched exploratory seed-1/2 replication of all three MSEF redesigns and both controls",
        "models": [{"name": name, "yaml": filename, "reference_seed0": str(reference_dir)} for name, filename, reference_dir in MODELS],
        "seeds": list(SEEDS),
        "train_args": {key: value for key, value in TRAIN_ARGS.items() if key != "project"},
        "bbox_loss_mode": "ciou",
        "selection": "Each best.pt selected by the same run's validation mAP50-95; never select by test",
        "environment": {"python": sys.version, "torch": torch.__version__, "ultralytics": ultralytics.__version__},
    }
    protocol_path = RUN_ROOT / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError(f"Protocol changed: {protocol_path}")
    else:
        save_json(protocol_path, protocol)

    summary: list[dict[str, object]] = []
    for seed in SEEDS:
        for name, filename, _ in MODELS:
            run_name = f"{name}_s{seed}"
            run_dir = RUN_ROOT / run_name
            best_weight = run_dir / "weights/best.pt"
            rows = read_rows(run_dir / "results.csv")
            if len(rows) < 60 or not best_weight.is_file():
                last_weight = run_dir / "weights/last.pt"
                if last_weight.is_file():
                    check_run_args(run_dir, seed)
                    print(f"RESUME {run_name} from {last_weight}", flush=True)
                    YOLO(str(last_weight)).train(resume=True)
                elif run_dir.exists():
                    raise RuntimeError(f"Incomplete run without last.pt: {run_dir}")
                else:
                    model_yaml = ROOT / "reproduction/ablation/models" / filename
                    print(f"START {run_name} from scratch", flush=True)
                    YOLO(str(model_yaml)).train(
                        name=run_name,
                        bbox_loss_mode="ciou",
                        exist_ok=False,
                        **{**TRAIN_ARGS, "seed": seed, "project": str(RUN_ROOT)},
                    )
                rows = read_rows(run_dir / "results.csv")
            if len(rows) != 60 or not best_weight.is_file():
                raise RuntimeError(f"Run incomplete: {run_name}, {len(rows)} / 60 epochs")
            check_run_args(run_dir, seed)
            best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
            record = {
                "name": name,
                "seed": seed,
                "epochs": len(rows),
                "best_epoch": int(best["epoch"]),
                "precision": float(best["metrics/precision(B)"]),
                "recall": float(best["metrics/recall(B)"]),
                "map50": float(best["metrics/mAP50(B)"]),
                "map50_95": float(best["metrics/mAP50-95(B)"]),
                "training_time_s": float(rows[-1]["time"]),
                "best_weight": str(best_weight),
            }
            summary.append(record)
            save_json(RUN_ROOT / "validation_summary.json", summary)
            print(f"DONE {run_name}: val mAP50-95={record['map50_95']:.6f}; {len(summary)}/10 runs", flush=True)

    if args.skip_test:
        return
    command = [
        sys.executable,
        "-B",
        str(ROOT / "reproduction/ablation/evaluate_frozen.py"),
        "--split", "combined_test",
        "--output", str(RESULT_ROOT / "combined_test.json"),
        "--device", "0",
        "--batch", "4",
    ]
    for record in summary:
        command.extend(["--weights", f"{record['name']}_s{record['seed']}={record['best_weight']}"])
    print("All validation selections frozen; evaluating 10 best.pt files on complete test split", flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
