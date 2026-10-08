#!/usr/bin/env python3
"""Run the frozen 200-epoch, multi-seed confirmation after validation selection.

The selected chain is provided explicitly as JSON. This keeps selection auditable
and prevents this script from consulting the test split or selecting by test score.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import ultralytics
from ultralytics import YOLO


DATA = ROOT / "DetectDataset_clean_v2" / "data.yaml"
RUN_ROOT = ROOT / "runs" / "ablation_confirmatory_v1"
STATUS = RUN_ROOT / "status.json"
SELECTION = RUN_ROOT / "frozen_selection.json"

TRAIN_ARGS = {
    "data": str(DATA),
    "epochs": 200,
    "patience": 200,
    "batch": 16,
    "imgsz": 640,
    "device": 0,
    "workers": 4,
    "pretrained": False,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "focaler_iou": False,
    "focaler_d": 0.0,
    "focaler_u": 0.95,
    "deterministic": True,
    "amp": False,
    "close_mosaic": 10,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "cutmix": 0.0,
    "cache": False,
    "plots": True,
    "save": True,
    "save_period": -1,
    "project": str(RUN_ROOT),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    return parser.parse_args()


def now() -> str:
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(run_name: str, model_name: str, seed: int, source: str, result_path: Path, best_weight: Path) -> dict:
    curve = rows(result_path)
    if len(curve) < 200:
        raise RuntimeError(f"{run_name} only has {len(curve)} epochs")
    best = max(curve, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    return {
        "run": run_name,
        "model": model_name,
        "seed": seed,
        "source": source,
        "best_epoch": int(best["epoch"]),
        "precision": float(best["metrics/precision(B)"]),
        "recall": float(best["metrics/recall(B)"]),
        "map50": float(best["metrics/mAP50(B)"]),
        "map50_95": float(best["metrics/mAP50-95(B)"]),
        "training_time_s": float(curve[-1]["time"]),
        "best_weight": str(best_weight.resolve()),
    }


def write_summaries(records: list[dict]) -> None:
    path = RUN_ROOT / "validation_runs.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    aggregate = []
    for model_name in sorted({record["model"] for record in records}):
        subset = [record for record in records if record["model"] == model_name]
        item: dict[str, object] = {"model": model_name, "seeds": len(subset)}
        for metric in ("precision", "recall", "map50", "map50_95"):
            values = [float(record[metric]) for record in subset]
            item[f"{metric}_mean"] = statistics.mean(values)
            item[f"{metric}_sample_sd"] = statistics.stdev(values) if len(values) > 1 else None
        aggregate.append(item)
    aggregate_path = RUN_ROOT / "validation_aggregate.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)


def main() -> None:
    args = parse_args()
    selection = json.loads(args.selection.resolve().read_text(encoding="utf-8"))
    required = {
        "selected_at",
        "screen_summary",
        "rationale",
        "main_model_yaml",
        "final_model_yaml",
        "bbox_loss_mode",
    }
    missing = required - set(selection)
    if missing:
        raise ValueError(f"selection is missing {sorted(missing)}")
    main_model_yaml = Path(selection["main_model_yaml"]).resolve()
    model_yaml = Path(selection["final_model_yaml"]).resolve()
    for path in (main_model_yaml, model_yaml):
        if not path.is_file():
            raise FileNotFoundError(path)

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if SELECTION.exists():
        existing = json.loads(SELECTION.read_text(encoding="utf-8"))
        if existing != selection:
            raise RuntimeError("confirmatory selection is already frozen and differs from the requested selection")
    else:
        write_json(SELECTION, selection)

    protocol = {
        "created_at": now(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "train_args": TRAIN_ARGS,
        "selection": selection,
        "selection_rule": "best.pt by validation mAP50:95 only",
        "test_access": "forbidden until all confirmation training and model choice are frozen",
    }
    protocol_path = RUN_ROOT / "protocol.json"
    if not protocol_path.exists():
        write_json(protocol_path, protocol)

    state = {"status": "running", "started_or_resumed_at": now(), "current": None, "completed": []}
    write_json(STATUS, state)
    records = [
        summarize(
            "yolo11n_seed0",
            "YOLO11n",
            0,
            "reused_identical_protocol",
            ROOT / "runs/clean_v2_scratch_pair/yolo11n_seed0/results.csv",
            ROOT / "runs/clean_v2_scratch_pair/yolo11n_seed0/weights/best.pt",
        )
    ]
    experiments = [
        (f"yolo11n_seed{seed}", "YOLO11n", Path("yolo11n.yaml"), "ciou", seed) for seed in (1, 2)
    ]
    # If an auxiliary loss was selected, confirm its increment on the chosen
    # structure at seed 0 before running the final three-seed model.
    if selection["bbox_loss_mode"] != "ciou" or main_model_yaml != model_yaml:
        experiments.append(("main_structure_seed0", "Main structure", main_model_yaml, "ciou", 0))
    experiments += [
        (f"final_seed{seed}", "Selected method", model_yaml, selection["bbox_loss_mode"], seed)
        for seed in (0, 1, 2)
    ]
    try:
        for run_name, model_name, yaml_path, loss_mode, seed in experiments:
            run_dir = RUN_ROOT / run_name
            result_path = run_dir / "results.csv"
            best_weight = run_dir / "weights/best.pt"
            if len(rows(result_path)) >= 200 and best_weight.is_file():
                state["completed"].append(run_name)
            else:
                state["current"] = run_name
                state["phase_started_at"] = now()
                write_json(STATUS, state)
                last = run_dir / "weights/last.pt"
                if last.is_file():
                    result = YOLO(str(last)).train(resume=True)
                else:
                    if run_dir.exists():
                        raise RuntimeError(f"incomplete run without last.pt: {run_dir}")
                    result = YOLO(str(yaml_path)).train(
                        name=run_name,
                        bbox_loss_mode=loss_mode,
                        seed=seed,
                        exist_ok=False,
                        **TRAIN_ARGS,
                    )
                    run_dir = Path(result.save_dir)
                    result_path = run_dir / "results.csv"
                    best_weight = run_dir / "weights/best.pt"
                state["completed"].append(run_name)
            records.append(summarize(run_name, model_name, seed, "new_confirmatory", result_path, best_weight))
            state["current"] = None
            write_json(STATUS, state)
            write_summaries(records)
        state["status"] = "completed"
        state["completed_at"] = now()
        write_json(STATUS, state)
    except BaseException as error:
        state["status"] = "failed"
        state["failed_at"] = now()
        state["error"] = f"{type(error).__name__}: {error}"
        write_json(STATUS, state)
        raise


if __name__ == "__main__":
    main()
