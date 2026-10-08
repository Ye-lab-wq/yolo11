#!/usr/bin/env python3
"""Fair scratch training and held-out evaluation on DetectDataset_clean_v2."""

from __future__ import annotations

import csv
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

# Always use this repository's author-modified Ultralytics implementation, even
# when the script is launched by its path from outside the repository root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import ultralytics
from ultralytics import YOLO


RUN_ROOT = ROOT / "runs" / "clean_v2_scratch_pair"
DATA_ROOT = ROOT / "DetectDataset_clean_v2"
STATUS = RUN_ROOT / "status.json"
SUMMARY_CSV = RUN_ROOT / "test_summary.csv"

MODELS = (
    ("yolo11n_seed0", "yolo11n.yaml", False),
    (
        "meda_w025_focaler_seed0",
        str(ROOT / "reproduction" / "dut" / "models" / "meda_pro_w025.yaml"),
        True,
    ),
)

TRAIN_ARGS = {
    "data": str(DATA_ROOT / "data.yaml"),
    "epochs": 200,
    "patience": 100,
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
    "seed": 0,
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
    "exist_ok": False,
    "project": str(RUN_ROOT),
}

EVALUATIONS = (
    ("public_test", DATA_ROOT / "data_public_test.yaml"),
    ("self_test", DATA_ROOT / "data_self_test.yaml"),
    ("combined_test", DATA_ROOT / "data.yaml"),
)


def now() -> str:
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def model_metadata(model_yaml: str, focaler_iou: bool) -> dict[str, object]:
    model = YOLO(model_yaml)
    module = model.model
    return {
        "yaml": model_yaml,
        "parameters": sum(parameter.numel() for parameter in module.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in module.parameters() if parameter.requires_grad
        ),
        "scales": module.yaml.get("scales"),
        "box_loss": "Focaler-IoU (gamma=1.5)" if focaler_iou else "standard CIoU",
    }


def metric_row(model_name: str, evaluation_name: str, metrics: object) -> dict[str, object]:
    results = metrics.results_dict
    speed = metrics.speed
    return {
        "model": model_name,
        "evaluation": evaluation_name,
        "precision": float(results["metrics/precision(B)"]),
        "recall": float(results["metrics/recall(B)"]),
        "map50": float(results["metrics/mAP50(B)"]),
        "map50_95": float(results["metrics/mAP50-95(B)"]),
        "fitness": float(results["fitness"]),
        "preprocess_ms_per_image": float(speed["preprocess"]),
        "inference_ms_per_image": float(speed["inference"]),
        "postprocess_ms_per_image": float(speed["postprocess"]),
        "evaluated_at": now(),
    }


def write_summary(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if RUN_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite experiment directory: {RUN_ROOT}")
    RUN_ROOT.mkdir(parents=True)
    metadata = {
        name: model_metadata(model_yaml, focaler_iou)
        for name, model_yaml, focaler_iou in MODELS
    }
    protocol = {
        "created_at": now(),
        "purpose": "paired scratch comparison on format-aware deduplicated frame-level dataset",
        "dataset": str(DATA_ROOT),
        "selection_rule": "Ultralytics best.pt selected only by the validation split",
        "test_rule": "public, self-video, and combined test sets evaluated only after best.pt is fixed",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "models": metadata,
        "train_args": TRAIN_ARGS,
        "evaluations": {name: str(path) for name, path in EVALUATIONS},
    }
    write_json(RUN_ROOT / "protocol.json", protocol)
    state: dict[str, object] = {
        "status": "running",
        "started_at": now(),
        "current_phase": None,
        "completed_phases": [],
        "models": metadata,
    }
    write_json(STATUS, state)
    rows: list[dict[str, object]] = []
    try:
        for model_name, model_yaml, focaler_iou in MODELS:
            state["current_phase"] = f"train:{model_name}"
            state["phase_started_at"] = now()
            write_json(STATUS, state)
            print(f"\n===== TRAIN {model_name} =====", flush=True)
            model = YOLO(model_yaml)
            train_result = model.train(name=model_name, focaler_iou=focaler_iou, **TRAIN_ARGS)
            save_dir = Path(train_result.save_dir)
            best = save_dir / "weights" / "best.pt"
            if not best.is_file():
                raise FileNotFoundError(f"training did not produce best.pt: {best}")
            state["completed_phases"].append(f"train:{model_name}")
            state.setdefault("best_weights", {})[model_name] = str(best)
            write_json(STATUS, state)

            for evaluation_name, data_yaml in EVALUATIONS:
                phase = f"eval:{model_name}:{evaluation_name}"
                state["current_phase"] = phase
                state["phase_started_at"] = now()
                write_json(STATUS, state)
                print(f"\n===== EVAL {model_name} / {evaluation_name} =====", flush=True)
                evaluator = YOLO(str(best))
                metrics = evaluator.val(
                    data=str(data_yaml),
                    split="test",
                    imgsz=640,
                    batch=16,
                    device=0,
                    workers=4,
                    amp=False,
                    plots=False,
                    project=str(RUN_ROOT / "evaluations"),
                    name=f"{model_name}_{evaluation_name}",
                    exist_ok=False,
                )
                rows.append(metric_row(model_name, evaluation_name, metrics))
                write_summary(rows)
                state["completed_phases"].append(phase)
                state["latest_metrics"] = rows[-1]
                write_json(STATUS, state)

        state["status"] = "completed"
        state["current_phase"] = None
        state["completed_at"] = now()
        write_json(STATUS, state)
        print(f"\nAll phases completed. Summary: {SUMMARY_CSV}", flush=True)
    except BaseException as error:
        state["status"] = "failed"
        state["failed_at"] = now()
        state["error"] = f"{type(error).__name__}: {error}"
        write_json(STATUS, state)
        raise


if __name__ == "__main__":
    main()
