#!/usr/bin/env python3
"""Train a frozen 200-epoch model, then test each seed.

This runner covers the final candidate, baseline, and the two missing sequential
ablation cells (P2 and P2+MSEFPaper). Validation selects ``best.pt``; the public
and self-video test splits are evaluated only after that checkpoint is frozen.
Re-running the same command skips completed runs and resumes incomplete runs
from their own ``last.pt``. The original final-model layouts are retained for
backward compatibility.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import ultralytics
from ultralytics import YOLO


# Defaults preserve the historical final-candidate layout. ``main`` switches
# these paths when the same controlled runner is used for the YOLO11n baseline.
RUN_ROOT = ROOT / "runs" / "final_candidate_p2_msef_adown_s0_200"
MODEL_YAML = ROOT / "reproduction" / "ablation" / "models" / "p2_msef_paper_adown.yaml"
MODEL_LABEL = "P2 + adapted MSEF + full-path ADown"
EVAL_LABEL = "P2_MSEFadapt_ADown"
DATA_YAML = ROOT / "DetectDataset_clean_v2" / "data.yaml"
EVALUATOR = ROOT / "reproduction" / "ablation" / "evaluate_frozen.py"

TRAIN_ARGS = {
    "data": str(DATA_YAML),
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
    "bbox_loss_mode": "ciou",
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
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("candidate", "baseline", "p2", "p2_msef", "p2_adown"),
        default=["candidate"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], choices=(0, 1, 2))
    return parser.parse_args()


def train_dir(seed: int) -> Path:
    return RUN_ROOT / f"train_seed{seed}"


def status_path(seed: int) -> Path:
    return RUN_ROOT / ("status.json" if seed == 0 else f"status_seed{seed}.json")


def protocol_path(seed: int) -> Path:
    return RUN_ROOT / ("protocol.json" if seed == 0 else f"protocol_seed{seed}.json")


def frozen_path(seed: int) -> Path:
    return RUN_ROOT / ("frozen_model.json" if seed == 0 else f"frozen_model_seed{seed}.json")


def now() -> str:
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def result_rows(seed: int) -> list[dict[str, str]]:
    path = train_dir(seed) / "results.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def freeze_best(seed: int) -> Path:
    rows = result_rows(seed)
    best_weight = train_dir(seed) / "weights" / "best.pt"
    if len(rows) != 200 or not best_weight.is_file():
        raise RuntimeError(f"expected 200 completed epochs and best.pt, found {len(rows)} epochs")
    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    frozen_metadata = {
        "frozen_at": now(),
        "seed": seed,
        "selection_split": "validation",
        "selection_rule": "maximum validation mAP50:95 over the fixed 200-epoch run",
        "best_epoch_zero_based": int(best["epoch"]),
        "precision": float(best["metrics/precision(B)"]),
        "recall": float(best["metrics/recall(B)"]),
        "map50": float(best["metrics/mAP50(B)"]),
        "map50_95": float(best["metrics/mAP50-95(B)"]),
        "weights": str(best_weight.resolve()),
        "test_access_before_freeze": "none",
    }
    frozen_file = frozen_path(seed)
    if frozen_file.exists():
        existing = json.loads(frozen_file.read_text(encoding="utf-8"))
        expected = {key: value for key, value in frozen_metadata.items() if key != "frozen_at"}
        actual = {key: value for key, value in existing.items() if key != "frozen_at"}
        # The original seed-0 record predates the explicit seed field.
        if seed == 0 and "seed" not in actual:
            expected.pop("seed")
        if actual != expected:
            raise RuntimeError("frozen checkpoint metadata changed")
    else:
        write_json(frozen_file, frozen_metadata)
    return best_weight


def evaluate(split: str, seed: int, best_weight: Path) -> Path:
    output = (
        RUN_ROOT / "test_results" / f"{split}.json"
        if seed == 0
        else RUN_ROOT / "test_results" / f"seed{seed}" / f"{split}.json"
    )
    if output.is_file():
        return output
    subprocess.run(
        [
            sys.executable,
            str(EVALUATOR),
            "--weights",
            f"{EVAL_LABEL}_seed{seed}={best_weight}",
            "--split",
            split,
            "--output",
            str(output),
            "--device",
            "0",
            "--batch",
            "4",
        ],
        cwd=ROOT,
        check=True,
    )
    return output


def run_seed(seed: int) -> None:
    if not MODEL_YAML.is_file() or not DATA_YAML.is_file():
        raise FileNotFoundError("model YAML or dataset YAML is missing")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    current_train_dir = train_dir(seed)
    current_status = status_path(seed)
    current_protocol = protocol_path(seed)
    model = YOLO(str(MODEL_YAML))
    train_args = {**TRAIN_ARGS, "seed": seed, "project": str(RUN_ROOT)}
    metadata = {
        "created_at": now(),
        "purpose": "three-seed full-budget final-model training",
        "seed": seed,
        "model_label": MODEL_LABEL,
        "model_yaml": str(MODEL_YAML),
        "parameters": sum(parameter.numel() for parameter in model.model.parameters()),
        "environment": {
            "python_executable": sys.executable,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "train_args": train_args,
        "selection_rule": "best.pt by validation mAP50:95 only",
        "test_rule": "public and self-video test evaluated separately after checkpoint freeze",
    }
    if current_protocol.exists():
        existing = json.loads(current_protocol.read_text(encoding="utf-8"))
        if existing["train_args"] != metadata["train_args"] or existing["model_yaml"] != metadata["model_yaml"]:
            raise RuntimeError("existing protocol differs from current frozen protocol")
    else:
        write_json(current_protocol, metadata)

    state: dict[str, object] = {
        "status": "running",
        "started_or_resumed_at": now(),
        "current_phase": "train",
        "completed_phases": [],
    }
    write_json(current_status, state)
    try:
        rows = result_rows(seed)
        if len(rows) < 200:
            last = current_train_dir / "weights" / "last.pt"
            if last.is_file():
                YOLO(str(last)).train(resume=True)
            else:
                if current_train_dir.exists():
                    raise RuntimeError("incomplete training directory has no resumable last.pt")
                model.train(name=current_train_dir.name, exist_ok=False, **train_args)
        state["completed_phases"].append("train_200_epochs")
        state["current_phase"] = "freeze_best"
        write_json(current_status, state)
        best = freeze_best(seed)
        state["completed_phases"].append("freeze_best_on_validation")
        for split in ("combined_test", "public_test", "self_test"):
            state["current_phase"] = f"evaluate_{split}"
            write_json(current_status, state)
            evaluate(split, seed, best)
            state["completed_phases"].append(f"evaluate_{split}")
        state["status"] = "completed"
        state["current_phase"] = None
        state["completed_at"] = now()
        write_json(current_status, state)
    except BaseException as error:
        state["status"] = "failed"
        state["failed_at"] = now()
        state["error"] = f"{type(error).__name__}: {error}"
        write_json(current_status, state)
        raise


def main() -> None:
    global RUN_ROOT, MODEL_YAML, MODEL_LABEL, EVAL_LABEL
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device 0 is unavailable; training was not started. "
            "Check nvidia-smi, then rerun the same command."
        )
    configurations = {
        "candidate": (
            ROOT / "runs" / "final_candidate_p2_msef_adown_s0_200",
            ROOT / "reproduction" / "ablation" / "models" / "p2_msef_paper_adown.yaml",
            "P2 + adapted MSEF + full-path ADown",
            "P2_MSEFadapt_ADown",
        ),
        "baseline": (
            ROOT / "runs" / "final_baseline_yolo11n_200",
            ROOT / "ultralytics" / "cfg" / "models" / "11" / "yolo11.yaml",
            "YOLO11n baseline",
            "YOLO11n",
        ),
        "p2": (
            ROOT / "runs" / "ablation_200_v1" / "p2",
            ROOT / "reproduction" / "ablation" / "models" / "p2_nearest.yaml",
            "YOLO11n + P2",
            "YOLO11n_P2",
        ),
        "p2_msef": (
            ROOT / "runs" / "ablation_200_v1" / "p2_msef_paper",
            ROOT / "reproduction" / "ablation" / "models" / "p2_msef_paper.yaml",
            "YOLO11n + P2 + MSEFPaper",
            "YOLO11n_P2_MSEFPaper",
        ),
        "p2_adown": (
            ROOT / "runs" / "ablation_200_v1" / "p2_adown",
            ROOT / "reproduction" / "ablation" / "models" / "p2_adown.yaml",
            "YOLO11n + P2 + full-path ADown",
            "YOLO11n_P2_ADown",
        ),
    }
    for model_name in args.models:
        RUN_ROOT, MODEL_YAML, MODEL_LABEL, EVAL_LABEL = configurations[model_name]
        for seed in args.seeds:
            run_seed(seed)


if __name__ == "__main__":
    main()
