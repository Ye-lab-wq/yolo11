#!/usr/bin/env python3
"""Run the frozen seed-0/60-epoch validation-only MEDA component screen."""

from __future__ import annotations

import csv
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import ultralytics
from ultralytics import YOLO


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"
DATA = ROOT / "DetectDataset_clean_v2" / "data.yaml"
RUN_ROOT = ROOT / "runs" / "ablation_screen_v1"
STATUS = RUN_ROOT / "status.json"
SUMMARY = RUN_ROOT / "screen_summary.csv"

# One parent plus one changed factor per row. The two loss rows deliberately reuse
# the identical p2_nearest graph, so their inference graph is exactly unchanged.
EXPERIMENTS = (
    ("yolo11n_ciou_s0", "@root:ultralytics/cfg/models/11/yolo11.yaml", "ciou"),
    (
        "full_meda_author_focal_s0",
        "@root:reproduction/dut/models/meda_pro_w025.yaml",
        "author_focal",
    ),
    ("p2_nearest_ciou_s0", "p2_nearest.yaml", "ciou"),
    ("p2_no_c2psa_ciou_s0", "p2_no_c2psa.yaml", "ciou"),
    ("p2_adown_ciou_s0", "p2_adown.yaml", "ciou"),
    ("p2_author_bilinear_ciou_s0", "p2_author_bilinear.yaml", "ciou"),
    ("p2_official_dysample_ciou_s0", "p2_official_dysample.yaml", "ciou"),
    ("p2_rep_ciou_s0", "p2_rep.yaml", "ciou"),
    ("p2_rep_official_ciou_s0", "p2_rep_official.yaml", "ciou"),
    ("p2_msef_author_ciou_s0", "p2_msef_author.yaml", "ciou"),
    ("p2_msef_paper_ciou_s0", "p2_msef_paper.yaml", "ciou"),
    ("p2_msef_author_ema_ciou_s0", "p2_msef_author_ema.yaml", "ciou"),
    ("p2_p5_msef_author_ciou_s0", "p2_p5_msef_author.yaml", "ciou"),
    ("p2_nearest_author_focal_s0", "p2_nearest.yaml", "author_focal"),
    ("p2_nearest_focaler_ciou_s0", "p2_nearest.yaml", "focaler_ciou"),
)

TRAIN_ARGS = {
    "data": str(DATA),
    "epochs": 60,
    "patience": 60,
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
    "plots": False,
    "save": True,
    "save_period": -1,
    "project": str(RUN_ROOT),
}

METRIC_COLUMNS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)
PARAMETER_CACHE: dict[str, int] = {}
REFERENCE_PARAMETERS = {
    # Counted from nc=1 checkpoints. Instantiating the stock yolo11.yaml
    # directly would retain its default 80-class head and overcount.
    "yolo11n_ciou_s0": 2_590_035,
    "full_meda_author_focal_s0": 2_844_348,
    "yolo11n_ciou_s0_reference": 2_590_035,
    "full_meda_author_focal_s0_reference": 2_844_348,
}


def model_path(yaml_name: str) -> Path:
    return ROOT / yaml_name.removeprefix("@root:") if yaml_name.startswith("@root:") else MODEL_DIR / yaml_name


def now() -> str:
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def result_rows(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:limit] if limit is not None else rows


def summarize(name: str, yaml_path: str, loss_mode: str, rows: list[dict[str, str]], source: str) -> dict[str, object]:
    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    if name in REFERENCE_PARAMETERS:
        parameters = REFERENCE_PARAMETERS[name]
    else:
        if yaml_path not in PARAMETER_CACHE:
            PARAMETER_CACHE[yaml_path] = sum(parameter.numel() for parameter in YOLO(yaml_path).model.parameters())
        parameters = PARAMETER_CACHE[yaml_path]
    return {
        "experiment": name,
        "source": source,
        "model_yaml": yaml_path,
        "bbox_loss_mode": loss_mode,
        "epochs_observed": len(rows),
        "best_epoch": int(best["epoch"]),
        "precision": float(best["metrics/precision(B)"]),
        "recall": float(best["metrics/recall(B)"]),
        "map50": float(best["metrics/mAP50(B)"]),
        "map50_95": float(best["metrics/mAP50-95(B)"]),
        "training_time_s_at_end": float(rows[-1]["time"]),
        "parameters": parameters,
        "best_weight": "" if source.startswith("noncomparable_") else str(RUN_ROOT / name / "weights" / "best.pt"),
    }


def collect_summary() -> list[dict[str, object]]:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    # Reuse parameter counts already audited in the summary. Re-instantiating
    # every completed architecture on each resume is expensive and can keep all
    # CPU cores busy for minutes before the CUDA training process is created.
    if SUMMARY.is_file():
        for cached in result_rows(SUMMARY):
            yaml_path = cached.get("model_yaml", "")
            parameters = cached.get("parameters", "")
            if yaml_path and parameters:
                PARAMETER_CACHE[yaml_path] = int(parameters)
    rows: list[dict[str, object]] = []
    # These legacy curves are retained separately for audit only. A 200-epoch
    # scheduler truncated at epoch 60 is not comparable to a true 60-epoch run.
    noncomparable_references = (
        (
            "yolo11n_ciou_s0_reference",
            "yolo11n.yaml",
            "ciou",
            ROOT / "runs" / "clean_v2_scratch_pair" / "yolo11n_seed0" / "results.csv",
        ),
        (
            "full_meda_author_focal_s0_reference",
            str(ROOT / "reproduction" / "dut" / "models" / "meda_pro_w025.yaml"),
            "author_focal",
            ROOT / "runs" / "clean_v2_scratch_pair" / "meda_w025_focaler_seed0" / "results.csv",
        ),
    )
    noncomparable_rows = []
    for name, yaml_path, loss_mode, csv_path in noncomparable_references:
        curve = result_rows(csv_path, 60)
        if len(curve) == 60:
            noncomparable_rows.append(summarize(name, yaml_path, loss_mode, curve, "noncomparable_200epoch_curve_first60"))
    if noncomparable_rows:
        with (RUN_ROOT / "noncomparable_first60_references.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(noncomparable_rows[0]))
            writer.writeheader()
            writer.writerows(noncomparable_rows)
    for name, yaml_name, loss_mode in EXPERIMENTS:
        curve = result_rows(RUN_ROOT / name / "results.csv")
        if len(curve) >= TRAIN_ARGS["epochs"]:
            rows.append(summarize(name, str(model_path(yaml_name)), loss_mode, curve, "new_screen"))
    if rows:
        with SUMMARY.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "created_at": now(),
        "frozen_protocol": str(HERE / "PROTOCOL.md"),
        "dataset": str(DATA),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "train_args": TRAIN_ARGS,
        "experiments": [
            {"name": name, "model": str(model_path(yaml_name)), "bbox_loss_mode": loss_mode}
            for name, yaml_name, loss_mode in EXPERIMENTS
        ],
        "test_access": "forbidden during screen",
    }
    protocol_path = RUN_ROOT / "protocol.json"
    if not protocol_path.exists():
        write_json(protocol_path, protocol)

    state = {
        "status": "running",
        "started_or_resumed_at": now(),
        "current_experiment": None,
        "completed": [],
    }
    write_json(STATUS, state)
    collect_summary()
    try:
        for name, yaml_name, loss_mode in EXPERIMENTS:
            run_dir = RUN_ROOT / name
            rows = result_rows(run_dir / "results.csv")
            if len(rows) >= TRAIN_ARGS["epochs"] and (run_dir / "weights" / "best.pt").is_file():
                state["completed"].append(name)
                write_json(STATUS, state)
                continue

            state["current_experiment"] = name
            state["phase_started_at"] = now()
            write_json(STATUS, state)
            last = run_dir / "weights" / "last.pt"
            if last.is_file():
                print(f"\n===== RESUME {name} =====", flush=True)
                result = YOLO(str(last)).train(resume=True)
            else:
                if run_dir.exists():
                    raise RuntimeError(f"incomplete run without resumable last.pt: {run_dir}")
                print(f"\n===== SCREEN {name} =====", flush=True)
                result = YOLO(str(model_path(yaml_name))).train(
                    name=name,
                    bbox_loss_mode=loss_mode,
                    exist_ok=False,
                    **TRAIN_ARGS,
                )
            completed_rows = result_rows(Path(result.save_dir) / "results.csv")
            if len(completed_rows) < TRAIN_ARGS["epochs"]:
                raise RuntimeError(f"{name} stopped at {len(completed_rows)} epochs")
            state["completed"].append(name)
            state["current_experiment"] = None
            write_json(STATUS, state)
            collect_summary()

        state["status"] = "completed"
        state["completed_at"] = now()
        write_json(STATUS, state)
        collect_summary()
    except BaseException as error:
        state["status"] = "failed"
        state["failed_at"] = now()
        state["error"] = f"{type(error).__name__}: {error}"
        write_json(STATUS, state)
        raise


if __name__ == "__main__":
    main()
