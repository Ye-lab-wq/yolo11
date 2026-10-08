#!/usr/bin/env python3
"""Run the frozen P2+MSEFPaper secondary one-factor validation screen."""

from __future__ import annotations

import csv
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import ultralytics
from ultralytics import YOLO


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"
DATA = ROOT / "DetectDataset_clean_v2" / "data.yaml"
PARENT_ROOT = ROOT / "runs" / "ablation_screen_v1"
PARENT_RUN = PARENT_ROOT / "p2_msef_paper_ciou_s0"
PARENT_SUMMARY = PARENT_ROOT / "screen_summary.csv"
RUN_ROOT = ROOT / "runs" / "ablation_secondary_v1"
STATUS = RUN_ROOT / "status.json"
SUMMARY = RUN_ROOT / "secondary_summary.csv"
VERIFICATION = RUN_ROOT / "protocol_verification.json"

# The loss-only rows intentionally share the exact parent inference graph.
EXPERIMENTS = (
    ("msefbase_adown_ciou_s0", "ADown", "p2_msef_paper_adown.yaml", "ciou"),
    (
        "msefbase_author_bilinear_ciou_s0",
        "Uploaded static bilinear upsampling",
        "p2_msef_paper_author_bilinear.yaml",
        "ciou",
    ),
    (
        "msefbase_official_dysample_ciou_s0",
        "Reference DySampleOfficial",
        "p2_msef_paper_official_dysample.yaml",
        "ciou",
    ),
    ("msefbase_author_elan_ciou_s0", "Uploaded ordinary-Conv ELAN", "p2_msef_paper_rep.yaml", "ciou"),
    (
        "msefbase_official_rep_ciou_s0",
        "RepNCSPELAN4Official",
        "p2_msef_paper_rep_official.yaml",
        "ciou",
    ),
    ("msefbase_ema_ciou_s0", "EMA after P2 MSEFPaper", "p2_msef_paper_ema.yaml", "ciou"),
    ("msefbase_p5_msef_paper_ciou_s0", "MSEFPaper at P5", "p2_msef_paper_p5.yaml", "ciou"),
    ("msefbase_no_c2psa_ciou_s0", "Remove C2PSA", "p2_msef_paper_no_c2psa.yaml", "ciou"),
    ("msefbase_author_power_iou_s0", "Uploaded power-IoU loss", "p2_msef_paper.yaml", "author_focal"),
    ("msefbase_focaler_ciou_s0", "Reference Focaler-CIoU", "p2_msef_paper.yaml", "focaler_ciou"),
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

ALLOWED_ARG_DIFFERENCES = {
    "bbox_loss_mode",
    "focaler_iou",
    "model",
    "name",
    "project",
    "resume",
    "save_dir",
}
PARAMETER_CACHE: dict[str, int] = {}


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


def best_metrics(curve: list[dict[str, str]]) -> dict[str, object]:
    best = max(curve, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    return {
        "best_epoch": int(best["epoch"]),
        "precision": float(best["metrics/precision(B)"]),
        "recall": float(best["metrics/recall(B)"]),
        "map50": float(best["metrics/mAP50(B)"]),
        "map50_95": float(best["metrics/mAP50-95(B)"]),
    }


def parent_record() -> dict[str, object]:
    candidates = [row for row in rows(PARENT_SUMMARY) if row["experiment"] == PARENT_RUN.name]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one frozen parent row, found {len(candidates)}")
    source = candidates[0]
    return {
        "experiment": PARENT_RUN.name,
        "factor": "Direct parent: P2+MSEFPaper",
        "source": "reused_stage1_parent",
        "model_yaml": source["model_yaml"],
        "bbox_loss_mode": source["bbox_loss_mode"],
        "epochs_observed": int(source["epochs_observed"]),
        "best_epoch": int(source["best_epoch"]),
        "precision": float(source["precision"]),
        "recall": float(source["recall"]),
        "map50": float(source["map50"]),
        "map50_95": float(source["map50_95"]),
        "delta_map50_95_pp": 0.0,
        "parameters": int(source["parameters"]),
        "best_weight": source["best_weight"],
        "decision_accuracy_only": "parent",
    }


def collect_summary() -> list[dict[str, object]]:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if SUMMARY.is_file():
        for cached in rows(SUMMARY):
            if cached.get("model_yaml") and cached.get("parameters"):
                PARAMETER_CACHE[cached["model_yaml"]] = int(cached["parameters"])
    parent = parent_record()
    records = [parent]
    for name, factor, yaml_name, loss_mode in EXPERIMENTS:
        run_dir = RUN_ROOT / name
        curve = rows(run_dir / "results.csv")
        best_weight = run_dir / "weights" / "best.pt"
        if len(curve) < 60 or not best_weight.is_file():
            continue
        yaml_path = str((MODEL_DIR / yaml_name).resolve())
        if yaml_path not in PARAMETER_CACHE:
            PARAMETER_CACHE[yaml_path] = sum(parameter.numel() for parameter in YOLO(str(best_weight)).model.parameters())
        metrics = best_metrics(curve)
        delta_pp = (float(metrics["map50_95"]) - float(parent["map50_95"])) * 100.0
        records.append(
            {
                "experiment": name,
                "factor": factor,
                "source": "new_secondary_screen",
                "model_yaml": yaml_path,
                "bbox_loss_mode": loss_mode,
                "epochs_observed": len(curve),
                **metrics,
                "delta_map50_95_pp": delta_pp,
                "parameters": PARAMETER_CACHE[yaml_path],
                "best_weight": str(best_weight.resolve()),
                "decision_accuracy_only": "advance" if delta_pp >= 0.30 else "do_not_advance",
            }
        )
    with SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return records


def verify() -> bool:
    parent_args = yaml.safe_load((PARENT_RUN / "args.yaml").read_text(encoding="utf-8"))
    records = []
    valid_all = True
    for name, factor, _yaml_name, loss_mode in EXPERIMENTS:
        run_dir = RUN_ROOT / name
        args_path = run_dir / "args.yaml"
        curve = rows(run_dir / "results.csv")
        best = run_dir / "weights" / "best.pt"
        last = run_dir / "weights" / "last.pt"
        if not args_path.is_file():
            valid = False
            differences = {"missing": str(args_path)}
        else:
            args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
            differences = {
                key: [parent_args.get(key), args.get(key)]
                for key in sorted(set(parent_args) | set(args))
                if key not in ALLOWED_ARG_DIFFERENCES and parent_args.get(key) != args.get(key)
            }
            valid = (
                len(curve) == 60
                and not differences
                and best.is_file()
                and last.is_file()
                and args.get("bbox_loss_mode") == loss_mode
                and args.get("pretrained") is False
                and args.get("seed") == 0
                and Path(args.get("data", "")).resolve() == DATA.resolve()
                and args.get("split") == "val"
            )
        valid_all &= valid
        records.append(
            {
                "run": name,
                "factor": factor,
                "valid": valid,
                "epochs": len(curve),
                "unexpected_argument_differences": differences,
            }
        )
    write_json(
        VERIFICATION,
        {
            "valid": valid_all,
            "direct_parent": str(PARENT_RUN.resolve()),
            "expected_epochs": 60,
            "allowed_argument_differences": sorted(ALLOWED_ARG_DIFFERENCES),
            "test_access": "none",
            "runs": records,
        },
    )
    return valid_all


def main() -> None:
    if len(rows(PARENT_RUN / "results.csv")) != 60:
        raise RuntimeError("frozen P2+MSEFPaper parent is not a complete true 60-epoch run")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    protocol_path = RUN_ROOT / "protocol.json"
    protocol = {
        "created_at": now(),
        "frozen_protocol": str((HERE / "SECONDARY_PROTOCOL.md").resolve()),
        "direct_parent": str(PARENT_RUN.resolve()),
        "dataset": str(DATA.resolve()),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "train_args": TRAIN_ARGS,
        "experiments": [
            {
                "name": name,
                "factor": factor,
                "model": str((MODEL_DIR / yaml_name).resolve()),
                "bbox_loss_mode": loss_mode,
            }
            for name, factor, yaml_name, loss_mode in EXPERIMENTS
        ],
        "selection_split": "val",
        "test_access": "forbidden during secondary screen",
    }
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing["experiments"] != protocol["experiments"] or existing["train_args"] != protocol["train_args"]:
            raise RuntimeError("secondary protocol is already frozen and differs from current code")
    else:
        write_json(protocol_path, protocol)

    state = {"status": "running", "started_or_resumed_at": now(), "current": None, "completed": []}
    write_json(STATUS, state)
    collect_summary()
    try:
        for name, factor, yaml_name, loss_mode in EXPERIMENTS:
            run_dir = RUN_ROOT / name
            curve = rows(run_dir / "results.csv")
            if len(curve) >= 60 and (run_dir / "weights" / "best.pt").is_file():
                state["completed"].append(name)
                write_json(STATUS, state)
                continue
            state["current"] = name
            state["factor"] = factor
            state["phase_started_at"] = now()
            write_json(STATUS, state)
            last = run_dir / "weights" / "last.pt"
            if last.is_file():
                print(f"\n===== RESUME {name}: {factor} =====", flush=True)
                result = YOLO(str(last)).train(resume=True)
            else:
                if run_dir.exists():
                    raise RuntimeError(f"incomplete run without last.pt: {run_dir}")
                print(f"\n===== SECONDARY {name}: {factor} =====", flush=True)
                result = YOLO(str(MODEL_DIR / yaml_name)).train(
                    name=name,
                    bbox_loss_mode=loss_mode,
                    exist_ok=False,
                    **TRAIN_ARGS,
                )
            if len(rows(Path(result.save_dir) / "results.csv")) < 60:
                raise RuntimeError(f"{name} stopped before 60 epochs")
            state["completed"].append(name)
            state["current"] = None
            write_json(STATUS, state)
            collect_summary()

        state["status"] = "completed"
        state["completed_at"] = now()
        state["protocol_valid"] = verify()
        if not state["protocol_valid"]:
            raise RuntimeError("secondary protocol verification failed")
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
