#!/usr/bin/env python3
"""Run the validation-only deep residual ADown screen with safe resume."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from ultralytics import YOLO

from reproduction.ablation.run_screen import TRAIN_ARGS as SCREEN_TRAIN_ARGS


HERE = Path(__file__).resolve().parent
MODEL = HERE / "models/p2_adown_residual_deep.yaml"
PROTOCOL = HERE / "ADOWN_RESIDUAL_PROTOCOL.md"
PARENT_RUN = ROOT / "runs/ablation_screen_v1/p2_adown_ciou_s0"
RUN_ROOT = ROOT / "runs/ablation_adown_residual_v1"
RUN_NAME = "p2_adown_residual_deep_s0"
RUN_DIR = RUN_ROOT / RUN_NAME
STATUS = RUN_ROOT / "status.json"
SUMMARY = RUN_ROOT / "residual_summary.json"
TRAIN_ARGS = {**SCREEN_TRAIN_ARGS, "project": str(RUN_ROOT)}


def now() -> str:
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def rows(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "results.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def best(run_dir: Path) -> dict[str, str]:
    curve = rows(run_dir)
    if len(curve) != 60:
        raise RuntimeError(f"{run_dir} has {len(curve)} epochs, expected 60")
    return max(curve, key=lambda row: float(row["metrics/mAP50-95(B)"]))


def summarize() -> dict[str, object]:
    parent = best(PARENT_RUN)
    child = best(RUN_DIR)
    parent_model = YOLO(str(PARENT_RUN / "weights/best.pt")).model
    child_model = YOLO(str(RUN_DIR / "weights/best.pt")).model
    parent_parameters = sum(parameter.numel() for parameter in parent_model.parameters())
    child_parameters = sum(parameter.numel() for parameter in child_model.parameters())
    residual_scales = {
        str(index): float(layer.alpha.detach().float().tanh().cpu())
        for index, layer in enumerate(child_model.model)
        if hasattr(layer, "alpha")
    }
    map_delta_pp = 100.0 * (
        float(child["metrics/mAP50-95(B)"]) - float(parent["metrics/mAP50-95(B)"])
    )
    map50_delta_pp = 100.0 * (
        float(child["metrics/mAP50(B)"]) - float(parent["metrics/mAP50(B)"])
    )
    parameter_change_percent = 100.0 * (child_parameters / parent_parameters - 1.0)
    criteria = {
        "map50_95_delta_at_least_0_30_pp": map_delta_pp >= 0.30,
        "map50_drop_no_more_than_0_50_pp": map50_delta_pp >= -0.50,
        "parameter_growth_no_more_than_2_percent": parameter_change_percent <= 2.0,
        "completed_60_epochs": len(rows(RUN_DIR)) == 60,
    }
    output = {
        "protocol": str(PROTOCOL.resolve()),
        "selection_split": "validation",
        "test_access": "none",
        "parent": {
            "run": str(PARENT_RUN.resolve()),
            "best_epoch_logged": int(parent["epoch"]),
            "precision": float(parent["metrics/precision(B)"]),
            "recall": float(parent["metrics/recall(B)"]),
            "map50": float(parent["metrics/mAP50(B)"]),
            "map50_95": float(parent["metrics/mAP50-95(B)"]),
            "parameters": parent_parameters,
        },
        "child": {
            "run": str(RUN_DIR.resolve()),
            "best_epoch_logged": int(child["epoch"]),
            "precision": float(child["metrics/precision(B)"]),
            "recall": float(child["metrics/recall(B)"]),
            "map50": float(child["metrics/mAP50(B)"]),
            "map50_95": float(child["metrics/mAP50-95(B)"]),
            "parameters": child_parameters,
            "learned_tanh_alpha_by_layer": residual_scales,
        },
        "effects": {
            "map50_95_delta_pp": map_delta_pp,
            "map50_delta_pp": map50_delta_pp,
            "parameter_change_percent": parameter_change_percent,
        },
        "criteria": criteria,
        "retain_for_confirmation": all(criteria.values()),
    }
    write_json(SUMMARY, output)
    return output


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no training state was changed")
    if not (PARENT_RUN / "weights/best.pt").is_file() or len(rows(PARENT_RUN)) != 60:
        raise RuntimeError("the frozen 60-epoch P2+ADown parent is incomplete")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    protocol_record = {
        "created_at": now(),
        "protocol": str(PROTOCOL.resolve()),
        "model": str(MODEL.resolve()),
        "parent": str(PARENT_RUN.resolve()),
        "train_args": TRAIN_ARGS,
        "test_access": "forbidden",
    }
    protocol_path = RUN_ROOT / "protocol.json"
    if protocol_path.is_file():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        for key in ("protocol", "model", "parent", "train_args", "test_access"):
            if existing[key] != protocol_record[key]:
                raise RuntimeError(f"frozen protocol mismatch: {key}")
    else:
        write_json(protocol_path, protocol_record)

    state: dict[str, object] = {"status": "running", "started_or_resumed_at": now()}
    write_json(STATUS, state)
    try:
        completed = rows(RUN_DIR)
        if len(completed) < 60:
            last = RUN_DIR / "weights/last.pt"
            if last.is_file():
                YOLO(str(last)).train(resume=True)
            else:
                if RUN_DIR.exists():
                    raise RuntimeError("incomplete run directory has no last.pt")
                YOLO(str(MODEL)).train(
                    name=RUN_NAME,
                    bbox_loss_mode="ciou",
                    exist_ok=False,
                    **TRAIN_ARGS,
                )
        if len(rows(RUN_DIR)) != 60 or not (RUN_DIR / "weights/best.pt").is_file():
            raise RuntimeError("residual screen did not complete all 60 epochs")
        result = summarize()
        state.update(
            {
                "status": "completed",
                "completed_at": now(),
                "retain_for_confirmation": result["retain_for_confirmation"],
            }
        )
        write_json(STATUS, state)
    except BaseException as error:
        state.update({"status": "failed", "failed_at": now(), "error": f"{type(error).__name__}: {error}"})
        write_json(STATUS, state)
        raise


if __name__ == "__main__":
    main()
