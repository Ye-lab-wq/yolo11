#!/usr/bin/env python3
"""Run the resumable P2 x MSEFPaper motion-blur training screen."""

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
RUN_ROOT = ROOT / "runs/ablation_motion_blur_v1"
STATUS = RUN_ROOT / "status.json"
SUMMARY = RUN_ROOT / "clean_validation_summary.csv"
PROTOCOL = HERE / "MOTION_BLUR_TRAINING_PROTOCOL.md"
EXPERIMENTS = (
    ("p2_motion_blur_s0", HERE / "models/p2_nearest.yaml"),
    ("p2_msef_motion_blur_s0", HERE / "models/p2_msef_paper.yaml"),
)
TRAIN_ARGS = {
    **SCREEN_TRAIN_ARGS,
    "project": str(RUN_ROOT),
    "motion_blur": 0.30,
    "motion_blur_min": 3,
    "motion_blur_max": 7,
}


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


def collect() -> list[dict[str, object]]:
    output = []
    for name, model_yaml in EXPERIMENTS:
        run_dir = RUN_ROOT / name
        curve = rows(run_dir)
        if len(curve) != 60 or not (run_dir / "weights/best.pt").is_file():
            continue
        best = max(curve, key=lambda item: float(item["metrics/mAP50-95(B)"]))
        output.append(
            {
                "experiment": name,
                "model": str(model_yaml.resolve()),
                "best_epoch_logged": int(best["epoch"]),
                "precision": float(best["metrics/precision(B)"]),
                "recall": float(best["metrics/recall(B)"]),
                "map50": float(best["metrics/mAP50(B)"]),
                "map50_95": float(best["metrics/mAP50-95(B)"]),
                "weights": str((run_dir / "weights/best.pt").resolve()),
            }
        )
    if output:
        with SUMMARY.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output[0]))
            writer.writeheader()
            writer.writerows(output)
    return output


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no motion-blur training state was changed")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    protocol_record = {
        "created_at": now(),
        "protocol": str(PROTOCOL.resolve()),
        "experiments": [{"name": name, "model": str(model.resolve())} for name, model in EXPERIMENTS],
        "train_args": TRAIN_ARGS,
        "selection": "clean validation mAP50:95",
        "test_access": "forbidden",
    }
    protocol_path = RUN_ROOT / "protocol.json"
    if protocol_path.is_file():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        for key in ("protocol", "experiments", "train_args", "selection", "test_access"):
            if existing[key] != protocol_record[key]:
                raise RuntimeError(f"frozen protocol mismatch: {key}")
    else:
        write_json(protocol_path, protocol_record)

    state: dict[str, object] = {"status": "running", "started_or_resumed_at": now(), "completed": []}
    write_json(STATUS, state)
    try:
        for name, model_yaml in EXPERIMENTS:
            run_dir = RUN_ROOT / name
            if len(rows(run_dir)) == 60 and (run_dir / "weights/best.pt").is_file():
                state["completed"].append(name)
                write_json(STATUS, state)
                continue
            state["current"] = name
            write_json(STATUS, state)
            last = run_dir / "weights/last.pt"
            if last.is_file():
                YOLO(str(last)).train(resume=True)
            else:
                if run_dir.exists():
                    raise RuntimeError(f"incomplete run without last.pt: {run_dir}")
                YOLO(str(model_yaml)).train(
                    name=name,
                    bbox_loss_mode="ciou",
                    exist_ok=False,
                    **TRAIN_ARGS,
                )
            if len(rows(run_dir)) != 60 or not (run_dir / "weights/best.pt").is_file():
                raise RuntimeError(f"{name} did not complete 60 epochs")
            state["completed"].append(name)
            state["current"] = None
            write_json(STATUS, state)
            collect()
        state.update({"status": "completed", "completed_at": now(), "current": None})
        write_json(STATUS, state)
        collect()
    except BaseException as error:
        state.update({"status": "failed", "failed_at": now(), "error": f"{type(error).__name__}: {error}"})
        write_json(STATUS, state)
        raise


if __name__ == "__main__":
    main()
