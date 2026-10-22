#!/usr/bin/env python3
"""Resumable seed-0/60-epoch validation screen for the exploratory MSEFHighPass."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import ultralytics
from ultralytics import YOLO

from reproduction.ablation.run_screen import TRAIN_ARGS

NAME = "p2_msef_highpass_ciou_s0"
SOURCE = ROOT / "reproduction/ablation/models/p2_msef_paper.yaml"
MODEL = ROOT / "reproduction/ablation/models/p2_msef_highpass.yaml"
RUN_ROOT = ROOT / "runs/msef_highpass_screen_v1"
RUN_DIR = RUN_ROOT / NAME


def write_new_model() -> None:
    source = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    replacements = 0
    for layer in source["head"]:
        if layer[2] == "MSEFPaper":
            layer[2] = "MSEFHighPass"
            replacements += 1
    if replacements != 1:
        raise RuntimeError(f"Expected exactly one P2 MSEFPaper layer, found {replacements}")
    expected = yaml.safe_dump(source, sort_keys=False)
    if MODEL.exists():
        if MODEL.read_text(encoding="utf-8") != expected:
            raise RuntimeError(f"Existing model YAML differs from frozen candidate: {MODEL}")
    else:
        MODEL.write_text(expected, encoding="utf-8")


def read_curve() -> list[dict[str, str]]:
    path = RUN_DIR / "results.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify model and module without training")
    args = parser.parse_args()
    write_new_model()
    model = YOLO(str(MODEL))
    parameters = sum(p.numel() for p in model.model.parameters())
    print(f"model={MODEL} parameters={parameters} torch={torch.__version__} ultralytics={ultralytics.__version__}", flush=True)
    if args.check:
        from ultralytics.nn.modules import MSEFHighPass

        block = MSEFHighPass(16, 16).eval()
        with torch.no_grad():
            constant = torch.full((1, 8, 16, 16), 3.0)
            assert torch.allclose(block._high_pass(constant, 3), torch.zeros_like(constant))
            assert torch.allclose(block._high_pass(constant, 5), torch.zeros_like(constant))
            sample = torch.randn((1, 16, 32, 32))
            assert torch.allclose(block(sample), block.base(block.local(sample)))
        print("High-pass constant-map and zero-init identity checks passed", flush=True)
        return

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    protocol_path = RUN_ROOT / "protocol.json"
    train_args = {**TRAIN_ARGS, "project": str(RUN_ROOT)}
    protocol = {
        "candidate": "MSEFHighPass: same-feature 3/5 high-pass, gated residual, zero-initialized scale",
        "status": "exploratory validation-only screen; not original MEDA-YOLO",
        "parent_model": str(SOURCE),
        "model": str(MODEL),
        "reference_run": str(ROOT / "runs/ablation_screen_v1/p2_msef_paper_ciou_s0"),
        "p2_reference_run": str(ROOT / "runs/ablation_screen_v1/p2_nearest_ciou_s0"),
        "train_args": train_args,
        "bbox_loss_mode": "ciou",
        "environment": {"python": sys.version, "torch": torch.__version__, "ultralytics": ultralytics.__version__},
        "selection": "best validation mAP50-95 within 60 epochs; no test access",
    }
    if protocol_path.exists():
        saved = json.loads(protocol_path.read_text(encoding="utf-8"))
        if saved != protocol:
            raise RuntimeError(f"Frozen protocol differs: {protocol_path}")
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rows = read_curve()
    best_weight = RUN_DIR / "weights/best.pt"
    if len(rows) < train_args["epochs"] or not best_weight.exists():
        last_weight = RUN_DIR / "weights/last.pt"
        if last_weight.exists():
            print(f"Resuming from {last_weight}", flush=True)
            YOLO(str(last_weight)).train(resume=True)
        elif RUN_DIR.exists():
            raise RuntimeError(f"Incomplete run without last.pt: {RUN_DIR}")
        else:
            model.train(name=NAME, bbox_loss_mode="ciou", exist_ok=False, **train_args)
        rows = read_curve()
    if len(rows) < train_args["epochs"] or not best_weight.exists():
        raise RuntimeError(f"Training incomplete: {len(rows)} / {train_args['epochs']} epochs")
    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    summary = {
        "run": NAME,
        "dataset": train_args["data"],
        "epochs": len(rows),
        "best_epoch": int(best["epoch"]),
        "precision": float(best["metrics/precision(B)"]),
        "recall": float(best["metrics/recall(B)"]),
        "map50": float(best["metrics/mAP50(B)"]),
        "map50_95": float(best["metrics/mAP50-95(B)"]),
        "parameters": parameters,
        "best_weight": str(best_weight),
    }
    (RUN_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
