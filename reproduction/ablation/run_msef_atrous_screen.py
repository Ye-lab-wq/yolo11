#!/usr/bin/env python3
"""Resumable seed-0/60-epoch validation screen for high-pass plus atrous MSEF."""

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

NAME = "p2_msef_highpass_atrous_ciou_s0"
SOURCE = ROOT / "reproduction/ablation/models/p2_msef_highpass.yaml"
MODEL = ROOT / "reproduction/ablation/models/p2_msef_highpass_atrous.yaml"
RUN_ROOT = ROOT / "runs/msef_atrous_screen_v1"
RUN_DIR = RUN_ROOT / NAME


def write_model() -> None:
    source = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    count = 0
    for layer in source["head"]:
        if layer[2] == "MSEFHighPass":
            layer[2] = "MSEFHighPassAtrous"
            count += 1
    if count != 1:
        raise RuntimeError(f"Expected one MSEFHighPass replacement, found {count}")
    expected = yaml.safe_dump(source, sort_keys=False)
    if MODEL.exists():
        if MODEL.read_text(encoding="utf-8") != expected:
            raise RuntimeError(f"Frozen candidate YAML differs: {MODEL}")
    else:
        MODEL.write_text(expected, encoding="utf-8")


def curve() -> list[dict[str, str]]:
    path = RUN_DIR / "results.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def check_block(model: YOLO) -> None:
    from ultralytics.nn.modules import MSEFHighPass, MSEFHighPassAtrous

    parent = MSEFHighPass(16, 16).eval()
    child = MSEFHighPassAtrous(16, 16).eval()
    missing, unexpected = child.load_state_dict(parent.state_dict(), strict=False)
    assert set(missing) == {
        "atrous_scale",
        "atrous_depthwise.weight",
        "atrous_project.weight",
    }, (missing, unexpected)
    assert not unexpected
    with torch.no_grad():
        sample = torch.randn(1, 16, 32, 32)
        assert torch.allclose(parent(sample), child(sample), atol=1e-6)
        constant = torch.full((1, 8, 16, 16), 3.0)
        assert torch.allclose(child._high_pass(constant, 5), torch.zeros_like(constant))
    parent_model = YOLO(str(SOURCE)).model.eval()
    candidate_model = model.model.eval()
    candidate_state = candidate_model.state_dict()
    for key, value in parent_model.state_dict().items():
        assert key in candidate_state and candidate_state[key].shape == value.shape
        candidate_state[key] = value
    candidate_model.load_state_dict(candidate_state)
    with torch.no_grad():
        image = torch.randn((1, 3, 64, 64))
        reference = parent_model(image)[0]
        candidate = candidate_model(image)[0]
        assert torch.allclose(reference, candidate, atol=1e-5, rtol=1e-5)
    print("Atrous child matches copied-weight parent at block and full-model levels", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify graph and module without training")
    args = parser.parse_args()
    write_model()
    model = YOLO(str(MODEL))
    parameters = sum(p.numel() for p in model.model.parameters())
    print(f"model={MODEL} parameters={parameters} torch={torch.__version__} ultralytics={ultralytics.__version__}", flush=True)
    if args.check:
        check_block(model)
        return

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    train_args = {**TRAIN_ARGS, "project": str(RUN_ROOT)}
    protocol = {
        "candidate": "MSEFHighPassAtrous: same-feature high-pass parent plus dilation-2 depthwise 3x3 branch",
        "status": "exploratory validation-only screen; not original MEDA-YOLO",
        "parent_model": str(SOURCE),
        "model": str(MODEL),
        "parent_run": str(ROOT / "runs/msef_highpass_screen_v1/p2_msef_highpass_ciou_s0"),
        "paper_reference_run": str(ROOT / "runs/ablation_screen_v1/p2_msef_paper_ciou_s0"),
        "train_args": train_args,
        "bbox_loss_mode": "ciou",
        "environment": {"python": sys.version, "torch": torch.__version__, "ultralytics": ultralytics.__version__},
        "selection": "best validation mAP50-95 within 60 epochs; no test access",
    }
    protocol_path = RUN_ROOT / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError(f"Frozen protocol differs: {protocol_path}")
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rows = curve()
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
        rows = curve()
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
