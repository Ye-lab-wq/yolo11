#!/usr/bin/env python3
"""Resumable validation screen: retain P2 C3k2 and append an identity-safe edge path."""

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

NAME = "p2_c3k2_msef_edge_addon_ciou_s0"
SOURCE = ROOT / "reproduction/ablation/models/p2_nearest.yaml"
MODEL = ROOT / "reproduction/ablation/models/p2_c3k2_msef_edge_addon.yaml"
RUN_ROOT = ROOT / "runs/msef_addon_screen_v1"
RUN_DIR = RUN_ROOT / NAME


def write_model() -> None:
    source = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    graph = source["backbone"] + source["head"]
    if len(source["backbone"]) != 11 or graph[19][2] != "C3k2" or graph[-1][0] != [19, 22, 25, 28]:
        raise RuntimeError("P2 parent graph has changed; inspect layer references before inserting an add-on")
    insert_index = 19
    head_insert_index = insert_index - len(source["backbone"]) + 1
    source["head"].insert(head_insert_index, [-1, 1, "MSEFEdgeAddOn", [128, False, 1, 0.5]])
    for layer in source["head"][head_insert_index + 1 :]:
        refs = layer[0]
        if isinstance(refs, int):
            layer[0] = refs + 1 if refs >= insert_index else refs
        else:
            layer[0] = [ref + 1 if isinstance(ref, int) and ref >= insert_index else ref for ref in refs]
    if source["head"][-1][0] != [20, 23, 26, 29]:
        raise RuntimeError(f"Unexpected Detect references after insertion: {source['head'][-1][0]}")
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
    from ultralytics.nn.modules import MSEFEdgeAddOn

    block = MSEFEdgeAddOn(32, 32).eval()
    with torch.no_grad():
        sample = torch.randn((1, 32, 32, 32))
        assert torch.equal(block(sample), sample)
    detector = model.model.model[-1]
    assert tuple(int(s) for s in detector.stride.tolist()) == (4, 8, 16, 32)
    assert model.model.model[19].__class__.__name__ == "C3k2"
    assert model.model.model[20].__class__.__name__ == "MSEFEdgeAddOn"
    assert detector.f == [20, 23, 26, 29]
    parent = YOLO(str(SOURCE)).model.eval()
    child = model.model.eval()
    child_state = child.state_dict()
    for key, value in parent.state_dict().items():
        prefix, index, remainder = key.split(".", 2)
        assert prefix == "model"
        target_index = int(index) + (int(index) > 19)
        target_key = f"model.{target_index}.{remainder}"
        assert target_key in child_state and child_state[target_key].shape == value.shape
        child_state[target_key] = value
    child.load_state_dict(child_state)
    with torch.no_grad():
        image = torch.randn((1, 3, 64, 64))
        reference = parent(image)[0]
        candidate = child(image)[0]
        assert torch.allclose(reference, candidate, atol=1e-5, rtol=1e-5)
    print("P2 C3k2 retained; copied-weight full-model output matches parent; Detect graph passed", flush=True)


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
        "candidate": "P2 C3k2 kept; MSEFEdgeAddOn identity-safe high-pass path appended",
        "status": "exploratory validation-only screen; not original MEDA-YOLO",
        "parent_model": str(SOURCE),
        "model": str(MODEL),
        "parent_run": str(ROOT / "runs/ablation_screen_v1/p2_nearest_ciou_s0"),
        "replaced_c3k2_comparison_run": str(ROOT / "runs/msef_highpass_screen_v1/p2_msef_highpass_ciou_s0"),
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
