#!/usr/bin/env python3
"""Validation-only ADown interaction screen after both MSEF redesign candidates finish."""

from __future__ import annotations

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

MODEL_DIR = ROOT / "reproduction/ablation/models"
RUN_ROOT = ROOT / "runs/msef_adown_screen_v1"
P2 = ROOT / "runs/ablation_screen_v1/screen_summary.csv"
PAPER_ADOWN = ROOT / "runs/ablation_secondary_v1/secondary_summary.csv"
HIGHPASS = ROOT / "runs/msef_highpass_screen_v1/summary.json"
ATROUS = ROOT / "runs/msef_atrous_screen_v1/summary.json"
ADDON = ROOT / "runs/msef_addon_screen_v1/summary.json"
MIN_GAIN = 0.003  # 0.30 percentage points


def record(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_row(path: Path, run: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        matches = [row for row in csv.DictReader(handle) if row.get("experiment") == run]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one reference row {run!r} in {path}, found {len(matches)}")
    return matches[0]


def select() -> dict:
    p2_map = float(csv_row(P2, "p2_nearest_ciou_s0")["map50_95"])
    paper_map = float(csv_row(P2, "p2_msef_paper_ciou_s0")["map50_95"])
    p2_adown_map = float(csv_row(P2, "p2_adown_ciou_s0")["map50_95"])
    paper_adown_map = float(csv_row(PAPER_ADOWN, "msefbase_adown_ciou_s0")["map50_95"])
    highpass_map = float(record(HIGHPASS)["map50_95"])
    candidates = [
        ("highpass_atrous", float(record(ATROUS)["map50_95"]), highpass_map),
        ("edge_addon", float(record(ADDON)["map50_95"]), p2_map),
    ]
    qualified = [
        (name, score)
        for name, score, parent in candidates
        if score >= parent + MIN_GAIN and score >= paper_map
    ]
    selected = max(qualified, key=lambda item: item[1])[0] if qualified else None
    return {
        "selection_metric": "best validation mAP50-95 from matched seed-0/60-epoch screens",
        "minimum_direct_parent_gain": MIN_GAIN,
        "references": {
            "p2": p2_map,
            "paper": paper_map,
            "highpass": highpass_map,
            "p2_adown": p2_adown_map,
            "paper_adown": paper_adown_map,
        },
        "candidates": [
            {"name": name, "map50_95": score, "parent_map50_95": parent}
            for name, score, parent in candidates
        ],
        "selected": selected,
        "test_access": "none",
    }


def write_frozen(path: Path, value: dict) -> None:
    expected = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != expected:
            raise RuntimeError(f"Frozen file differs: {path}")
    else:
        path.write_text(expected, encoding="utf-8")


def candidate_model(name: str) -> Path:
    if name == "highpass_atrous":
        source = yaml.safe_load((MODEL_DIR / "p2_msef_paper_adown.yaml").read_text(encoding="utf-8"))
        count = 0
        for layer in source["head"]:
            if layer[2] == "MSEFPaper":
                layer[2] = "MSEFHighPassAtrous"
                count += 1
        if count != 1:
            raise RuntimeError(f"Expected one P2 MSEFPaper layer, found {count}")
        path = MODEL_DIR / "p2_msef_highpass_atrous_adown.yaml"
    elif name == "edge_addon":
        source = yaml.safe_load((MODEL_DIR / "p2_adown.yaml").read_text(encoding="utf-8"))
        graph = source["backbone"] + source["head"]
        if len(source["backbone"]) != 11 or graph[19][2] != "C3k2" or graph[-1][0] != [19, 22, 25, 28]:
            raise RuntimeError("P2+ADown parent graph changed; inspect layer references")
        insert_index = 19
        head_insert = insert_index - len(source["backbone"]) + 1
        source["head"].insert(head_insert, [-1, 1, "MSEFEdgeAddOn", [128, False, 1, 0.5]])
        for layer in source["head"][head_insert + 1 :]:
            refs = layer[0]
            if isinstance(refs, int):
                layer[0] = refs + 1 if refs >= insert_index else refs
            else:
                layer[0] = [ref + 1 if isinstance(ref, int) and ref >= insert_index else ref for ref in refs]
        if source["head"][-1][0] != [20, 23, 26, 29]:
            raise RuntimeError("Unexpected Detect references after inserting edge add-on")
        path = MODEL_DIR / "p2_c3k2_msef_edge_addon_adown.yaml"
    else:
        raise ValueError(name)
    expected = yaml.safe_dump(source, sort_keys=False)
    if path.exists():
        if path.read_text(encoding="utf-8") != expected:
            raise RuntimeError(f"Frozen candidate YAML differs: {path}")
    else:
        path.write_text(expected, encoding="utf-8")
    return path


def rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no interaction training was started")
    selection = select()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    write_frozen(RUN_ROOT / "selection.json", selection)
    name = selection["selected"]
    if name is None:
        print("No candidate met the predeclared validation-only P2 gate; ADown screen skipped", flush=True)
        return
    model_path = candidate_model(name)
    model = YOLO(str(model_path))
    detector = model.model.model[-1]
    if tuple(int(s) for s in detector.stride.tolist()) != (4, 8, 16, 32):
        raise RuntimeError("Candidate Detect strides differ from the parent")
    run_name = f"{name}_adown_ciou_s0"
    run_dir = RUN_ROOT / run_name
    train_args = {**TRAIN_ARGS, "project": str(RUN_ROOT)}
    protocol = {
        "model": str(model_path),
        "model_parameters": sum(p.numel() for p in model.model.parameters()),
        "selection": selection,
        "train_args": train_args,
        "bbox_loss_mode": "ciou",
        "environment": {"python": sys.version, "torch": torch.__version__, "ultralytics": ultralytics.__version__},
        "selection_rule": "best validation mAP50-95 within 60 epochs; no test access",
    }
    write_frozen(RUN_ROOT / "protocol.json", protocol)
    curve = rows(run_dir / "results.csv")
    best_weight = run_dir / "weights/best.pt"
    if len(curve) < 60 or not best_weight.exists():
        last_weight = run_dir / "weights/last.pt"
        if last_weight.exists():
            YOLO(str(last_weight)).train(resume=True)
        elif run_dir.exists():
            raise RuntimeError(f"Incomplete run without last.pt: {run_dir}")
        else:
            model.train(name=run_name, bbox_loss_mode="ciou", exist_ok=False, **train_args)
        curve = rows(run_dir / "results.csv")
    if len(curve) < 60 or not best_weight.exists():
        raise RuntimeError(f"Training incomplete: {len(curve)} / 60 epochs")
    best = max(curve, key=lambda item: float(item["metrics/mAP50-95(B)"]))
    score = float(best["metrics/mAP50-95(B)"])
    parent = float(selection["references"]["p2_adown"])
    paper = float(selection["references"]["paper_adown"])
    result = {
        "candidate": name,
        "run": run_name,
        "epochs": len(curve),
        "best_epoch": int(best["epoch"]),
        "precision": float(best["metrics/precision(B)"]),
        "recall": float(best["metrics/recall(B)"]),
        "map50": float(best["metrics/mAP50(B)"]),
        "map50_95": score,
        "parameters": protocol["model_parameters"],
        "best_weight": str(best_weight),
        "confirmatory_200_gate": score >= parent + MIN_GAIN and score >= paper,
        "gate_references": {"p2_adown": parent, "paper_adown": paper},
    }
    write_frozen(RUN_ROOT / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
