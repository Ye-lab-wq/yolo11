#!/usr/bin/env python3
"""Resume seed-1/200-epoch MSEF redesigns, then compare frozen best.pt on full test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import ultralytics
import yaml
from ultralytics import YOLO

from reproduction.ablation.run_final_candidate import TRAIN_ARGS as FROZEN_200_ARGS


SEED = 1
EPOCHS = 200
RUN_ROOT = ROOT / "runs/msef_redesign_seed1_200_v1"
RESULT_ROOT = ROOT / "reproduction/results/msef_redesign_seed1_200_v1"
MODEL_DIR = ROOT / "reproduction/ablation/models"
EVALUATOR = ROOT / "reproduction/ablation/evaluate_frozen.py"
MODULE_SOURCE = ROOT / "ultralytics/nn/modules/meda_modules.py"
CONTROLS = (
    ("P2", "p2_nearest.yaml", ROOT / "runs/ablation_200_v1/p2/train_seed1"),
    ("P2_MSEFPaper", "p2_msef_paper.yaml", ROOT / "runs/ablation_200_v1/p2_msef_paper/train_seed1"),
)
CANDIDATES = (
    ("MSEFHighPass", "p2_msef_highpass.yaml"),
    ("MSEFHighPassAtrous", "p2_msef_highpass_atrous.yaml"),
    ("MSEFEdgeAddOn", "p2_c3k2_msef_edge_addon.yaml"),
)
MATCHED_KEYS = tuple(key for key in FROZEN_200_ARGS if key not in {"project", "device"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def check_args(run_dir: Path, expected_yaml: Path) -> None:
    saved = yaml.safe_load((run_dir / "args.yaml").read_text(encoding="utf-8"))
    if saved["seed"] != SEED or Path(saved["model"]).resolve() != expected_yaml.resolve():
        raise RuntimeError(f"Seed/model mismatch in {run_dir}")
    differences = {
        key: (saved.get(key), FROZEN_200_ARGS[key])
        for key in MATCHED_KEYS
        if saved.get(key) != FROZEN_200_ARGS[key]
    }
    if differences:
        raise RuntimeError(f"200-epoch training controls differ in {run_dir}: {differences}")


def frozen_record(name: str, run_dir: Path) -> dict[str, object]:
    rows = read_rows(run_dir / "results.csv")
    best_weight = run_dir / "weights/best.pt"
    if len(rows) != EPOCHS or not best_weight.is_file():
        raise RuntimeError(f"Incomplete run: {name} at {run_dir}, {len(rows)}/{EPOCHS} epochs")
    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    return {
        "name": name,
        "seed": SEED,
        "epochs": len(rows),
        "selection_split": "val",
        "selection_metric": "mAP50-95",
        "best_epoch_one_based": int(best["epoch"]),
        "precision": float(best["metrics/precision(B)"]),
        "recall": float(best["metrics/recall(B)"]),
        "map50": float(best["metrics/mAP50(B)"]),
        "map50_95": float(best["metrics/mAP50-95(B)"]),
        "weights": str(best_weight.resolve()),
        "weights_sha256": sha256(best_weight),
    }


def check_controls() -> list[dict[str, object]]:
    records = []
    for name, yaml_name, run_dir in CONTROLS:
        expected_yaml = MODEL_DIR / yaml_name
        check_args(run_dir, expected_yaml)
        records.append(frozen_record(name, run_dir))
    for _, yaml_name in CANDIDATES:
        if not (MODEL_DIR / yaml_name).is_file():
            raise FileNotFoundError(MODEL_DIR / yaml_name)
    return records


def run_candidate(name: str, yaml_name: str) -> dict[str, object]:
    run_dir = RUN_ROOT / f"{name}_seed1"
    model_yaml = MODEL_DIR / yaml_name
    rows = read_rows(run_dir / "results.csv")
    best_weight = run_dir / "weights/best.pt"
    if len(rows) != EPOCHS or not best_weight.is_file():
        last_weight = run_dir / "weights/last.pt"
        if last_weight.is_file():
            check_args(run_dir, model_yaml)
            print(f"RESUME {name}: {len(rows)}/{EPOCHS} epochs", flush=True)
            YOLO(str(last_weight)).train(resume=True)
        elif run_dir.exists():
            raise RuntimeError(f"Incomplete run without last.pt: {run_dir}")
        else:
            print(f"START {name}: seed {SEED}, {EPOCHS} epochs from scratch", flush=True)
            YOLO(str(model_yaml)).train(
                name=run_dir.name,
                seed=SEED,
                exist_ok=False,
                **{**FROZEN_200_ARGS, "project": str(RUN_ROOT)},
            )
    check_args(run_dir, model_yaml)
    record = frozen_record(name, run_dir)
    print(f"DONE {name}: best val mAP50-95={record['map50_95']:.6f}", flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Read-only control and model check")
    args = parser.parse_args()
    controls = check_controls()
    print("Two existing seed-1/200-epoch controls verified; three candidate YAMLs present", flush=True)
    if args.check:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing to silently switch to CPU")

    protocol = {
        "purpose": "Exploratory seed-1/200-epoch follow-up to the 60-epoch MSEF redesign screen",
        "seed": SEED,
        "epochs": EPOCHS,
        "controls": [{"name": name, "yaml": str(MODEL_DIR / yaml_name), "run": str(run_dir)} for name, yaml_name, run_dir in CONTROLS],
        "candidates": [{"name": name, "yaml": str(MODEL_DIR / yaml_name), "yaml_sha256": sha256(MODEL_DIR / yaml_name)} for name, yaml_name in CANDIDATES],
        "module_source_sha256": sha256(MODULE_SOURCE),
        "training": {key: value for key, value in FROZEN_200_ARGS.items() if key != "project"},
        "checkpoint_selection": "Validation mAP50-95 selects one best.pt independently for each 200-epoch run",
        "evaluation": "All five frozen weights on the full 641-image combined test split; no selection by test",
        "environment": {"python": sys.version, "torch": torch.__version__, "ultralytics": ultralytics.__version__},
    }
    protocol_path = RUN_ROOT / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError(f"Frozen protocol differs: {protocol_path}")
    else:
        write_json(protocol_path, protocol)

    records = list(controls)
    write_json(RUN_ROOT / "validation_selection.json", records)
    for name, yaml_name in CANDIDATES:
        record = run_candidate(name, yaml_name)
        records.append(record)
        write_json(RUN_ROOT / "validation_selection.json", records)

    output = RESULT_ROOT / "combined_test.json"
    expected = {str(record["name"]): str(record["weights_sha256"]) for record in records}
    if output.is_file():
        report = json.loads(output.read_text(encoding="utf-8"))
        observed = {name: value.get("weights_sha256") for name, value in report.get("models", {}).items()}
        if report.get("split") == "combined_test" and observed == expected:
            print(f"Existing test report verified: {output}", flush=True)
            return
        raise RuntimeError(f"Existing test report is incomplete or uses other weights: {output}")
    command = [sys.executable, "-B", str(EVALUATOR)]
    for record in records:
        command.extend(["--weights", f"{record['name']}={record['weights']}"])
    command.extend(["--split", "combined_test", "--output", str(output), "--device", "0", "--batch", "4"])
    print("All five validation-selected weights frozen; evaluating full combined test", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    report = json.loads(output.read_text(encoding="utf-8"))
    observed = {name: value.get("weights_sha256") for name, value in report.get("models", {}).items()}
    if observed != expected or any(value["coco_style"]["images"] != 641 for value in report["models"].values()):
        raise RuntimeError(f"Incomplete or mismatched test report: {output}")
    print(f"COMPLETE {output}", flush=True)


if __name__ == "__main__":
    main()
