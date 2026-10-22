#!/usr/bin/env python3
"""Gate-controlled, resumable three-seed MSEF confirmation and paired full-test audit."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import ultralytics
from ultralytics import YOLO

from reproduction.ablation.run_final_candidate import TRAIN_ARGS as FROZEN_200_ARGS

SCREEN = ROOT / "runs/msef_adown_screen_v1"
RUN_ROOT = ROOT / "runs/msef_confirm_200_v1"
PARENT_ROOT = ROOT / "runs/ablation_200_v1/p2_adown"
EVALUATOR = ROOT / "reproduction/ablation/evaluate_frozen.py"
BENCHMARK = ROOT / "reproduction/ablation/benchmark_frozen.py"
MODULE_SOURCE = ROOT / "ultralytics/nn/modules/meda_modules.py"
MATCHED_KEYS = (
    "data", "epochs", "patience", "batch", "imgsz", "workers", "pretrained",
    "optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
    "warmup_momentum", "warmup_bias_lr", "box", "cls", "dfl", "bbox_loss_mode",
    "focaler_iou", "focaler_d", "focaler_u", "deterministic", "amp",
    "close_mosaic", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate",
    "scale", "shear", "perspective", "flipud", "fliplr", "mosaic",
    "mixup", "copy_paste", "cutmix", "cache", "plots", "save", "save_period",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_frozen(path: Path, value: dict) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"Frozen record differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def curve(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parent_weight(seed: int) -> Path:
    run = PARENT_ROOT / f"train_seed{seed}"
    weight = run / "weights/best.pt"
    if len(curve(run / "results.csv")) != 200 or not weight.is_file():
        raise RuntimeError(f"Parent seed {seed} is not a complete 200-epoch frozen run: {run}")
    return weight


def check_parent_protocol(train_args: dict) -> None:
    for seed in (0, 1, 2):
        parent_weight(seed)
        path = PARENT_ROOT / f"train_seed{seed}/args.yaml"
        parent = yaml.safe_load(path.read_text(encoding="utf-8"))
        differences = {
            key: (parent.get(key), train_args[key])
            for key in MATCHED_KEYS
            if parent.get(key) != train_args[key]
        }
        if differences:
            raise RuntimeError(f"Parent seed {seed} training args differ: {differences}")


def frozen_best(seed: int) -> Path:
    run = RUN_ROOT / f"train_seed{seed}"
    rows = curve(run / "results.csv")
    best_weight = run / "weights/best.pt"
    if len(rows) != 200 or not best_weight.is_file():
        raise RuntimeError(f"Candidate seed {seed} incomplete: {len(rows)} / 200 epochs")
    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    record = {
        "seed": seed,
        "selection_split": "validation",
        "selection_metric": "mAP50-95",
        "epochs": len(rows),
        "best_epoch_one_based": int(best["epoch"]),
        "precision": float(best["metrics/precision(B)"]),
        "recall": float(best["metrics/recall(B)"]),
        "map50": float(best["metrics/mAP50(B)"]),
        "map50_95": float(best["metrics/mAP50-95(B)"]),
        "weights": str(best_weight),
        "weights_sha256": sha256(best_weight),
        "test_access_before_freeze": "none",
    }
    write_frozen(RUN_ROOT / f"frozen_seed{seed}.json", record)
    return best_weight


def train_seed(seed: int, model_yaml: Path, train_args: dict) -> Path:
    run = RUN_ROOT / f"train_seed{seed}"
    rows = curve(run / "results.csv")
    best = run / "weights/best.pt"
    if len(rows) != 200 or not best.is_file():
        last = run / "weights/last.pt"
        if last.is_file():
            print(f"Resuming candidate seed {seed} from {last}", flush=True)
            YOLO(str(last)).train(resume=True)
        elif run.exists():
            raise RuntimeError(f"Incomplete candidate run without last.pt: {run}")
        else:
            print(f"Training candidate seed {seed}/200 from scratch", flush=True)
            YOLO(str(model_yaml)).train(
                name=run.name,
                seed=seed,
                exist_ok=False,
                **train_args,
            )
    return frozen_best(seed)


def evaluate_seed(seed: int, candidate: Path) -> dict:
    output = RUN_ROOT / "test_results" / f"seed{seed}" / "combined_test.json"
    parent = parent_weight(seed)
    if output.exists():
        report = read_json(output)
        models = report.get("models", {})
        if (
            report.get("split") == "combined_test"
            and set(models) == {"parent", "candidate"}
            and models["parent"].get("weights_sha256") == sha256(parent)
            and models["candidate"].get("weights_sha256") == sha256(candidate)
        ):
            return report
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable, str(EVALUATOR),
            "--weights", f"parent={parent}",
            "--weights", f"candidate={candidate}",
            "--split", "combined_test",
            "--output", str(output),
            "--device", "0",
            "--batch", "4",
        ],
        cwd=ROOT,
        check=True,
    )
    report = read_json(output)
    if set(report.get("models", {})) != {"parent", "candidate"}:
        raise RuntimeError(f"Incomplete paired full-test report: {output}")
    return report


def summarize(reports: list[dict]) -> dict:
    output = {"dataset": "DetectDataset_clean_v2", "split": "combined_test", "seeds": []}
    for seed, report in enumerate(reports):
        pair = report["models"]
        a = pair["parent"]
        b = pair["candidate"]
        if a["coco_style"]["images"] != 641 or b["coco_style"]["images"] != 641:
            raise RuntimeError(f"Seed {seed} did not evaluate the full 641-image test split")
        metrics = {}
        for evaluator, names in (
            ("ultralytics", ("precision", "recall", "mAP50", "mAP75", "mAP50_95")),
            ("coco_style", ("AP", "AP50", "AP75", "AP_tiny_lt16", "AP_small_16_32")),
        ):
            metrics[evaluator] = {
                name: {
                    "parent": a[evaluator][name],
                    "candidate": b[evaluator][name],
                    "delta_percentage_points": 100 * (b[evaluator][name] - a[evaluator][name])
                    if a[evaluator][name] is not None and b[evaluator][name] is not None else None,
                }
                for name in names
            }
        output["seeds"].append({"seed": seed, "images": 641, "metrics": metrics})
    return output


def wait_for_screen() -> dict | None:
    """Wait for the queued ADown screen, or stop if P2 screening rejects all candidates."""
    summary_path = SCREEN / "summary.json"
    selection_path = SCREEN / "selection.json"
    deadline = time.monotonic() + 6 * 3600
    while time.monotonic() < deadline:
        if summary_path.is_file():
            return read_json(summary_path)
        if selection_path.is_file() and read_json(selection_path).get("selected") is None:
            return None
        time.sleep(30)
    raise TimeoutError(f"No terminal ADown screening outcome within 6 hours: {SCREEN}")


def main() -> None:
    screen_summary = wait_for_screen()
    if screen_summary is None:
        print("No P2 redesign candidate passed the first validation-only gate; confirmation skipped", flush=True)
        return
    if not screen_summary.get("confirmatory_200_gate"):
        print("Validation-only ADown screen did not pass; no 200-epoch confirmation started", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; confirmation was not started")
    screen_protocol = read_json(SCREEN / "protocol.json")
    model_yaml = Path(screen_protocol["model"])
    if not model_yaml.is_file():
        raise FileNotFoundError(model_yaml)
    train_args = {**FROZEN_200_ARGS, "project": str(RUN_ROOT)}
    check_parent_protocol(train_args)
    protocol = {
        "status": "confirmatory training gated by clean validation only",
        "model_yaml": str(model_yaml),
        "model_yaml_sha256": sha256(model_yaml),
        "module_source_sha256": sha256(MODULE_SOURCE),
        "screen_summary_sha256": sha256(SCREEN / "summary.json"),
        "train_args": train_args,
        "environment": {"python": sys.version, "torch": torch.__version__, "ultralytics": ultralytics.__version__},
        "checkpoint_selection": "best.pt by validation mAP50-95 within each 200-epoch seed run",
        "evaluation": "all seeds frozen before paired parent/candidate evaluation on the full 641-image test",
        "efficiency": "paired seed-1 FP32 fused forward benchmark on the same GPU, excluding preprocessing and NMS",
    }
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    write_frozen(RUN_ROOT / "protocol.json", protocol)
    frozen = [train_seed(seed, model_yaml, train_args) for seed in (0, 1, 2)]
    reports = [evaluate_seed(seed, frozen[seed]) for seed in (0, 1, 2)]
    result = summarize(reports)
    write_frozen(RUN_ROOT / "paired_results.json", result)
    efficiency = RUN_ROOT / "efficiency_seed1.csv"
    if not efficiency.is_file():
        subprocess.run(
            [
                sys.executable, str(BENCHMARK),
                "--weights", f"parent={parent_weight(1)}",
                "--weights", f"candidate={frozen[1]}",
                "--output", str(efficiency),
            ],
            cwd=ROOT,
            check=True,
        )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
