#!/usr/bin/env python3
"""Verify protocol invariants for every completed screening run."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "runs" / "ablation_screen_v1"
# ``resume`` is execution provenance (False for uninterrupted runs and the
# checkpoint path for a resumed run), not a training hyperparameter.
ALLOWED_ARG_DIFFERENCES = {"model", "name", "save_dir", "bbox_loss_mode", "focaler_iou", "resume"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    run_dirs = sorted(
        path for path in RUN_ROOT.iterdir() if path.is_dir() and (path / "args.yaml").is_file()
    )
    if not run_dirs:
        raise RuntimeError("no screen runs found")
    baseline_dir = RUN_ROOT / "yolo11n_ciou_s0"
    if baseline_dir not in run_dirs:
        raise RuntimeError(f"frozen baseline missing: {baseline_dir}")
    baseline_args = yaml.safe_load((baseline_dir / "args.yaml").read_text(encoding="utf-8"))
    records = []
    all_valid = True
    for run_dir in run_dirs:
        args = yaml.safe_load((run_dir / "args.yaml").read_text(encoding="utf-8"))
        differences = {
            key: [baseline_args.get(key), args.get(key)]
            for key in sorted(set(baseline_args) | set(args))
            if key not in ALLOWED_ARG_DIFFERENCES and baseline_args.get(key) != args.get(key)
        }
        curve = csv_rows(run_dir / "results.csv") if (run_dir / "results.csv").is_file() else []
        best = run_dir / "weights" / "best.pt"
        last = run_dir / "weights" / "last.pt"
        valid = (
            len(curve) == 60
            and not differences
            and best.is_file()
            and last.is_file()
            and args.get("split") == "val"
            and Path(args.get("data", "")).resolve() == (ROOT / "DetectDataset_clean_v2/data.yaml").resolve()
            and args.get("pretrained") is False
            and args.get("seed") == 0
        )
        all_valid &= valid
        records.append(
            {
                "run": run_dir.name,
                "valid": valid,
                "epochs": len(curve),
                "unexpected_argument_differences": differences,
                "model": args.get("model"),
                "bbox_loss_mode": args.get("bbox_loss_mode"),
                "best_pt_sha256": sha256(best) if best.is_file() else None,
                "last_pt_sha256": sha256(last) if last.is_file() else None,
            }
        )
    report = {
        "valid": all_valid,
        "expected_epochs": 60,
        "allowed_argument_differences": sorted(ALLOWED_ARG_DIFFERENCES),
        "training_data": str((ROOT / "DetectDataset_clean_v2/data.yaml").resolve()),
        "selection_split": "val",
        "test_access_during_screen": "none in run arguments or screen script",
        "runs": records,
    }
    output = RUN_ROOT / "protocol_verification.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    if not all_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
