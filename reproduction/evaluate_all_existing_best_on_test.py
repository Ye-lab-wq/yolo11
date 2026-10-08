#!/usr/bin/env python3
"""Evaluate every migrated historical best.pt on a fixed dataset split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
SOURCE = PROJECT
RUNS = PROJECT / "runs"
DATA = ROOT / "data.yaml"
sys.path.insert(0, str(SOURCE))
# Some early checkpoints serialized DySample as ``dysample.DySample`` rather
# than through its package path.  The original project contains this exact
# module, so expose that directory under the historical import path.
sys.path.insert(0, str(SOURCE / "ultralytics" / "nn" / "modules"))

import torch
import torch.nn.functional as F
import ultralytics
import yaml
import dysample  # noqa: F401 - historical top-level pickle import path
from CAA import CAA  # noqa: F401 - required to unpickle historical CAA models
from ultralytics import YOLO
from ultralytics.nn.modules.meda_modules import MSEF


_CURRENT_MSEF_FORWARD = MSEF.forward


def _version_compatible_msef_forward(self: MSEF, x: torch.Tensor) -> torch.Tensor:
    """Dispatch between the two MSEF implementations preserved in source.

    Early submitted checkpoints contain ``scales`` but no ``branch_convs`` and
    therefore require the adaptive-pooling forward still present (commented)
    in the original meda_modules.py. Later checkpoints use the active fixed-
    kernel implementation unchanged.
    """
    if hasattr(self, "branch_convs"):
        return _CURRENT_MSEF_FORWARD(self, x)
    if hasattr(self, "scales"):
        _, _, h, w = x.shape
        local = self.cv_local(x)
        feats = [local]
        for pool, enhancer in zip(self.poolings, self.edge_enhancers):
            pooled = pool(local)
            up = F.interpolate(pooled, size=(h, w), mode="bilinear", align_corners=False)
            feats.append(enhancer(up))
        return self.cv_final(torch.cat(feats, 1))
    raise AttributeError("Unsupported historical MSEF object layout")


MSEF.forward = _version_compatible_msef_forward


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test"), default="test")
    return parser.parse_args()


def write_outputs(report: dict[str, object], out: Path, split: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"all_best_{split}_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    columns = [
        "run", "model_arg", "status", "checkpoint_version", "sha256", "reused_from",
        "compatibility_adapter", "parameters", "precision", "recall", "mAP50", "mAP50-95", "error",
    ]
    with (out / f"all_best_{split}_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for item in report["results"]:
            writer.writerow({key: item.get(key) for key in columns})


def main() -> int:
    args = parse_args()
    split = args.split
    out = ROOT / "results" / f"all_existing_best_{split}"
    split_counts = {"val": (842, 993), "test": (560, 608)}
    images, instances = split_counts[split]
    weights = sorted(RUNS.rglob("weights/best.pt"))
    report: dict[str, object] = {
        "policy": f"Every migrated historical best.pt evaluated on {split}; no training; no last.pt",
        "environment": {
            "conda": "/home/b520/anaconda3/envs/YHP",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "ultralytics": ultralytics.__version__,
            "ultralytics_file": str(Path(ultralytics.__file__).resolve()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "data": str(DATA.resolve()),
        "split": split,
        "images": images,
        "instances": instances,
        "discovered_best_checkpoints": len(weights),
        "results": [],
    }
    by_hash: dict[str, dict[str, object]] = {}

    for index, path in enumerate(weights, 1):
        run_dir = path.parent.parent
        run = run_dir.relative_to(RUNS).as_posix()
        args_path = run_dir / "args.yaml"
        args = yaml.safe_load(args_path.read_text(encoding="utf-8")) if args_path.exists() else {}
        digest = sha256(path)
        item: dict[str, object] = {
            "run": run,
            "weights": str(path.resolve()),
            "model_arg": args.get("model"),
            "sha256": digest,
            "reused_from": None,
            "status": "pending",
            "error": None,
        }
        print(f"[{index}/{len(weights)}] {run}", flush=True)

        if digest in by_hash:
            previous = by_hash[digest]
            for key in ("checkpoint_version", "parameters", "precision", "recall", "mAP50", "mAP50-95"):
                item[key] = previous.get(key)
            item["reused_from"] = previous["run"]
            item["status"] = "reused_identical_sha256"
        else:
            try:
                model = YOLO(str(path))
                item["checkpoint_version"] = (model.ckpt or {}).get("version")
                item["parameters"] = sum(p.numel() for p in model.model.parameters())
                adapters = []
                if any(type(module).__module__ == "dysample" for module in model.model.modules()):
                    adapters.append("historical top-level dysample import alias")
                if any(isinstance(module, MSEF) and not hasattr(module, "branch_convs") for module in model.model.modules()):
                    adapters.append("historical adaptive-pooling MSEF forward from original source")
                item["compatibility_adapter"] = "; ".join(adapters) or None
                metrics = model.val(
                    data=str(DATA),
                    split=split,
                    imgsz=640,
                    batch=16,
                    device=0,
                    workers=4,
                    plots=False,
                    project=str(out / "runs"),
                    name=run.replace("/", "__"),
                    exist_ok=True,
                    verbose=False,
                )
                item.update(
                    {
                        "precision": float(metrics.box.mp),
                        "recall": float(metrics.box.mr),
                        "mAP50": float(metrics.box.map50),
                        "mAP50-95": float(metrics.box.map),
                        "status": "evaluated",
                    }
                )
                by_hash[digest] = item
            except Exception as exc:  # preserve all failures for audit
                item["status"] = "failed"
                item["error"] = f"{type(exc).__name__}: {exc}"
                item["traceback"] = traceback.format_exc()

        report["results"].append(item)
        write_outputs(report, out, split)
        print(
            f"  status={item['status']} mAP50-95={item.get('mAP50-95')} error={item.get('error')}",
            flush=True,
        )

    report["successful_records"] = sum(item["status"] != "failed" for item in report["results"])
    report["failed_records"] = sum(item["status"] == "failed" for item in report["results"])
    report["unique_weight_hashes_evaluated"] = len(by_hash)
    write_outputs(report, out, split)
    return 1 if report["failed_records"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
