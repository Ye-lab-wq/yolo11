#!/usr/bin/env python3
"""Confirm the MSEFPaper x full-path ADown interaction on held-out seeds."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import statistics
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

import torch
import ultralytics
import yaml
from ultralytics import YOLO

from reproduction.ablation.run_screen import TRAIN_ARGS as SCREEN_TRAIN_ARGS


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"
DATA = ROOT / "DetectDataset_clean_v2" / "data.yaml"
RUN_ROOT = ROOT / "runs" / "ablation_interaction_confirm_v1"
STATUS = RUN_ROOT / "status.json"
SUMMARY = RUN_ROOT / "seed_results.csv"
AGGREGATE = RUN_ROOT / "cell_aggregate.csv"
INTERACTIONS = RUN_ROOT / "interaction_by_seed.csv"
ANALYSIS = RUN_ROOT / "interaction_analysis.json"
VERIFICATION = RUN_ROOT / "protocol_verification.json"

METRICS = ("precision", "recall", "map50", "map50_95")
CSV_METRICS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}

# Factor order is MSEFPaper, ADown. Seed-0 paths are the frozen discovery runs.
CELLS = (
    {
        "cell": "00",
        "msef": False,
        "adown": False,
        "label": "P2",
        "model": MODEL_DIR / "p2_nearest.yaml",
        "seed0": ROOT / "runs" / "ablation_screen_v1" / "p2_nearest_ciou_s0",
    },
    {
        "cell": "10",
        "msef": True,
        "adown": False,
        "label": "P2+MSEFPaper",
        "model": MODEL_DIR / "p2_msef_paper.yaml",
        "seed0": ROOT / "runs" / "ablation_screen_v1" / "p2_msef_paper_ciou_s0",
    },
    {
        "cell": "01",
        "msef": False,
        "adown": True,
        "label": "P2+ADown",
        "model": MODEL_DIR / "p2_adown.yaml",
        "seed0": ROOT / "runs" / "ablation_screen_v1" / "p2_adown_ciou_s0",
    },
    {
        "cell": "11",
        "msef": True,
        "adown": True,
        "label": "P2+MSEFPaper+ADown",
        "model": MODEL_DIR / "p2_msef_paper_adown.yaml",
        "seed0": ROOT / "runs" / "ablation_secondary_v1" / "msefbase_adown_ciou_s0",
    },
)

TRAIN_ARGS = {
    **{key: value for key, value in SCREEN_TRAIN_ARGS.items() if key != "seed"},
    "project": str(RUN_ROOT),
    "plots": False,
}
ALLOWED_ARG_DIFFERENCES = {"model", "name", "project", "resume", "save_dir", "seed"}


def now() -> str:
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def run_dir(cell: dict[str, object], seed: int) -> Path:
    if seed == 0:
        return Path(cell["seed0"])
    return RUN_ROOT / f"cell_{cell['cell']}_s{seed}"


def summarize(cell: dict[str, object], seed: int) -> dict[str, object]:
    directory = run_dir(cell, seed)
    curve = read_rows(directory / "results.csv")
    if len(curve) != 60:
        raise RuntimeError(f"{directory.name} has {len(curve)} epochs, expected 60")
    weight = directory / "weights" / "best.pt"
    if not weight.is_file():
        raise FileNotFoundError(weight)
    best = max(curve, key=lambda row: float(row[CSV_METRICS["map50_95"]]))
    parameters = sum(parameter.numel() for parameter in YOLO(str(weight)).model.parameters())
    return {
        "cell": cell["cell"],
        "msef": cell["msef"],
        "adown": cell["adown"],
        "label": cell["label"],
        "seed": seed,
        "role": "discovery" if seed == 0 else "held_out_replication",
        "source": "reused_seed0" if seed == 0 else "new_replication",
        "best_epoch": int(best["epoch"]),
        **{metric: float(best[column]) for metric, column in CSV_METRICS.items()},
        "parameters": parameters,
        "training_time_s": float(curve[-1]["time"]),
        "model_yaml": str(Path(cell["model"]).resolve()),
        "best_weight": str(weight.resolve()),
    }


def completed(cell: dict[str, object], seed: int) -> bool:
    directory = run_dir(cell, seed)
    return len(read_rows(directory / "results.csv")) == 60 and (directory / "weights" / "best.pt").is_file()


def collect() -> list[dict[str, object]]:
    records = []
    for seed in (0, 1, 2):
        for cell in CELLS:
            if completed(cell, seed):
                records.append(summarize(cell, seed))
    write_csv(SUMMARY, records)

    aggregate = []
    for cell in CELLS:
        subset = [record for record in records if record["cell"] == cell["cell"]]
        if not subset:
            continue
        item: dict[str, object] = {
            "cell": cell["cell"],
            "msef": cell["msef"],
            "adown": cell["adown"],
            "label": cell["label"],
            "seeds": len(subset),
        }
        for metric in METRICS:
            values = [float(record[metric]) for record in subset]
            item[f"{metric}_mean"] = statistics.mean(values)
            item[f"{metric}_sample_sd"] = statistics.stdev(values) if len(values) > 1 else None
        aggregate.append(item)
    write_csv(AGGREGATE, aggregate)

    interactions = []
    for seed in (0, 1, 2):
        by_cell = {record["cell"]: record for record in records if record["seed"] == seed}
        if set(by_cell) != {"00", "10", "01", "11"}:
            continue
        item: dict[str, object] = {"seed": seed, "role": "discovery" if seed == 0 else "held_out_replication"}
        for metric in METRICS:
            item[f"{metric}_interaction_pp"] = 100.0 * (
                float(by_cell["11"][metric])
                - float(by_cell["10"][metric])
                - float(by_cell["01"][metric])
                + float(by_cell["00"][metric])
            )
        interactions.append(item)
    write_csv(INTERACTIONS, interactions)

    if len(interactions) == 3:
        replication = [row for row in interactions if row["role"] == "held_out_replication"]
        primary = [float(row["map50_95_interaction_pp"]) for row in replication]
        all_primary = [float(row["map50_95_interaction_pp"]) for row in interactions]
        directional = all(value > 0.0 for value in primary)
        replication_mean = statistics.mean(primary)
        practical = replication_mean >= 0.30
        write_json(
            ANALYSIS,
            {
                "design": "2x2 MSEFPaper x full-path ADown on fixed YOLO11n-P2 parent",
                "primary_metric": "best validation mAP50:95 within true 60-epoch budget",
                "interaction_formula": "100 * (cell_11 - cell_10 - cell_01 + cell_00)",
                "discovery_seed0_interaction_pp": all_primary[0],
                "held_out_seed_interactions_pp": primary,
                "held_out_interaction_mean_pp": replication_mean,
                "held_out_interaction_sample_sd_pp": statistics.stdev(primary),
                "all_three_interaction_mean_pp": statistics.mean(all_primary),
                "all_three_interaction_sample_sd_pp": statistics.stdev(all_primary),
                "directionally_replicated": directional,
                "practically_stable_threshold_pp": 0.30,
                "practically_stable": practical,
                "advance_to_200_epoch_confirmation": directional and practical,
                "inference_note": "No p-value: only two held-out replication seeds; report seed-wise effects and dispersion.",
                "test_access": "none",
            },
        )
    return records


def verify() -> bool:
    canonical = yaml.safe_load((Path(CELLS[0]["seed0"]) / "args.yaml").read_text(encoding="utf-8"))
    output = []
    valid_all = True
    for seed in (0, 1, 2):
        for cell in CELLS:
            directory = run_dir(cell, seed)
            args_path = directory / "args.yaml"
            if args_path.is_file():
                args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
                differences = {
                    key: [canonical.get(key), args.get(key)]
                    for key in sorted(set(canonical) | set(args))
                    if key not in ALLOWED_ARG_DIFFERENCES and canonical.get(key) != args.get(key)
                }
            else:
                args, differences = {}, {"missing": str(args_path)}
            valid = (
                completed(cell, seed)
                and not differences
                and (directory / "weights" / "last.pt").is_file()
                and args.get("pretrained") is False
                and args.get("seed") == seed
                and args.get("bbox_loss_mode") == "ciou"
                and Path(args.get("data", "")).resolve() == DATA.resolve()
                and args.get("split") == "val"
            )
            valid_all &= valid
            output.append(
                {
                    "cell": cell["cell"],
                    "seed": seed,
                    "run": str(directory.resolve()),
                    "valid": valid,
                    "epochs": len(read_rows(directory / "results.csv")),
                    "unexpected_argument_differences": differences,
                }
            )
    write_json(
        VERIFICATION,
        {
            "valid": valid_all,
            "design": "2x2 MSEFPaper x full-path ADown, seeds 0/1/2",
            "environment_expected": {"python": "3.12.0", "torch": "2.10.0+cu128", "ultralytics": "8.3.241"},
            "test_access": "none",
            "runs": output,
        },
    )
    return valid_all


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "created_at": now(),
        "frozen_protocol": str((HERE / "INTERACTION_CONFIRMATION_PROTOCOL.md").resolve()),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "train_args": TRAIN_ARGS,
        "cells": [
            {
                "cell": cell["cell"],
                "msef": cell["msef"],
                "adown": cell["adown"],
                "label": cell["label"],
                "model": str(Path(cell["model"]).resolve()),
                "model_sha256": sha256(Path(cell["model"])),
                "seed0_run": str(Path(cell["seed0"]).resolve()),
            }
            for cell in CELLS
        ],
        "replication_seeds": [1, 2],
        "test_access": "forbidden",
    }
    protocol_path = RUN_ROOT / "protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        for key in ("train_args", "cells", "replication_seeds"):
            if existing[key] != protocol[key]:
                raise RuntimeError(f"frozen interaction protocol differs in {key}")
    else:
        write_json(protocol_path, protocol)

    state = {"status": "running", "started_or_resumed_at": now(), "current": None, "completed": []}
    write_json(STATUS, state)
    collect()
    try:
        for seed in (1, 2):
            for cell in CELLS:
                name = f"cell_{cell['cell']}_s{seed}"
                directory = run_dir(cell, seed)
                if completed(cell, seed):
                    state["completed"].append(name)
                    write_json(STATUS, state)
                    continue
                state["current"] = name
                state["cell"] = cell["cell"]
                state["seed"] = seed
                state["phase_started_at"] = now()
                write_json(STATUS, state)
                last = directory / "weights" / "last.pt"
                if last.is_file():
                    print(f"\n===== RESUME {name} =====", flush=True)
                    result = YOLO(str(last)).train(resume=True)
                else:
                    if directory.exists():
                        raise RuntimeError(f"incomplete run without last.pt: {directory}")
                    print(f"\n===== INTERACTION REPLICATION {name}: {cell['label']} =====", flush=True)
                    result = YOLO(str(cell["model"])).train(
                        name=name,
                        bbox_loss_mode="ciou",
                        seed=seed,
                        exist_ok=False,
                        **TRAIN_ARGS,
                    )
                if len(read_rows(Path(result.save_dir) / "results.csv")) != 60:
                    raise RuntimeError(f"{name} did not finish 60 epochs")
                state["completed"].append(name)
                state["current"] = None
                write_json(STATUS, state)
                collect()
        state["protocol_valid"] = verify()
        if not state["protocol_valid"]:
            raise RuntimeError("interaction confirmation protocol verification failed")
        state["status"] = "completed"
        state["completed_at"] = now()
        state["analysis"] = str(ANALYSIS.resolve())
        write_json(STATUS, state)
        collect()
    except BaseException as error:
        state["status"] = "failed"
        state["failed_at"] = now()
        state["error"] = f"{type(error).__name__}: {error}"
        write_json(STATUS, state)
        raise


if __name__ == "__main__":
    main()
