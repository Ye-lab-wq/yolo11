#!/usr/bin/env python3
"""Run the two missing cells of the frozen ADown backbone/neck 2x2 screen."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from reproduction.ablation.run_secondary_screen import TRAIN_ARGS as SECONDARY_TRAIN_ARGS


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"
DATA = ROOT / "DetectDataset_clean_v2" / "data.yaml"
PARENT_RUN = ROOT / "runs" / "ablation_screen_v1" / "p2_msef_paper_ciou_s0"
JOINT_RUN = ROOT / "runs" / "ablation_secondary_v1" / "msefbase_adown_ciou_s0"
RUN_ROOT = ROOT / "runs" / "ablation_adown_placement_v1"
STATUS = RUN_ROOT / "status.json"
SUMMARY = RUN_ROOT / "placement_summary.csv"
ANALYSIS = RUN_ROOT / "factorial_analysis.json"
VERIFICATION = RUN_ROOT / "protocol_verification.json"
TRAIN_ARGS = {**SECONDARY_TRAIN_ARGS, "project": str(RUN_ROOT)}

EXPERIMENTS = (
    (
        "adown_backbone_only_s0",
        "Backbone ADown + neck Conv",
        MODEL_DIR / "p2_msef_paper_adown_backbone.yaml",
    ),
    (
        "adown_neck_only_s0",
        "Backbone Conv + neck ADown",
        MODEL_DIR / "p2_msef_paper_adown_neck.yaml",
    ),
)
ALLOWED_ARG_DIFFERENCES = {"model", "name", "project", "resume", "save_dir"}


def now() -> str:
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(name: str, placement: str, source: str, run_dir: Path, model_yaml: Path) -> dict[str, object]:
    curve = rows(run_dir / "results.csv")
    if len(curve) != 60:
        raise RuntimeError(f"{name} has {len(curve)} epochs, expected 60")
    best = max(curve, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    weight = run_dir / "weights" / "best.pt"
    if not weight.is_file():
        raise FileNotFoundError(weight)
    parameters = sum(parameter.numel() for parameter in YOLO(str(weight)).model.parameters())
    return {
        "experiment": name,
        "placement": placement,
        "source": source,
        "model_yaml": str(model_yaml.resolve()),
        "best_epoch": int(best["epoch"]),
        "precision": float(best["metrics/precision(B)"]),
        "recall": float(best["metrics/recall(B)"]),
        "map50": float(best["metrics/mAP50(B)"]),
        "map50_95": float(best["metrics/mAP50-95(B)"]),
        "parameters": parameters,
        "best_weight": str(weight.resolve()),
    }


def collect() -> list[dict[str, object]]:
    records = [
        summarize(
            "p2_msef_paper_ciou_s0",
            "backbone_conv__neck_conv",
            "reused_neither_endpoint",
            PARENT_RUN,
            MODEL_DIR / "p2_msef_paper.yaml",
        )
    ]
    for name, placement, model_yaml in EXPERIMENTS:
        run_dir = RUN_ROOT / name
        if len(rows(run_dir / "results.csv")) == 60 and (run_dir / "weights" / "best.pt").is_file():
            records.append(summarize(name, placement, "new_placement_screen", run_dir, model_yaml))
    records.append(
        summarize(
            "msefbase_adown_ciou_s0",
            "backbone_adown__neck_adown",
            "reused_joint_endpoint",
            JOINT_RUN,
            MODEL_DIR / "p2_msef_paper_adown.yaml",
        )
    )
    with SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    if len(records) == 4:
        values = {record["experiment"]: float(record["map50_95"]) for record in records}
        neither = values["p2_msef_paper_ciou_s0"]
        backbone = values["adown_backbone_only_s0"]
        neck = values["adown_neck_only_s0"]
        joint = values["msefbase_adown_ciou_s0"]
        write_json(
            ANALYSIS,
            {
                "metric": "best validation mAP50:95",
                "backbone_only_delta_pp": (backbone - neither) * 100.0,
                "neck_only_delta_pp": (neck - neither) * 100.0,
                "joint_delta_pp": (joint - neither) * 100.0,
                "interaction_pp": (joint - backbone - neck + neither) * 100.0,
            },
        )
    return records


def verify() -> bool:
    parent_args = yaml.safe_load((PARENT_RUN / "args.yaml").read_text(encoding="utf-8"))
    output = []
    valid_all = True
    for name, placement, _model_yaml in EXPERIMENTS:
        run_dir = RUN_ROOT / name
        args_path = run_dir / "args.yaml"
        if args_path.is_file():
            args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
            differences = {
                key: [parent_args.get(key), args.get(key)]
                for key in sorted(set(parent_args) | set(args))
                if key not in ALLOWED_ARG_DIFFERENCES and parent_args.get(key) != args.get(key)
            }
        else:
            args, differences = {}, {"missing": str(args_path)}
        valid = (
            len(rows(run_dir / "results.csv")) == 60
            and not differences
            and (run_dir / "weights" / "best.pt").is_file()
            and (run_dir / "weights" / "last.pt").is_file()
            and args.get("pretrained") is False
            and args.get("seed") == 0
            and args.get("bbox_loss_mode") == "ciou"
            and Path(args.get("data", "")).resolve() == DATA.resolve()
        )
        valid_all &= valid
        output.append(
            {
                "run": name,
                "placement": placement,
                "valid": valid,
                "epochs": len(rows(run_dir / "results.csv")),
                "unexpected_argument_differences": differences,
            }
        )
    write_json(
        VERIFICATION,
        {
            "valid": valid_all,
            "design": "2x2 backbone ADown x neck ADown",
            "test_access": "none",
            "runs": output,
        },
    )
    return valid_all


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "created_at": now(),
        "frozen_protocol": str((HERE / "ADOWN_PLACEMENT_PROTOCOL.md").resolve()),
        "train_args": TRAIN_ARGS,
        "reused_endpoints": {"neither": str(PARENT_RUN.resolve()), "joint": str(JOINT_RUN.resolve())},
        "experiments": [
            {"name": name, "placement": placement, "model": str(model_yaml.resolve())}
            for name, placement, model_yaml in EXPERIMENTS
        ],
        "test_access": "forbidden",
    }
    protocol_path = RUN_ROOT / "protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing["experiments"] != protocol["experiments"] or existing["train_args"] != protocol["train_args"]:
            raise RuntimeError("frozen ADown placement protocol differs from current code")
    else:
        write_json(protocol_path, protocol)

    state = {"status": "running", "started_or_resumed_at": now(), "current": None, "completed": []}
    write_json(STATUS, state)
    collect()
    try:
        for name, placement, model_yaml in EXPERIMENTS:
            run_dir = RUN_ROOT / name
            if len(rows(run_dir / "results.csv")) == 60 and (run_dir / "weights" / "best.pt").is_file():
                state["completed"].append(name)
                write_json(STATUS, state)
                continue
            state["current"] = name
            state["placement"] = placement
            state["phase_started_at"] = now()
            write_json(STATUS, state)
            last = run_dir / "weights" / "last.pt"
            if last.is_file():
                print(f"\n===== RESUME {name}: {placement} =====", flush=True)
                result = YOLO(str(last)).train(resume=True)
            else:
                if run_dir.exists():
                    raise RuntimeError(f"incomplete run without last.pt: {run_dir}")
                print(f"\n===== ADOWN PLACEMENT {name}: {placement} =====", flush=True)
                result = YOLO(str(model_yaml)).train(
                    name=name,
                    bbox_loss_mode="ciou",
                    exist_ok=False,
                    **TRAIN_ARGS,
                )
            if len(rows(Path(result.save_dir) / "results.csv")) != 60:
                raise RuntimeError(f"{name} did not finish 60 epochs")
            state["completed"].append(name)
            state["current"] = None
            write_json(STATUS, state)
            collect()
        state["protocol_valid"] = verify()
        if not state["protocol_valid"]:
            raise RuntimeError("ADown placement protocol verification failed")
        state["status"] = "completed"
        state["completed_at"] = now()
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
