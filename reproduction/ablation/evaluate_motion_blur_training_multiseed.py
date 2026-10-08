#!/usr/bin/env python3
"""Evaluate the frozen three-seed motion-blur-training 2x2 experiment."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from ultralytics import YOLO

from reproduction.ablation.evaluate_frozen import coco_metrics, image_paths, make_ground_truth, predictions


BENCHMARK = ROOT / "DetectDataset_clean_v2_motion_blur"
OUTPUT = ROOT / "reproduction/results/motion_blur_training_3seed"
SEED0_SOURCE = ROOT / "reproduction/results/motion_blur_training_2x2_seed0/results.json"
CONDITIONS = ("clean", "light", "moderate", "strong")
MODEL_ORDER = ("P2_standard", "P2_blur_train", "P2_MSEF_standard", "P2_MSEF_blur_train")
MODELS = {
    1: OrderedDict(
        [
            ("P2_standard", ROOT / "runs/ablation_interaction_confirm_v1/cell_00_s1/weights/best.pt"),
            ("P2_blur_train", ROOT / "runs/ablation_motion_blur_replicates_v1/seed1/p2_motion_blur_s1/weights/best.pt"),
            ("P2_MSEF_standard", ROOT / "runs/ablation_interaction_confirm_v1/cell_10_s1/weights/best.pt"),
            ("P2_MSEF_blur_train", ROOT / "runs/ablation_motion_blur_replicates_v1/seed1/p2_msef_motion_blur_s1/weights/best.pt"),
        ]
    ),
    2: OrderedDict(
        [
            ("P2_standard", ROOT / "runs/ablation_interaction_confirm_v1/cell_00_s2/weights/best.pt"),
            ("P2_blur_train", ROOT / "runs/ablation_motion_blur_replicates_v1/seed2/p2_motion_blur_s2/weights/best.pt"),
            ("P2_MSEF_standard", ROOT / "runs/ablation_interaction_confirm_v1/cell_10_s2/weights/best.pt"),
            ("P2_MSEF_blur_train", ROOT / "runs/ablation_motion_blur_replicates_v1/seed2/p2_msef_motion_blur_s2/weights/best.pt"),
        ]
    ),
}
METRICS = (
    "AP50_95_percent",
    "AP50_percent",
    "AP75_percent",
    "AP_tiny_lt16_percent",
    "AP_small_16_32_percent",
    "AP_medium_32_96_percent",
    "AP_large_ge96_percent",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int, default=4)
    return parser.parse_args()


def now() -> str:
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def paths_for(condition: str) -> list[Path]:
    directory = ROOT / "DetectDataset_clean_v2/test/images" if condition == "clean" else BENCHMARK / condition / "images"
    return image_paths((directory,))


def result_row(seed: int, model: str, condition: str, metrics: dict[str, object]) -> dict[str, object]:
    return {
        "seed": seed,
        "model": model,
        "condition": condition,
        "images": int(metrics["images"]),
        "instances": int(metrics["instances"]),
        "AP50_95_percent": 100.0 * float(metrics["AP"]),
        "AP50_percent": 100.0 * float(metrics["AP50"]),
        "AP75_percent": 100.0 * float(metrics["AP75"]),
        "AP_tiny_lt16_percent": 100.0 * float(metrics["AP_tiny_lt16"]),
        "AP_small_16_32_percent": 100.0 * float(metrics["AP_small_16_32"]),
        "AP_medium_32_96_percent": 100.0 * float(metrics["AP_coco_medium_32_96"]),
        "AP_large_ge96_percent": 100.0 * float(metrics["AP_coco_large_ge96"]),
    }


def seed0_records() -> list[dict[str, object]]:
    payload = json.loads(SEED0_SOURCE.read_text(encoding="utf-8"))
    return [{"seed": 0, **record} for record in payload["records"]]


def summarize(records: list[dict[str, object]], device: str, batch: int) -> None:
    records.sort(key=lambda row: (int(row["seed"]), MODEL_ORDER.index(str(row["model"])), CONDITIONS.index(str(row["condition"]))))
    write_csv(OUTPUT / "seed_results.csv", records)
    by_key = {(int(row["seed"]), str(row["model"]), str(row["condition"])): row for row in records}

    aggregate = []
    for model in MODEL_ORDER:
        for condition in CONDITIONS:
            item: dict[str, object] = {"model": model, "condition": condition, "seeds": 3}
            for metric in METRICS:
                values = [float(by_key[(seed, model, condition)][metric]) for seed in (0, 1, 2)]
                item[f"{metric}_mean"] = statistics.mean(values)
                item[f"{metric}_sample_sd"] = statistics.stdev(values)
            aggregate.append(item)
    write_csv(OUTPUT / "aggregate.csv", aggregate)

    effects = []
    for seed in (0, 1, 2):
        for condition in CONDITIONS:
            row: dict[str, object] = {"seed": seed, "condition": condition}
            for metric in ("AP50_95_percent", "AP75_percent"):
                value = lambda model: float(by_key[(seed, model, condition)][metric])
                blur_p2 = value("P2_blur_train") - value("P2_standard")
                blur_msef = value("P2_MSEF_blur_train") - value("P2_MSEF_standard")
                msef_standard = value("P2_MSEF_standard") - value("P2_standard")
                msef_blur = value("P2_MSEF_blur_train") - value("P2_blur_train")
                prefix = metric.removesuffix("_percent")
                row[f"{prefix}_blur_effect_P2_pp"] = blur_p2
                row[f"{prefix}_blur_effect_MSEF_pp"] = blur_msef
                row[f"{prefix}_MSEF_effect_standard_pp"] = msef_standard
                row[f"{prefix}_MSEF_effect_blur_train_pp"] = msef_blur
                row[f"{prefix}_interaction_pp"] = msef_blur - msef_standard
            effects.append(row)
    write_csv(OUTPUT / "effects_by_seed.csv", effects)

    effect_aggregate = []
    effect_keys = [key for key in effects[0] if key not in {"seed", "condition"}]
    for condition in CONDITIONS:
        subset = [row for row in effects if row["condition"] == condition]
        item = {"condition": condition, "seeds": 3}
        for key in effect_keys:
            values = [float(row[key]) for row in subset]
            item[f"{key}_mean"] = statistics.mean(values)
            item[f"{key}_sample_sd"] = statistics.stdev(values)
        effect_aggregate.append(item)
    write_csv(OUTPUT / "effects_aggregate.csv", effect_aggregate)

    lines = [
        "# Motion-blur training 2x2 — three-seed public-test audit",
        "",
        "All checkpoints are 60-epoch from-scratch runs selected only by clean-validation mAP50:95. Seed 0 results are reused from the frozen identical-protocol evaluation; seeds 1 and 2 were evaluated on the same 625-image/685-instance public test and deterministic derivatives.",
        "",
        "## AP50:95 by seed",
        "",
        "| Seed | Model | Clean/% | Light/% | Moderate/% | Strong/% | Blur mean/% |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for seed in (0, 1, 2):
        for model in MODEL_ORDER:
            values = [float(by_key[(seed, model, condition)]["AP50_95_percent"]) for condition in CONDITIONS]
            lines.append(f"| {seed} | {model} | {values[0]:.2f} | {values[1]:.2f} | {values[2]:.2f} | {values[3]:.2f} | {statistics.mean(values[1:]):.2f} |")

    lines.extend(["", "## AP50:95 mean ± sample SD", "", "| Model | Clean/% | Light/% | Moderate/% | Strong/% |", "|---|---:|---:|---:|---:|"])
    for model in MODEL_ORDER:
        cells = []
        for condition in CONDITIONS:
            values = [float(by_key[(seed, model, condition)]["AP50_95_percent"]) for seed in (0, 1, 2)]
            cells.append(f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## AP50:95 factorial effects by seed",
            "",
            "Interaction = MSEF effect after blur training minus MSEF effect under standard training. Values are percentage points.",
            "",
            "| Seed | Condition | Blur effect on P2 | Blur effect on P2+MSEF | MSEF effect, standard | MSEF effect, blur-trained | Interaction |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in effects:
        lines.append(
            f"| {row['seed']} | {row['condition']} | {row['AP50_95_blur_effect_P2_pp']:+.2f} | "
            f"{row['AP50_95_blur_effect_MSEF_pp']:+.2f} | {row['AP50_95_MSEF_effect_standard_pp']:+.2f} | "
            f"{row['AP50_95_MSEF_effect_blur_train_pp']:+.2f} | {row['AP50_95_interaction_pp']:+.2f} |"
        )

    lines.extend(["", "## Mean AP50:95 effects", "", "| Condition | Blur effect on P2 | Blur effect on P2+MSEF | MSEF effect, standard | MSEF effect, blur-trained | Interaction |", "|---|---:|---:|---:|---:|---:|"])
    for condition in CONDITIONS:
        subset = [row for row in effects if row["condition"] == condition]
        keys = ("AP50_95_blur_effect_P2_pp", "AP50_95_blur_effect_MSEF_pp", "AP50_95_MSEF_effect_standard_pp", "AP50_95_MSEF_effect_blur_train_pp", "AP50_95_interaction_pp")
        cells = []
        for key in keys:
            values = [float(row[key]) for row in subset]
            cells.append(f"{statistics.mean(values):+.2f} ± {statistics.stdev(values):.2f}")
        lines.append(f"| {condition} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "No p-value is reported because n=3 seeds is too small for reliable distributional inference. Report seed-wise effects and dispersion. This synthetic public-test corruption audit does not replace a separately collected real-motion-blur test set, and it must not be used for further model selection.",
        ]
    )
    (OUTPUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        OUTPUT / "results.json",
        {
            "protocol": "frozen three-seed 2x2; public test and deterministic derivatives; no model selection",
            "seed0_source": str(SEED0_SOURCE.resolve()),
            "models": {str(seed): {name: str(path.resolve()) for name, path in models.items()} for seed, models in MODELS.items()},
            "device": device,
            "batch": batch,
            "confidence_floor": 0.001,
            "nms_iou": 0.7,
            "max_det": 300,
            "records": records,
        },
    )


def main() -> None:
    args = parse_args()
    device = "0" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if not (BENCHMARK / "metadata.json").is_file():
        raise FileNotFoundError("build the motion-blur benchmark first")
    if not SEED0_SOURCE.is_file():
        raise FileNotFoundError(SEED0_SOURCE)
    for models in MODELS.values():
        for weight in models.values():
            if not weight.is_file():
                raise FileNotFoundError(weight)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    partial = OUTPUT / "partial_results.json"
    records = json.loads(partial.read_text(encoding="utf-8")) if partial.is_file() else seed0_records()
    completed = {(int(row["seed"]), str(row["model"]), str(row["condition"])) for row in records}
    write_json(OUTPUT / "status.json", {"status": "running", "updated_at": now(), "completed": len(completed), "total": 48, "device": device})
    try:
        for seed, models in MODELS.items():
            for model_name, weight in models.items():
                pending = [condition for condition in CONDITIONS if (seed, model_name, condition) not in completed]
                if not pending:
                    continue
                model = YOLO(str(weight))
                for condition in pending:
                    paths = paths_for(condition)
                    if len(paths) != 625:
                        raise RuntimeError(f"{condition}: expected 625 images, found {len(paths)}")
                    print(f"{now()} | seed={seed} | {model_name} | {condition} | {len(completed)}/48 | device={device}", flush=True)
                    ground_truth, path_to_id = make_ground_truth(paths)
                    detections = predictions(model, paths, path_to_id, device, args.batch)
                    metrics = coco_metrics(ground_truth, detections)
                    records.append(result_row(seed, model_name, condition, metrics))
                    completed.add((seed, model_name, condition))
                    write_json(partial, records)
                    write_json(OUTPUT / "status.json", {"status": "running", "updated_at": now(), "completed": len(completed), "total": 48, "last_completed": {"seed": seed, "model": model_name, "condition": condition}})
                    print(f"DONE | AP50:95={100.0 * float(metrics['AP']):.2f}", flush=True)
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        summarize(records, device, args.batch)
        write_json(OUTPUT / "status.json", {"status": "completed", "completed_at": now(), "completed": len(records), "total": 48, "device": device})
        print(OUTPUT, flush=True)
    except BaseException as error:
        write_json(OUTPUT / "status.json", {"status": "failed", "failed_at": now(), "completed": len(completed), "total": 48, "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
