#!/usr/bin/env python3
"""Build reproducible visual evidence for the seed-1 P2, MSEF, and ADown ablation.

The script evaluates the frozen YOLO11n -> +P2 -> +P2+MSEF ->
+P2+MSEF+ADown chain on the combined clean-v2 test split. It exports raw
predictions, COCO-style AP curves and size metrics, and an objectively selected
qualitative grid containing one direct-parent gain per module and one ADown
counterexample. Selection uses only box/score/ground-truth geometry at a fixed
confidence threshold; images are not manually chosen or cosmetically altered.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from PIL import Image
from pycocotools.cocoeval import COCOeval


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from reproduction.ablation.evaluate_frozen import coco_metrics, make_ground_truth, mean_precision


MODELS = OrderedDict(
    [
        ("YOLO11n", ROOT / "runs/final_baseline_yolo11n_200/train_seed1/weights/best.pt"),
        ("+P2", ROOT / "runs/ablation_200_v1/p2/train_seed1/weights/best.pt"),
        (
            "+P2+MSEF",
            ROOT / "runs/ablation_200_v1/p2_msef_paper/train_seed1/weights/best.pt",
        ),
        ("+P2+ADown", ROOT / "runs/ablation_200_v1/p2_adown/train_seed1/weights/best.pt"),
        (
            "+P2+MSEF+ADown",
            ROOT / "runs/final_candidate_p2_msef_adown_s0_200/train_seed1/weights/best.pt",
        ),
    ]
)
COLORS = {
    "YOLO11n": "#D55E00",
    "+P2": "#0072B2",
    "+P2+MSEF": "#CC79A7",
    "+P2+ADown": "#000000",
    "+P2+MSEF+ADown": "#009E73",
}
MARKERS = {
    "YOLO11n": "o",
    "+P2": "s",
    "+P2+MSEF": "D",
    "+P2+ADown": "v",
    "+P2+MSEF+ADown": "^",
}
COMPLEXITY = {
    "YOLO11n": {"parameters_m": 2.590035, "gflops": 6.4},
    "+P2": {"parameters_m": 2.903812, "gflops": 10.8},
    "+P2+MSEF": {"parameters_m": 2.920468, "gflops": 11.7},
    "+P2+ADown": {"parameters_m": 2.278148, "gflops": 9.5},
    "+P2+MSEF+ADown": {"parameters_m": 2.294804, "gflops": 10.3},
}
IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reproduction/results/p2_msef_adown_visual_evidence_seed1",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--display-conf", type=float, default=0.25)
    parser.add_argument(
        "--reuse-predictions",
        action="store_true",
        help="Reuse this output directory's raw_predictions.json and only rebuild evaluation/figures.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_paths() -> list[Path]:
    directories = (
        ROOT / "DetectDataset_clean_v2/test/images",
        ROOT / "DetectDataset_clean_v2/self_test/images",
    )
    return sorted(
        path.resolve()
        for directory in directories
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def load_ground_truth(paths: list[Path]) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for path in paths:
        with Image.open(path) as image:
            width, height = image.size
        boxes = []
        label_path = path.parent.parent / "labels" / f"{path.stem}.txt"
        for line in label_path.read_text(encoding="utf-8").splitlines():
            class_id, x, y, w, h = map(float, line.split())
            if int(class_id) != 0:
                raise ValueError(f"Unexpected class in {label_path}: {class_id}")
            boxes.append(
                [
                    (x - w / 2) * width,
                    (y - h / 2) * height,
                    (x + w / 2) * width,
                    (y + h / 2) * height,
                ]
            )
        records[str(path)] = {
            "path": path,
            "width": width,
            "height": height,
            "boxes": np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
        }
    return records


def infer(model_path: Path, paths: list[Path], device: str, batch: int) -> dict[str, np.ndarray]:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = YOLO(str(model_path))
    predictions: dict[str, np.ndarray] = {}
    for start in range(0, len(paths), batch):
        chunk = paths[start : start + batch]
        results = model.predict(
            source=[str(path) for path in chunk],
            imgsz=640,
            conf=0.001,
            iou=0.7,
            max_det=300,
            batch=batch,
            device=device,
            workers=4,
            stream=True,
            verbose=False,
        )
        for result in results:
            key = str(Path(result.path).resolve())
            boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float64)
            scores = result.boxes.conf.detach().cpu().numpy().astype(np.float64)[:, None]
            predictions[key] = np.concatenate((boxes, scores), axis=1)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float64)
    top_left = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = np.clip(bottom_right - top_left, 0, None).prod(2)
    area1 = np.clip(boxes1[:, 2:] - boxes1[:, :2], 0, None).prod(1)
    area2 = np.clip(boxes2[:, 2:] - boxes2[:, :2], 0, None).prod(1)
    return intersection / np.maximum(area1[:, None] + area2[None, :] - intersection, 1e-12)


def detection_records(
    predictions: dict[str, np.ndarray], path_to_id: dict[str, int]
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path, pred in predictions.items():
        image_id = path_to_id[path]
        for box in pred:
            x1, y1, x2, y2, score = map(float, box)
            records.append(
                {
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score,
                }
            )
    return records


def evaluate_predictions(ground_truth, records: list[dict[str, object]]) -> tuple[list[float], dict]:
    detections = ground_truth.loadRes(records)
    evaluator = COCOeval(ground_truth, detections, "bbox")
    evaluator.params.imgIds = sorted(ground_truth.getImgIds())
    evaluator.params.maxDets = [1, 10, 300]
    evaluator.evaluate()
    evaluator.accumulate()
    curve = [100.0 * mean_precision(evaluator, 0, index) for index in range(10)]
    metrics = coco_metrics(ground_truth, records)
    return curve, metrics


def best_prediction_for_target(pred: np.ndarray, target: np.ndarray, confidence: float) -> dict[str, object]:
    kept = pred[pred[:, 4] >= confidence]
    if len(kept) == 0:
        return {"iou": 0.0, "confidence": 0.0, "box": None}
    ious = box_iou(kept[:, :4], target[None, :])[:, 0]
    index = int(np.argmax(ious))
    return {
        "iou": float(ious[index]),
        "confidence": float(kept[index, 4]),
        "box": [float(value) for value in kept[index, :4]],
    }


def target_records(
    ground_truth: dict[str, dict[str, object]],
    predictions: dict[str, dict[str, np.ndarray]],
    confidence: float,
) -> list[dict[str, object]]:
    targets: list[dict[str, object]] = []
    for path, record in ground_truth.items():
        for target_index, box in enumerate(record["boxes"]):
            width = float(box[2] - box[0])
            height = float(box[3] - box[1])
            model_results = {
                name: best_prediction_for_target(model_predictions[path], box, confidence)
                for name, model_predictions in predictions.items()
            }
            raw_model_results = {
                name: best_prediction_for_target(model_predictions[path], box, 0.001)
                for name, model_predictions in predictions.items()
            }
            targets.append(
                {
                    "path": path,
                    "target_index": target_index,
                    "gt_box": [float(value) for value in box],
                    "equivalent_side_px": float(np.sqrt(max(width * height, 0.0))),
                    "models": model_results,
                    "raw_models": raw_model_results,
                    "p2_delta_iou": model_results["+P2"]["iou"] - model_results["YOLO11n"]["iou"],
                    "msef_delta_iou": model_results["+P2+MSEF"]["iou"]
                    - model_results["+P2"]["iou"],
                    "adown_without_msef_delta_iou": model_results["+P2+ADown"]["iou"]
                    - model_results["+P2"]["iou"],
                    "adown_with_msef_delta_iou": model_results["+P2+MSEF+ADown"]["iou"]
                    - model_results["+P2+MSEF"]["iou"],
                    "full_over_best_single_delta_iou": model_results["+P2+MSEF+ADown"]["iou"]
                    - max(
                        model_results["+P2+MSEF"]["iou"],
                        model_results["+P2+ADown"]["iou"],
                    ),
                }
            )
    return targets


def adown_score_shift_analysis(
    targets: list[dict[str, object]], parent_name: str, final_name: str
) -> dict[str, object]:
    """Audit whether ADown score suppression is concentrated at large target sizes."""
    rows = []
    for target in targets:
        parent = target["raw_models"][parent_name]
        final = target["raw_models"][final_name]
        rows.append(
            {
                "path": str(Path(target["path"]).relative_to(ROOT)),
                "target_index": int(target["target_index"]),
                "equivalent_side_px": float(target["equivalent_side_px"]),
                "parent_iou": float(parent["iou"]),
                "parent_confidence": float(parent["confidence"]),
                "final_iou": float(final["iou"]),
                "final_confidence": float(final["confidence"]),
            }
        )
    grouped = {}
    for label, low, high in (
        ("lt32", 0.0, 32.0),
        ("32_96", 32.0, 96.0),
        ("96_300", 96.0, 300.0),
        ("ge300", 300.0, float("inf")),
    ):
        subset = [row for row in rows if low <= row["equivalent_side_px"] < high]
        matched = [row for row in subset if row["parent_iou"] >= 0.50 and row["final_iou"] >= 0.50]
        lost = [
            row
            for row in subset
            if row["parent_iou"] >= 0.50
            and row["parent_confidence"] >= 0.25
            and not (row["final_iou"] >= 0.50 and row["final_confidence"] >= 0.25)
        ]
        gained = [
            row
            for row in subset
            if row["final_iou"] >= 0.50
            and row["final_confidence"] >= 0.25
            and not (row["parent_iou"] >= 0.50 and row["parent_confidence"] >= 0.25)
        ]
        score_changes = [row["final_confidence"] - row["parent_confidence"] for row in matched]
        grouped[label] = {
            "targets": len(subset),
            "matched_by_both_at_iou50": len(matched),
            "median_final_minus_parent_confidence": float(np.median(score_changes))
            if score_changes
            else None,
            "lost_at_confidence_025": len(lost),
            "gained_at_confidence_025": len(gained),
        }
    largest_drops = sorted(
        (
            row
            for row in rows
            if row["parent_iou"] >= 0.50
            and row["final_iou"] >= 0.50
            and row["parent_confidence"] >= 0.25
        ),
        key=lambda row: row["final_confidence"] - row["parent_confidence"],
    )[:10]
    for row in largest_drops:
        row["final_minus_parent_confidence"] = (
            row["final_confidence"] - row["parent_confidence"]
        )
    return {
        "parent": parent_name,
        "child": final_name,
        "definition": "best post-NMS candidate per target from confidence-floor 0.001 inference",
        "grouped_by_equivalent_side_px": grouped,
        "largest_matching_score_drops": largest_drops,
    }


def choose_unique(
    candidates: list[dict[str, object]],
    count: int,
    used_paths: set[str],
) -> list[dict[str, object]]:
    selected = []
    for candidate in candidates:
        if candidate["path"] in used_paths:
            continue
        selected.append(candidate)
        used_paths.add(candidate["path"])
        if len(selected) == count:
            break
    return selected


def select_examples(targets: list[dict[str, object]]) -> list[dict[str, object]]:
    used: set[str] = set()
    p2_candidates = sorted(
        (
            target
            for target in targets
            if target["equivalent_side_px"] < 96
            and target["models"]["+P2"]["iou"] >= 0.50
            and target["p2_delta_iou"] > 0
        ),
        key=lambda target: (target["p2_delta_iou"], -target["equivalent_side_px"]),
        reverse=True,
    )
    msef_candidates = sorted(
        (
            target
            for target in targets
            if target["models"]["+P2+MSEF"]["iou"] >= 0.75
            and target["msef_delta_iou"] > 0
        ),
        key=lambda target: target["msef_delta_iou"],
        reverse=True,
    )
    adown_candidates = sorted(
        (
            target
            for target in targets
            if target["models"]["+P2+ADown"]["iou"] >= 0.75
            and target["adown_without_msef_delta_iou"] > 0
        ),
        key=lambda target: target["adown_without_msef_delta_iou"],
        reverse=True,
    )
    interaction_candidates = sorted(
        (
            target
            for target in targets
            if target["models"]["+P2+MSEF+ADown"]["iou"] >= 0.75
            and target["full_over_best_single_delta_iou"] > 0
        ),
        key=lambda target: target["full_over_best_single_delta_iou"],
        reverse=True,
    )
    regressions = sorted(
        (
            target
            for target in targets
            if target["models"]["+P2+MSEF"]["iou"] >= 0.50
            and target["adown_with_msef_delta_iou"] < 0
        ),
        key=lambda target: target["adown_with_msef_delta_iou"],
    )
    selected = []
    for label, candidates, count in (
        ("P2 improvement", p2_candidates, 1),
        ("MSEF improvement", msef_candidates, 1),
        ("ADown improvement", adown_candidates, 1),
        ("Combined improvement", interaction_candidates, 1),
        ("Counterexample", regressions, 1),
    ):
        items = choose_unique(candidates, count, used)
        for item in items:
            copy = dict(item)
            copy["selection_group"] = label
            selected.append(copy)
    return selected


def crop_bounds(target: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    box_width = float(target[2] - target[0])
    box_height = float(target[3] - target[1])
    side = int(np.ceil(max(96.0, 3.5 * max(box_width, box_height))))
    side = min(side, width, height)
    center_x = float((target[0] + target[2]) / 2)
    center_y = float((target[1] + target[3]) / 2)
    left = int(round(center_x - side / 2))
    top = int(round(center_y - side / 2))
    left = min(max(left, 0), width - side)
    top = min(max(top, 0), height - side)
    return left, top, left + side, top + side


def add_box(ax, box, crop, color, linestyle="-", linewidth=1.8, label=None):
    left, top, right, bottom = crop
    x1, y1, x2, y2 = box
    if x2 <= left or x1 >= right or y2 <= top or y1 >= bottom:
        return
    x1, y1 = max(x1, left) - left, max(y1, top) - top
    x2, y2 = min(x2, right) - left, min(y2, bottom) - top
    patch = Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
    )
    patch.set_path_effects([pe.Stroke(linewidth=linewidth + 1.5, foreground="black"), pe.Normal()])
    ax.add_patch(patch)


def qualitative_figure(
    selected: list[dict[str, object]],
    ground_truth: dict[str, dict[str, object]],
    predictions: dict[str, dict[str, np.ndarray]],
    confidence: float,
    output_dir: Path,
) -> list[dict[str, object]]:
    if not selected:
        raise RuntimeError("No qualitative examples satisfied the deterministic selection rules")
    columns = ["Ground truth", *MODELS.keys()]
    short_names = {
        "Ground truth": "Ground truth",
        "YOLO11n": "YOLO11n",
        "+P2": "+P2",
        "+P2+MSEF": "+P2+MSEF",
        "+P2+ADown": "+P2+ADown",
        "+P2+MSEF+ADown": "Full model",
    }
    with mpl.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        fig, axes = plt.subplots(
            len(selected),
            len(columns),
            figsize=(7.08, 1.62 * len(selected)),
            layout="constrained",
            squeeze=False,
        )
        manifest_rows = []
        for row, example in enumerate(selected):
            path = Path(example["path"])
            record = ground_truth[str(path)]
            target = np.asarray(example["gt_box"], dtype=float)
            crop = crop_bounds(target, int(record["width"]), int(record["height"]))
            with Image.open(path) as source:
                image = np.asarray(source.convert("RGB").crop(crop))
            for col, column in enumerate(columns):
                ax = axes[row, col]
                ax.imshow(image, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if column != "Ground truth":
                    visible = predictions[column][str(path)]
                    visible = visible[visible[:, 4] >= confidence]
                    for prediction in visible:
                        add_box(ax, prediction[:4], crop, "#D62728", linewidth=1.8)
                    result = example["models"][column]
                    if result["box"] is None:
                        status = "Miss"
                    else:
                        status = f"IoU {result['iou']:.2f} | C {result['confidence']:.2f}"
                    ax.text(
                        0.02,
                        0.02,
                        status,
                        transform=ax.transAxes,
                        va="bottom",
                        ha="left",
                        fontsize=5.5,
                        color="#B22222",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
                    )
                else:
                    for gt_index, gt_box in enumerate(record["boxes"]):
                        add_box(
                            ax,
                            gt_box,
                            crop,
                            "#00A65A",
                            linestyle="-",
                            linewidth=2.0 if gt_index == example["target_index"] else 1.2,
                        )
                    ax.text(
                        0.02,
                        0.02,
                        f"GT size {example['equivalent_side_px']:.1f}px",
                        transform=ax.transAxes,
                        va="bottom",
                        ha="left",
                        fontsize=5.5,
                        color="black",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
                    )
                if row == 0:
                    ax.set_title(
                        short_names[column],
                        color="black",
                    )
                if col == 0:
                    relative = path.relative_to(ROOT)
                    ax.set_ylabel(
                        f"{example['selection_group']}\n{relative.parent.parent.name}/{path.stem[:18]}",
                        rotation=0,
                        ha="right",
                        va="center",
                    )
            manifest_rows.append(
                {
                    **example,
                    "path": str(path.relative_to(ROOT)),
                    "crop_xyxy": list(crop),
                }
            )
        fig.suptitle(
            "Seed-1 qualitative comparison on the combined test set (confidence >= 0.25)\n"
            "Ground truth: green solid box; predictions: red solid boxes",
            fontsize=8,
        )
        fig.savefig(output_dir / "qualitative_p2_msef_adown_seed1.png", dpi=400)
        fig.savefig(output_dir / "qualitative_p2_msef_adown_seed1.pdf")
        plt.close(fig)
    return manifest_rows


def large_target_diagnostic_figure(
    ground_truth: dict[str, dict[str, object]],
    predictions: dict[str, dict[str, np.ndarray]],
    confidence: float,
    output_dir: Path,
) -> dict[str, object]:
    path = (ROOT / "DetectDataset_clean_v2/test/images/pic_1032__cff54d60821402c5.jpg").resolve()
    record = ground_truth[str(path)]
    target = np.asarray(record["boxes"][0], dtype=float)
    crop = crop_bounds(target, int(record["width"]), int(record["height"]))
    with Image.open(path) as source:
        image = np.asarray(source.convert("RGB").crop(crop))
    columns = ["Ground truth", *MODELS.keys()]
    short_names = ["Ground truth", "YOLO11n", "+P2", "+P2+MSEF", "+P2+ADown", "Full model"]
    diagnostics: dict[str, object] = {}
    with mpl.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.titlesize": 7.5,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(1, len(columns), figsize=(7.08, 1.70), layout="constrained")
        for ax, column, short_name in zip(axes, columns, short_names):
            ax.imshow(image, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if column == "Ground truth":
                add_box(ax, target, crop, "#00A65A", linestyle="-", linewidth=2.0)
                status = "GT: 457 x 601 px"
                title_color = "black"
            else:
                display = best_prediction_for_target(predictions[column][str(path)], target, confidence)
                raw = best_prediction_for_target(predictions[column][str(path)], target, 0.001)
                diagnostics[column] = {"display": display, "raw": raw}
                if display["box"] is not None:
                    add_box(ax, display["box"], crop, "#D62728", linewidth=1.8)
                    status = f"IoU {display['iou']:.2f}\nC {display['confidence']:.3f}"
                else:
                    if raw["box"] is not None:
                        add_box(ax, raw["box"], crop, "#E69F00", linestyle=":", linewidth=1.5)
                    status = f"miss @.25\nraw IoU {raw['iou']:.2f} | C {raw['confidence']:.3f}"
                title_color = "black"
            ax.set_title(short_name, color=title_color)
            ax.text(
                0.02,
                0.02,
                status,
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                fontsize=5.2,
                color=title_color,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
            )
        fig.suptitle(
            "Large-target score diagnostic (red solid >=0.25; orange dotted <0.25)",
            fontsize=8,
        )
        fig.savefig(output_dir / "large_target_threshold_diagnostic_seed1.png", dpi=400)
        fig.savefig(output_dir / "large_target_threshold_diagnostic_seed1.pdf")
        plt.close(fig)
    return {
        "path": str(path.relative_to(ROOT)),
        "image_size": [int(record["width"]), int(record["height"])],
        "gt_box": [float(value) for value in target],
        "gt_width_px": float(target[2] - target[0]),
        "gt_height_px": float(target[3] - target[1]),
        "gt_area_fraction": float(
            (target[2] - target[0])
            * (target[3] - target[1])
            / (float(record["width"]) * float(record["height"]))
        ),
        "equivalent_side_px": float(np.sqrt((target[2] - target[0]) * (target[3] - target[1]))),
        "equivalent_side_percentile_in_test": float(
            100.0
            * np.mean(
                [
                    np.sqrt((box[2] - box[0]) * (box[3] - box[1]))
                    <= np.sqrt((target[2] - target[0]) * (target[3] - target[1]))
                    for item in ground_truth.values()
                    for box in item["boxes"]
                ]
            )
        ),
        "models": diagnostics,
    }


def quantitative_figure(
    curves: dict[str, list[float]],
    metrics: dict[str, dict[str, object]],
    output_dir: Path,
) -> None:
    size_keys = [
        ("Tiny", "AP_tiny_lt16", "instances_tiny_lt16"),
        ("Small", "AP_small_16_32", "instances_small_16_32"),
        ("Medium", "AP_coco_medium_32_96", "instances_coco_medium_32_96"),
        ("Large", "AP_coco_large_ge96", "instances_coco_large_ge96"),
    ]
    with mpl.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(7.08, 5.15), layout="constrained")
        ax = axes[0, 0]
        for name, values in curves.items():
            ax.plot(
                IOU_THRESHOLDS,
                values,
                color=COLORS[name],
                marker=MARKERS[name],
                linewidth=1.6,
                markersize=3.7,
                label=name,
            )
        ax.set(xlabel="IoU threshold", ylabel="COCO-style AP (%)", title="a  Localization strictness")
        ax.set_xticks(IOU_THRESHOLDS[::2])
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        ax.legend(frameon=False)

        ax = axes[0, 1]
        x = np.arange(len(size_keys))
        width = 0.16
        for model_index, (name, model_metrics) in enumerate(metrics.items()):
            values = [100.0 * float(model_metrics[key]) for _, key, _ in size_keys]
            ax.bar(
                x + (model_index - (len(metrics) - 1) / 2) * width,
                values,
                width,
                color=COLORS[name],
                edgecolor="black",
                linewidth=0.35,
                label=name,
            )
        counts = [int(next(iter(metrics.values()))[count_key]) for _, _, count_key in size_keys]
        labels = [f"{label}\n(n={count})" for (label, _, _), count in zip(size_keys, counts)]
        ax.set_xticks(x, labels)
        ax.set(ylabel="COCO-style AP50:95 (%)", title="b  Target-size performance")
        ax.set_ylim(0, 75)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        ax.text(0.02, 0.98, "Tiny is exploratory (n=9)", transform=ax.transAxes, va="top", fontsize=6.5)

        ax = axes[1, 0]
        delta_metrics = [
            ("AP50:95", "AP"),
            ("AP50", "AP50"),
            ("AP75", "AP75"),
            ("Small", "AP_small_16_32"),
        ]
        effects = [
            ("P2 vs base", "+P2", "YOLO11n"),
            ("MSEF | Conv", "+P2+MSEF", "+P2"),
            ("ADown | no MSEF", "+P2+ADown", "+P2"),
            ("MSEF | ADown", "+P2+MSEF+ADown", "+P2+ADown"),
            ("ADown | MSEF", "+P2+MSEF+ADown", "+P2+MSEF"),
        ]
        effect_values = np.asarray(
            [
                [
                    100.0 * (float(metrics[child][key]) - float(metrics[parent][key]))
                    for _, key in delta_metrics
                ]
                for _, child, parent in effects
            ],
            dtype=float,
        )
        limit = max(0.5, float(np.max(np.abs(effect_values))))
        image = ax.imshow(
            effect_values,
            cmap="RdBu",
            norm=mpl.colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            aspect="auto",
        )
        for row in range(effect_values.shape[0]):
            for col in range(effect_values.shape[1]):
                value = effect_values[row, col]
                ax.text(
                    col,
                    row,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color="white" if abs(value) > 0.58 * limit else "black",
                )
        ax.set_xticks(np.arange(len(delta_metrics)), [label for label, _ in delta_metrics])
        ax.set_yticks(np.arange(len(effects)), [label for label, _, _ in effects])
        ax.set_title("c  Direct-parent module effects (points)")
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        colorbar.set_label("Change (points)", fontsize=7)
        colorbar.ax.tick_params(labelsize=6)

        ax = axes[1, 1]
        for name, model_metrics in metrics.items():
            x_value = COMPLEXITY[name]["parameters_m"]
            y_value = 100.0 * float(model_metrics["AP75"])
            size = 16.0 * COMPLEXITY[name]["gflops"]
            ax.scatter(
                x_value,
                y_value,
                s=size,
                color=COLORS[name],
                marker=MARKERS[name],
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )
            short_name = {
                "YOLO11n": "YOLO11n",
                "+P2": "+P2",
                "+P2+MSEF": "+MSEF",
                "+P2+ADown": "+ADown",
                "+P2+MSEF+ADown": "+MSEF+ADown",
            }[name]
            ax.annotate(
                f"{short_name}\n{COMPLEXITY[name]['gflops']:.1f} GFLOPs",
                (x_value, y_value),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=6.5,
            )
        ax.set(
            xlabel="Parameters (million)",
            ylabel="COCO-style AP75 (%)",
            title="d  Localization-efficiency trade-off",
        )
        parameter_values = [item["parameters_m"] for item in COMPLEXITY.values()]
        ap75_values = [100.0 * float(item["AP75"]) for item in metrics.values()]
        ax.set_xlim(min(parameter_values) - 0.10, max(parameter_values) + 0.14)
        ax.set_ylim(min(ap75_values) - 1.5, max(ap75_values) + 2.0)
        ax.grid(color="#DDDDDD", linewidth=0.6)
        ax.text(0.02, 0.02, "Marker area encodes GFLOPs", transform=ax.transAxes, fontsize=6.5)

        fig.savefig(output_dir / "quantitative_p2_msef_adown_seed1.png", dpi=400)
        fig.savefig(output_dir / "quantitative_p2_msef_adown_seed1.pdf")
        plt.close(fig)


def write_source_table(
    curves: dict[str, list[float]], metrics: dict[str, dict[str, object]], output_dir: Path
) -> None:
    path = output_dir / "figure_source_data.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "metric_family", "metric", "value_percent", "instances"])
        for name, values in curves.items():
            for threshold, value in zip(IOU_THRESHOLDS, values):
                writer.writerow([name, "iou_curve", f"AP@{threshold:.2f}", f"{value:.8f}", ""])
        size_keys = [
            ("tiny_lt16", "AP_tiny_lt16", "instances_tiny_lt16"),
            ("small_16_32", "AP_small_16_32", "instances_small_16_32"),
            ("medium_32_96", "AP_coco_medium_32_96", "instances_coco_medium_32_96"),
            ("large_ge96", "AP_coco_large_ge96", "instances_coco_large_ge96"),
        ]
        for name, model_metrics in metrics.items():
            for label, metric_key, count_key in size_keys:
                writer.writerow(
                    [
                        name,
                        "size",
                        label,
                        f"{100.0 * float(model_metrics[metric_key]):.8f}",
                        int(model_metrics[count_key]),
                    ]
                )
            writer.writerow([name, "complexity", "parameters_m", COMPLEXITY[name]["parameters_m"], ""])
            writer.writerow([name, "complexity", "gflops", COMPLEXITY[name]["gflops"], ""])


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = image_paths()
    ground_truth_records = load_ground_truth(paths)
    ground_truth, path_to_id = make_ground_truth(paths)

    cached_predictions: dict[str, dict[str, np.ndarray]] = {}
    raw_predictions_path = output_dir / "raw_predictions.json"
    if args.reuse_predictions:
        if not raw_predictions_path.is_file():
            raise FileNotFoundError(raw_predictions_path)
        serialized = json.loads(raw_predictions_path.read_text(encoding="utf-8"))
        missing = set(MODELS) - set(serialized)
        if missing:
            print(f"Cached predictions are missing {sorted(missing)}; only those models will be inferred.")
        cached_predictions = {
            name: {
                str((ROOT / path).resolve()): np.asarray(values, dtype=np.float64).reshape(-1, 5)
                for path, values in serialized[name].items()
            }
            for name in MODELS
            if name in serialized
        }

    all_predictions: dict[str, dict[str, np.ndarray]] = {}
    curves: dict[str, list[float]] = {}
    metrics: dict[str, dict[str, object]] = {}
    for name, weights in MODELS.items():
        if not weights.is_file():
            raise FileNotFoundError(weights)
        if name in cached_predictions:
            print(f"Reusing cached predictions for {name}", flush=True)
            model_predictions = cached_predictions[name]
        else:
            print(f"Inferring {name}: {weights}", flush=True)
            model_predictions = infer(weights, paths, args.device, args.batch)
        all_predictions[name] = model_predictions
        records = detection_records(model_predictions, path_to_id)
        curves[name], metrics[name] = evaluate_predictions(ground_truth, records)

    serializable_predictions = {
        name: {
            str(Path(path).relative_to(ROOT)): np.round(values, 6).tolist()
            for path, values in model_predictions.items()
        }
        for name, model_predictions in all_predictions.items()
    }
    (output_dir / "raw_predictions.json").write_text(
        json.dumps(serializable_predictions, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    targets = target_records(ground_truth_records, all_predictions, args.display_conf)
    adown_score_shift = {
        "without_msef": adown_score_shift_analysis(targets, "+P2", "+P2+ADown"),
        "with_msef": adown_score_shift_analysis(
            targets, "+P2+MSEF", "+P2+MSEF+ADown"
        ),
    }
    selected = select_examples(targets)
    selection_manifest = qualitative_figure(
        selected, ground_truth_records, all_predictions, args.display_conf, output_dir
    )
    large_target_diagnostic = large_target_diagnostic_figure(
        ground_truth_records, all_predictions, args.display_conf, output_dir
    )
    quantitative_figure(curves, metrics, output_dir)
    write_source_table(curves, metrics, output_dir)

    manifest = {
        "purpose": "seed-1 visual evidence for the P2, MSEFPaper, and full-path ADown chain",
        "split": "DetectDataset_clean_v2 combined test (public test + 16 self-video frames)",
        "images": len(paths),
        "instances": sum(len(record["boxes"]) for record in ground_truth_records.values()),
        "inference": {
            "imgsz": 640,
            "prediction_confidence_floor": 0.001,
            "nms_iou": 0.7,
            "max_det": 300,
            "display_confidence": args.display_conf,
        },
        "weights": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for name, path in MODELS.items()
        },
        "selection_rules": {
            "P2 improvement": "top unique image among targets <96 px with P2 IoU>=0.50 and largest positive P2-minus-baseline IoU at confidence>=0.25",
            "MSEF improvement": "top remaining unique image with P2+MSEF IoU>=0.75 and largest positive MSEF-minus-P2 IoU at confidence>=0.25",
            "ADown improvement": "top remaining unique image with P2+ADown IoU>=0.75 and largest positive P2+ADown-minus-P2 IoU at confidence>=0.25",
            "Combined improvement": "top remaining unique image with full-model IoU>=0.75 and largest positive full-minus-better-single-branch IoU at confidence>=0.25",
            "Counterexample": "remaining unique image with the most negative final-minus-P2+MSEF IoU while P2+MSEF IoU>=0.50 at confidence>=0.25",
            "manual_image_selection": False,
        },
        "selected_examples": selection_manifest,
        "user_flagged_large_target_diagnostic": large_target_diagnostic,
        "adown_score_shift_audit": adown_score_shift,
        "coco_style_metrics": metrics,
        "ap_by_iou_percent": curves,
        "complexity": COMPLEXITY,
        "figure_processing": "native RGB crops only; identical crop per model; nearest-neighbor display interpolation; no brightness, contrast, denoising, or selective image adjustment",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    def percent(model: str, key: str) -> float:
        return 100.0 * float(metrics[model][key])

    def delta(child: str, parent: str, key: str) -> float:
        return percent(child, key) - percent(parent, key)

    diagnostic_final = large_target_diagnostic["models"]["+P2+MSEF+ADown"]["raw"]
    diagnostic_adown = large_target_diagnostic["models"]["+P2+ADown"]["raw"]
    large_score_group = adown_score_shift["with_msef"]["grouped_by_equivalent_side_px"]["ge300"]
    caption = f"""# Figure notes

## Same-protocol results

Single-run seed-1 COCO-style results on 641 images / 701 instances:

| Model | AP50:95/% | AP50/% | AP75/% | AP 16-32 px/% | AP >=96 px/% | Params/M | GFLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n | {percent('YOLO11n', 'AP'):.2f} | {percent('YOLO11n', 'AP50'):.2f} | {percent('YOLO11n', 'AP75'):.2f} | {percent('YOLO11n', 'AP_small_16_32'):.2f} | {percent('YOLO11n', 'AP_coco_large_ge96'):.2f} | {COMPLEXITY['YOLO11n']['parameters_m']:.3f} | {COMPLEXITY['YOLO11n']['gflops']:.1f} |
| +P2 | {percent('+P2', 'AP'):.2f} | {percent('+P2', 'AP50'):.2f} | {percent('+P2', 'AP75'):.2f} | {percent('+P2', 'AP_small_16_32'):.2f} | {percent('+P2', 'AP_coco_large_ge96'):.2f} | {COMPLEXITY['+P2']['parameters_m']:.3f} | {COMPLEXITY['+P2']['gflops']:.1f} |
| +P2+MSEF | {percent('+P2+MSEF', 'AP'):.2f} | {percent('+P2+MSEF', 'AP50'):.2f} | {percent('+P2+MSEF', 'AP75'):.2f} | {percent('+P2+MSEF', 'AP_small_16_32'):.2f} | {percent('+P2+MSEF', 'AP_coco_large_ge96'):.2f} | {COMPLEXITY['+P2+MSEF']['parameters_m']:.3f} | {COMPLEXITY['+P2+MSEF']['gflops']:.1f} |
| +P2+ADown | {percent('+P2+ADown', 'AP'):.2f} | {percent('+P2+ADown', 'AP50'):.2f} | {percent('+P2+ADown', 'AP75'):.2f} | {percent('+P2+ADown', 'AP_small_16_32'):.2f} | {percent('+P2+ADown', 'AP_coco_large_ge96'):.2f} | {COMPLEXITY['+P2+ADown']['parameters_m']:.3f} | {COMPLEXITY['+P2+ADown']['gflops']:.1f} |
| +P2+MSEF+ADown | {percent('+P2+MSEF+ADown', 'AP'):.2f} | {percent('+P2+MSEF+ADown', 'AP50'):.2f} | {percent('+P2+MSEF+ADown', 'AP75'):.2f} | {percent('+P2+MSEF+ADown', 'AP_small_16_32'):.2f} | {percent('+P2+MSEF+ADown', 'AP_coco_large_ge96'):.2f} | {COMPLEXITY['+P2+MSEF+ADown']['parameters_m']:.3f} | {COMPLEXITY['+P2+MSEF+ADown']['gflops']:.1f} |

## Mechanism assessment

- **P2:** AP50 changes by {delta('+P2', 'YOLO11n', 'AP50'):+.2f} points and 16-32 px AP by {delta('+P2', 'YOLO11n', 'AP_small_16_32'):+.2f} points, while AP75 changes by {delta('+P2', 'YOLO11n', 'AP75'):+.2f}. This supports improved small-target/lenient-IoU detection, not uniformly better localization.
- **MSEF:** relative to P2, AP75 changes by {delta('+P2+MSEF', '+P2', 'AP75'):+.2f} points, but AP50 by {delta('+P2+MSEF', '+P2', 'AP50'):+.2f} and 16-32 px AP by {delta('+P2+MSEF', '+P2', 'AP_small_16_32'):+.2f}. Its clearest observed effect is stricter box localization, not a broad small-target gain.
- **ADown without MSEF:** relative to P2, parameters change by {(COMPLEXITY['+P2+ADown']['parameters_m'] / COMPLEXITY['+P2']['parameters_m'] - 1) * 100:+.1f}% and GFLOPs by {(COMPLEXITY['+P2+ADown']['gflops'] / COMPLEXITY['+P2']['gflops'] - 1) * 100:+.1f}%, while AP50:95 changes by {delta('+P2+ADown', '+P2', 'AP'):+.2f}, AP75 by {delta('+P2+ADown', '+P2', 'AP75'):+.2f}, and 16-32 px AP by {delta('+P2+ADown', '+P2', 'AP_small_16_32'):+.2f}.
- **ADown with MSEF:** relative to P2+MSEF, AP50:95 changes by {delta('+P2+MSEF+ADown', '+P2+MSEF', 'AP'):+.2f}, AP75 by {delta('+P2+MSEF+ADown', '+P2+MSEF', 'AP75'):+.2f}, and 16-32 px AP by {delta('+P2+MSEF+ADown', '+P2+MSEF', 'AP_small_16_32'):+.2f}. The difference between the two ADown effects is the MSEF x ADown interaction; non-additivity means the modules must not be credited independently from only the full model.
- **Combination check:** adding MSEF to P2+ADown changes AP50:95 by {delta('+P2+MSEF+ADown', '+P2+ADown', 'AP'):+.2f}, AP75 by {delta('+P2+MSEF+ADown', '+P2+ADown', 'AP75'):+.2f}, and 16-32 px AP by {delta('+P2+MSEF+ADown', '+P2+ADown', 'AP_small_16_32'):+.2f}, while increasing GFLOPs by {(COMPLEXITY['+P2+MSEF+ADown']['gflops'] / COMPLEXITY['+P2+ADown']['gflops'] - 1) * 100:+.1f}%. Under this protocol P2+ADown is the stronger compact candidate; the full combination is not justified as an additive improvement.

## Large-target threshold audit

The user-flagged `pic_1032` target is 457 x 600.5 px, occupies 67.0% of the image area, and lies at target-size percentile {large_target_diagnostic['equivalent_side_percentile_in_test']:.2f}. P2+ADown produces a geometrically correct candidate at IoU={diagnostic_adown['iou']:.3f}, confidence={diagnostic_adown['confidence']:.3f}; the full model produces IoU={diagnostic_final['iou']:.3f}, confidence={diagnostic_final['confidence']:.3f}. Both survive low-floor NMS but fall below the 0.25 decision threshold. The separate diagnostic figure shows these candidates as dotted boxes.

This is not evidence of a systematic large-target collapse: among {large_score_group['targets']} targets with equivalent side >=300 px, the median final-minus-parent confidence change is {large_score_group['median_final_minus_parent_confidence']:+.3f}; {large_score_group['lost_at_confidence_025']} parent detections cross below 0.25 and {large_score_group['gained_at_confidence_025']} cross above it. Large-target AP also changes by {delta('+P2+MSEF+ADown', '+P2+MSEF', 'AP_coco_large_ge96'):+.2f} points. The correct interpretation is an isolated score-calibration failure on an extreme-scale, unusual grayscale sample, not proof that ADown generally cannot detect large objects.

## Figure construction

The paper-style qualitative grid uses deterministic geometry-based selection rather than manual image picking: one P2 gain, one MSEF gain, one ADown gain, one combined-model gain, and one ADown regression. Ground truth is shown only in its own column with a green solid box; each model column uses red solid prediction boxes at confidence >=0.25. Low-score candidates are excluded from the main qualitative grid and shown only in the separately labelled diagnostic figure. Each row uses an identical native-pixel crop across models, with no brightness, contrast, denoising, or selective adjustment.

The quantitative figure reports AP-versus-IoU, size-stratified AP, direct-parent changes, and the parameter/AP75 trade-off. The <16 px subset is exploratory because it contains only nine instances. No uncertainty interval is shown because these figures use the requested single selected seed. All plotted values use the COCO-style evaluator consistently.

Alt text: Four-panel quantitative comparison and five-row qualitative crop grid compare seed-1 YOLO11n, YOLO11n+P2, YOLO11n+P2+MSEF, YOLO11n+P2+ADown, and the full combination. The main grid uses conventional separate solid ground-truth and prediction boxes. The figures expose both direct module effects and the non-additive MSEF-ADown interaction, retain an ADown regression, and separately diagnose the low-score but well-localized candidate for the user-flagged very large target.
"""
    (output_dir / "FIGURE_NOTES.md").write_text(caption, encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
