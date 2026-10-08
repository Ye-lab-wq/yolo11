#!/usr/bin/env python3
"""Build deterministic multi-seed qualitative evidence for motion-blur training."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from PIL import Image
from ultralytics import YOLO

from reproduction.ablation.evaluate_frozen import image_paths


OUTPUT = ROOT / "reproduction/results/motion_blur_visual_analysis_3seed"
BENCHMARK = ROOT / "DetectDataset_clean_v2_motion_blur"
METRICS = ROOT / "reproduction/results/motion_blur_training_3seed"
CONDITIONS = ("clean", "moderate", "strong")
MODELS = {
    0: OrderedDict(
        [
            ("P2_standard", ROOT / "runs/ablation_screen_v1/p2_nearest_ciou_s0/weights/best.pt"),
            ("P2_blur_train", ROOT / "runs/ablation_motion_blur_v1/p2_motion_blur_s0/weights/best.pt"),
            ("P2_MSEF_standard", ROOT / "runs/ablation_screen_v1/p2_msef_paper_ciou_s0/weights/best.pt"),
            ("P2_MSEF_blur_train", ROOT / "runs/ablation_motion_blur_v1/p2_msef_motion_blur_s0/weights/best.pt"),
        ]
    ),
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
MODEL_ORDER = tuple(MODELS[0])
DISPLAY_CONFIDENCE = 0.25
NMS_IOU = 0.70
GT_COLOR = "#009E73"
PRED_COLOR = "#D55E00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int, default=4)
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def paths_for(condition: str) -> list[Path]:
    directory = ROOT / "DetectDataset_clean_v2/test/images" if condition == "clean" else BENCHMARK / condition / "images"
    return image_paths((directory,))


def image_map(condition: str) -> dict[str, Path]:
    return {path.stem: path for path in paths_for(condition)}


def ground_truth(path: Path) -> list[list[float]]:
    with Image.open(path) as image:
        width, height = image.size
    label = path.parent.parent / "labels" / f"{path.stem}.txt"
    boxes = []
    for line in label.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) != 5:
            continue
        _, xc, yc, bw, bh = map(float, values)
        boxes.append([(xc - bw / 2) * width, (yc - bh / 2) * height, (xc + bw / 2) * width, (yc + bh / 2) * height])
    return boxes


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_counts(gt: list[list[float]], predictions: list[dict[str, object]], threshold: float) -> tuple[int, int, int]:
    candidates = []
    for gt_index, gt_box in enumerate(gt):
        for pred_index, prediction in enumerate(predictions):
            overlap = iou(gt_box, prediction["box"])
            if overlap >= threshold:
                candidates.append((overlap, gt_index, pred_index))
    used_gt, used_pred = set(), set()
    for _, gt_index, pred_index in sorted(candidates, reverse=True):
        if gt_index not in used_gt and pred_index not in used_pred:
            used_gt.add(gt_index)
            used_pred.add(pred_index)
    true_positive = len(used_gt)
    return true_positive, len(predictions) - true_positive, len(gt) - true_positive


def image_metrics(gt: list[list[float]], predictions: list[dict[str, object]]) -> dict[str, float | int]:
    tp50, fp50, fn50 = match_counts(gt, predictions, 0.50)
    tp75, fp75, fn75 = match_counts(gt, predictions, 0.75)
    best = [max((iou(box, prediction["box"]) for prediction in predictions), default=0.0) for box in gt]
    return {
        "gt": len(gt),
        "pred": len(predictions),
        "tp50": tp50,
        "fp50": fp50,
        "fn50": fn50,
        "tp75": tp75,
        "fp75": fp75,
        "fn75": fn75,
        "mean_best_iou": float(np.mean(best)) if best else 0.0,
    }


def collect_predictions(device: str, batch: int) -> dict[str, dict[str, list[dict[str, object]]]]:
    raw_path = OUTPUT / "raw_predictions.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.is_file() else {}
    total = len(MODELS) * len(MODEL_ORDER) * len(CONDITIONS)
    completed = len(raw)
    for seed, models in MODELS.items():
        for model_name, weight in models.items():
            pending = [condition for condition in CONDITIONS if f"{seed}|{model_name}|{condition}" not in raw]
            if not pending:
                continue
            model = YOLO(str(weight))
            for condition in pending:
                paths = paths_for(condition)
                key = f"{seed}|{model_name}|{condition}"
                print(f"qualitative inference {completed}/{total}: {key}", flush=True)
                records: dict[str, list[dict[str, object]]] = {}
                for start in range(0, len(paths), batch):
                    chunk = paths[start : start + batch]
                    results = model.predict(
                        source=[str(path) for path in chunk],
                        imgsz=640,
                        conf=DISPLAY_CONFIDENCE,
                        iou=NMS_IOU,
                        max_det=300,
                        batch=batch,
                        device=device,
                        workers=4,
                        stream=True,
                        verbose=False,
                    )
                    for result in results:
                        boxes = result.boxes.xyxy.detach().cpu().numpy()
                        scores = result.boxes.conf.detach().cpu().numpy()
                        records[Path(result.path).stem] = [
                            {"box": [float(value) for value in box], "confidence": float(score)}
                            for box, score in zip(boxes, scores)
                        ]
                raw[key] = records
                completed += 1
                write_json(raw_path, raw)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return raw


def prediction(raw: dict[str, object], seed: int, model: str, condition: str, stem: str) -> list[dict[str, object]]:
    return raw[f"{seed}|{model}|{condition}"].get(stem, [])


def build_per_image(raw: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for condition in CONDITIONS:
        paths = image_map(condition)
        for stem, path in paths.items():
            gt = ground_truth(path)
            for seed in MODELS:
                values = {model: image_metrics(gt, prediction(raw, seed, model, condition, stem)) for model in MODEL_ORDER}
                row: dict[str, object] = {"seed": seed, "condition": condition, "stem": stem, "path": str(path.resolve()), "gt": len(gt)}
                for model, metrics in values.items():
                    for key, value in metrics.items():
                        row[f"{model}_{key}"] = value
                base, final = values["P2_standard"], values["P2_MSEF_blur_train"]
                row["final_minus_base_tp50"] = int(final["tp50"]) - int(base["tp50"])
                row["final_minus_base_tp75"] = int(final["tp75"]) - int(base["tp75"])
                row["final_minus_base_mean_best_iou"] = float(final["mean_best_iou"]) - float(base["mean_best_iou"])
                rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def choose_examples(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["condition"]), str(row["stem"])), []).append(row)

    candidates: dict[str, list[dict[str, object]]] = {condition: [] for condition in CONDITIONS}
    for (condition, stem), subset in grouped.items():
        gain_votes = sum(int(row["final_minus_base_tp50"]) > 0 or int(row["final_minus_base_tp75"]) > 0 for row in subset)
        loss_votes = sum(int(row["final_minus_base_tp50"]) < 0 or int(row["final_minus_base_tp75"]) < 0 for row in subset)
        mean_tp50 = float(np.mean([int(row["final_minus_base_tp50"]) for row in subset]))
        mean_tp75 = float(np.mean([int(row["final_minus_base_tp75"]) for row in subset]))
        mean_iou = float(np.mean([float(row["final_minus_base_mean_best_iou"]) for row in subset]))
        candidates[condition].append(
            {
                "condition": condition,
                "stem": stem,
                "gain_votes": gain_votes,
                "loss_votes": loss_votes,
                "mean_delta_tp50": mean_tp50,
                "mean_delta_tp75": mean_tp75,
                "mean_delta_iou": mean_iou,
                "gain_score": gain_votes * 10 + mean_tp75 * 4 + mean_tp50 * 2 + mean_iou,
                "loss_score": loss_votes * 10 - mean_tp75 * 4 - mean_tp50 * 2 - mean_iou,
            }
        )

    blurred = []
    used = set()
    for condition in ("strong", "moderate"):
        eligible = [item for item in candidates[condition] if item["gain_votes"] >= 2 and item["stem"] not in used]
        for item in sorted(eligible, key=lambda value: (value["gain_score"], value["mean_delta_iou"]), reverse=True)[:2]:
            blurred.append(item)
            used.add(item["stem"])
    if len(blurred) < 4:
        fallback = [item for condition in ("strong", "moderate") for item in candidates[condition] if item["stem"] not in used and item["gain_votes"] >= 1]
        for item in sorted(fallback, key=lambda value: value["gain_score"], reverse=True)[: 4 - len(blurred)]:
            blurred.append(item)
            used.add(item["stem"])

    clear = [item for item in candidates["clean"] if item["loss_votes"] >= 2]
    clear = sorted(clear, key=lambda value: (value["loss_score"], -value["mean_delta_iou"]), reverse=True)[:4]
    if len(clear) < 4:
        used_clear = {item["stem"] for item in clear}
        fallback = [item for item in candidates["clean"] if item["stem"] not in used_clear and item["loss_votes"] >= 1]
        clear.extend(sorted(fallback, key=lambda value: value["loss_score"], reverse=True)[: 4 - len(clear)])

    summary = {}
    for condition in CONDITIONS:
        items = candidates[condition]
        metric_consensus = {}
        condition_subsets = [subset for (item_condition, _), subset in grouped.items() if item_condition == condition]
        for metric in ("tp50", "tp75", "mean_best_iou"):
            gain = loss = 0
            for subset in condition_subsets:
                differences = [
                    float(row[f"P2_MSEF_blur_train_{metric}"]) - float(row[f"P2_standard_{metric}"])
                    for row in subset
                ]
                gain += sum(value > 1e-12 for value in differences) >= 2
                loss += sum(value < -1e-12 for value in differences) >= 2
            metric_consensus[metric] = {"gain_images": gain, "loss_images": loss}
        summary[condition] = {
            "images": len(items),
            "consensus_gain_images": sum(item["gain_votes"] >= 2 for item in items),
            "consensus_loss_images": sum(item["loss_votes"] >= 2 for item in items),
            "any_seed_gain_images": sum(item["gain_votes"] >= 1 for item in items),
            "any_seed_loss_images": sum(item["loss_votes"] >= 1 for item in items),
            "metric_consensus": metric_consensus,
        }
    return blurred, clear, summary


def crop_box(path: Path, gt: list[list[float]]) -> tuple[int, int, int, int]:
    with Image.open(path) as image:
        width, height = image.size
    if not gt:
        return 0, 0, width, height
    x1 = min(box[0] for box in gt)
    y1 = min(box[1] for box in gt)
    x2 = max(box[2] for box in gt)
    y2 = max(box[3] for box in gt)
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(180.0, 4.0 * max(x2 - x1, y2 - y1))
    side = min(side, float(max(width, height)))
    left = max(0.0, min(center_x - side / 2, width - side))
    top = max(0.0, min(center_y - side / 2, height - side))
    return int(left), int(top), int(min(width, left + side)), int(min(height, top + side))


def clear_focus_box(
    raw: dict[str, object], stem: str, gt: list[list[float]]
) -> list[float]:
    """Choose the GT target with the largest mean localization regression."""
    if len(gt) == 1:
        return gt[0]
    losses = []
    for gt_box in gt:
        per_seed = []
        for seed in MODELS:
            standard = prediction(raw, seed, "P2_standard", "clean", stem)
            final = prediction(raw, seed, "P2_MSEF_blur_train", "clean", stem)
            standard_iou = max((iou(gt_box, item["box"]) for item in standard), default=0.0)
            final_iou = max((iou(gt_box, item["box"]) for item in final), default=0.0)
            per_seed.append(standard_iou - final_iou)
        losses.append(float(np.mean(per_seed)))
    return gt[int(np.argmax(losses))]


def draw_boxes(ax: plt.Axes, boxes: list[list[float]], crop: tuple[int, int, int, int], color: str, scores: list[float] | None = None) -> None:
    left, top, right, bottom = crop
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        if x2 <= left or x1 >= right or y2 <= top or y1 >= bottom:
            continue
        ax.add_patch(Rectangle((x1 - left, y1 - top), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2.4))
        label = "GT" if scores is None else f"{scores[index]:.2f}"
        ax.text(
            max(2, x1 - left + 2),
            max(2, y1 - top + 2),
            label,
            color="white",
            fontsize=7,
            va="top",
            bbox={"facecolor": color, "edgecolor": "none", "pad": 1.2, "alpha": 0.92},
            clip_on=True,
        )


def show_panel(ax: plt.Axes, path: Path, crop: tuple[int, int, int, int], boxes: list[list[float]], color: str, scores: list[float] | None, subtitle: str) -> None:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB").crop(crop))
    ax.imshow(array)
    draw_boxes(ax, boxes, crop, color, scores)
    ax.text(0.02, 0.02, subtitle, transform=ax.transAxes, fontsize=7, color="white", va="bottom", bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none", "pad": 1.5})
    ax.set_axis_off()


def prediction_panel(ax: plt.Axes, raw: dict[str, object], seed: int, model: str, condition: str, stem: str, path: Path, crop: tuple[int, int, int, int], gt: list[list[float]]) -> None:
    values = prediction(raw, seed, model, condition, stem)
    metrics = image_metrics(gt, values)
    show_panel(
        ax,
        path,
        crop,
        [item["box"] for item in values],
        PRED_COLOR,
        [float(item["confidence"]) for item in values],
        f"TP50 {metrics['tp50']}/{metrics['gt']} · TP75 {metrics['tp75']}/{metrics['gt']} · mIoU {metrics['mean_best_iou']:.2f}",
    )


def qualitative_grid(raw: dict[str, object], examples: list[dict[str, object]], blurred: bool, output_name: str) -> None:
    maps = {condition: image_map(condition) for condition in CONDITIONS}
    columns = 8 if blurred else 7
    titles = (["Clean + GT", "Blurred + GT"] if blurred else ["Clean + GT"]) + [f"S{seed} P2 std" for seed in MODELS for _ in (0,)]
    # Interleave standard and final within each seed.
    titles = (["Clean + GT", "Blurred + GT"] if blurred else ["Clean + GT"]) + [title for seed in MODELS for title in (f"S{seed} P2 std", f"S{seed} Blur+MSEF")]
    with mpl.rc_context({"font.size": 8, "axes.titlesize": 9, "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(len(examples), columns, figsize=(18, 3.0 * len(examples)), squeeze=False, layout="constrained")
        for column, title in enumerate(titles):
            axes[0, column].set_title(title, fontweight="bold")
        for row_index, example in enumerate(examples):
            condition = str(example["condition"])
            stem = str(example["stem"])
            path = maps[condition][stem]
            clean_path = maps["clean"][stem]
            gt = ground_truth(path)
            crop_targets = gt if blurred else [clear_focus_box(raw, stem, gt)]
            crop = crop_box(path, crop_targets)
            show_panel(axes[row_index, 0], clean_path, crop, gt, GT_COLOR, None, f"{condition} · {stem[:24]}")
            offset = 1
            if blurred:
                show_panel(axes[row_index, 1], path, crop, gt, GT_COLOR, None, f"votes +{example['gain_votes']}/3")
                offset = 2
            for seed_index, seed in enumerate(MODELS):
                prediction_panel(axes[row_index, offset + seed_index * 2], raw, seed, "P2_standard", condition, stem, path, crop, gt)
                prediction_panel(axes[row_index, offset + seed_index * 2 + 1], raw, seed, "P2_MSEF_blur_train", condition, stem, path, crop, gt)
        fig.suptitle(
            ("Consensus improvements under deterministic motion blur" if blurred else "Clear-image counterexamples: standard P2 versus blur-trained P2+MSEF")
            + f"\nSolid green = ground truth; solid red = prediction; confidence ≥ {DISPLAY_CONFIDENCE:.2f}; identical crop within each row",
            fontsize=13,
            fontweight="bold",
        )
        fig.savefig(OUTPUT / f"{output_name}.png", dpi=300)
        fig.savefig(OUTPUT / f"{output_name}.pdf")
        plt.close(fig)


def quantitative_figure() -> None:
    rows = list(csv.DictReader((METRICS / "seed_results.csv").open(encoding="utf-8")))
    conditions = ("clean", "light", "moderate", "strong")
    labels = ("Clean", "Light", "Moderate", "Strong")
    colors = {"P2_standard": "#0072B2", "P2_blur_train": "#56B4E9", "P2_MSEF_standard": "#D55E00", "P2_MSEF_blur_train": "#009E73"}
    names = {"P2_standard": "P2 standard", "P2_blur_train": "P2 + blur train", "P2_MSEF_standard": "P2 + MSEF", "P2_MSEF_blur_train": "P2 + MSEF + blur train"}
    by_key = {(int(row["seed"]), row["model"], row["condition"]): float(row["AP50_95_percent"]) for row in rows}
    effects = list(csv.DictReader((METRICS / "effects_by_seed.csv").open(encoding="utf-8")))
    with mpl.rc_context({"font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11, "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), layout="constrained")
        x = np.arange(len(conditions))
        for model in MODEL_ORDER:
            matrix = np.asarray([[by_key[(seed, model, condition)] for condition in conditions] for seed in (0, 1, 2)])
            for values in matrix:
                axes[0].plot(x, values, color=colors[model], alpha=0.18, linewidth=1)
                axes[0].scatter(x, values, color=colors[model], alpha=0.35, s=13)
            axes[0].errorbar(x, matrix.mean(axis=0), yerr=matrix.std(axis=0, ddof=1), color=colors[model], marker="o", linewidth=2.0, capsize=3, label=names[model])
        axes[0].set_xticks(x, labels)
        axes[0].set_ylabel("COCO AP50–95 (%)")
        axes[0].set_title("(a) Robustness curve (mean ± sample SD, n=3 seeds)")
        axes[0].grid(axis="y", alpha=0.25)
        axes[0].legend(frameon=False, loc="lower left")

        effect_specs = (
            ("AP50_95_blur_effect_P2_pp", "Blur training on P2", "#0072B2", "o"),
            ("AP50_95_MSEF_effect_blur_train_pp", "MSEF after blur training", "#009E73", "s"),
            ("AP50_95_interaction_pp", "Interaction", "#CC79A7", "^"),
        )
        offsets = (-0.18, 0.0, 0.18)
        for offset, (key, label, color, marker) in zip(offsets, effect_specs):
            matrix = np.asarray([[float(next(row for row in effects if int(row["seed"]) == seed and row["condition"] == condition)[key]) for condition in conditions] for seed in (0, 1, 2)])
            for condition_index in range(len(conditions)):
                axes[1].scatter(np.full(3, x[condition_index] + offset), matrix[:, condition_index], color=color, marker=marker, s=22, alpha=0.45)
            axes[1].errorbar(x + offset, matrix.mean(axis=0), yerr=matrix.std(axis=0, ddof=1), color=color, marker=marker, linewidth=1.8, capsize=3, label=label)
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_xticks(x, labels)
        axes[1].set_ylabel("AP50–95 change (percentage points)")
        axes[1].set_title("(b) Paired effects; dots are individual seeds")
        axes[1].grid(axis="y", alpha=0.25)
        axes[1].legend(frameon=False, loc="upper left")
        fig.savefig(OUTPUT / "motion_blur_quantitative_3seed.png", dpi=400)
        fig.savefig(OUTPUT / "motion_blur_quantitative_3seed.pdf")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    device = "0" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for models in MODELS.values():
        for path in models.values():
            if not path.is_file():
                raise FileNotFoundError(path)
    raw = collect_predictions(device, args.batch)
    rows = build_per_image(raw)
    write_csv(OUTPUT / "per_image_analysis.csv", rows)
    blurred, clear, summary = choose_examples(rows)
    if len(blurred) < 4 or len(clear) < 4:
        raise RuntimeError(f"insufficient deterministic examples: blurred={len(blurred)}, clear={len(clear)}")
    write_json(
        OUTPUT / "selection_manifest.json",
        {
            "manual_image_selection": False,
            "display_confidence": DISPLAY_CONFIDENCE,
            "nms_iou": NMS_IOU,
            "selection_rule": "rank images by paired P2+MSEF+blur-training minus standard-P2 TP50/TP75/IoU change; require same direction in >=2 of 3 seeds before one-seed fallback",
            "blurred_examples": blurred,
            "clear_regression_examples": clear,
            "population_summary": summary,
        },
    )
    qualitative_grid(raw, blurred, True, "qualitative_blur_consensus_3seed")
    qualitative_grid(raw, clear, False, "qualitative_clear_regressions_3seed")
    quantitative_figure()
    notes = f"""# Motion-blur visual analysis

The figures use deterministic geometry-based selection, not manual image picking. A consensus example requires the paired direction to occur in at least two of three seeds. All model panels use the same native-pixel crop, confidence threshold {DISPLAY_CONFIDENCE:.2f}, and NMS IoU {NMS_IOU:.2f}. Ground truth is a solid green box in dedicated columns; predictions are solid vermillion boxes. No brightness, contrast, sharpening, denoising, or selective image adjustment was applied.

## Population counts

| Condition | Images | Consensus gain | Consensus loss | Any-seed gain | Any-seed loss |
|---|---:|---:|---:|---:|---:|
"""
    for condition in CONDITIONS:
        item = summary[condition]
        notes += f"| {condition} | {item['images']} | {item['consensus_gain_images']} | {item['consensus_loss_images']} | {item['any_seed_gain_images']} | {item['any_seed_loss_images']} |\n"
    notes += """

The broad gain/loss columns above are not mutually exclusive for multi-object images: one target can improve while another degrades. The threshold-specific consensus counts below are easier to interpret.

| Condition | TP50 gain/loss | TP75 gain/loss | mean-best-IoU gain/loss |
|---|---:|---:|---:|
"""
    for condition in CONDITIONS:
        metric = summary[condition]["metric_consensus"]
        notes += (
            f"| {condition} | {metric['tp50']['gain_images']}/{metric['tp50']['loss_images']} "
            f"| {metric['tp75']['gain_images']}/{metric['tp75']['loss_images']} "
            f"| {metric['mean_best_iou']['gain_images']}/{metric['mean_best_iou']['loss_images']} |\n"
        )
    notes += """

The qualitative examples demonstrate mechanisms and counterexamples, not prevalence; prevalence is represented by the complete per-image CSV and the aggregate AP figure. Because the corruption is synthetic and derived from the public test images, these figures cannot establish real-world motion-blur generalization and must not be used for additional model selection.

Alt text: The quantitative two-panel figure shows three-seed AP50–95 curves across clean, light, moderate, and strong blur and paired effects for blur training, MSEF after blur training, and their interaction. The qualitative blur grid shows clean and blurred ground truth beside standard-P2 and blur-trained P2+MSEF predictions for all three seeds. The clear-regression grid shows counterexamples where standard P2 more consistently detects or localizes a clean-image target than the blur-trained P2+MSEF model.
"""
    (OUTPUT / "FIGURE_NOTES.md").write_text(notes, encoding="utf-8")
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
