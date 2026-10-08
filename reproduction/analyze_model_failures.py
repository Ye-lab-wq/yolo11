#!/usr/bin/env python3
"""Size-stratified and split-leakage-aware failure analysis for submitted checkpoints."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
REPRODUCTION = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ultralytics" / "nn" / "modules"))

import dysample  # noqa: F401 - historical checkpoint import path
from CAA import CAA  # noqa: F401 - historical checkpoint import path
from ultralytics import YOLO
from ultralytics.utils.metrics import compute_ap


MODELS = {
    "YOLO11n": ROOT / "runs/detect/yolo11_100/weights/best.pt",
    "MEDA": ROOT / "runs/detect/meda_200/weights/best.pt",
}
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def size_group(box: np.ndarray) -> str:
    side = float(max((box[2] - box[0]) * (box[3] - box[1]), 0.0) ** 0.5)
    if side < 16:
        return "tiny"
    if side < 32:
        return "small"
    if side < 96:
        return "medium"
    return "large"


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float64)
    top_left = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = np.clip(bottom_right - top_left, 0, None).prod(2)
    area1 = np.clip(boxes1[:, 2:] - boxes1[:, :2], 0, None).prod(1)
    area2 = np.clip(boxes2[:, 2:] - boxes2[:, :2], 0, None).prod(1)
    return intersection / np.maximum(area1[:, None] + area2[None, :] - intersection, 1e-12)


def load_ground_truth(split: str) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    image_dir = ROOT / "DetectDataset" / split / "images"
    for path in sorted(p for p in image_dir.iterdir() if p.is_file()):
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
        boxes = []
        label_path = ROOT / "DetectDataset" / split / "labels" / f"{path.stem}.txt"
        for line in label_path.read_text(encoding="utf-8").splitlines():
            _, x, y, bw, bh = map(float, line.split())
            boxes.append(
                [
                    (x - bw / 2) * width,
                    (y - bh / 2) * height,
                    (x + bw / 2) * width,
                    (y + bh / 2) * height,
                ]
            )
        array = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        records[str(path.resolve())] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "boxes": array,
            "groups": np.asarray([size_group(box) for box in array], dtype=object),
        }
    return records


def infer(weights: Path, ground_truth: dict[str, dict[str, object]]) -> dict[str, np.ndarray]:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = YOLO(str(weights))
    paths = list(ground_truth)
    predictions: dict[str, np.ndarray] = {}
    batch_size = 4
    for start in range(0, len(paths), batch_size):
        results = model.predict(
            source=paths[start : start + batch_size],
            imgsz=640,
            conf=0.001,
            iou=0.7,
            max_det=300,
            batch=batch_size,
            device=0,
            workers=4,
            stream=False,
            verbose=False,
        )
        for result in results:
            path = str(Path(result.path).resolve())
            boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float64)
            confidence = result.boxes.conf.detach().cpu().numpy().astype(np.float64)[:, None]
            predictions[path] = np.concatenate((boxes, confidence), axis=1)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions


def evaluate_ap(
    ground_truth: dict[str, dict[str, object]],
    predictions: dict[str, np.ndarray],
    group: str = "all",
    selected_paths: set[str] | None = None,
) -> dict[str, float | int | None]:
    detections_by_iou: list[list[tuple[float, int, int]]] = [[] for _ in IOU_THRESHOLDS]
    positives = 0
    for path, record in ground_truth.items():
        if selected_paths is not None and path not in selected_paths:
            continue
        gt_boxes = record["boxes"]
        gt_groups = record["groups"]
        in_group = np.ones(len(gt_boxes), dtype=bool) if group == "all" else gt_groups == group
        positives += int(in_group.sum())
        pred = predictions.get(path, np.zeros((0, 5), dtype=np.float64))
        pred = pred[np.argsort(-pred[:, 4])]
        ious = box_iou(pred[:, :4], gt_boxes)
        pred_groups = np.asarray([size_group(box) for box in pred[:, :4]], dtype=object)
        for threshold_index, threshold in enumerate(IOU_THRESHOLDS):
            matched = np.zeros(len(gt_boxes), dtype=bool)
            for prediction_index, prediction in enumerate(pred):
                candidates = np.flatnonzero((ious[prediction_index] >= threshold) & ~matched & in_group)
                if len(candidates):
                    match = candidates[np.argmax(ious[prediction_index, candidates])]
                    matched[match] = True
                    detections_by_iou[threshold_index].append((float(prediction[4]), 1, 0))
                    continue
                ignored = np.flatnonzero((ious[prediction_index] >= threshold) & ~matched & ~in_group)
                if len(ignored):
                    match = ignored[np.argmax(ious[prediction_index, ignored])]
                    matched[match] = True
                    continue
                if group == "all" or pred_groups[prediction_index] == group:
                    detections_by_iou[threshold_index].append((float(prediction[4]), 0, 1))

    aps = []
    max_recalls = []
    for detections in detections_by_iou:
        detections.sort(key=lambda item: item[0], reverse=True)
        tp = np.cumsum([item[1] for item in detections])
        fp = np.cumsum([item[2] for item in detections])
        recall = tp / max(positives, 1)
        precision = tp / np.maximum(tp + fp, 1)
        aps.append(float(compute_ap(recall, precision)[0]) if positives else None)
        max_recalls.append(float(recall[-1]) if len(recall) and positives else None)
    valid_aps = [value for value in aps if value is not None]
    return {
        "ground_truth": positives,
        "AP50": aps[0],
        "AP75": aps[5],
        "AP50_95": float(np.mean(valid_aps)) if valid_aps else None,
        "max_recall_iou50": max_recalls[0],
        "max_recall_iou75": max_recalls[5],
    }


def fixed_threshold_errors(
    ground_truth: dict[str, dict[str, object]], predictions: dict[str, np.ndarray], confidence: float = 0.25
) -> dict[str, object]:
    matched_by_group = {threshold: defaultdict(int) for threshold in (0.5, 0.75)}
    total_by_group = defaultdict(int)
    misses: list[dict[str, object]] = []
    false_positives: list[dict[str, object]] = []
    for path, record in ground_truth.items():
        gt_boxes = record["boxes"]
        gt_groups = record["groups"]
        pred = predictions.get(path, np.zeros((0, 5), dtype=np.float64))
        pred = pred[pred[:, 4] >= confidence]
        pred = pred[np.argsort(-pred[:, 4])]
        ious = box_iou(pred[:, :4], gt_boxes)
        for group in gt_groups:
            total_by_group[str(group)] += 1
        for threshold in (0.5, 0.75):
            matched_gt: set[int] = set()
            for prediction_index in range(len(pred)):
                candidates = [
                    index for index in range(len(gt_boxes))
                    if index not in matched_gt and ious[prediction_index, index] >= threshold
                ]
                if candidates:
                    match = max(candidates, key=lambda index: ious[prediction_index, index])
                    matched_gt.add(match)
                    matched_by_group[threshold][str(gt_groups[match])] += 1
            if threshold == 0.5:
                for gt_index, (box, group) in enumerate(zip(gt_boxes, gt_groups)):
                    if gt_index not in matched_gt:
                        best_iou = float(ious[:, gt_index].max()) if len(pred) else 0.0
                        misses.append(
                            {
                                "path": str(Path(path).relative_to(ROOT)),
                                "group": str(group),
                                "box": [round(float(value), 2) for value in box],
                                "best_iou_at_conf025": best_iou,
                            }
                        )
                matched_predictions = set()
                matched_gt_fp: set[int] = set()
                for prediction_index in range(len(pred)):
                    candidates = [
                        index for index in range(len(gt_boxes))
                        if index not in matched_gt_fp and ious[prediction_index, index] >= 0.5
                    ]
                    if candidates:
                        match = max(candidates, key=lambda index: ious[prediction_index, index])
                        matched_gt_fp.add(match)
                        matched_predictions.add(prediction_index)
                for prediction_index, prediction in enumerate(pred):
                    if prediction_index not in matched_predictions:
                        best_iou = float(ious[prediction_index].max()) if len(gt_boxes) else 0.0
                        false_positives.append(
                            {
                                "path": str(Path(path).relative_to(ROOT)),
                                "confidence": float(prediction[4]),
                                "box": [round(float(value), 2) for value in prediction[:4]],
                                "best_iou_to_any_gt": best_iou,
                                "error_type": "localization" if best_iou >= 0.1 else "background_or_unlabeled",
                            }
                        )
    recall = {
        group: {
            "ground_truth": count,
            "recall_iou50_conf025": matched_by_group[0.5][group] / count,
            "recall_iou75_conf025": matched_by_group[0.75][group] / count,
        }
        for group, count in sorted(total_by_group.items())
    }
    misses.sort(key=lambda item: (item["best_iou_at_conf025"], item["group"]))
    false_positives.sort(key=lambda item: item["confidence"], reverse=True)
    return {
        "confidence": confidence,
        "recall_by_size": recall,
        "misses_iou50": len(misses),
        "false_positives_iou50": len(false_positives),
        "false_positive_error_types": dict(Counter(item["error_type"] for item in false_positives)),
        "top_misses": misses[:40],
        "top_false_positives": false_positives[:40],
    }


def main() -> None:
    args = parse_args()
    ground_truth = load_ground_truth(args.split)
    train_hashes = {
        sha256(path)
        for path in (ROOT / "DetectDataset/train/images").iterdir()
        if path.is_file()
    }
    duplicate_paths = {path for path, record in ground_truth.items() if record["sha256"] in train_hashes}
    clean_paths = set(ground_truth) - duplicate_paths
    report: dict[str, object] = {
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "imgsz": 640,
            "prediction_confidence_floor": 0.001,
            "nms_iou": 0.7,
            "inference_batch": 4,
        },
        "split": args.split,
        "images": len(ground_truth),
        "exact_train_duplicate_images": len(duplicate_paths),
        "nonduplicate_images": len(clean_paths),
        "area_definition": "tiny <16 px; small 16-32; medium 32-96; large >=96 by sqrt(box area)",
        "models": {},
    }
    output = REPRODUCTION / "results" / f"failure_analysis_{args.split}.json"
    if output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous.get("split") == args.split:
            report["models"].update(previous.get("models", {}))
    for model_name in args.models:
        print(f"Evaluating {model_name}", flush=True)
        predictions = infer(MODELS[model_name], ground_truth)
        model_report = {
            "weights": str(MODELS[model_name].relative_to(ROOT)),
            "all": evaluate_ap(ground_truth, predictions),
            "by_size": {
                group: evaluate_ap(ground_truth, predictions, group=group)
                for group in ("tiny", "small", "medium", "large")
            },
            "exact_train_duplicates": evaluate_ap(
                ground_truth, predictions, selected_paths=duplicate_paths
            ),
            "nonduplicates": evaluate_ap(ground_truth, predictions, selected_paths=clean_paths),
            "by_size_exact_train_duplicates": {
                group: evaluate_ap(
                    ground_truth, predictions, group=group, selected_paths=duplicate_paths
                )
                for group in ("tiny", "small", "medium", "large")
            },
            "by_size_nonduplicates": {
                group: evaluate_ap(
                    ground_truth, predictions, group=group, selected_paths=clean_paths
                )
                for group in ("tiny", "small", "medium", "large")
            },
            "fixed_threshold": fixed_threshold_errors(ground_truth, predictions),
        }
        report["models"][model_name] = model_report
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    model_name: {
                        "all": model_report["all"],
                        "by_size": model_report["by_size"],
                        "exact_train_duplicates": model_report["exact_train_duplicates"],
                        "nonduplicates": model_report["nonduplicates"],
                        "false_positive_error_types": model_report["fixed_threshold"][
                            "false_positive_error_types"
                        ],
                    }
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
