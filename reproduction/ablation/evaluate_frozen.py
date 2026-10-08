#!/usr/bin/env python3
"""Evaluate frozen checkpoints with Ultralytics and COCO-style size metrics.

This script is intentionally separate from ``run_screen.py``. It must only be used
after validation-only model selection has been frozen.
"""

from __future__ import annotations

import gc
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


DATASET = ROOT / "DetectDataset_clean_v2"
SPLITS = {
    "val": (DATASET / "data.yaml", "val", (DATASET / "valid" / "images",)),
    "combined_test": (
        DATASET / "data.yaml",
        "test",
        (DATASET / "test" / "images", DATASET / "self_test" / "images"),
    ),
    "public_test": (DATASET / "data_public_test.yaml", "test", (DATASET / "test" / "images",)),
    "self_test": (DATASET / "data_self_test.yaml", "test", (DATASET / "self_test" / "images",)),
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--split", choices=tuple(SPLITS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    return parser.parse_args()


def parse_weights(values: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, received {value!r}")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        parsed.append((name, path))
    return parsed


def image_paths(directories: tuple[Path, ...]) -> list[Path]:
    return sorted(
        path.resolve()
        for directory in directories
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def make_ground_truth(paths: list[Path]) -> tuple[COCO, dict[str, int]]:
    dataset: dict[str, object] = {
        "info": {"description": "DetectDataset_clean_v2"},
        "licenses": [],
        "categories": [{"id": 1, "name": "drone"}],
        "images": [],
        "annotations": [],
    }
    path_to_id: dict[str, int] = {}
    annotation_id = 1
    for image_id, path in enumerate(paths, start=1):
        with Image.open(path) as image:
            width, height = image.size
        path_to_id[str(path)] = image_id
        dataset["images"].append(
            {"id": image_id, "file_name": path.name, "width": width, "height": height}
        )
        label_path = path.parent.parent / "labels" / f"{path.stem}.txt"
        lines = label_path.read_text(encoding="utf-8").splitlines() if label_path.is_file() else []
        for line in lines:
            values = line.split()
            if len(values) != 5:
                raise ValueError(f"Invalid YOLO label in {label_path}: {line!r}")
            class_id, x_center, y_center, box_width, box_height = map(float, values)
            if int(class_id) != 0:
                raise ValueError(f"Unexpected class {class_id} in {label_path}")
            x = (x_center - box_width / 2) * width
            y = (y_center - box_height / 2) * height
            w = box_width * width
            h = box_height * height
            dataset["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    ground_truth = COCO()
    ground_truth.dataset = dataset
    ground_truth.createIndex()
    return ground_truth, path_to_id


def predictions(model: YOLO, paths: list[Path], path_to_id: dict[str, int], device: str, batch: int) -> list[dict]:
    records: list[dict] = []
    # A Python list of images is treated by Ultralytics as one in-memory batch;
    # its ``batch`` argument does not split that list. Chunk explicitly so P2
    # feature maps stay inside the declared evaluation memory budget.
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
            path = str(Path(result.path).resolve())
            image_id = path_to_id[path]
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            for box, score in zip(xyxy, scores):
                x1, y1, x2, y2 = map(float, box)
                records.append(
                    {
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    }
                )
    return records


def mean_precision(evaluator: COCOeval, area_index: int, iou_index: int | None = None) -> float | None:
    # precision dimensions: IoU, recall, category, area range, max detections.
    values = evaluator.eval["precision"][:, :, :, area_index, -1]
    if iou_index is not None:
        values = values[iou_index : iou_index + 1]
    values = values[values > -1]
    return float(values.mean()) if values.size else None


def coco_metrics(ground_truth: COCO, detection_records: list[dict]) -> dict[str, float | int | None]:
    areas = np.asarray([annotation["area"] for annotation in ground_truth.dataset["annotations"]], dtype=float)
    counts = {
        "instances_tiny_lt16": int((areas < 16.0**2).sum()),
        "instances_small_16_32": int(((areas >= 16.0**2) & (areas < 32.0**2)).sum()),
        "instances_coco_small_lt32": int((areas < 32.0**2).sum()),
        "instances_coco_medium_32_96": int(((areas >= 32.0**2) & (areas < 96.0**2)).sum()),
        "instances_coco_large_ge96": int((areas >= 96.0**2).sum()),
    }
    if not detection_records:
        return {
            "images": len(ground_truth.getImgIds()),
            "instances": len(ground_truth.getAnnIds()),
            **counts,
            **{
                key: 0.0
                for key in (
                    "AP",
                    "AP50",
                    "AP75",
                    "AP_tiny_lt16",
                    "AP_small_16_32",
                    "AP_coco_small_lt32",
                    "AP_coco_medium_32_96",
                    "AP_coco_large_ge96",
                )
            },
        }
    detections = ground_truth.loadRes(detection_records)

    standard = COCOeval(ground_truth, detections, "bbox")
    standard.params.imgIds = sorted(ground_truth.getImgIds())
    standard.params.maxDets = [1, 10, 300]
    standard.evaluate()
    standard.accumulate()

    custom = COCOeval(ground_truth, detections, "bbox")
    custom.params.imgIds = sorted(ground_truth.getImgIds())
    custom.params.maxDets = [1, 10, 300]
    custom.params.areaRng = [[0.0, 16.0**2], [16.0**2, 32.0**2]]
    custom.params.areaRngLbl = ["tiny", "small_only"]
    custom.evaluate()
    custom.accumulate()
    return {
        "images": len(ground_truth.getImgIds()),
        "instances": len(ground_truth.getAnnIds()),
        **counts,
        "AP": mean_precision(standard, 0),
        "AP50": mean_precision(standard, 0, 0),
        "AP75": mean_precision(standard, 0, 5),
        "AP_tiny_lt16": mean_precision(custom, 0),
        "AP_small_16_32": mean_precision(custom, 1),
        "AP_coco_small_lt32": mean_precision(standard, 1),
        "AP_coco_medium_32_96": mean_precision(standard, 2),
        "AP_coco_large_ge96": mean_precision(standard, 3),
    }


def main() -> None:
    args = parse_args()
    weights = parse_weights(args.weights)
    data_yaml, ultralytics_split, directories = SPLITS[args.split]
    paths = image_paths(directories)
    ground_truth, path_to_id = make_ground_truth(paths)
    report: dict[str, object] = {
        "split": args.split,
        "data": str(data_yaml),
        "selection_status": "frozen_before_test" if "test" in args.split else "validation_only",
        "imgsz": 640,
        "batch": args.batch,
        "device": args.device,
        "confidence_floor": 0.001,
        "nms_iou": 0.7,
        "max_det": 300,
        "size_ranges": {
            "tiny": "area < 16^2 pixels",
            "small_only": "16^2 <= area < 32^2 pixels",
            "coco_small": "area < 32^2 pixels",
            "coco_medium": "32^2 <= area < 96^2 pixels",
            "coco_large": "area >= 96^2 pixels",
        },
        "models": {},
    }
    for name, path in weights:
        print(f"Evaluating {name}: {path}", flush=True)
        model = YOLO(str(path))
        validation = model.val(
            data=str(data_yaml),
            split=ultralytics_split,
            imgsz=640,
            batch=args.batch,
            device=args.device,
            workers=4,
            conf=0.001,
            iou=0.7,
            max_det=300,
            plots=False,
            save_json=False,
            verbose=False,
            project=str(args.output.parent / "ultralytics"),
            name=f"{name}_{args.split}",
            exist_ok=True,
        )
        ultralytics_metrics = {
            "precision": float(validation.box.mp),
            "recall": float(validation.box.mr),
            "mAP50": float(validation.box.map50),
            "mAP75": float(validation.box.map75),
            "mAP50_95": float(validation.box.map),
        }
        # The following COCO-style prediction pass is intentionally separate.
        # Release validation tensors first so high-resolution P2 features do not
        # overlap in GPU memory with the prediction batch.
        del validation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        records = predictions(model, paths, path_to_id, args.device, args.batch)
        report["models"][name] = {
            "weights": str(path),
            "ultralytics": ultralytics_metrics,
            "coco_style": coco_metrics(ground_truth, records),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
