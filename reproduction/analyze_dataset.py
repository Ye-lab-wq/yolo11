#!/usr/bin/env python3
"""Audit dataset geometry, split integrity, and video-frame leakage risks."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "DetectDataset"
OUTPUT = Path(__file__).resolve().parent / "results" / "dataset_audit.json"
SPLITS = ("train", "valid", "test")
RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$")
VIDEO_FRAME = re.compile(r"^(video\d+)_(\d+)(?:_|$)", re.IGNORECASE)


def image_files(split: str) -> list[Path]:
    return sorted(p for p in (DATASET / split / "images").iterdir() if p.is_file())


def base_name(path: Path) -> str:
    return RF_SUFFIX.sub("", path.stem)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("min", "p05", "p25", "p50", "p75", "p95", "max")}
    array = np.asarray(values, dtype=np.float64)
    keys = ("min", "p05", "p25", "p50", "p75", "p95", "max")
    points = (0, 5, 25, 50, 75, 95, 100)
    return {key: float(np.percentile(array, point)) for key, point in zip(keys, points)}


def main() -> None:
    report: dict[str, object] = {
        "dataset": str(DATASET),
        "network_input": 640,
        "size_definition": {
            "lt8x8": "both resized box width and height < 8 px",
            "tiny": "sqrt(resized box area) < 16 px",
            "small": "16 <= sqrt(area) < 32 px",
            "medium": "32 <= sqrt(area) < 96 px",
            "large": "sqrt(area) >= 96 px",
        },
        "splits": {},
    }
    all_images: list[dict[str, object]] = []
    all_boxes: list[dict[str, object]] = []
    invalid_labels: list[dict[str, object]] = []

    for split in SPLITS:
        files = image_files(split)
        split_images: list[dict[str, object]] = []
        split_boxes: list[dict[str, object]] = []
        background_images = 0
        missing_labels = 0
        for path in files:
            with Image.open(path) as image:
                width, height = image.size
            scale = min(640 / width, 640 / height)
            label_path = DATASET / split / "labels" / f"{path.stem}.txt"
            lines = label_path.read_text(encoding="utf-8").splitlines() if label_path.exists() else []
            if not label_path.exists():
                missing_labels += 1
            if not lines:
                background_images += 1
            base = base_name(path)
            video_match = VIDEO_FRAME.match(base)
            image_row = {
                "split": split,
                "path": str(path.relative_to(ROOT)),
                "width": width,
                "height": height,
                "base_name": base,
                "sha256": sha256(path),
                "label_sha256": hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest(),
                "video": video_match.group(1).lower() if video_match else None,
                "frame": int(video_match.group(2)) if video_match else None,
                "instances": len(lines),
            }
            split_images.append(image_row)
            all_images.append(image_row)
            for line_number, line in enumerate(lines, 1):
                fields = line.split()
                if len(fields) != 5:
                    invalid_labels.append({"path": str(label_path), "line": line_number, "reason": "field_count"})
                    continue
                try:
                    cls, x, y, bw, bh = map(float, fields)
                except ValueError:
                    invalid_labels.append({"path": str(label_path), "line": line_number, "reason": "non_numeric"})
                    continue
                valid = cls == 0 and bw > 0 and bh > 0 and 0 <= x <= 1 and 0 <= y <= 1
                valid = valid and x - bw / 2 >= 0 and y - bh / 2 >= 0 and x + bw / 2 <= 1 and y + bh / 2 <= 1
                if not valid:
                    invalid_labels.append({"path": str(label_path), "line": line_number, "reason": "bounds_or_class"})
                box_width = bw * width * scale
                box_height = bh * height * scale
                area = box_width * box_height
                equivalent_side = area**0.5
                if equivalent_side < 16:
                    size_group = "tiny"
                elif equivalent_side < 32:
                    size_group = "small"
                elif equivalent_side < 96:
                    size_group = "medium"
                else:
                    size_group = "large"
                box_row = {
                    "split": split,
                    "path": image_row["path"],
                    "width_640": box_width,
                    "height_640": box_height,
                    "area_640": area,
                    "equivalent_side_640": equivalent_side,
                    "relative_area": bw * bh,
                    "size_group": size_group,
                    "lt8x8": box_width < 8 and box_height < 8,
                    "touches_border_1px": min(
                        (x - bw / 2) * width * scale,
                        (y - bh / 2) * height * scale,
                        (1 - x - bw / 2) * width * scale,
                        (1 - y - bh / 2) * height * scale,
                    ) <= 1,
                }
                split_boxes.append(box_row)
                all_boxes.append(box_row)

        size_counts = Counter(row["size_group"] for row in split_boxes)
        report["splits"][split] = {
            "images": len(split_images),
            "instances": len(split_boxes),
            "background_images": background_images,
            "missing_label_files": missing_labels,
            "image_dimensions": Counter(f"{row['width']}x{row['height']}" for row in split_images),
            "instances_per_image": percentiles([row["instances"] for row in split_images]),
            "box_width_640": percentiles([row["width_640"] for row in split_boxes]),
            "box_height_640": percentiles([row["height_640"] for row in split_boxes]),
            "equivalent_side_640": percentiles([row["equivalent_side_640"] for row in split_boxes]),
            "relative_area": percentiles([row["relative_area"] for row in split_boxes]),
            "size_counts": dict(size_counts),
            "size_fractions": {key: value / len(split_boxes) for key, value in size_counts.items()},
            "lt8x8_count": sum(row["lt8x8"] for row in split_boxes),
            "lt8x8_fraction": sum(row["lt8x8"] for row in split_boxes) / len(split_boxes),
            "touches_border_1px_count": sum(row["touches_border_1px"] for row in split_boxes),
        }

    report["invalid_label_records"] = invalid_labels

    def overlap_groups(key: str) -> dict[str, object]:
        groups: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for row in all_images:
            groups[str(row[key])][str(row["split"])] += 1
        cross = {name: dict(counts) for name, counts in groups.items() if len(counts) > 1}
        return {
            "unique_groups": len(groups),
            "cross_split_groups": len(cross),
            "cross_split_images": sum(sum(counts.values()) for counts in cross.values()),
            "examples": dict(list(sorted(cross.items()))[:25]),
        }

    report["exact_sha256_overlap"] = overlap_groups("sha256")
    report["roboflow_base_name_overlap"] = overlap_groups("base_name")

    sha_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in all_images:
        sha_rows[str(row["sha256"])].append(row)
    pairwise: dict[str, dict[str, int]] = {}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        shared = [
            rows for rows in sha_rows.values()
            if any(row["split"] == left for row in rows) and any(row["split"] == right for row in rows)
        ]
        pairwise[f"{left}_{right}"] = {
            "shared_sha256_groups": len(shared),
            f"{left}_images_with_match": sum(
                sum(row["split"] == left for row in rows) for rows in shared
            ),
            f"{right}_images_with_match": sum(
                sum(row["split"] == right for row in rows) for rows in shared
            ),
            "groups_with_identical_labels": sum(
                len({str(row["label_sha256"]) for row in rows if row["split"] in {left, right}}) == 1
                for rows in shared
            ),
            "groups_with_different_labels": sum(
                len({str(row["label_sha256"]) for row in rows if row["split"] in {left, right}}) > 1
                for rows in shared
            ),
        }
    report["exact_sha256_pairwise"] = pairwise
    report["within_split_exact_duplicates"] = {
        split: {
            "images": sum(row["split"] == split for row in all_images),
            "unique_sha256": len({str(row["sha256"]) for row in all_images if row["split"] == split}),
            "duplicate_copies": sum(row["split"] == split for row in all_images)
            - len({str(row["sha256"]) for row in all_images if row["split"] == split}),
        }
        for split in SPLITS
    }
    train_sha_counts = Counter(
        str(row["sha256"]) for row in all_images if row["split"] == "train"
    )
    repeated_train_paths = {
        str(row["path"])
        for row in all_images
        if row["split"] == "train" and train_sha_counts[str(row["sha256"])] > 1
    }
    oversampling_groups: dict[str, dict[str, object]] = {}
    for name, predicate in (
        ("images_in_repeated_sha_groups", lambda path: path in repeated_train_paths),
        ("images_with_unique_sha", lambda path: path not in repeated_train_paths),
    ):
        boxes = [
            row for row in all_boxes
            if row["split"] == "train" and predicate(str(row["path"]))
        ]
        counts = Counter(str(row["size_group"]) for row in boxes)
        oversampling_groups[name] = {
            "images": sum(
                row["split"] == "train" and predicate(str(row["path"])) for row in all_images
            ),
            "instances": len(boxes),
            "size_counts": dict(counts),
            "tiny_small_fraction": (
                (counts.get("tiny", 0) + counts.get("small", 0)) / len(boxes) if boxes else None
            ),
        }
    report["train_exact_repeat_size_audit"] = oversampling_groups
    train_base_counts = Counter(
        str(row["base_name"]) for row in all_images if row["split"] == "train"
    )
    repeated_base_paths = {
        str(row["path"])
        for row in all_images
        if row["split"] == "train" and train_base_counts[str(row["base_name"])] > 1
    }
    base_groups: dict[str, dict[str, object]] = {}
    for name, predicate in (
        ("images_in_repeated_base_groups", lambda path: path in repeated_base_paths),
        ("images_with_unique_base", lambda path: path not in repeated_base_paths),
    ):
        boxes = [
            row for row in all_boxes
            if row["split"] == "train" and predicate(str(row["path"]))
        ]
        counts = Counter(str(row["size_group"]) for row in boxes)
        base_groups[name] = {
            "images": sum(
                row["split"] == "train" and predicate(str(row["path"])) for row in all_images
            ),
            "instances": len(boxes),
            "size_counts": dict(counts),
            "tiny_small_fraction": (
                (counts.get("tiny", 0) + counts.get("small", 0)) / len(boxes) if boxes else None
            ),
        }
    report["train_roboflow_base_repeat_size_audit"] = base_groups

    video_rows = [row for row in all_images if row["video"] is not None]
    video_splits: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in video_rows:
        video_splits[str(row["video"])][str(row["split"])].append(int(row["frame"]))
    cross_videos = {video: frames for video, frames in video_splits.items() if len(frames) > 1}
    proximity: dict[str, dict[str, int | float | None]] = {}
    for target_split in ("valid", "test"):
        distances: list[int] = []
        for video, frames_by_split in video_splits.items():
            train_frames = frames_by_split.get("train", [])
            target_frames = frames_by_split.get(target_split, [])
            if not train_frames or not target_frames:
                continue
            train_array = np.asarray(train_frames)
            distances.extend(int(np.min(np.abs(train_array - frame))) for frame in target_frames)
        proximity[target_split] = {
            "frames_with_train_video_source": len(distances),
            "min_frame_distance": min(distances) if distances else None,
            "median_min_frame_distance": float(np.median(distances)) if distances else None,
            "distance_le_1": sum(distance <= 1 for distance in distances),
            "distance_le_5": sum(distance <= 5 for distance in distances),
            "distance_le_10": sum(distance <= 10 for distance in distances),
        }
    report["video_frame_audit"] = {
        "matched_video_images": len(video_rows),
        "unique_videos": len(video_splits),
        "videos_in_multiple_splits": len(cross_videos),
        "video_split_counts": {
            video: {split: len(frames) for split, frames in frames_by_split.items()}
            for video, frames_by_split in sorted(cross_videos.items())
        },
        "train_proximity": proximity,
    }

    all_size_counts = Counter(row["size_group"] for row in all_boxes)
    report["overall"] = {
        "images": len(all_images),
        "instances": len(all_boxes),
        "size_counts": dict(all_size_counts),
        "size_fractions": {key: value / len(all_boxes) for key, value in all_size_counts.items()},
        "lt8x8_count": sum(row["lt8x8"] for row in all_boxes),
        "lt8x8_fraction": sum(row["lt8x8"] for row in all_boxes) / len(all_boxes),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=dict) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
