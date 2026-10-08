#!/usr/bin/env python3
"""Compare a recovered Roboflow export with the current DetectDataset tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


SPLITS = ("train", "valid", "test")
RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$")
VIDEO_FRAME = re.compile(r"^(video\d+)_(\d+)(?:_|$)", re.IGNORECASE)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def inventory(root: Path) -> list[dict[str, object]]:
    rows = []
    for split in SPLITS:
        for path in sorted((root / split / "images").iterdir()):
            if not path.is_file():
                continue
            with Image.open(path) as image:
                width, height = image.size
            label = root / split / "labels" / f"{path.stem}.txt"
            text = label.read_text(encoding="utf-8").strip() if label.exists() else ""
            lines = [line for line in text.splitlines() if line.strip()]
            base = RF_SUFFIX.sub("", path.stem)
            match = VIDEO_FRAME.match(base)
            lt8 = 0
            for line in lines:
                fields = line.split()
                if len(fields) == 5:
                    _, _, _, bw, bh = map(float, fields)
                    lt8 += int(bw * 640 < 8 and bh * 640 < 8)
            rows.append({
                "split": split,
                "relative_image": str(path.relative_to(root)),
                "relative_label": str(label.relative_to(root)),
                "image_sha256": digest(path),
                "label_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "label_text": text,
                "instances": len(lines),
                "width": width,
                "height": height,
                "base": base,
                "video": match.group(1).lower() if match else None,
                "frame": int(match.group(2)) if match else None,
                "lt8x8": lt8,
            })
    return rows


def audit(rows: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {"splits": {}}
    for split in SPLITS:
        subset = [row for row in rows if row["split"] == split]
        counts = Counter(int(row["instances"]) for row in subset)
        result["splits"][split] = {
            "images": len(subset),
            "instances": sum(int(row["instances"]) for row in subset),
            "empty_images": counts[0],
            "multi_instance_images": sum(value for count, value in counts.items() if count > 1),
            "max_instances": max(counts, default=0),
            "dimensions": dict(Counter(f"{row['width']}x{row['height']}" for row in subset)),
            "lt8x8_boxes": sum(int(row["lt8x8"]) for row in subset),
        }

    sha_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        sha_groups[str(row["image_sha256"])].append(row)
    result["within_split_exact_duplicates"] = {}
    for split in SPLITS:
        subset = [row for row in rows if row["split"] == split]
        result["within_split_exact_duplicates"][split] = {
            "images": len(subset),
            "unique_hashes": len({str(row["image_sha256"]) for row in subset}),
            "duplicate_copies": len(subset) - len({str(row["image_sha256"]) for row in subset}),
        }
    pairwise = {}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        shared = [
            group for group in sha_groups.values()
            if any(row["split"] == left for row in group) and any(row["split"] == right for row in group)
        ]
        pairwise[f"{left}_{right}"] = {
            "shared_hash_groups": len(shared),
            f"{left}_images": sum(sum(row["split"] == left for row in group) for group in shared),
            f"{right}_images": sum(sum(row["split"] == right for row in group) for group in shared),
            "same_labels": sum(
                len({str(row["label_sha256"]) for row in group if row["split"] in {left, right}}) == 1
                for group in shared
            ),
            "different_labels": sum(
                len({str(row["label_sha256"]) for row in group if row["split"] in {left, right}}) > 1
                for group in shared
            ),
            "examples": [
                [str(row["relative_image"]) for row in group if row["split"] in {left, right}]
                for group in shared[:5]
            ],
        }
    result["cross_split_exact_duplicates"] = pairwise

    train_frames: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        if row["split"] == "train" and row["video"] is not None:
            train_frames[str(row["video"])].append(int(row["frame"]))
    distance_counts: Counter[int] = Counter()
    test_video_rows = 0
    for row in rows:
        if row["split"] != "test" or row["video"] is None:
            continue
        candidates = train_frames.get(str(row["video"]), [])
        if candidates:
            test_video_rows += 1
            distance_counts[min(abs(int(row["frame"]) - value) for value in candidates)] += 1
    result["test_to_train_video_frame_distance"] = {
        "test_video_images_with_train_source": test_video_rows,
        "distance_0": distance_counts[0],
        "distance_1": distance_counts[1],
        "distance_2_to_5": sum(distance_counts[value] for value in range(2, 6)),
        "distance_gt5": sum(count for distance, count in distance_counts.items() if distance > 5),
    }
    video_splits: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row["video"] is not None:
            video_splits[str(row["video"])][str(row["split"])] += 1
    result["video_split_counts"] = {video: dict(counts) for video, counts in sorted(video_splits.items())}
    return result


def compare(original: list[dict[str, object]], current: list[dict[str, object]]) -> dict[str, object]:
    original_by_path = {str(row["relative_image"]): row for row in original}
    current_by_path = {str(row["relative_image"]): row for row in current}
    shared_paths = sorted(original_by_path.keys() & current_by_path.keys())
    extra_paths = sorted(current_by_path.keys() - original_by_path.keys())
    missing_paths = sorted(original_by_path.keys() - current_by_path.keys())
    return {
        "original_images": len(original),
        "current_images": len(current),
        "shared_relative_paths": len(shared_paths),
        "shared_images_same_sha256": sum(
            original_by_path[path]["image_sha256"] == current_by_path[path]["image_sha256"]
            for path in shared_paths
        ),
        "shared_labels_same_sha256": sum(
            original_by_path[path]["label_sha256"] == current_by_path[path]["label_sha256"]
            for path in shared_paths
        ),
        "original_paths_missing_from_current": missing_paths,
        "current_extra_paths_count": len(extra_paths),
        "current_extra_paths": extra_paths,
        "current_extra_split_counts": dict(Counter(path.split("/", 1)[0] for path in extra_paths)),
        "current_extra_dimensions": dict(Counter(
            f"{current_by_path[path]['width']}x{current_by_path[path]['height']}" for path in extra_paths
        )),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = inventory(args.original)
    current = inventory(args.current)
    report = {
        "original_root": str(args.original),
        "current_root": str(args.current),
        "original_audit": audit(original),
        "current_audit": audit(current),
        "comparison": compare(original, current),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
